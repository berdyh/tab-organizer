"""URL and content deduplication utilities."""

import hashlib
import re
from typing import Optional
from urllib.parse import urlparse

import numpy as np


class ContentDeduplicator:
    """Detect duplicate and near-duplicate content."""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self._content_hashes: dict[str, str] = {}  # hash → url
        self._embeddings: dict[str, np.ndarray] = {}  # url → embedding
    
    def compute_content_hash(self, content: str) -> str:
        """Compute SHA-256 hash of content."""
        # Normalize whitespace before hashing
        normalized = re.sub(r'\s+', ' ', content.strip())
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def compute_simhash(self, content: str, num_bits: int = 64) -> int:
        """
        Compute SimHash for near-duplicate detection.
        
        SimHash is a locality-sensitive hash that produces similar
        hashes for similar content.
        """
        # Tokenize content
        tokens = re.findall(r'\w+', content.lower())
        
        # Initialize bit vector
        v = [0] * num_bits
        
        for token in tokens:
            # Get hash of token
            token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
            
            # Update bit vector
            for i in range(num_bits):
                if token_hash & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        
        # Convert to fingerprint
        fingerprint = 0
        for i in range(num_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        
        return fingerprint
    
    def hamming_distance(self, hash1: int, hash2: int, num_bits: int = 64) -> int:
        """Compute Hamming distance between two hashes."""
        xor = hash1 ^ hash2
        return bin(xor).count('1')
    
    def is_near_duplicate_simhash(
        self, 
        hash1: int, 
        hash2: int, 
        threshold: int = 3
    ) -> bool:
        """Check if two SimHashes indicate near-duplicate content."""
        return self.hamming_distance(hash1, hash2) <= threshold
    
    def check_exact_duplicate(self, content: str, url: str) -> Optional[str]:
        """
        Check if content is an exact duplicate.
        
        Returns:
            URL of duplicate if found, None otherwise.
        """
        content_hash = self.compute_content_hash(content)
        
        if content_hash in self._content_hashes:
            return self._content_hashes[content_hash]
        
        self._content_hashes[content_hash] = url
        return None
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def check_near_duplicate_embedding(
        self, 
        embedding: np.ndarray, 
        url: str
    ) -> Optional[tuple[str, float]]:
        """
        Check if embedding indicates near-duplicate content.
        
        Returns:
            Tuple of (duplicate_url, similarity) if found, None otherwise.
        """
        for existing_url, existing_embedding in self._embeddings.items():
            similarity = self.cosine_similarity(embedding, existing_embedding)
            if similarity >= self.similarity_threshold:
                return existing_url, similarity
        
        self._embeddings[url] = embedding
        return None
    
    def add_embedding(self, url: str, embedding: np.ndarray) -> None:
        """Add an embedding for a URL."""
        self._embeddings[url] = embedding
    
    def find_similar(
        self, 
        embedding: np.ndarray, 
        top_k: int = 10,
        min_similarity: float = 0.5
    ) -> list[tuple[str, float]]:
        """
        Find similar URLs based on embedding similarity.
        
        Returns:
            List of (url, similarity) tuples, sorted by similarity descending.
        """
        similarities = []
        
        for url, existing_embedding in self._embeddings.items():
            similarity = self.cosine_similarity(embedding, existing_embedding)
            if similarity >= min_similarity:
                similarities.append((url, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def clear(self) -> None:
        """Clear all stored hashes and embeddings."""
        self._content_hashes.clear()
        self._embeddings.clear()


class URLDeduplicator:
    """URL-level deduplication utilities."""
    
    @staticmethod
    def extract_domain(url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    
    @staticmethod
    def extract_base_domain(url: str) -> str:
        """Extract base domain (without subdomains) from URL."""
        domain = URLDeduplicator.extract_domain(url)
        parts = domain.split(".")
        
        # Handle common TLDs
        if len(parts) >= 2:
            # Check for two-part TLDs like .co.uk, .com.au
            two_part_tlds = {"co.uk", "com.au", "co.nz", "co.jp", "com.br"}
            if len(parts) >= 3:
                potential_tld = f"{parts[-2]}.{parts[-1]}"
                if potential_tld in two_part_tlds:
                    return f"{parts[-3]}.{potential_tld}"
            return f"{parts[-2]}.{parts[-1]}"
        
        return domain
    
    @staticmethod
    def is_same_page(url1: str, url2: str) -> bool:
        """Check if two URLs point to the same page (ignoring fragments)."""
        parsed1 = urlparse(url1)
        parsed2 = urlparse(url2)
        
        return (
            parsed1.scheme == parsed2.scheme and
            parsed1.netloc.lower() == parsed2.netloc.lower() and
            parsed1.path.rstrip("/") == parsed2.path.rstrip("/") and
            parsed1.query == parsed2.query
        )
    
    @staticmethod
    def group_by_domain(urls: list[str]) -> dict[str, list[str]]:
        """Group URLs by their domain."""
        groups: dict[str, list[str]] = {}
        
        for url in urls:
            domain = URLDeduplicator.extract_domain(url)
            if domain not in groups:
                groups[domain] = []
            groups[domain].append(url)
        
        return groups
