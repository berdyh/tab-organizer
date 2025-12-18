"""Hybrid clustering pipeline for tab organization."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse


@dataclass
class Tab:
    """Represents a browser tab."""
    url: str
    title: str = ""
    content: str = ""
    embedding: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Cluster:
    """Represents a cluster of related tabs."""
    id: int
    name: str = ""
    tabs: list[Tab] = field(default_factory=list)
    subclusters: list["Cluster"] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)


class TabClusterer:
    """Hybrid clustering pipeline for organizing browser tabs."""
    
    def __init__(
        self,
        min_cluster_size: int = 3,
        min_samples: int = 2,
        cluster_selection_epsilon: float = 0.5,
        umap_n_neighbors: int = 15,
        umap_n_components: int = 5,
        umap_min_dist: float = 0.1,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.umap_min_dist = umap_min_dist
        self._llm_client = None
    
    def set_llm_client(self, client) -> None:
        """Set LLM client for label generation."""
        self._llm_client = client
    
    def extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    
    def group_by_domain(self, tabs: list[Tab]) -> dict[str, list[Tab]]:
        """Group tabs by their domain."""
        groups: dict[str, list[Tab]] = {}
        for tab in tabs:
            domain = self.extract_domain(tab.url)
            if domain not in groups:
                groups[domain] = []
            groups[domain].append(tab)
        return groups
    
    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce embedding dimensions using UMAP."""
        try:
            import umap
            
            n_samples = embeddings.shape[0]
            n_neighbors = min(self.umap_n_neighbors, n_samples - 1)
            
            if n_samples < 4:
                # Too few samples for UMAP, return as-is or truncate
                return embeddings[:, :self.umap_n_components] if embeddings.shape[1] > self.umap_n_components else embeddings
            
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=min(self.umap_n_components, n_samples - 1),
                min_dist=self.umap_min_dist,
                metric='cosine',
                random_state=42,
            )
            return reducer.fit_transform(embeddings)
        except ImportError:
            # Fallback: simple PCA-like reduction using SVD
            if embeddings.shape[1] <= self.umap_n_components:
                return embeddings
            
            # Center the data
            centered = embeddings - np.mean(embeddings, axis=0)
            # SVD
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            # Project to lower dimensions
            return U[:, :self.umap_n_components] * S[:self.umap_n_components]
    
    def cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster embeddings using HDBSCAN."""
        try:
            import hdbscan
            
            n_samples = embeddings.shape[0]
            min_cluster_size = min(self.min_cluster_size, max(2, n_samples // 3))
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min(self.min_samples, min_cluster_size),
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                metric='euclidean',
            )
            return clusterer.fit_predict(embeddings)
        except ImportError:
            # Fallback: simple k-means clustering
            return self._kmeans_cluster(embeddings)
    
    def _kmeans_cluster(self, embeddings: np.ndarray, max_clusters: int = 10) -> np.ndarray:
        """Simple k-means clustering fallback."""
        n_samples = embeddings.shape[0]
        
        if n_samples < 3:
            return np.zeros(n_samples, dtype=int)
        
        # Estimate number of clusters using elbow method approximation
        k = min(max(2, n_samples // 5), max_clusters)
        
        # Initialize centroids randomly
        np.random.seed(42)
        indices = np.random.choice(n_samples, k, replace=False)
        centroids = embeddings[indices].copy()
        
        # Iterate
        for _ in range(100):
            # Assign points to nearest centroid
            distances = np.zeros((n_samples, k))
            for i, centroid in enumerate(centroids):
                distances[:, i] = np.linalg.norm(embeddings - centroid, axis=1)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = labels == i
                if np.any(mask):
                    new_centroids[i] = embeddings[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        return labels
    
    def _group_by_labels(self, tabs: list[Tab], labels: np.ndarray) -> list[Cluster]:
        """Group tabs by cluster labels."""
        clusters: dict[int, list[Tab]] = {}
        
        for tab, label in zip(tabs, labels):
            if label == -1:
                # Noise point - create individual cluster or skip
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(tab)
        
        # Handle noise points (label == -1)
        noise_tabs = [tab for tab, label in zip(tabs, labels) if label == -1]
        
        result = []
        for label, cluster_tabs in sorted(clusters.items()):
            cluster = Cluster(
                id=label,
                tabs=cluster_tabs,
            )
            # Calculate centroid
            embeddings = [t.embedding for t in cluster_tabs if t.embedding is not None]
            if embeddings:
                cluster.centroid = np.mean(embeddings, axis=0)
            result.append(cluster)
        
        # Add unclustered tabs as individual clusters
        for i, tab in enumerate(noise_tabs):
            result.append(Cluster(
                id=len(clusters) + i,
                name="Uncategorized",
                tabs=[tab],
            ))
        
        return result
    
    async def generate_cluster_label(self, cluster: Cluster) -> str:
        """Generate a descriptive label for a cluster using LLM."""
        if not self._llm_client:
            # Fallback: use domain or first title
            if cluster.tabs:
                domains = set(self.extract_domain(t.url) for t in cluster.tabs)
                if len(domains) == 1:
                    return list(domains)[0]
                titles = [t.title for t in cluster.tabs[:3] if t.title]
                if titles:
                    return titles[0][:50]
            return f"Cluster {cluster.id}"
        
        # Build context from tabs
        tab_summaries = []
        for tab in cluster.tabs[:10]:  # Limit to 10 tabs
            summary = f"- {tab.title or tab.url}"
            if tab.content:
                summary += f": {tab.content[:200]}..."
            tab_summaries.append(summary)
        
        prompt = f"""Generate a short, descriptive label (2-5 words) for a group of related browser tabs.

Tabs in this group:
{chr(10).join(tab_summaries)}

Respond with ONLY the label, nothing else. Examples: "Python Async Programming", "React Hooks Tutorial", "Machine Learning Papers"
"""
        
        try:
            label = await self._llm_client.generate(prompt)
            return label.strip().strip('"\'')[:50]
        except Exception:
            return f"Cluster {cluster.id}"
    
    async def cluster(self, tabs: list[Tab]) -> list[Cluster]:
        """
        Main clustering pipeline.
        
        1. Generate embeddings (if not present)
        2. Reduce dimensions with UMAP
        3. Cluster with HDBSCAN
        4. Generate labels with LLM
        5. Optionally sub-cluster large clusters
        """
        if not tabs:
            return []
        
        # Ensure all tabs have embeddings
        tabs_with_embeddings = [t for t in tabs if t.embedding is not None]
        if not tabs_with_embeddings:
            # Need to generate embeddings first
            if self._llm_client:
                contents = [t.content or t.title or t.url for t in tabs]
                embeddings = await self._llm_client.embed(contents)
                for tab, emb in zip(tabs, embeddings):
                    tab.embedding = np.array(emb)
                tabs_with_embeddings = tabs
            else:
                # Cannot cluster without embeddings
                return [Cluster(id=0, name="All Tabs", tabs=tabs)]
        
        # Stack embeddings
        embeddings = np.vstack([t.embedding for t in tabs_with_embeddings])
        
        # Reduce dimensions
        reduced = self.reduce_dimensions(embeddings)
        
        # Cluster
        labels = self.cluster_embeddings(reduced)
        
        # Group by labels
        clusters = self._group_by_labels(tabs_with_embeddings, labels)
        
        # Generate labels
        for cluster in clusters:
            if cluster.name != "Uncategorized":
                cluster.name = await self.generate_cluster_label(cluster)
        
        # Sub-cluster large clusters
        for cluster in clusters:
            if len(cluster.tabs) > 10:
                cluster.subclusters = await self.cluster(cluster.tabs)
        
        return clusters
    
    def cluster_sync(self, tabs: list[Tab], embeddings: np.ndarray) -> list[Cluster]:
        """
        Synchronous clustering (without LLM label generation).
        
        Useful when embeddings are pre-computed.
        """
        if len(tabs) != embeddings.shape[0]:
            raise ValueError("Number of tabs must match number of embeddings")
        
        # Assign embeddings to tabs
        for tab, emb in zip(tabs, embeddings):
            tab.embedding = emb
        
        # Reduce dimensions
        reduced = self.reduce_dimensions(embeddings)
        
        # Cluster
        labels = self.cluster_embeddings(reduced)
        
        # Group by labels
        clusters = self._group_by_labels(tabs, labels)
        
        # Generate simple labels based on domains
        for cluster in clusters:
            if cluster.name != "Uncategorized":
                domains = set(self.extract_domain(t.url) for t in cluster.tabs)
                if len(domains) == 1:
                    cluster.name = list(domains)[0]
                else:
                    # Use most common domain
                    domain_counts = {}
                    for t in cluster.tabs:
                        d = self.extract_domain(t.url)
                        domain_counts[d] = domain_counts.get(d, 0) + 1
                    cluster.name = max(domain_counts, key=domain_counts.get)
        
        return clusters
    
    def to_dict(self, clusters: list[Cluster]) -> list[dict]:
        """Convert clusters to dictionary format."""
        result = []
        for cluster in clusters:
            cluster_dict = {
                "id": cluster.id,
                "name": cluster.name,
                "urls": [
                    {
                        "url": t.url,
                        "title": t.title,
                        "metadata": t.metadata,
                    }
                    for t in cluster.tabs
                ],
                "tab_count": len(cluster.tabs),
            }
            if cluster.subclusters:
                cluster_dict["subclusters"] = self.to_dict(cluster.subclusters)
            result.append(cluster_dict)
        return result
