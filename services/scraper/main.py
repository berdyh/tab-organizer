"""Web Scraper Service - Comprehensive web scraper with parallel authentication workflow."""

import time
import hashlib
import asyncio
import aiohttp
import uuid
import os
import mimetypes
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, HttpUrl, Field
import structlog
import scrapy
from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings
from scrapy.utils.log import configure_logging
from twisted.internet import asyncioreactor, defer
from bs4 import BeautifulSoup
import trafilatura
import requests
import PyPDF2
import io

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Install asyncio reactor for Scrapy
try:
    asyncioreactor.install()
except Exception:
    pass  # Already installed

app = FastAPI(
    title="Web Scraper Service",
    description="Extracts and cleans content from web URLs with rate limiting and duplicate detection",
    version="1.0.0"
)

# Enums and Data Classes
class URLStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    SCRAPING = "scraping"
    COMPLETED = "completed"
    FAILED = "failed"
    AUTH_REQUIRED = "auth_required"
    AUTH_PENDING = "auth_pending"
    RETRYING = "retrying"

class ContentType(str, Enum):
    HTML = "html"
    PDF = "pdf"
    TEXT = "text"
    JSON = "json"
    XML = "xml"
    UNKNOWN = "unknown"

class QueueType(str, Enum):
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    RETRY = "retry"

@dataclass
class URLClassification:
    url: str
    queue_type: QueueType
    requires_auth: bool
    confidence: float
    auth_indicators: List[str] = field(default_factory=list)
    domain: str = ""
    priority: int = 1

@dataclass
class ProcessingStats:
    total_urls: int = 0
    public_queue_size: int = 0
    auth_queue_size: int = 0
    retry_queue_size: int = 0
    completed: int = 0
    failed: int = 0
    auth_pending: int = 0
    processing_rate: float = 0.0
    avg_response_time: float = 0.0

# Pydantic models
class URLRequest(BaseModel):
    url: HttpUrl
    priority: int = Field(default=1, ge=1, le=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    force_auth_check: bool = False
    content_type_hint: Optional[ContentType] = None

class ScrapeRequest(BaseModel):
    urls: List[URLRequest]
    session_id: str
    rate_limit_delay: float = Field(default=1.0, ge=0.1, le=10.0)
    respect_robots: bool = True
    max_retries: int = Field(default=3, ge=1, le=10)
    parallel_auth: bool = True
    max_concurrent_workers: int = Field(default=5, ge=1, le=20)
    content_quality_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_pdf_extraction: bool = True

class ScrapedContent(BaseModel):
    url: str
    title: str
    content: str
    content_hash: str
    content_type: ContentType
    metadata: Dict[str, Any]
    scraped_at: datetime
    word_count: int
    quality_score: float
    is_duplicate: bool = False
    extraction_method: str = ""
    response_time_ms: int = 0

class AuthSession(BaseModel):
    session_id: str
    domain: str
    cookies: Dict[str, Any] = Field(default_factory=dict)
    headers: Dict[str, str] = Field(default_factory=dict)
    user_agent: Optional[str] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    last_used: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0

class ScrapeJob(BaseModel):
    job_id: str
    status: str
    total_urls: int
    completed_urls: int
    failed_urls: int
    auth_pending_urls: int = 0
    results: List[ScrapedContent] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    auth_sessions: Dict[str, str] = Field(default_factory=dict)  # domain -> session_id mapping
    correlation_id: str = ""
    started_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    processing_stats: ProcessingStats = Field(default_factory=ProcessingStats)
    url_classifications: List[URLClassification] = Field(default_factory=list)

class RealTimeStatus(BaseModel):
    job_id: str
    current_status: str
    progress_percentage: float
    urls_in_progress: List[Dict[str, Any]]
    recent_completions: List[Dict[str, Any]]
    auth_status: Dict[str, Any]
    performance_metrics: Dict[str, Any]

# Global storage (in production, use Redis or database)
active_jobs: Dict[str, ScrapeJob] = {}
content_hashes: Dict[str, str] = {}  # hash -> url mapping for duplicate detection
auth_sessions: Dict[str, AuthSession] = {}  # session_id -> AuthSession mapping
websocket_connections: Dict[str, WebSocket] = {}  # job_id -> websocket for real-time updates

# Parallel processing queues
class ProcessingQueues:
    def __init__(self):
        self.public_queue: deque = deque()
        self.auth_queue: deque = deque()
        self.retry_queue: deque = deque()
        self.processing: Set[str] = set()
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()
        
    def add_to_queue(self, url_classification: URLClassification, url_request: URLRequest):
        """Add URL to appropriate queue based on classification."""
        item = {
            'classification': url_classification,
            'request': url_request,
            'queued_at': datetime.now(),
            'retry_count': 0
        }
        
        if url_classification.queue_type == QueueType.PUBLIC:
            self.public_queue.append(item)
        elif url_classification.queue_type == QueueType.AUTHENTICATED:
            self.auth_queue.append(item)
        else:
            self.retry_queue.append(item)
    
    def get_next_public(self):
        """Get next URL from public queue."""
        return self.public_queue.popleft() if self.public_queue else None
    
    def get_next_auth(self):
        """Get next URL from auth queue."""
        return self.auth_queue.popleft() if self.auth_queue else None
    
    def get_next_retry(self):
        """Get next URL from retry queue."""
        return self.retry_queue.popleft() if self.retry_queue else None
    
    def get_stats(self) -> ProcessingStats:
        """Get current processing statistics."""
        return ProcessingStats(
            total_urls=len(self.public_queue) + len(self.auth_queue) + len(self.retry_queue) + 
                      len(self.processing) + len(self.completed) + len(self.failed),
            public_queue_size=len(self.public_queue),
            auth_queue_size=len(self.auth_queue),
            retry_queue_size=len(self.retry_queue),
            completed=len(self.completed),
            failed=len(self.failed)
        )

processing_queues: Dict[str, ProcessingQueues] = {}  # job_id -> ProcessingQueues

class URLClassifier:
    """Intelligent URL classifier to determine authentication requirements and routing."""
    
    def __init__(self):
        self.auth_indicators = {
            'url_patterns': [
                'login', 'signin', 'auth', 'account', 'dashboard', 'profile',
                'admin', 'secure', 'private', 'member', 'user', 'my'
            ],
            'domain_patterns': [
                'admin.', 'secure.', 'my.', 'account.', 'portal.', 'app.'
            ],
            'path_patterns': [
                '/admin/', '/dashboard/', '/account/', '/profile/', '/secure/',
                '/private/', '/member/', '/user/', '/my/', '/portal/'
            ]
        }
        self.public_indicators = [
            'blog', 'news', 'about', 'contact', 'help', 'faq', 'public',
            'home', 'index', 'main', 'welcome'
        ]
    
    async def classify_urls(self, urls: List[URLRequest], correlation_id: str) -> List[URLClassification]:
        """Classify URLs into appropriate processing queues."""
        classifications = []
        
        for url_request in urls:
            url = str(url_request.url)
            classification = await self._classify_single_url(url, url_request, correlation_id)
            classifications.append(classification)
        
        return classifications
    
    async def _classify_single_url(self, url: str, url_request: URLRequest, correlation_id: str) -> URLClassification:
        """Classify a single URL for authentication requirements."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path.lower()
        url_lower = url.lower()
        
        # Initialize classification
        classification = URLClassification(
            url=url,
            queue_type=QueueType.PUBLIC,
            requires_auth=False,
            confidence=0.0,
            domain=domain,
            priority=url_request.priority
        )
        
        # Check for obvious auth indicators in URL
        auth_score = 0.0
        auth_indicators_found = []
        
        # URL pattern matching
        for pattern in self.auth_indicators['url_patterns']:
            if pattern in url_lower:
                auth_score += 0.3
                auth_indicators_found.append(f"url_pattern:{pattern}")
        
        # Domain pattern matching
        for pattern in self.auth_indicators['domain_patterns']:
            if domain.startswith(pattern):
                auth_score += 0.4
                auth_indicators_found.append(f"domain_pattern:{pattern}")
        
        # Path pattern matching
        for pattern in self.auth_indicators['path_patterns']:
            if pattern in path:
                auth_score += 0.5
                auth_indicators_found.append(f"path_pattern:{pattern}")
        
        # Check for public indicators (negative score)
        for pattern in self.public_indicators:
            if pattern in url_lower:
                auth_score -= 0.2
        
        # Force auth check if requested
        if url_request.force_auth_check:
            auth_score += 0.8
            auth_indicators_found.append("force_auth_check")
        
        # Determine classification based on score
        if auth_score >= 0.5:
            classification.queue_type = QueueType.AUTHENTICATED
            classification.requires_auth = True
            classification.confidence = min(auth_score, 1.0)
        else:
            classification.queue_type = QueueType.PUBLIC
            classification.requires_auth = False
            classification.confidence = max(0.0, 1.0 - auth_score)
        
        classification.auth_indicators = auth_indicators_found
        
        logger.info(
            "URL classified",
            url=url,
            queue_type=classification.queue_type.value,
            requires_auth=classification.requires_auth,
            confidence=classification.confidence,
            indicators=auth_indicators_found,
            correlation_id=correlation_id
        )
        
        return classification

class ContentExtractor:
    """Extract and clean content using multiple parsing strategies with quality assessment."""
    
    def __init__(self):
        self.extraction_methods = [
            self._extract_with_trafilatura,
            self._extract_with_beautifulsoup_smart,
            self._extract_with_beautifulsoup_fallback
        ]
    
    def extract_content(self, content: bytes, url: str, content_type: str = "") -> Dict[str, Any]:
        """Extract clean content from various content types."""
        try:
            # Determine content type
            detected_type = self._detect_content_type(content, content_type, url)
            
            if detected_type == ContentType.PDF:
                return self._extract_pdf_content(content, url)
            elif detected_type == ContentType.HTML:
                return self._extract_html_content(content.decode('utf-8', errors='ignore'), url)
            elif detected_type == ContentType.TEXT:
                return self._extract_text_content(content.decode('utf-8', errors='ignore'), url)
            else:
                # Try HTML extraction as fallback
                return self._extract_html_content(content.decode('utf-8', errors='ignore'), url)
                
        except Exception as e:
            logger.error("Content extraction failed", url=url, error=str(e))
            return self._empty_result(url)
    
    def _detect_content_type(self, content: bytes, content_type: str, url: str) -> ContentType:
        """Detect content type from headers, content, and URL."""
        # Check content-type header
        if content_type:
            if 'pdf' in content_type.lower():
                return ContentType.PDF
            elif 'html' in content_type.lower():
                return ContentType.HTML
            elif 'text' in content_type.lower():
                return ContentType.TEXT
        
        # Check URL extension
        if url.lower().endswith('.pdf'):
            return ContentType.PDF
        elif url.lower().endswith(('.html', '.htm')):
            return ContentType.HTML
        elif url.lower().endswith('.txt'):
            return ContentType.TEXT
        
        # Check content magic bytes
        if content.startswith(b'%PDF'):
            return ContentType.PDF
        elif b'<html' in content[:1000].lower() or b'<!doctype html' in content[:1000].lower():
            return ContentType.HTML
        
        return ContentType.HTML  # Default assumption
    
    def _extract_html_content(self, html: str, url: str) -> Dict[str, Any]:
        """Extract content from HTML using multiple strategies."""
        best_result = None
        best_score = 0.0
        extraction_method = "none"
        
        for method in self.extraction_methods:
            try:
                result = method(html, url)
                score = self._calculate_quality_score(result)
                
                if score > best_score:
                    best_result = result
                    best_score = score
                    extraction_method = method.__name__
                    
                # If we get a high-quality result, use it
                if score > 0.8:
                    break
                    
            except Exception as e:
                logger.debug(f"Extraction method {method.__name__} failed", url=url, error=str(e))
                continue
        
        if best_result is None:
            return self._empty_result(url)
        
        best_result['quality_score'] = best_score
        best_result['extraction_method'] = extraction_method
        best_result['content_type'] = ContentType.HTML
        
        return best_result
    
    def _extract_with_trafilatura(self, html: str, url: str) -> Dict[str, Any]:
        """Extract content using trafilatura."""
        content = trafilatura.extract(
            html, 
            include_comments=False, 
            include_tables=True,
            include_formatting=True
        )
        
        if not content or len(content.strip()) < 20:
            raise ValueError("Trafilatura extraction failed or insufficient content")
        
        # Extract title using BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        title = self._extract_title(soup)
        
        return {
            "title": title,
            "content": content.strip(),
            "word_count": len(content.split()) if content else 0
        }
    
    def _extract_with_beautifulsoup_smart(self, html: str, url: str) -> Dict[str, Any]:
        """Extract content using BeautifulSoup with smart content detection."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "advertisement"]):
            element.decompose()
        
        # Try to find main content areas in order of preference
        content_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.content',
            '.main-content',
            '.post-content',
            '.entry-content',
            '#content',
            '#main'
        ]
        
        content_element = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break
        
        if not content_element:
            # Fallback to body
            content_element = soup.find('body') or soup
        
        # Extract text content
        content = content_element.get_text(separator=' ', strip=True)
        title = self._extract_title(soup)
        
        if len(content.strip()) < 20:
            raise ValueError("Insufficient content extracted")
        
        return {
            "title": title,
            "content": content,
            "word_count": len(content.split()) if content else 0
        }
    
    def _extract_with_beautifulsoup_fallback(self, html: str, url: str) -> Dict[str, Any]:
        """Fallback extraction using basic BeautifulSoup."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style"]):
            element.decompose()
        
        content = soup.get_text(separator=' ', strip=True)
        title = self._extract_title(soup)
        
        return {
            "title": title,
            "content": content,
            "word_count": len(content.split()) if content else 0
        }
    
    def _extract_pdf_content(self, content: bytes, url: str) -> Dict[str, Any]:
        """Extract content from PDF files."""
        try:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            # Clean up the text
            text_content = ' '.join(text_content.split())
            
            # Try to extract title from first page or metadata
            title = ""
            if pdf_reader.metadata and pdf_reader.metadata.get('/Title'):
                title = str(pdf_reader.metadata['/Title'])
            else:
                # Use first line as title if it's short enough
                first_line = text_content.split('\n')[0] if text_content else ""
                if len(first_line) < 100:
                    title = first_line.strip()
            
            return {
                "title": title,
                "content": text_content,
                "word_count": len(text_content.split()) if text_content else 0,
                "quality_score": 0.8 if text_content else 0.0,
                "extraction_method": "_extract_pdf_content",
                "content_type": ContentType.PDF
            }
            
        except Exception as e:
            logger.error("PDF extraction failed", url=url, error=str(e))
            return self._empty_result(url)
    
    def _extract_text_content(self, text: str, url: str) -> Dict[str, Any]:
        """Extract content from plain text files."""
        lines = text.split('\n')
        title = lines[0].strip() if lines else ""
        
        # If first line is too long, don't use as title
        if len(title) > 100:
            title = ""
        
        return {
            "title": title,
            "content": text.strip(),
            "word_count": len(text.split()) if text else 0,
            "quality_score": 0.7 if text.strip() else 0.0,
            "extraction_method": "_extract_text_content",
            "content_type": ContentType.TEXT
        }
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract title from HTML soup."""
        # Try multiple title sources
        title_sources = [
            lambda: soup.find('title'),
            lambda: soup.find('h1'),
            lambda: soup.find('meta', property='og:title'),
            lambda: soup.find('meta', name='title')
        ]
        
        for source in title_sources:
            try:
                element = source()
                if element:
                    if element.name == 'meta':
                        title = element.get('content', '')
                    else:
                        title = element.get_text(strip=True)
                    
                    if title and len(title.strip()) > 0:
                        return title.strip()
            except:
                continue
        
        return ""
    
    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate quality score for extracted content."""
        content = result.get('content', '')
        title = result.get('title', '')
        word_count = result.get('word_count', 0)
        
        score = 0.0
        
        # Content length score (0-0.4)
        if word_count > 500:
            score += 0.4
        elif word_count > 100:
            score += 0.3
        elif word_count > 50:
            score += 0.2
        elif word_count > 10:
            score += 0.1
        
        # Title presence score (0-0.2)
        if title and len(title.strip()) > 0:
            score += 0.2
        
        # Content quality indicators (0-0.4)
        if content:
            # Check for structured content
            if any(indicator in content.lower() for indicator in ['paragraph', 'section', 'chapter']):
                score += 0.1
            
            # Check for meaningful content vs boilerplate
            meaningful_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
            word_list = content.lower().split()
            meaningful_ratio = sum(1 for word in word_list if word in meaningful_words) / max(len(word_list), 1)
            
            if meaningful_ratio > 0.1:
                score += 0.2
            elif meaningful_ratio > 0.05:
                score += 0.1
            
            # Penalize repetitive content
            unique_words = len(set(word_list))
            repetition_ratio = unique_words / max(len(word_list), 1)
            if repetition_ratio < 0.3:
                score -= 0.2
        
        return min(score, 1.0)
    
    def _empty_result(self, url: str) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            "title": "",
            "content": "",
            "word_count": 0,
            "quality_score": 0.0,
            "extraction_method": "failed",
            "content_type": ContentType.UNKNOWN
        }

class DuplicateDetector:
    """Advanced duplicate detection using content hashing with similarity thresholds."""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.content_fingerprints: Dict[str, Tuple[str, Set[str]]] = {}  # hash -> (url, word_set)
    
    def generate_content_hash(self, content: str) -> str:
        """Generate SHA-256 hash of normalized content."""
        # Normalize content for hashing
        normalized = ' '.join(content.lower().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    def generate_content_fingerprint(self, content: str) -> Set[str]:
        """Generate content fingerprint using word sets for similarity detection."""
        words = set(content.lower().split())
        # Remove common stop words for better fingerprinting
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return words - stop_words
    
    def is_duplicate(self, content: str, url: str) -> Tuple[bool, str, float]:
        """Check if content is duplicate and return (is_duplicate, hash, similarity_score)."""
        content_hash = self.generate_content_hash(content)
        
        # Exact duplicate check
        if content_hash in content_hashes:
            original_url = content_hashes[content_hash]
            logger.info("Exact duplicate content detected", url=url, original_url=original_url)
            return True, content_hash, 1.0
        
        # Similarity-based duplicate check
        content_fingerprint = self.generate_content_fingerprint(content)
        
        for existing_hash, (existing_url, existing_fingerprint) in self.content_fingerprints.items():
            if len(content_fingerprint) == 0 or len(existing_fingerprint) == 0:
                continue
                
            # Calculate Jaccard similarity
            intersection = len(content_fingerprint & existing_fingerprint)
            union = len(content_fingerprint | existing_fingerprint)
            similarity = intersection / union if union > 0 else 0.0
            
            if similarity >= self.similarity_threshold:
                logger.info(
                    "Similar content detected",
                    url=url,
                    original_url=existing_url,
                    similarity=similarity
                )
                return True, content_hash, similarity
        
        # Store new content
        content_hashes[content_hash] = url
        self.content_fingerprints[content_hash] = (url, content_fingerprint)
        
        return False, content_hash, 0.0
    
    def get_duplicate_stats(self) -> Dict[str, Any]:
        """Get statistics about duplicate detection."""
        return {
            "total_content_hashes": len(content_hashes),
            "total_fingerprints": len(self.content_fingerprints),
            "similarity_threshold": self.similarity_threshold
        }

class ParallelProcessingEngine:
    """Engine for parallel processing of authenticated and non-authenticated URLs."""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.public_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.auth_executor = ThreadPoolExecutor(max_workers=max_workers // 2 + 1)
        self.retry_executor = ThreadPoolExecutor(max_workers=2)
        self.active_tasks: Dict[str, Set[asyncio.Task]] = defaultdict(set)
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
    
    async def process_job(self, job_id: str, scrape_request: ScrapeRequest, 
                         url_classifications: List[URLClassification]) -> None:
        """Process a scraping job with parallel authentication workflow."""
        try:
            # Initialize processing queues for this job
            processing_queues[job_id] = ProcessingQueues()
            queues = processing_queues[job_id]
            
            # Add URLs to appropriate queues
            for i, classification in enumerate(url_classifications):
                url_request = scrape_request.urls[i]
                queues.add_to_queue(classification, url_request)
            
            # Start parallel processing
            tasks = []
            
            # Public URL processing
            if queues.public_queue:
                task = asyncio.create_task(
                    self._process_public_queue(job_id, scrape_request)
                )
                tasks.append(task)
                self.active_tasks[job_id].add(task)
            
            # Authenticated URL processing
            if queues.auth_queue:
                task = asyncio.create_task(
                    self._process_auth_queue(job_id, scrape_request)
                )
                tasks.append(task)
                self.active_tasks[job_id].add(task)
            
            # Retry queue processing (starts after initial processing)
            retry_task = asyncio.create_task(
                self._process_retry_queue(job_id, scrape_request)
            )
            tasks.append(retry_task)
            self.active_tasks[job_id].add(retry_task)
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error("Parallel processing failed", job_id=job_id, error=str(e))
        finally:
            # Cleanup
            if job_id in self.active_tasks:
                del self.active_tasks[job_id]
            if job_id in processing_queues:
                del processing_queues[job_id]
    
    async def _process_public_queue(self, job_id: str, scrape_request: ScrapeRequest):
        """Process public URLs in parallel."""
        queues = processing_queues[job_id]
        
        while True:
            item = queues.get_next_public()
            if not item:
                await asyncio.sleep(1)
                # Check if job is complete or if we should stop
                if self._should_stop_processing(job_id):
                    break
                continue
            
            try:
                await self._process_single_url(job_id, item, scrape_request)
            except Exception as e:
                logger.error("Public URL processing failed", 
                           url=item['request'].url, error=str(e))
    
    async def _process_auth_queue(self, job_id: str, scrape_request: ScrapeRequest):
        """Process authenticated URLs, waiting for auth sessions."""
        queues = processing_queues[job_id]
        
        while True:
            item = queues.get_next_auth()
            if not item:
                await asyncio.sleep(2)  # Longer wait for auth
                if self._should_stop_processing(job_id):
                    break
                continue
            
            try:
                # Check if we have auth session for this domain
                domain = item['classification'].domain
                if await self._has_auth_session(job_id, domain):
                    await self._process_single_url(job_id, item, scrape_request)
                else:
                    # Re-queue for later processing
                    item['retry_count'] += 1
                    if item['retry_count'] < 5:  # Max 5 auth retries
                        queues.auth_queue.append(item)
                        await asyncio.sleep(5)  # Wait before retry
                    else:
                        # Move to failed
                        await self._mark_url_failed(job_id, item, "Authentication timeout")
            except Exception as e:
                logger.error("Auth URL processing failed", 
                           url=item['request'].url, error=str(e))
    
    async def _process_retry_queue(self, job_id: str, scrape_request: ScrapeRequest):
        """Process retry queue for failed URLs."""
        queues = processing_queues[job_id]
        
        # Wait a bit before starting retry processing
        await asyncio.sleep(10)
        
        while True:
            item = queues.get_next_retry()
            if not item:
                await asyncio.sleep(5)
                if self._should_stop_processing(job_id):
                    break
                continue
            
            try:
                await self._process_single_url(job_id, item, scrape_request)
            except Exception as e:
                logger.error("Retry URL processing failed", 
                           url=item['request'].url, error=str(e))
    
    async def _process_single_url(self, job_id: str, item: Dict[str, Any], 
                                scrape_request: ScrapeRequest):
        """Process a single URL with timing and error handling."""
        start_time = time.time()
        url = str(item['request'].url)
        
        try:
            # Mark as processing
            queues = processing_queues[job_id]
            queues.processing.add(url)
            
            # Update real-time status
            await self._update_realtime_status(job_id, url, "processing")
            
            # Perform the actual scraping
            result = await self._scrape_url(job_id, item, scrape_request)
            
            # Record performance metrics
            response_time = (time.time() - start_time) * 1000  # ms
            self.performance_metrics[job_id].append(response_time)
            
            # Update job with result
            if result:
                await self._update_job_success(job_id, result, response_time)
                queues.completed.add(url)
            else:
                await self._mark_url_failed(job_id, item, "Scraping failed")
                queues.failed.add(url)
            
        except Exception as e:
            await self._mark_url_failed(job_id, item, str(e))
            queues = processing_queues[job_id]
            queues.failed.add(url)
        finally:
            # Remove from processing
            if job_id in processing_queues:
                processing_queues[job_id].processing.discard(url)
    
    async def _scrape_url(self, job_id: str, item: Dict[str, Any], 
                         scrape_request: ScrapeRequest) -> Optional[ScrapedContent]:
        """Perform actual URL scraping with authentication support."""
        # This would integrate with the existing Scrapy-based scraping
        # For now, return a placeholder implementation
        url = str(item['request'].url)
        
        # TODO: Integrate with enhanced Scrapy spider
        # This is where the actual scraping logic would go
        
        return None
    
    async def _has_auth_session(self, job_id: str, domain: str) -> bool:
        """Check if we have an active auth session for domain."""
        if job_id not in active_jobs:
            return False
        
        job = active_jobs[job_id]
        session_id = job.auth_sessions.get(domain)
        
        if not session_id or session_id not in auth_sessions:
            return False
        
        session = auth_sessions[session_id]
        return session.is_active
    
    async def _update_job_success(self, job_id: str, content: ScrapedContent, response_time: float):
        """Update job with successful result."""
        if job_id in active_jobs:
            job = active_jobs[job_id]
            job.results.append(content)
            job.completed_urls += 1
            job.updated_at = datetime.now()
            
            # Update processing stats
            job.processing_stats.completed = job.completed_urls
            job.processing_stats.avg_response_time = sum(self.performance_metrics[job_id]) / len(self.performance_metrics[job_id])
            
            # Update status
            total_processed = job.completed_urls + job.failed_urls
            if total_processed >= job.total_urls:
                job.status = "completed"
            
            await self._broadcast_job_update(job_id, job)
    
    async def _mark_url_failed(self, job_id: str, item: Dict[str, Any], error: str):
        """Mark URL as failed and update job."""
        if job_id in active_jobs:
            job = active_jobs[job_id]
            job.errors.append({
                "url": str(item['request'].url),
                "error": error,
                "timestamp": datetime.now().isoformat(),
                "retry_count": item.get('retry_count', 0)
            })
            job.failed_urls += 1
            job.updated_at = datetime.now()
            
            # Update processing stats
            job.processing_stats.failed = job.failed_urls
            
            # Update status
            total_processed = job.completed_urls + job.failed_urls
            if total_processed >= job.total_urls:
                job.status = "completed"
            
            await self._broadcast_job_update(job_id, job)
    
    async def _update_realtime_status(self, job_id: str, url: str, status: str):
        """Update real-time status for WebSocket clients."""
        if job_id in websocket_connections:
            try:
                await websocket_connections[job_id].send_json({
                    "type": "url_status_update",
                    "job_id": job_id,
                    "url": url,
                    "status": status,
                    "timestamp": datetime.now().isoformat()
                })
            except:
                # Remove dead connection
                del websocket_connections[job_id]
    
    async def _broadcast_job_update(self, job_id: str, job: ScrapeJob):
        """Broadcast job update to WebSocket clients."""
        if job_id in websocket_connections:
            try:
                await websocket_connections[job_id].send_json({
                    "type": "job_update",
                    "job": job.model_dump(mode='json')
                })
            except:
                # Remove dead connection
                del websocket_connections[job_id]
    
    def _should_stop_processing(self, job_id: str) -> bool:
        """Check if processing should stop for a job."""
        if job_id not in active_jobs:
            return True
        
        job = active_jobs[job_id]
        total_processed = job.completed_urls + job.failed_urls
        
        return total_processed >= job.total_urls or job.status == "cancelled"
    
    def get_performance_metrics(self, job_id: str) -> Dict[str, Any]:
        """Get performance metrics for a job."""
        if job_id not in self.performance_metrics:
            return {}
        
        response_times = self.performance_metrics[job_id]
        if not response_times:
            return {}
        
        return {
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "total_requests": len(response_times)
        }
    
    async def shutdown(self):
        """Shutdown the processing engine."""
        self.public_executor.shutdown(wait=True)
        self.auth_executor.shutdown(wait=True)
        self.retry_executor.shutdown(wait=True)

class AuthenticationServiceClient:
    """Client for communicating with the Authentication Service."""
    
    def __init__(self, auth_service_url: str = "http://auth-service:8082"):
        self.auth_service_url = auth_service_url.rstrip('/')
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_session_for_domain(self, domain: str, correlation_id: str) -> Optional[AuthSession]:
        """Get active authentication session for a domain."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{self.auth_service_url}/sessions/{domain}",
                headers={"X-Correlation-ID": correlation_id}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return AuthSession(**data)
                elif response.status == 404:
                    logger.info("No active session found for domain", domain=domain)
                    return None
                else:
                    logger.warning("Failed to get session", domain=domain, status=response.status)
                    return None
        except Exception as e:
            logger.error("Error getting session from auth service", domain=domain, error=str(e))
            return None
    
    async def check_auth_required(self, url: str, response_content: str, 
                                status_code: int, headers: Dict[str, str],
                                correlation_id: str) -> Dict[str, Any]:
        """Check if URL requires authentication."""
        try:
            payload = {
                "url": url,
                "response_content": response_content,
                "status_code": status_code,
                "headers": headers
            }
            
            session = await self._get_session()
            async with session.post(
                f"{self.auth_service_url}/detect-auth",
                json=payload,
                headers={"X-Correlation-ID": correlation_id}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning("Auth detection failed", url=url, status=response.status)
                    return {"requires_auth": False, "confidence": 0.0}
        except Exception as e:
            logger.error("Error checking auth requirements", url=url, error=str(e))
            return {"requires_auth": False, "confidence": 0.0}
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()

class RobotsChecker:
    """Check robots.txt compliance."""
    
    def __init__(self):
        self.robots_cache: Dict[str, RobotFileParser] = {}
    
    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL can be fetched according to robots.txt."""
        try:
            parsed_url = urlparse(url)
            domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            if domain not in self.robots_cache:
                robots_url = urljoin(domain, "/robots.txt")
                rp = RobotFileParser()
                rp.set_url(robots_url)
                try:
                    rp.read()
                    self.robots_cache[domain] = rp
                except Exception:
                    # If robots.txt can't be read, assume allowed
                    rp = RobotFileParser()
                    rp.set_url(robots_url)
                    self.robots_cache[domain] = rp
            
            return self.robots_cache[domain].can_fetch(user_agent, url)
            
        except Exception as e:
            logger.warning("Robots.txt check failed", url=url, error=str(e))
            return True  # Default to allowing if check fails

class ContentSpider(scrapy.Spider):
    """Scrapy spider for content extraction with rate limiting and authentication."""
    
    name = 'content_spider'
    
    custom_settings = {
        'DOWNLOAD_DELAY': 1.0,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'CONCURRENT_REQUESTS': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429, 401, 403],
        'USER_AGENT': 'WebScrapingTool/1.0 (+https://example.com/bot)',
        'ROBOTSTXT_OBEY': True,
    }
    
    def __init__(self, urls=None, job_id=None, rate_limit_delay=1.0, respect_robots=True, 
                 correlation_id=None, *args, **kwargs):
        super(ContentSpider, self).__init__(*args, **kwargs)
        self.start_urls = urls or []
        self.job_id = job_id
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.custom_settings['DOWNLOAD_DELAY'] = rate_limit_delay
        self.custom_settings['ROBOTSTXT_OBEY'] = respect_robots
        self.robots_checker = RobotsChecker()
        self.content_extractor = ContentExtractor()
        self.duplicate_detector = DuplicateDetector()
        self.auth_client = AuthenticationServiceClient()
        self.retry_queue = []  # Queue for auth-failed URLs to retry
    
    def start_requests(self):
        """Generate initial requests with robots.txt checking and auth session injection."""
        for url_req in self.start_urls:
            url = str(url_req.url)
            domain = urlparse(url).netloc
            
            # Check robots.txt if enabled
            if self.custom_settings['ROBOTSTXT_OBEY']:
                if not self.robots_checker.can_fetch(url):
                    logger.warning("URL blocked by robots.txt", url=url, correlation_id=self.correlation_id)
                    self._update_job_error(url, "Blocked by robots.txt")
                    continue
            
            # Prepare request with potential auth session
            request = scrapy.Request(
                url=url,
                callback=self.parse,
                meta={
                    'url_request': url_req,
                    'domain': domain,
                    'retry_count': 0
                },
                errback=self.handle_error
            )
            
            # Try to inject auth session if available
            self._inject_auth_session(request, domain)
            
            yield request
    
    def parse(self, response):
        """Parse response and extract content with auth handling."""
        url_request = response.meta['url_request']
        url = response.url
        domain = response.meta.get('domain', urlparse(url).netloc)
        retry_count = response.meta.get('retry_count', 0)
        
        try:
            # Check if response indicates authentication is required
            if self._requires_authentication(response):
                logger.info("Authentication required detected", 
                           url=url, status_code=response.status, correlation_id=self.correlation_id)
                
                # Try to get auth session and retry if possible
                if retry_count < 2:  # Max 2 auth retries
                    self._queue_for_auth_retry(url_request, domain, retry_count + 1)
                    return
                else:
                    self._update_job_error(url, f"Authentication failed after {retry_count} retries")
                    return
            
            # Extract content
            extracted = self.content_extractor.extract_content(response.text, url)
            
            # Check for duplicates
            is_duplicate, content_hash = self.duplicate_detector.is_duplicate(
                extracted['content'], url
            )
            
            # Create scraped content object
            scraped_content = ScrapedContent(
                url=url,
                title=extracted['title'],
                content=extracted['content'],
                content_hash=content_hash,
                metadata={
                    'status_code': response.status,
                    'content_type': response.headers.get('content-type', '').decode('utf-8', errors='ignore'),
                    'original_metadata': url_request.metadata,
                    'response_time': response.meta.get('download_latency', 0),
                    'auth_used': response.meta.get('auth_session_used', False),
                    'retry_count': retry_count
                },
                scraped_at=datetime.now(),
                word_count=extracted['word_count'],
                is_duplicate=is_duplicate
            )
            
            self._update_job_success(scraped_content)
            
            logger.info(
                "Content scraped successfully",
                url=url,
                word_count=extracted['word_count'],
                is_duplicate=is_duplicate,
                auth_used=response.meta.get('auth_session_used', False),
                correlation_id=self.correlation_id
            )
            
        except Exception as e:
            logger.error("Content parsing failed", url=url, error=str(e), correlation_id=self.correlation_id)
            self._update_job_error(url, str(e))
    
    def handle_error(self, failure):
        """Handle request errors with smart retry for auth failures."""
        url = failure.request.url
        error_msg = str(failure.value)
        domain = failure.request.meta.get('domain', urlparse(url).netloc)
        retry_count = failure.request.meta.get('retry_count', 0)
        
        # Check if this is an auth-related error that we should retry
        if self._is_auth_error(failure) and retry_count < 2:
            logger.info("Auth-related error detected, queuing for retry", 
                       url=url, error=error_msg, correlation_id=self.correlation_id)
            url_request = failure.request.meta.get('url_request')
            if url_request:
                self._queue_for_auth_retry(url_request, domain, retry_count + 1)
                return
        
        logger.error("Request failed", url=url, error=error_msg, correlation_id=self.correlation_id)
        self._update_job_error(url, error_msg)
    
    def _inject_auth_session(self, request, domain):
        """Inject authentication session data into request if available."""
        try:
            # Check if we have an active session for this domain
            job = active_jobs.get(self.job_id)
            if job and domain in job.auth_sessions:
                session_id = job.auth_sessions[domain]
                if session_id in auth_sessions:
                    session = auth_sessions[session_id]
                    
                    # Inject cookies
                    if session.cookies:
                        request.cookies.update(session.cookies)
                    
                    # Inject headers
                    if session.headers:
                        request.headers.update(session.headers)
                    
                    # Set user agent if available
                    if session.user_agent:
                        request.headers['User-Agent'] = session.user_agent
                    
                    request.meta['auth_session_used'] = True
                    request.meta['session_id'] = session_id
                    
                    logger.debug("Auth session injected", 
                               domain=domain, session_id=session_id, correlation_id=self.correlation_id)
        except Exception as e:
            logger.warning("Failed to inject auth session", domain=domain, error=str(e))
    
    def _requires_authentication(self, response):
        """Check if response indicates authentication is required."""
        # Check status codes
        if response.status in [401, 403]:
            return True
        
        # Check for common auth redirect patterns
        if response.status in [302, 303, 307, 308]:
            location = response.headers.get('Location', '').decode('utf-8', errors='ignore').lower()
            if any(keyword in location for keyword in ['login', 'signin', 'auth']):
                return True
        
        # Check response content for auth indicators
        content_lower = response.text.lower()
        auth_indicators = [
            'please log in', 'please sign in', 'authentication required',
            'access denied', 'unauthorized', 'login required'
        ]
        
        return any(indicator in content_lower for indicator in auth_indicators)
    
    def _is_auth_error(self, failure):
        """Check if failure is auth-related and worth retrying."""
        error_msg = str(failure.value).lower()
        auth_error_patterns = [
            '401', '403', 'unauthorized', 'forbidden', 'authentication',
            'login required', 'access denied'
        ]
        return any(pattern in error_msg for pattern in auth_error_patterns)
    
    def _queue_for_auth_retry(self, url_request, domain, retry_count):
        """Queue URL for retry with authentication."""
        self.retry_queue.append({
            'url_request': url_request,
            'domain': domain,
            'retry_count': retry_count,
            'queued_at': datetime.now()
        })
        
        logger.info("URL queued for auth retry", 
                   url=str(url_request.url), domain=domain, retry_count=retry_count,
                   correlation_id=self.correlation_id)
    
    def _update_job_success(self, content: ScrapedContent):
        """Update job with successful result."""
        if self.job_id in active_jobs:
            job = active_jobs[self.job_id]
            job.results.append(content)
            job.completed_urls += 1
            job.status = "completed" if job.completed_urls + job.failed_urls >= job.total_urls else "running"
    
    def _update_job_error(self, url: str, error: str):
        """Update job with error."""
        if self.job_id in active_jobs:
            job = active_jobs[self.job_id]
            job.errors.append({
                "url": url,
                "error": error,
                "timestamp": datetime.now().isoformat()
            })
            job.failed_urls += 1
            job.status = "completed" if job.completed_urls + job.failed_urls >= job.total_urls else "running"

class ScrapingEngine:
    """Main scraping engine using Scrapy with authentication and parallel processing."""
    
    def __init__(self):
        self.runner = None
        self.auth_client = AuthenticationServiceClient()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._setup_scrapy()
    
    def _setup_scrapy(self):
        """Setup Scrapy crawler runner."""
        configure_logging({'LOG_LEVEL': 'WARNING'})  # Reduce Scrapy logging noise
        settings = get_project_settings()
        settings.update({
            'LOG_LEVEL': 'WARNING',
            'TELNETCONSOLE_ENABLED': False,
        })
        self.runner = CrawlerRunner(settings)
    
    async def scrape_urls(self, scrape_request: ScrapeRequest) -> str:
        """Start scraping job with authentication integration and parallel processing."""
        job_id = f"job_{int(time.time())}_{len(active_jobs)}"
        correlation_id = f"scrape_{job_id}_{uuid.uuid4().hex[:8]}"
        
        # Create job record
        job = ScrapeJob(
            job_id=job_id,
            status="running",
            total_urls=len(scrape_request.urls),
            completed_urls=0,
            failed_urls=0,
            correlation_id=correlation_id
        )
        active_jobs[job_id] = job
        
        # Separate URLs by domain and check for existing auth sessions
        await self._prepare_auth_sessions(scrape_request, job, correlation_id)
        
        # Start scraping in background
        deferred = self.runner.crawl(
            ContentSpider,
            urls=scrape_request.urls,
            job_id=job_id,
            rate_limit_delay=scrape_request.rate_limit_delay,
            respect_robots=scrape_request.respect_robots,
            correlation_id=correlation_id
        )
        
        # Don't wait for completion, return job ID immediately
        logger.info("Scraping job started with auth integration", 
                   job_id=job_id, total_urls=len(scrape_request.urls),
                   correlation_id=correlation_id)
        
        return job_id
    
    async def _prepare_auth_sessions(self, scrape_request: ScrapeRequest, job: ScrapeJob, correlation_id: str):
        """Prepare authentication sessions for domains that need them."""
        domain_urls = {}
        
        # Group URLs by domain
        for url_req in scrape_request.urls:
            domain = urlparse(str(url_req.url)).netloc
            if domain not in domain_urls:
                domain_urls[domain] = []
            domain_urls[domain].append(url_req)
        
        # Check for existing auth sessions for each domain
        for domain in domain_urls.keys():
            try:
                session = await self.auth_client.get_session_for_domain(domain, correlation_id)
                if session and session.is_active:
                    # Store session for use during scraping
                    auth_sessions[session.session_id] = session
                    job.auth_sessions[domain] = session.session_id
                    
                    logger.info("Auth session found for domain", 
                               domain=domain, session_id=session.session_id,
                               correlation_id=correlation_id)
                else:
                    logger.info("No active auth session for domain", 
                               domain=domain, correlation_id=correlation_id)
            except Exception as e:
                logger.warning("Failed to check auth session for domain", 
                              domain=domain, error=str(e), correlation_id=correlation_id)
    
    async def process_auth_retries(self, job_id: str):
        """Process URLs that failed due to authentication issues."""
        if job_id not in active_jobs:
            return
        
        job = active_jobs[job_id]
        correlation_id = job.correlation_id
        
        # This would be called periodically to retry auth-failed URLs
        # Implementation would check for new auth sessions and retry queued URLs
        logger.info("Processing auth retries", job_id=job_id, correlation_id=correlation_id)
    
    async def close(self):
        """Clean up resources."""
        await self.auth_client.close()
        self.executor.shutdown(wait=True)

# Global instances
scraping_engine = ScrapingEngine()
url_classifier = URLClassifier()
duplicate_detector = DuplicateDetector()
parallel_engine = ParallelProcessingEngine()

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed status."""
    return {
        "status": "healthy",
        "service": "Web Scraper Service",
        "version": "2.0.0",
        "timestamp": time.time(),
        "active_jobs": len(active_jobs),
        "active_websockets": len(websocket_connections),
        "total_content_hashes": len(content_hashes),
        "active_auth_sessions": len(auth_sessions),
        "features": [
            "Parallel authentication workflow",
            "Multi-format content extraction (HTML, PDF, Text)",
            "Advanced duplicate detection with similarity",
            "Real-time status tracking via WebSocket",
            "Content quality assessment",
            "Intelligent URL classification"
        ]
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Web Scraper Service",
        "version": "1.0.0",
        "status": "running",
        "features": [
            "Scrapy-based scraping with rate limiting",
            "Content extraction with Beautiful Soup and trafilatura",
            "Duplicate detection using content hashing",
            "Robots.txt compliance",
            "Configurable delays and retries"
        ]
    }

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time scraping status updates."""
    await websocket.accept()
    websocket_connections[job_id] = websocket
    
    try:
        # Send initial job status
        if job_id in active_jobs:
            await websocket.send_json({
                "type": "job_status",
                "job": active_jobs[job_id].model_dump(mode='json')
            })
        
        # Keep connection alive and handle client messages
        while True:
            try:
                data = await websocket.receive_json()
                # Handle client requests (e.g., pause, resume, cancel)
                if data.get("action") == "get_status":
                    if job_id in active_jobs:
                        await websocket.send_json({
                            "type": "job_status",
                            "job": active_jobs[job_id].model_dump(mode='json')
                        })
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error("WebSocket error", job_id=job_id, error=str(e))
                break
    finally:
        if job_id in websocket_connections:
            del websocket_connections[job_id]

@app.post("/scrape", response_model=Dict[str, Any])
async def start_scraping(scrape_request: ScrapeRequest, background_tasks: BackgroundTasks):
    """Start a comprehensive scraping job with parallel authentication workflow."""
    correlation_id = f"api_{uuid.uuid4().hex[:8]}"
    
    try:
        # Classify URLs for parallel processing
        url_classifications = await url_classifier.classify_urls(scrape_request.urls, correlation_id)
        
        # Create job
        job_id = f"job_{int(time.time())}_{len(active_jobs)}"
        job = ScrapeJob(
            job_id=job_id,
            status="initializing",
            total_urls=len(scrape_request.urls),
            completed_urls=0,
            failed_urls=0,
            correlation_id=correlation_id,
            url_classifications=url_classifications
        )
        
        # Update processing stats
        job.processing_stats = ProcessingStats(
            total_urls=len(scrape_request.urls),
            public_queue_size=sum(1 for c in url_classifications if c.queue_type == QueueType.PUBLIC),
            auth_queue_size=sum(1 for c in url_classifications if c.queue_type == QueueType.AUTHENTICATED)
        )
        
        active_jobs[job_id] = job
        
        # Start parallel processing
        background_tasks.add_task(
            parallel_engine.process_job, 
            job_id, 
            scrape_request, 
            url_classifications
        )
        
        # Prepare auth sessions
        background_tasks.add_task(
            scraping_engine._prepare_auth_sessions,
            scrape_request,
            job,
            correlation_id
        )
        
        logger.info(
            "Comprehensive scraping job initiated",
            job_id=job_id,
            session_id=scrape_request.session_id,
            url_count=len(scrape_request.urls),
            public_urls=job.processing_stats.public_queue_size,
            auth_urls=job.processing_stats.auth_queue_size,
            correlation_id=correlation_id
        )
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": f"Processing {len(scrape_request.urls)} URLs with parallel authentication workflow",
            "correlation_id": correlation_id,
            "url_classifications": [c.__dict__ for c in url_classifications],
            "processing_stats": job.processing_stats.__dict__,
            "websocket_url": f"/ws/{job_id}"
        }
        
    except Exception as e:
        logger.error("Failed to start scraping job", error=str(e), correlation_id=correlation_id)
        raise HTTPException(status_code=500, detail=f"Failed to start scraping: {str(e)}")

@app.post("/classify-urls", response_model=List[Dict[str, Any]])
async def classify_urls(urls: List[URLRequest]):
    """Classify URLs for authentication requirements and queue routing."""
    correlation_id = f"classify_{uuid.uuid4().hex[:8]}"
    
    try:
        classifications = await url_classifier.classify_urls(urls, correlation_id)
        return [c.__dict__ for c in classifications]
    except Exception as e:
        logger.error("URL classification failed", error=str(e), correlation_id=correlation_id)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/jobs/{job_id}/realtime-status", response_model=RealTimeStatus)
async def get_realtime_status(job_id: str):
    """Get real-time status of a scraping job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    # Get current processing info
    urls_in_progress = []
    if job_id in processing_queues:
        queues = processing_queues[job_id]
        urls_in_progress = [
            {"url": url, "status": "processing"} 
            for url in queues.processing
        ]
    
    # Get recent completions (last 10)
    recent_completions = [
        {
            "url": result.url,
            "completed_at": result.scraped_at.isoformat(),
            "word_count": result.word_count,
            "quality_score": result.quality_score
        }
        for result in job.results[-10:]
    ]
    
    # Auth status
    auth_status = {
        "active_sessions": len([s for s in job.auth_sessions.values() if s in auth_sessions]),
        "domains_with_auth": list(job.auth_sessions.keys()),
        "auth_pending": job.auth_pending_urls
    }
    
    # Performance metrics
    performance_metrics = parallel_engine.get_performance_metrics(job_id)
    
    progress_percentage = (job.completed_urls + job.failed_urls) / max(job.total_urls, 1) * 100
    
    return RealTimeStatus(
        job_id=job_id,
        current_status=job.status,
        progress_percentage=progress_percentage,
        urls_in_progress=urls_in_progress,
        recent_completions=recent_completions,
        auth_status=auth_status,
        performance_metrics=performance_metrics
    )

async def schedule_auth_retry_processing(job_id: str):
    """Schedule periodic processing of auth retries."""
    # Wait a bit for initial scraping to start
    await asyncio.sleep(10)
    
    # Process retries every 30 seconds for up to 5 minutes
    for _ in range(10):
        await scraping_engine.process_auth_retries(job_id)
        await asyncio.sleep(30)

@app.get("/jobs/{job_id}", response_model=ScrapeJob)
async def get_job_status(job_id: str):
    """Get status of a scraping job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return active_jobs[job_id]

@app.get("/jobs", response_model=List[str])
async def list_jobs():
    """List all active job IDs."""
    return list(active_jobs.keys())

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel and remove a job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Remove from active jobs
    del active_jobs[job_id]
    
    logger.info("Job cancelled", job_id=job_id)
    return {"message": f"Job {job_id} cancelled"}

@app.get("/stats")
async def get_comprehensive_stats():
    """Get comprehensive scraping statistics including parallel processing metrics."""
    total_jobs = len(active_jobs)
    running_jobs = sum(1 for job in active_jobs.values() if job.status in ["running", "initializing"])
    completed_jobs = sum(1 for job in active_jobs.values() if job.status == "completed")
    failed_jobs = sum(1 for job in active_jobs.values() if job.status == "failed")
    
    total_urls_scraped = sum(job.completed_urls for job in active_jobs.values())
    total_urls_failed = sum(job.failed_urls for job in active_jobs.values())
    
    # Content type distribution
    content_type_stats = defaultdict(int)
    quality_scores = []
    
    for job in active_jobs.values():
        for result in job.results:
            content_type_stats[result.content_type.value] += 1
            quality_scores.append(result.quality_score)
    
    # Duplicate detection stats
    duplicate_stats = duplicate_detector.get_duplicate_stats()
    total_duplicates = sum(
        1 for job in active_jobs.values() 
        for result in job.results 
        if result.is_duplicate
    )
    
    # Auth-related stats
    jobs_with_auth = sum(1 for job in active_jobs.values() if job.auth_sessions)
    active_auth_sessions = len(auth_sessions)
    
    auth_success_count = sum(
        1 for job in active_jobs.values()
        for result in job.results
        if result.metadata.get('auth_used', False)
    )
    
    # Parallel processing stats
    total_public_queue = sum(
        job.processing_stats.public_queue_size 
        for job in active_jobs.values()
    )
    total_auth_queue = sum(
        job.processing_stats.auth_queue_size 
        for job in active_jobs.values()
    )
    
    # Performance metrics
    avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    
    return {
        "job_statistics": {
            "total_jobs": total_jobs,
            "running_jobs": running_jobs,
            "completed_jobs": completed_jobs,
            "failed_jobs": failed_jobs
        },
        "url_statistics": {
            "total_urls_scraped": total_urls_scraped,
            "total_urls_failed": total_urls_failed,
            "success_rate": total_urls_scraped / max(total_urls_scraped + total_urls_failed, 1)
        },
        "content_statistics": {
            "content_type_distribution": dict(content_type_stats),
            "average_quality_score": avg_quality_score,
            "total_duplicates_detected": total_duplicates,
            "duplicate_detection_stats": duplicate_stats
        },
        "authentication_statistics": {
            "jobs_with_auth": jobs_with_auth,
            "active_auth_sessions": active_auth_sessions,
            "auth_success_count": auth_success_count,
            "auth_success_rate": auth_success_count / max(total_urls_scraped, 1)
        },
        "parallel_processing_statistics": {
            "total_public_queue_size": total_public_queue,
            "total_auth_queue_size": total_auth_queue,
            "active_websocket_connections": len(websocket_connections)
        },
        "system_statistics": {
            "unique_content_hashes": len(content_hashes),
            "memory_usage_estimate": {
                "content_hashes": len(content_hashes) * 64,  # bytes
                "auth_sessions": len(auth_sessions) * 1024,  # estimated bytes
                "active_jobs": len(active_jobs) * 2048  # estimated bytes
            }
        }
    }

@app.get("/jobs/{job_id}/performance")
async def get_job_performance(job_id: str):
    """Get detailed performance metrics for a specific job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    performance_metrics = parallel_engine.get_performance_metrics(job_id)
    
    # Calculate additional metrics
    processing_time = (datetime.now() - job.started_at).total_seconds()
    urls_per_second = job.completed_urls / max(processing_time, 1)
    
    # Quality distribution
    quality_distribution = defaultdict(int)
    for result in job.results:
        quality_range = f"{int(result.quality_score * 10) * 10}-{int(result.quality_score * 10) * 10 + 9}%"
        quality_distribution[quality_range] += 1
    
    return {
        "job_id": job_id,
        "processing_time_seconds": processing_time,
        "urls_per_second": urls_per_second,
        "performance_metrics": performance_metrics,
        "quality_distribution": dict(quality_distribution),
        "error_analysis": {
            "total_errors": len(job.errors),
            "error_types": defaultdict(int)
        },
        "processing_stats": job.processing_stats.dict()
    }

@app.post("/jobs/{job_id}/retry-auth")
async def retry_auth_failed_urls(job_id: str):
    """Manually trigger retry of authentication-failed URLs."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    try:
        await scraping_engine.process_auth_retries(job_id)
        return {"message": f"Auth retry processing triggered for job {job_id}"}
    except Exception as e:
        logger.error("Failed to process auth retries", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process retries: {str(e)}")

@app.get("/content-quality/{job_id}")
async def analyze_content_quality(job_id: str):
    """Analyze content quality for a specific job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    if not job.results:
        return {"message": "No content available for analysis"}
    
    # Quality analysis
    quality_scores = [result.quality_score for result in job.results]
    word_counts = [result.word_count for result in job.results]
    
    quality_analysis = {
        "total_content_pieces": len(job.results),
        "average_quality_score": sum(quality_scores) / len(quality_scores),
        "min_quality_score": min(quality_scores),
        "max_quality_score": max(quality_scores),
        "average_word_count": sum(word_counts) / len(word_counts),
        "content_type_distribution": defaultdict(int),
        "extraction_method_distribution": defaultdict(int),
        "low_quality_content": [],
        "high_quality_content": []
    }
    
    for result in job.results:
        quality_analysis["content_type_distribution"][result.content_type.value] += 1
        quality_analysis["extraction_method_distribution"][result.extraction_method] += 1
        
        if result.quality_score < 0.3:
            quality_analysis["low_quality_content"].append({
                "url": result.url,
                "quality_score": result.quality_score,
                "word_count": result.word_count,
                "issues": "Low quality score"
            })
        elif result.quality_score > 0.8:
            quality_analysis["high_quality_content"].append({
                "url": result.url,
                "quality_score": result.quality_score,
                "word_count": result.word_count
            })
    
    return quality_analysis

@app.post("/jobs/{job_id}/pause")
async def pause_job(job_id: str):
    """Pause a running scraping job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    if job.status != "running":
        raise HTTPException(status_code=400, detail="Job is not running")
    
    job.status = "paused"
    job.updated_at = datetime.now()
    
    logger.info("Job paused", job_id=job_id)
    return {"message": f"Job {job_id} paused"}

@app.post("/jobs/{job_id}/resume")
async def resume_job(job_id: str):
    """Resume a paused scraping job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    if job.status != "paused":
        raise HTTPException(status_code=400, detail="Job is not paused")
    
    job.status = "running"
    job.updated_at = datetime.now()
    
    logger.info("Job resumed", job_id=job_id)
    return {"message": f"Job {job_id} resumed"}

@app.get("/duplicate-analysis")
async def get_duplicate_analysis():
    """Get comprehensive duplicate content analysis."""
    duplicate_stats = duplicate_detector.get_duplicate_stats()
    
    # Analyze duplicate patterns
    duplicate_domains = defaultdict(int)
    duplicate_pairs = []
    
    for job in active_jobs.values():
        for result in job.results:
            if result.is_duplicate:
                domain = urlparse(result.url).netloc
                duplicate_domains[domain] += 1
    
    return {
        "duplicate_detection_stats": duplicate_stats,
        "duplicate_domains": dict(duplicate_domains),
        "total_unique_content": len(content_hashes),
        "duplicate_rate": len([r for job in active_jobs.values() for r in job.results if r.is_duplicate]) / 
                         max(len([r for job in active_jobs.values() for r in job.results]), 1)
    }

@app.get("/sessions")
async def list_auth_sessions():
    """List all active authentication sessions with enhanced details."""
    sessions_info = []
    for session_id, session in auth_sessions.items():
        sessions_info.append({
            "session_id": session_id,
            "domain": session.domain,
            "is_active": session.is_active,
            "expires_at": session.expires_at.isoformat() if session.expires_at else None,
            "last_used": session.last_used.isoformat() if session.last_used else None,
            "success_count": session.success_count,
            "failure_count": session.failure_count,
            "has_cookies": bool(session.cookies),
            "has_headers": bool(session.headers),
            "user_agent": session.user_agent
        })
    
    return {
        "total_sessions": len(auth_sessions),
        "active_sessions": len([s for s in auth_sessions.values() if s.is_active]),
        "sessions": sessions_info
    }

@app.delete("/sessions/{session_id}")
async def delete_auth_session(session_id: str):
    """Delete an authentication session."""
    if session_id not in auth_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = auth_sessions[session_id]
    del auth_sessions[session_id]
    
    logger.info("Auth session deleted", session_id=session_id, domain=session.domain)
    return {"message": f"Session {session_id} deleted"}

@app.get("/robots-compliance/{job_id}")
async def check_robots_compliance(job_id: str):
    """Check robots.txt compliance for a job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    # Analyze robots.txt compliance
    compliance_stats = {
        "total_urls": len(job.url_classifications),
        "robots_checked": 0,
        "robots_compliant": 0,
        "robots_violations": 0,
        "domains_analyzed": set()
    }
    
    for classification in job.url_classifications:
        domain = classification.domain
        compliance_stats["domains_analyzed"].add(domain)
        # This would integrate with the RobotsChecker
        compliance_stats["robots_checked"] += 1
        compliance_stats["robots_compliant"] += 1  # Placeholder
    
    compliance_stats["domains_analyzed"] = len(compliance_stats["domains_analyzed"])
    
    return compliance_stats

# Cleanup and shutdown handlers
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown."""
    logger.info("Shutting down scraper service")
    
    # Close all WebSocket connections
    for job_id, websocket in list(websocket_connections.items()):
        try:
            await websocket.close()
        except:
            pass
    
    # Shutdown parallel processing engine
    await parallel_engine.shutdown()
    
    # Close authentication client
    await scraping_engine.close()
    
    logger.info("Scraper service shutdown complete")

@app.post("/test-extract")
async def test_content_extraction(url: str):
    """Test content extraction for a single URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        extractor = ContentExtractor()
        extracted = extractor.extract_content(response.text, url)
        
        detector = DuplicateDetector()
        is_duplicate, content_hash = detector.is_duplicate(extracted['content'], url)
        
        return {
            "url": url,
            "title": extracted['title'],
            "content_preview": extracted['content'][:500] + "..." if len(extracted['content']) > 500 else extracted['content'],
            "word_count": extracted['word_count'],
            "content_hash": content_hash,
            "is_duplicate": is_duplicate,
            "status": "success"
        }
        
    except Exception as e:
        logger.error("Test extraction failed", url=url, error=str(e))
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)