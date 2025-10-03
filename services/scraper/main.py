"""Web Scraper Service - Extracts content from web URLs."""

import time
import hashlib
import asyncio
import aiohttp
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
import structlog
import scrapy
from scrapy.crawler import CrawlerRunner
from scrapy.utils.project import get_project_settings
from scrapy.utils.log import configure_logging
from twisted.internet import asyncioreactor
from bs4 import BeautifulSoup
import trafilatura
import requests

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

# Pydantic models
class URLRequest(BaseModel):
    url: HttpUrl
    priority: int = 1
    metadata: Dict[str, Any] = {}

class ScrapeRequest(BaseModel):
    urls: List[URLRequest]
    session_id: str
    rate_limit_delay: float = 1.0
    respect_robots: bool = True
    max_retries: int = 3

class ScrapedContent(BaseModel):
    url: str
    title: str
    content: str
    content_hash: str
    metadata: Dict[str, Any]
    scraped_at: datetime
    word_count: int
    is_duplicate: bool = False

class AuthSession(BaseModel):
    session_id: str
    domain: str
    cookies: Dict[str, Any] = {}
    headers: Dict[str, str] = {}
    user_agent: Optional[str] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True

class ScrapeJob(BaseModel):
    job_id: str
    status: str
    total_urls: int
    completed_urls: int
    failed_urls: int
    results: List[ScrapedContent] = []
    errors: List[Dict[str, Any]] = []
    auth_sessions: Dict[str, str] = {}  # domain -> session_id mapping
    correlation_id: str = ""

# Global job storage (in production, use Redis or database)
active_jobs: Dict[str, ScrapeJob] = {}
content_hashes: Dict[str, str] = {}  # hash -> url mapping for duplicate detection
auth_sessions: Dict[str, AuthSession] = {}  # session_id -> AuthSession mapping

class ContentExtractor:
    """Extract and clean content using Beautiful Soup and trafilatura."""
    
    @staticmethod
    def extract_content(html: str, url: str) -> Dict[str, Any]:
        """Extract clean content from HTML using multiple methods."""
        try:
            # Primary extraction with trafilatura
            content = trafilatura.extract(html, include_comments=False, include_tables=True)
            
            # Fallback to Beautiful Soup if trafilatura fails
            if not content or len(content.strip()) < 50:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Extract text from main content areas
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                if main_content:
                    content = main_content.get_text(separator=' ', strip=True)
                else:
                    content = soup.get_text(separator=' ', strip=True)
            
            # Extract title
            soup = BeautifulSoup(html, 'html.parser')
            title = ""
            if soup.title:
                title = soup.title.string.strip() if soup.title.string else ""
            
            # Clean and normalize content
            content = ' '.join(content.split()) if content else ""
            
            return {
                "title": title,
                "content": content,
                "word_count": len(content.split()) if content else 0
            }
            
        except Exception as e:
            logger.error("Content extraction failed", url=url, error=str(e))
            return {
                "title": "",
                "content": "",
                "word_count": 0
            }

class DuplicateDetector:
    """Detect duplicate content using content hashing."""
    
    @staticmethod
    def generate_content_hash(content: str) -> str:
        """Generate SHA-256 hash of normalized content."""
        # Normalize content for hashing
        normalized = ' '.join(content.lower().split())
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    
    @staticmethod
    def is_duplicate(content: str, url: str) -> tuple[bool, str]:
        """Check if content is duplicate and return (is_duplicate, hash)."""
        content_hash = DuplicateDetector.generate_content_hash(content)
        
        if content_hash in content_hashes:
            original_url = content_hashes[content_hash]
            logger.info("Duplicate content detected", url=url, original_url=original_url)
            return True, content_hash
        
        content_hashes[content_hash] = url
        return False, content_hash

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

# Global scraping engine instance
scraping_engine = ScrapingEngine()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "scraper",
        "timestamp": time.time(),
        "active_jobs": len(active_jobs)
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

@app.post("/scrape", response_model=Dict[str, str])
async def start_scraping(scrape_request: ScrapeRequest, background_tasks: BackgroundTasks):
    """Start a scraping job with authentication integration."""
    correlation_id = f"api_{uuid.uuid4().hex[:8]}"
    
    try:
        job_id = await scraping_engine.scrape_urls(scrape_request)
        
        # Schedule periodic auth retry processing
        background_tasks.add_task(schedule_auth_retry_processing, job_id)
        
        logger.info(
            "Scraping job initiated with auth support",
            job_id=job_id,
            session_id=scrape_request.session_id,
            url_count=len(scrape_request.urls),
            correlation_id=correlation_id
        )
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": f"Scraping {len(scrape_request.urls)} URLs with auth integration",
            "correlation_id": correlation_id
        }
        
    except Exception as e:
        logger.error("Failed to start scraping job", error=str(e), correlation_id=correlation_id)
        raise HTTPException(status_code=500, detail=f"Failed to start scraping: {str(e)}")

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
async def get_stats():
    """Get scraping statistics including authentication metrics."""
    total_jobs = len(active_jobs)
    running_jobs = sum(1 for job in active_jobs.values() if job.status == "running")
    completed_jobs = sum(1 for job in active_jobs.values() if job.status == "completed")
    
    total_urls_scraped = sum(job.completed_urls for job in active_jobs.values())
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
    
    return {
        "total_jobs": total_jobs,
        "running_jobs": running_jobs,
        "completed_jobs": completed_jobs,
        "total_urls_scraped": total_urls_scraped,
        "total_duplicates_detected": total_duplicates,
        "unique_content_hashes": len(content_hashes),
        "jobs_with_auth": jobs_with_auth,
        "active_auth_sessions": active_auth_sessions,
        "auth_success_count": auth_success_count
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

@app.get("/sessions")
async def list_auth_sessions():
    """List all active authentication sessions."""
    sessions_info = []
    for session_id, session in auth_sessions.items():
        sessions_info.append({
            "session_id": session_id,
            "domain": session.domain,
            "is_active": session.is_active,
            "expires_at": session.expires_at.isoformat() if session.expires_at else None,
            "has_cookies": bool(session.cookies),
            "has_headers": bool(session.headers)
        })
    
    return {
        "total_sessions": len(sessions_info),
        "sessions": sessions_info
    }

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