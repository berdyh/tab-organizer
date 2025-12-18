"""Non-blocking web scraper with parallel auth support."""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Awaitable
from urllib.parse import urlparse, urljoin

import httpx
from playwright.async_api import async_playwright, Browser, Page, TimeoutError as PlaywrightTimeout


@dataclass
class ScrapeResult:
    """Result of scraping a URL."""
    url: str
    status: str  # success, failed, auth_required, timeout, blocked
    title: Optional[str] = None
    content: Optional[str] = None
    html: Optional[str] = None
    status_code: Optional[int] = None
    error: Optional[str] = None
    scraped_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


class ContentExtractor:
    """Extract readable content from HTML."""
    
    # Tags to remove entirely
    REMOVE_TAGS = [
        'script', 'style', 'noscript', 'iframe', 'svg', 'canvas',
        'video', 'audio', 'nav', 'footer', 'header', 'aside',
    ]
    
    # Tags that indicate main content
    CONTENT_TAGS = ['article', 'main', 'section', 'div']
    
    def extract_title(self, html: str) -> str:
        """Extract page title from HTML."""
        # Try <title> tag
        match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try <h1> tag
        match = re.search(r'<h1[^>]*>([^<]+)</h1>', html, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return ""
    
    def extract_text(self, html: str) -> str:
        """Extract readable text from HTML."""
        # Remove unwanted tags
        text = html
        for tag in self.REMOVE_TAGS:
            text = re.sub(
                rf'<{tag}[^>]*>.*?</{tag}>',
                '',
                text,
                flags=re.IGNORECASE | re.DOTALL
            )
        
        # Remove all remaining HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Decode HTML entities
        text = self._decode_entities(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _decode_entities(self, text: str) -> str:
        """Decode common HTML entities."""
        entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&apos;': "'",
        }
        for entity, char in entities.items():
            text = text.replace(entity, char)
        return text
    
    def extract_metadata(self, html: str) -> dict:
        """Extract metadata from HTML."""
        metadata = {}
        
        # Extract meta description
        match = re.search(
            r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if match:
            metadata['description'] = match.group(1)
        
        # Extract meta keywords
        match = re.search(
            r'<meta[^>]*name=["\']keywords["\'][^>]*content=["\']([^"\']+)["\']',
            html, re.IGNORECASE
        )
        if match:
            metadata['keywords'] = match.group(1).split(',')
        
        # Extract Open Graph data
        og_patterns = [
            ('og_title', r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']'),
            ('og_description', r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']'),
            ('og_image', r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']'),
        ]
        
        for key, pattern in og_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1)
        
        return metadata


class RobotsChecker:
    """Check robots.txt compliance."""
    
    def __init__(self):
        self._cache: dict[str, dict] = {}  # domain → rules
    
    async def is_allowed(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL is allowed by robots.txt."""
        parsed = urlparse(url)
        domain = f"{parsed.scheme}://{parsed.netloc}"
        
        if domain not in self._cache:
            await self._fetch_robots(domain)
        
        rules = self._cache.get(domain, {})
        path = parsed.path or "/"
        
        # Check disallow rules
        disallow = rules.get("disallow", [])
        for pattern in disallow:
            if path.startswith(pattern):
                return False
        
        return True
    
    async def _fetch_robots(self, domain: str) -> None:
        """Fetch and parse robots.txt."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{domain}/robots.txt",
                    timeout=10.0,
                    follow_redirects=True,
                )
                if response.status_code == 200:
                    self._cache[domain] = self._parse_robots(response.text)
                else:
                    self._cache[domain] = {}
        except Exception:
            self._cache[domain] = {}
    
    def _parse_robots(self, content: str) -> dict:
        """Parse robots.txt content."""
        rules = {"disallow": [], "allow": []}
        current_agent = None
        
        for line in content.split('\n'):
            line = line.strip().lower()
            
            if line.startswith('user-agent:'):
                current_agent = line.split(':', 1)[1].strip()
            elif current_agent in ('*', 'tab-organizer'):
                if line.startswith('disallow:'):
                    path = line.split(':', 1)[1].strip()
                    if path:
                        rules["disallow"].append(path)
                elif line.startswith('allow:'):
                    path = line.split(':', 1)[1].strip()
                    if path:
                        rules["allow"].append(path)
        
        return rules


class ScraperEngine:
    """Non-blocking web scraper with parallel auth support."""
    
    USER_AGENT = "Mozilla/5.0 (compatible; TabOrganizer/1.0; +https://github.com/tab-organizer)"
    
    def __init__(
        self,
        max_concurrent: int = 10,
        timeout: int = 30,
        respect_robots: bool = True,
    ):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.respect_robots = respect_robots
        
        self._browser: Optional[Browser] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._extractor = ContentExtractor()
        self._robots = RobotsChecker()
        self._auth_queue = None
        self._auth_detector = None
    
    def set_auth_queue(self, queue) -> None:
        """Set the auth queue for handling authentication."""
        self._auth_queue = queue
    
    def set_auth_detector(self, detector) -> None:
        """Set the auth detector."""
        self._auth_detector = detector
    
    async def _get_browser(self) -> Browser:
        """Get or create browser instance."""
        if self._browser is None:
            playwright = await async_playwright().start()
            self._browser = await playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage'],
            )
        return self._browser
    
    async def close(self) -> None:
        """Close browser instance."""
        if self._browser:
            await self._browser.close()
            self._browser = None
    
    async def scrape_url(
        self,
        url: str,
        session_id: Optional[str] = None,
        use_browser: bool = False,
    ) -> ScrapeResult:
        """
        Scrape a single URL.
        
        Args:
            url: URL to scrape
            session_id: Session ID for auth queue
            use_browser: Use Playwright instead of httpx
        """
        async with self._semaphore:
            # Check robots.txt
            if self.respect_robots:
                if not await self._robots.is_allowed(url):
                    return ScrapeResult(
                        url=url,
                        status="blocked",
                        error="Blocked by robots.txt",
                    )
            
            # Check for existing credentials
            if self._auth_queue and self._auth_queue.has_credentials(url):
                credentials = self._auth_queue.get_credentials(url)
                return await self._scrape_with_auth(url, credentials, session_id)
            
            # Try scraping
            if use_browser:
                return await self._scrape_with_browser(url, session_id)
            else:
                return await self._scrape_with_httpx(url, session_id)
    
    async def _scrape_with_httpx(
        self,
        url: str,
        session_id: Optional[str] = None,
    ) -> ScrapeResult:
        """Scrape URL using httpx."""
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.timeout,
            ) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": self.USER_AGENT},
                )
                
                # Check for auth requirement
                if self._auth_detector:
                    auth_result = self._auth_detector.detect_from_response(
                        response.status_code,
                        dict(response.headers),
                        url,
                    )
                    
                    if auth_result.requires_auth:
                        if self._auth_queue:
                            await self._auth_queue.request_auth(
                                url=url,
                                auth_type=auth_result.auth_type,
                                session_id=session_id,
                                form_fields=auth_result.form_fields,
                                oauth_provider=auth_result.oauth_provider,
                            )
                        return ScrapeResult(
                            url=url,
                            status="auth_required",
                            status_code=response.status_code,
                            metadata={"auth_type": auth_result.auth_type},
                        )
                
                html = response.text
                
                # Check HTML for auth indicators
                if self._auth_detector:
                    auth_result = self._auth_detector.detect_from_html(html, url)
                    if auth_result.requires_auth and auth_result.confidence > 0.7:
                        if self._auth_queue:
                            await self._auth_queue.request_auth(
                                url=url,
                                auth_type=auth_result.auth_type,
                                session_id=session_id,
                                form_fields=auth_result.form_fields,
                            )
                        return ScrapeResult(
                            url=url,
                            status="auth_required",
                            status_code=response.status_code,
                            html=html,
                            metadata={"auth_type": auth_result.auth_type},
                        )
                
                # Extract content
                title = self._extractor.extract_title(html)
                content = self._extractor.extract_text(html)
                metadata = self._extractor.extract_metadata(html)
                
                return ScrapeResult(
                    url=url,
                    status="success",
                    title=title,
                    content=content,
                    html=html,
                    status_code=response.status_code,
                    metadata=metadata,
                )
                
        except httpx.TimeoutException:
            return ScrapeResult(
                url=url,
                status="timeout",
                error="Request timed out",
            )
        except Exception as e:
            return ScrapeResult(
                url=url,
                status="failed",
                error=str(e),
            )
    
    async def _scrape_with_browser(
        self,
        url: str,
        session_id: Optional[str] = None,
    ) -> ScrapeResult:
        """Scrape URL using Playwright browser."""
        try:
            browser = await self._get_browser()
            page = await browser.new_page()
            
            try:
                await page.set_extra_http_headers({"User-Agent": self.USER_AGENT})
                
                response = await page.goto(url, timeout=self.timeout * 1000)
                
                if response is None:
                    return ScrapeResult(
                        url=url,
                        status="failed",
                        error="No response received",
                    )
                
                status_code = response.status
                
                # Wait for content to load
                await page.wait_for_load_state("domcontentloaded")
                
                html = await page.content()
                
                # Check for auth
                if self._auth_detector:
                    auth_result = self._auth_detector.detect(
                        url=url,
                        status_code=status_code,
                        headers=dict(response.headers),
                        html=html,
                    )
                    
                    if auth_result.requires_auth:
                        if self._auth_queue:
                            await self._auth_queue.request_auth(
                                url=url,
                                auth_type=auth_result.auth_type,
                                session_id=session_id,
                                form_fields=auth_result.form_fields,
                                oauth_provider=auth_result.oauth_provider,
                            )
                        return ScrapeResult(
                            url=url,
                            status="auth_required",
                            status_code=status_code,
                            html=html,
                            metadata={"auth_type": auth_result.auth_type},
                        )
                
                title = await page.title()
                content = self._extractor.extract_text(html)
                metadata = self._extractor.extract_metadata(html)
                
                return ScrapeResult(
                    url=url,
                    status="success",
                    title=title,
                    content=content,
                    html=html,
                    status_code=status_code,
                    metadata=metadata,
                )
                
            finally:
                await page.close()
                
        except PlaywrightTimeout:
            return ScrapeResult(
                url=url,
                status="timeout",
                error="Browser timeout",
            )
        except Exception as e:
            return ScrapeResult(
                url=url,
                status="failed",
                error=str(e),
            )
    
    async def _scrape_with_auth(
        self,
        url: str,
        credentials: dict,
        session_id: Optional[str] = None,
    ) -> ScrapeResult:
        """Scrape URL with authentication."""
        auth_type = credentials.get("type", "basic")
        
        if auth_type == "basic":
            return await self._scrape_basic_auth(url, credentials)
        elif auth_type == "cookie":
            return await self._scrape_cookie_auth(url, credentials)
        elif auth_type == "form":
            return await self._scrape_form_auth(url, credentials, session_id)
        else:
            return await self._scrape_with_httpx(url, session_id)
    
    async def _scrape_basic_auth(
        self,
        url: str,
        credentials: dict,
    ) -> ScrapeResult:
        """Scrape with HTTP Basic Auth."""
        try:
            auth = httpx.BasicAuth(
                credentials.get("username", ""),
                credentials.get("password", ""),
            )
            
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.timeout,
                auth=auth,
            ) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": self.USER_AGENT},
                )
                
                html = response.text
                title = self._extractor.extract_title(html)
                content = self._extractor.extract_text(html)
                metadata = self._extractor.extract_metadata(html)
                
                return ScrapeResult(
                    url=url,
                    status="success",
                    title=title,
                    content=content,
                    html=html,
                    status_code=response.status_code,
                    metadata=metadata,
                )
                
        except Exception as e:
            return ScrapeResult(
                url=url,
                status="failed",
                error=str(e),
            )
    
    async def _scrape_cookie_auth(
        self,
        url: str,
        credentials: dict,
    ) -> ScrapeResult:
        """Scrape with cookie authentication."""
        try:
            cookies = credentials.get("cookies", {})
            
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.timeout,
                cookies=cookies,
            ) as client:
                response = await client.get(
                    url,
                    headers={"User-Agent": self.USER_AGENT},
                )
                
                html = response.text
                title = self._extractor.extract_title(html)
                content = self._extractor.extract_text(html)
                metadata = self._extractor.extract_metadata(html)
                
                return ScrapeResult(
                    url=url,
                    status="success",
                    title=title,
                    content=content,
                    html=html,
                    status_code=response.status_code,
                    metadata=metadata,
                )
                
        except Exception as e:
            return ScrapeResult(
                url=url,
                status="failed",
                error=str(e),
            )
    
    async def _scrape_form_auth(
        self,
        url: str,
        credentials: dict,
        session_id: Optional[str] = None,
    ) -> ScrapeResult:
        """Scrape with form-based authentication using browser."""
        try:
            browser = await self._get_browser()
            page = await browser.new_page()
            
            try:
                login_url = credentials.get("login_url", url)
                await page.goto(login_url, timeout=self.timeout * 1000)
                
                # Fill form fields
                username_field = credentials.get("username_field", "username")
                password_field = credentials.get("password_field", "password")
                
                await page.fill(
                    f'input[name="{username_field}"], input[type="email"], input[type="text"]',
                    credentials.get("username", ""),
                )
                await page.fill(
                    f'input[name="{password_field}"], input[type="password"]',
                    credentials.get("password", ""),
                )
                
                # Submit form
                await page.click('button[type="submit"], input[type="submit"]')
                await page.wait_for_load_state("networkidle")
                
                # Navigate to target URL
                if page.url != url:
                    await page.goto(url, timeout=self.timeout * 1000)
                
                html = await page.content()
                title = await page.title()
                content = self._extractor.extract_text(html)
                metadata = self._extractor.extract_metadata(html)
                
                return ScrapeResult(
                    url=url,
                    status="success",
                    title=title,
                    content=content,
                    html=html,
                    metadata=metadata,
                )
                
            finally:
                await page.close()
                
        except Exception as e:
            return ScrapeResult(
                url=url,
                status="failed",
                error=str(e),
            )
    
    async def scrape_batch(
        self,
        urls: list[str],
        session_id: Optional[str] = None,
        callback: Optional[Callable[[ScrapeResult], Awaitable[None]]] = None,
    ) -> list[ScrapeResult]:
        """
        Scrape multiple URLs in parallel.
        
        Non-blocking: continues scraping public sites while
        waiting for auth on protected sites.
        """
        tasks = []
        
        for url in urls:
            task = asyncio.create_task(
                self._scrape_with_callback(url, session_id, callback)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        final_results = []
        for url, result in zip(urls, results):
            if isinstance(result, Exception):
                final_results.append(ScrapeResult(
                    url=url,
                    status="failed",
                    error=str(result),
                ))
            else:
                final_results.append(result)
        
        return final_results
    
    async def _scrape_with_callback(
        self,
        url: str,
        session_id: Optional[str],
        callback: Optional[Callable[[ScrapeResult], Awaitable[None]]],
    ) -> ScrapeResult:
        """Scrape URL and call callback with result."""
        result = await self.scrape_url(url, session_id)
        
        if callback:
            try:
                await callback(result)
            except Exception:
                pass
        
        return result
