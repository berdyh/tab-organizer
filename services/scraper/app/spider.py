"""Scrapy spider integration."""

from __future__ import annotations

from datetime import datetime
import uuid
from typing import Any, Dict, List
from urllib.parse import urlparse

import scrapy

from .auth_client import AuthenticationServiceClient
from .duplicates import DuplicateDetector
from .extraction import ContentExtractor
from .logging import get_logger
from .models import ScrapedContent
from .robots import RobotsChecker
from .state import state

logger = get_logger()


class ContentSpider(scrapy.Spider):
    """Scrapy spider for content extraction with rate limiting and authentication."""

    name = "content_spider"

    custom_settings = {
        "DOWNLOAD_DELAY": 1.0,
        "RANDOMIZE_DOWNLOAD_DELAY": True,
        "CONCURRENT_REQUESTS": 2,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "RETRY_TIMES": 3,
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429, 401, 403],
        "USER_AGENT": "WebScrapingTool/1.0 (+https://example.com/bot)",
        "ROBOTSTXT_OBEY": True,
    }

    def __init__(
        self,
        urls: List[Any] | None = None,
        job_id: str | None = None,
        rate_limit_delay: float = 1.0,
        respect_robots: bool = True,
        correlation_id: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.start_urls = urls or []
        self.job_id = job_id
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.custom_settings["DOWNLOAD_DELAY"] = rate_limit_delay
        self.custom_settings["ROBOTSTXT_OBEY"] = respect_robots
        self.robots_checker = RobotsChecker()
        self.content_extractor = ContentExtractor()
        self.duplicate_detector = DuplicateDetector()
        self.auth_client = AuthenticationServiceClient()
        self.retry_queue: List[Dict[str, Any]] = []

    def start_requests(self):
        """Generate initial requests with robots.txt checking and auth session injection."""
        for url_req in self.start_urls:
            url = str(url_req.url)
            domain = urlparse(url).netloc

            if self.custom_settings["ROBOTSTXT_OBEY"]:
                if not self.robots_checker.can_fetch(url):
                    logger.warning(
                        "URL blocked by robots.txt",
                        url=url,
                        correlation_id=self.correlation_id,
                    )
                    self._update_job_error(url, "Blocked by robots.txt")
                    continue

            request = scrapy.Request(
                url=url,
                callback=self.parse,
                meta={"url_request": url_req, "domain": domain, "retry_count": 0},
                errback=self.handle_error,
            )

            self._inject_auth_session(request, domain)
            yield request

    def parse(self, response):
        """Parse response and extract content with auth handling."""
        url_request = response.meta["url_request"]
        url = response.url
        domain = response.meta.get("domain", urlparse(url).netloc)
        retry_count = response.meta.get("retry_count", 0)

        try:
            if self._requires_authentication(response):
                logger.info(
                    "Authentication required detected",
                    url=url,
                    status_code=response.status,
                    correlation_id=self.correlation_id,
                )

                if retry_count < 2:
                    self._queue_for_auth_retry(url_request, domain, retry_count + 1)
                    return

                self._update_job_error(
                    url, f"Authentication failed after {retry_count} retries"
                )
                return

            extracted = self.content_extractor.extract_content(
                response.body, url, response.headers.get("content-type", b"").decode("utf-8", errors="ignore")
            )

            is_duplicate, content_hash, similarity = self.duplicate_detector.is_duplicate(
                extracted["content"], url
            )

            scraped_content = ScrapedContent(
                url=url,
                title=extracted["title"],
                content=extracted["content"],
                content_hash=content_hash,
                metadata={
                    "status_code": response.status,
                    "content_type": response.headers.get("content-type", b"").decode(
                        "utf-8", errors="ignore"
                    ),
                    "original_metadata": url_request.metadata,
                    "response_time": response.meta.get("download_latency", 0),
                    "auth_used": response.meta.get("auth_session_used", False),
                    "retry_count": retry_count,
                    "duplicate_similarity": similarity,
                },
                scraped_at=datetime.now(),
                word_count=extracted["word_count"],
                quality_score=extracted.get("quality_score", 0.0),
                is_duplicate=is_duplicate,
                extraction_method=extracted.get("extraction_method", ""),
            )

            self._update_job_success(scraped_content)

            logger.info(
                "Content scraped successfully",
                url=url,
                word_count=extracted["word_count"],
                is_duplicate=is_duplicate,
                auth_used=response.meta.get("auth_session_used", False),
                correlation_id=self.correlation_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(
                "Content parsing failed",
                url=url,
                error=str(exc),
                correlation_id=self.correlation_id,
            )
            self._update_job_error(url, str(exc))

    def handle_error(self, failure):
        """Handle request errors with smart retry for auth failures."""
        url = failure.request.url
        error_msg = str(failure.value)
        domain = failure.request.meta.get("domain", urlparse(url).netloc)
        retry_count = failure.request.meta.get("retry_count", 0)

        if self._is_auth_error(failure) and retry_count < 2:
            logger.info(
                "Auth-related error detected, queuing for retry",
                url=url,
                error=error_msg,
                correlation_id=self.correlation_id,
            )
            url_request = failure.request.meta.get("url_request")
            if url_request:
                self._queue_for_auth_retry(url_request, domain, retry_count + 1)
                return

        logger.error(
            "Request failed",
            url=url,
            error=error_msg,
            correlation_id=self.correlation_id,
        )
        self._update_job_error(url, error_msg)

    def _inject_auth_session(self, request, domain: str) -> None:
        """Inject authentication session data into request if available."""
        try:
            job = state.active_jobs.get(self.job_id)
            if not job:
                return

            session_id = job.auth_sessions.get(domain)
            if not session_id:
                return

            session = state.auth_sessions.get(session_id)
            if not session:
                return

            if session.cookies:
                request.cookies.update(session.cookies)
            if session.headers:
                request.headers.update(session.headers)
            if session.user_agent:
                request.headers["User-Agent"] = session.user_agent

            request.meta["auth_session_used"] = True
            request.meta["session_id"] = session_id

            logger.debug(
                "Auth session injected",
                domain=domain,
                session_id=session_id,
                correlation_id=self.correlation_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to inject auth session", domain=domain, error=str(exc)
            )

    def _requires_authentication(self, response) -> bool:
        """Check if response indicates authentication is required."""
        if response.status in [401, 403]:
            return True

        if response.status in [302, 303, 307, 308]:
            location = response.headers.get("Location", b"").decode(
                "utf-8", errors="ignore"
            )
            if any(keyword in location.lower() for keyword in ["login", "signin", "auth"]):
                return True

        content_lower = response.text.lower()
        auth_indicators = [
            "please log in",
            "please sign in",
            "authentication required",
            "access denied",
            "unauthorized",
            "login required",
        ]

        return any(indicator in content_lower for indicator in auth_indicators)

    def _is_auth_error(self, failure) -> bool:
        """Check if failure is auth-related and worth retrying."""
        error_msg = str(failure.value).lower()
        auth_error_patterns = [
            "401",
            "403",
            "unauthorized",
            "forbidden",
            "authentication",
            "login required",
            "access denied",
        ]
        return any(pattern in error_msg for pattern in auth_error_patterns)

    def _queue_for_auth_retry(self, url_request, domain: str, retry_count: int) -> None:
        """Queue URL for retry with authentication."""
        self.retry_queue.append(
            {
                "url_request": url_request,
                "domain": domain,
                "retry_count": retry_count,
                "queued_at": datetime.now(),
            }
        )

        logger.info(
            "URL queued for auth retry",
            url=str(url_request.url),
            domain=domain,
            retry_count=retry_count,
            correlation_id=self.correlation_id,
        )

    def _update_job_success(self, content: ScrapedContent) -> None:
        """Update job with successful result."""
        job = state.active_jobs.get(self.job_id)
        if not job:
            return

        job.results.append(content)
        job.completed_urls += 1
        job.status = (
            "completed"
            if job.completed_urls + job.failed_urls >= job.total_urls
            else "running"
        )
        job.updated_at = datetime.now()

    def _update_job_error(self, url: str, error: str) -> None:
        """Update job with error."""
        job = state.active_jobs.get(self.job_id)
        if not job:
            return

        job.errors.append(
            {
                "url": url,
                "error": error,
                "timestamp": datetime.now().isoformat(),
            }
        )
        job.failed_urls += 1
        job.status = (
            "completed"
            if job.completed_urls + job.failed_urls >= job.total_urls
            else "running"
        )
        job.updated_at = datetime.now()
