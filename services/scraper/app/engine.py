"""Scraping engine orchestrating Scrapy integration and auth prep."""

from __future__ import annotations

import time
import uuid
from typing import Dict, List
from urllib.parse import urlparse

from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings

from .auth_client import AuthenticationServiceClient
from .logging import get_logger
from .models import ScrapeJob, ScrapeRequest
from .spider import ContentSpider
from .state import state

logger = get_logger()


class ScrapingEngine:
    """Main scraping engine using Scrapy with authentication support."""

    def __init__(self) -> None:
        self.runner: CrawlerRunner | None = None
        self.auth_client = AuthenticationServiceClient()
        self._setup_scrapy()

    def _setup_scrapy(self) -> None:
        configure_logging({"LOG_LEVEL": "WARNING"})
        settings = get_project_settings()
        settings.update({"LOG_LEVEL": "WARNING", "TELNETCONSOLE_ENABLED": False})
        self.runner = CrawlerRunner(settings)

    async def scrape_urls(self, scrape_request: ScrapeRequest) -> str:
        """Start scraping job with authentication integration and parallel processing."""
        job_id = f"job_{int(time.time())}_{len(state.active_jobs)}"
        correlation_id = f"scrape_{job_id}_{uuid.uuid4().hex[:8]}"

        job = ScrapeJob(
            job_id=job_id,
            status="running",
            total_urls=len(scrape_request.urls),
            completed_urls=0,
            failed_urls=0,
            correlation_id=correlation_id,
        )
        state.reset_job(job)

        await self._prepare_auth_sessions(scrape_request, job, correlation_id)

        if self.runner:
            self.runner.crawl(
                ContentSpider,
                urls=scrape_request.urls,
                job_id=job_id,
                rate_limit_delay=scrape_request.rate_limit_delay,
                respect_robots=scrape_request.respect_robots,
                correlation_id=correlation_id,
            )

        logger.info(
            "Scraping job started with auth integration",
            job_id=job_id,
            total_urls=len(scrape_request.urls),
            correlation_id=correlation_id,
        )
        return job_id

    async def _prepare_auth_sessions(
        self, scrape_request: ScrapeRequest, job: ScrapeJob, correlation_id: str
    ) -> None:
        """Prepare authentication sessions for domains that need them."""
        domain_urls: Dict[str, List] = {}

        for url_req in scrape_request.urls:
            domain = urlparse(str(url_req.url)).netloc
            domain_urls.setdefault(domain, []).append(url_req)

        for domain in domain_urls:
            try:
                session = await self.auth_client.get_session_for_domain(
                    domain, correlation_id
                )
                if session and session.is_active:
                    state.auth_sessions[session.session_id] = session
                    job.auth_sessions[domain] = session.session_id
                    logger.info(
                        "Auth session found for domain",
                        domain=domain,
                        session_id=session.session_id,
                        correlation_id=correlation_id,
                    )
                else:
                    logger.info(
                        "No active auth session for domain",
                        domain=domain,
                        correlation_id=correlation_id,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to check auth session for domain",
                    domain=domain,
                    error=str(exc),
                    correlation_id=correlation_id,
                )

    async def process_auth_retries(self, job_id: str) -> None:
        """Process URLs that failed due to authentication issues."""
        job = state.active_jobs.get(job_id)
        if not job:
            return

        logger.info(
            "Processing auth retries", job_id=job_id, correlation_id=job.correlation_id
        )

    async def close(self) -> None:
        """Clean up resources."""
        await self.auth_client.close()
