"""URL metadata extraction and enrichment utilities."""

from __future__ import annotations

import hashlib
from typing import List
from urllib.parse import parse_qs, urlparse

from .logging import get_logger
from .models import URLEntry, URLMetadata


logger = get_logger(__name__)


class URLEnricher:
    """URL metadata extraction and enrichment utilities."""

    @classmethod
    def extract_metadata(cls, url: str) -> URLMetadata:
        """Extract comprehensive metadata from a URL."""
        try:
            parsed = urlparse(url)

            if "localhost" in parsed.netloc or parsed.netloc.replace(".", "").replace(":", "").isdigit():
                domain = parsed.netloc
                subdomain = None
                tld = None
            else:
                netloc_without_port = parsed.netloc.split(":")[0] if ":" in parsed.netloc else parsed.netloc
                domain_parts = netloc_without_port.split(".")
                domain = netloc_without_port
                subdomain = None
                tld = None

                if len(domain_parts) > 2:
                    subdomain = ".".join(domain_parts[:-2])
                    domain = ".".join(domain_parts[-2:])
                    tld = domain_parts[-1]
                elif len(domain_parts) == 2:
                    tld = domain_parts[-1]

            path = parsed.path or "/"
            path_segments = [seg for seg in path.split("/") if seg]
            path_depth = len(path_segments)

            parameters = parse_qs(parsed.query)
            parameter_count = len(parameters)

            url_hash = hashlib.md5(url.encode()).hexdigest()

            return URLMetadata(
                domain=domain,
                subdomain=subdomain,
                path=path,
                parameters=parameters,
                fragment=parsed.fragment or None,
                port=parsed.port,
                scheme=parsed.scheme,
                tld=tld,
                path_segments=path_segments,
                parameter_count=parameter_count,
                path_depth=path_depth,
                url_hash=url_hash,
            )

        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Failed to extract metadata", error=str(exc), url=url)
            return URLMetadata(domain=url, url_hash=hashlib.md5(url.encode()).hexdigest())

    @classmethod
    def categorize_url(cls, url_entry: URLEntry) -> str:
        """Automatically categorize URL based on domain and path."""
        if not url_entry.metadata:
            return "unknown"

        domain = url_entry.metadata.domain.lower()
        subdomain = url_entry.metadata.subdomain.lower() if url_entry.metadata.subdomain else ""
        path = url_entry.metadata.path.lower()

        full_domain = f"{subdomain}.{domain}" if subdomain else domain

        social_domains = [
            "twitter.com",
            "facebook.com",
            "linkedin.com",
            "instagram.com",
            "youtube.com",
            "tiktok.com",
            "reddit.com",
        ]
        if any(social in full_domain for social in social_domains):
            return "social_media"

        news_indicators = ["news", "blog", "article", "post", "story"]
        if any(indicator in full_domain for indicator in news_indicators) or any(
            indicator in path for indicator in news_indicators
        ):
            return "news_media"

        tech_indicators = ["docs", "api", "github", "stackoverflow", "wiki"]
        if any(indicator in full_domain or indicator in path for indicator in tech_indicators):
            return "documentation"

        ecommerce_indicators = ["shop", "store", "buy", "cart", "product"]
        if any(indicator in full_domain or indicator in path for indicator in ecommerce_indicators):
            return "ecommerce"

        edu_indicators = [".edu", "university", "college", "course", "learn"]
        if any(indicator in full_domain or indicator in path for indicator in edu_indicators):
            return "education"

        return "general"

    @classmethod
    def enrich_url_entry(cls, url_entry: URLEntry) -> URLEntry:
        """Enrich URL entry with metadata and categorization."""
        if not url_entry.validated:
            return url_entry

        url_entry.metadata = cls.extract_metadata(url_entry.url)

        if not url_entry.category:
            url_entry.category = cls.categorize_url(url_entry)

        url_entry.enriched = True
        return url_entry

    @classmethod
    def enrich_entry(cls, url_entry: URLEntry) -> URLEntry:
        """Backward compatible alias for enrich_url_entry."""
        return cls.enrich_url_entry(url_entry)


__all__ = ["URLEnricher"]
