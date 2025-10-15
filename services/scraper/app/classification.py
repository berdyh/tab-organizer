"""URL classification utilities."""

from __future__ import annotations

from typing import List
from urllib.parse import urlparse

from .logging import get_logger
from .models import QueueType, URLClassification, URLRequest

logger = get_logger()


class URLClassifier:
    """Determine which queue a URL should belong to and whether it needs auth."""

    def __init__(self) -> None:
        self.auth_indicators = {
            "url_patterns": [
                "login",
                "signin",
                "auth",
                "account",
                "dashboard",
                "profile",
                "admin",
                "secure",
                "private",
                "member",
                "user",
                "my",
            ],
            "domain_patterns": [
                "admin.",
                "secure.",
                "my.",
                "account.",
                "portal.",
                "app.",
            ],
            "path_patterns": [
                "/admin/",
                "/dashboard/",
                "/account/",
                "/profile/",
                "/secure/",
                "/private/",
                "/member/",
                "/user/",
                "/my/",
                "/portal/",
            ],
        }
        self.public_indicators = [
            "blog",
            "news",
            "about",
            "contact",
            "help",
            "faq",
            "public",
            "home",
            "index",
            "main",
            "welcome",
        ]

    async def classify_urls(
        self, urls: List[URLRequest], correlation_id: str
    ) -> List[URLClassification]:
        """Classify URLs into appropriate processing queues."""
        classifications: List[URLClassification] = []
        for url_request in urls:
            url = str(url_request.url)
            classification = await self._classify_single_url(
                url, url_request, correlation_id
            )
            classifications.append(classification)
        return classifications

    async def _classify_single_url(
        self, url: str, url_request: URLRequest, correlation_id: str
    ) -> URLClassification:
        """Classify a single URL for authentication requirements."""
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path.lower()
        url_lower = url.lower()

        classification = URLClassification(
            url=url,
            queue_type=QueueType.PUBLIC,
            requires_auth=False,
            confidence=0.0,
            domain=domain,
            priority=url_request.priority,
        )

        auth_score = 0.0
        auth_indicators_found = []

        for pattern in self.auth_indicators["url_patterns"]:
            if pattern in url_lower:
                auth_score += 0.3
                auth_indicators_found.append(f"url_pattern:{pattern}")

        for pattern in self.auth_indicators["domain_patterns"]:
            if domain.startswith(pattern):
                auth_score += 0.4
                auth_indicators_found.append(f"domain_pattern:{pattern}")

        for pattern in self.auth_indicators["path_patterns"]:
            if pattern in path:
                auth_score += 0.5
                auth_indicators_found.append(f"path_pattern:{pattern}")

        for pattern in self.public_indicators:
            if pattern in url_lower:
                auth_score -= 0.2

        if url_request.force_auth_check:
            auth_score += 0.8
            auth_indicators_found.append("force_auth_check")

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
            correlation_id=correlation_id,
        )

        return classification
