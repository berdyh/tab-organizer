"""Robots.txt compliance helpers."""

from __future__ import annotations

import sys
from typing import Dict
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from .logging import get_logger

logger = get_logger()


class RobotsChecker:
    """Check robots.txt compliance."""

    def __init__(self) -> None:
        self.robots_cache: Dict[str, RobotFileParser] = {}

    def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """Check if URL can be fetched according to robots.txt."""
        try:
            parsed_url = urlparse(url)
            domain = f"{parsed_url.scheme}://{parsed_url.netloc}"

            parser_factory = self._get_parser_factory()

            if domain not in self.robots_cache:
                robots_url = urljoin(domain, "/robots.txt")
                parser = parser_factory()
                parser.set_url(robots_url)
                try:
                    parser.read()
                    self.robots_cache[domain] = parser
                except Exception:
                    parser = parser_factory()
                    parser.set_url(robots_url)
                    self.robots_cache[domain] = parser

            return self.robots_cache[domain].can_fetch(user_agent, url)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Robots.txt check failed", url=url, error=str(exc))
            return True

    @staticmethod
    def _get_parser_factory():
        """Return RobotFileParser factory, allowing tests to patch `main.RobotFileParser`."""
        main_module = sys.modules.get("main")
        if main_module and hasattr(main_module, "RobotFileParser"):
            return getattr(main_module, "RobotFileParser")
        return RobotFileParser
