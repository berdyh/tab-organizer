"""Domain authentication mapping utilities."""

from datetime import datetime
from typing import Dict, Optional

import structlog

from .models import AuthenticationRequirement, DomainAuthMapping


logger = structlog.get_logger()


class DomainAuthMapper:
    """Manages domain authentication mapping and learning."""

    def __init__(self) -> None:
        self.domain_mappings: Dict[str, DomainAuthMapping] = {}

    def learn_domain_auth(self, domain: str, auth_requirement: AuthenticationRequirement) -> None:
        """Learn authentication requirements for a domain."""
        if domain not in self.domain_mappings:
            self.domain_mappings[domain] = DomainAuthMapping(
                domain=domain,
                auth_method=auth_requirement.detected_method,
                requires_auth=auth_requirement.detection_confidence > 0.5,
            )
        else:
            mapping = self.domain_mappings[domain]
            if auth_requirement.detection_confidence > 0.7:
                mapping.auth_method = auth_requirement.detected_method
                mapping.requires_auth = True
                mapping.last_verified = datetime.now()

        logger.info(
            "Domain auth mapping updated",
            domain=domain,
            method=auth_requirement.detected_method,
            confidence=auth_requirement.detection_confidence,
        )

    def get_domain_auth_info(self, domain: str) -> Optional[DomainAuthMapping]:
        """Get authentication information for a domain."""
        return self.domain_mappings.get(domain)

    def mark_auth_success(self, domain: str) -> None:
        """Mark successful authentication for a domain."""
        if domain in self.domain_mappings:
            self.domain_mappings[domain].success_count += 1
            self.domain_mappings[domain].last_verified = datetime.now()

    def mark_auth_failure(self, domain: str) -> None:
        """Mark failed authentication for a domain."""
        if domain in self.domain_mappings:
            self.domain_mappings[domain].failure_count += 1

    def get_all_mappings(self) -> Dict[str, DomainAuthMapping]:
        """Get all domain authentication mappings."""
        return self.domain_mappings.copy()


__all__ = ["DomainAuthMapper"]
