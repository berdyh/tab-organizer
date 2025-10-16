"""Routing helpers for service alias resolution."""

from typing import Dict, Tuple

SERVICE_ALIASES: Dict[str, Tuple[str, str]] = {
    # Frontend-friendly aliases that map to internal service names and optional path prefixes
    "input": ("url-input-service", "input"),
    "url-input": ("url-input-service", "input"),
    "auth": ("auth-service", "auth"),
}


def resolve_service_alias(service_name: str) -> Tuple[str, str]:
    """Resolve service aliases to canonical names and additional path prefixes."""
    return SERVICE_ALIASES.get(service_name, (service_name, ""))


__all__ = ["SERVICE_ALIASES", "resolve_service_alias"]

