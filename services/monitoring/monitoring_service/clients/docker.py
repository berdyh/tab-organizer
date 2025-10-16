"""Utilities for initialising Docker SDK clients inside containers."""

from __future__ import annotations

from typing import Optional

import docker
from docker import DockerClient


def init_docker_client(socket_url: Optional[str] = None) -> DockerClient:
    """Return a ready-to-use Docker client.

    Preference is given to the configured socket URL; if that fails we fall
    back to Docker's built-in environment discovery. A short `ping` ensures
    the client is actually usable before returning it to callers.
    """

    last_error: Optional[Exception] = None

    if socket_url:
        try:
            client = docker.DockerClient(base_url=socket_url)
            client.ping()
            return client
        except Exception as exc:  # pragma: no cover - depends on host runtime
            last_error = exc

    try:
        client = docker.from_env()
        client.ping()
        return client
    except Exception as exc:  # pragma: no cover - depends on host runtime
        last_error = exc

    raise RuntimeError(
        f"Failed to initialise Docker client: {last_error}"
    ) from last_error
