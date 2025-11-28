"""In-memory storage helpers for URL inputs."""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Optional

from .models import URLInput


class URLInputStorage:
    """Simple in-memory storage for URL inputs."""

    def __init__(self) -> None:
        self._inputs: Dict[str, URLInput] = {}

    @property
    def data(self) -> Dict[str, URLInput]:
        """Direct access to the underlying storage dictionary."""
        return self._inputs

    def all(self) -> Dict[str, URLInput]:
        """Return all stored URL inputs."""
        return self._inputs

    def items(self) -> Iterable[tuple[str, URLInput]]:
        """Return storage items."""
        return self._inputs.items()

    def values(self) -> Iterable[URLInput]:
        """Return stored URL input values."""
        return self._inputs.values()

    def get(self, input_id: str) -> Optional[URLInput]:
        """Return a URL input by id if present."""
        return self._inputs.get(input_id)

    def set(self, input_id: str, url_input: URLInput) -> None:
        """Store or replace a URL input."""
        self._inputs[input_id] = url_input

    def delete(self, input_id: str) -> None:
        """Remove a URL input if it exists."""
        self._inputs.pop(input_id, None)

    def exists(self, input_id: str) -> bool:
        """Return True if an input with the given id exists."""
        return input_id in self._inputs

    def clear(self) -> None:
        """Remove all stored URL inputs."""
        self._inputs.clear()

    def __contains__(self, input_id: str) -> bool:
        return self.exists(input_id)

    def __iter__(self) -> Iterator[tuple[str, URLInput]]:
        return iter(self._inputs.items())


url_input_storage = URLInputStorage()


__all__ = ["URLInputStorage", "url_input_storage"]
