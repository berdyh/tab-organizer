"""Compatibility package mapping `services.web_ui` to `services/web-ui`."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "web-ui")]
