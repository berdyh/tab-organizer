"""Compatibility package mapping `services.browser_engine` to `services/browser-engine`."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "browser-engine")]
