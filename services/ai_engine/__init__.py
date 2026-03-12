"""Compatibility package mapping `services.ai_engine` to `services/ai-engine`."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "ai-engine")]
