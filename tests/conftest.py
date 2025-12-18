"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add services to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def anyio_backend():
    """Configure async backend."""
    return "asyncio"
