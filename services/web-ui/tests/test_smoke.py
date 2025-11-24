import unittest
import sys
import os
from unittest.mock import MagicMock

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock streamlit
sys.modules["streamlit"] = MagicMock()
sys.modules["streamlit.runtime.scriptrunner"] = MagicMock()

class TestImports(unittest.TestCase):
    def test_import_pages(self):
        """Test that page modules can be imported."""
        try:
            from pages import sessions
            from pages import url_input
            from pages import scraping
            from pages import analysis
            from pages import clustering
            from pages import chatbot
            from pages import export
        except Exception as e:
            self.fail(f"Failed to import pages: {e}")

if __name__ == '__main__':
    unittest.main()
