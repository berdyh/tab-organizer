# Browser Engine Service

This service handles browser automation tasks:
- **Web Scraping**: Batch scraping of URLs.
- **Authentication Detection**: Checking if websites require login.

## Architecture

Built with FastAPI, utilizing Playwright/Selenium/Scrapy.

## Structure

- `main.py`: Application entry point.
- `app/scraper/`: Scraper service logic.
- `app/auth/`: Browser-based authentication logic.
