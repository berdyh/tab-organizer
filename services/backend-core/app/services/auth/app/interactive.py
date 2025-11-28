"""Interactive authentication workflows using browser automation."""

import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import HTTPException
import structlog
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions

from .models import AuthSession


logger = structlog.get_logger()


class InteractiveAuthenticator:
    """Handles popup-based authentication using browser automation."""

    def __init__(self) -> None:
        self.active_sessions: Dict[str, AuthSession] = {}
        self.browser_pool: Dict[str, Any] = {}
        self.max_concurrent_browsers = 5

    async def authenticate_with_popup(
        self,
        domain: str,
        auth_method: str,
        credentials: Dict[str, Any],
        login_url: str,
        browser_type: str = "chrome",
        headless: bool = True,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """Perform authentication using browser automation."""
        try:
            if browser_type == "playwright":
                return await self._authenticate_with_playwright(
                    domain, auth_method, credentials, login_url, headless, timeout
                )
            return await self._authenticate_with_selenium(
                domain, auth_method, credentials, login_url, browser_type, headless, timeout
            )
        except Exception as exc:
            logger.error(
                "Interactive authentication failed",
                domain=domain,
                error=str(exc),
                auth_method=auth_method,
            )
            raise HTTPException(status_code=500, detail=f"Authentication failed: {str(exc)}") from exc

    async def _authenticate_with_playwright(
        self,
        domain: str,
        auth_method: str,
        credentials: Dict[str, Any],
        login_url: str,
        headless: bool,
        timeout: int,
    ) -> Dict[str, Any]:
        """Authenticate using Playwright (Chromium)."""
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=headless)
            context = await browser.new_context()
            page = await context.new_page()

            try:
                await page.goto(login_url, timeout=timeout * 1000)
                await page.wait_for_load_state("networkidle")

                if auth_method == "form":
                    success = await self._handle_form_auth_playwright(page, credentials, timeout)
                elif auth_method == "oauth":
                    success = await self._handle_oauth_playwright(page, credentials, timeout)
                else:
                    raise ValueError(f"Unsupported auth method: {auth_method}")

                if success:
                    cookies = await context.cookies()
                    session_data = {
                        "cookies": {cookie["name"]: cookie["value"] for cookie in cookies},
                        "user_agent": await page.evaluate("navigator.userAgent"),
                        "current_url": page.url,
                    }
                    session_id = self._create_session(domain, auth_method, session_data)
                    return {
                        "success": True,
                        "session_id": session_id,
                        "message": "Authentication successful",
                        "session_data": session_data,
                    }

                return {
                    "success": False,
                    "message": "Authentication failed - invalid credentials or form not found",
                }
            finally:
                await browser.close()

    async def _authenticate_with_selenium(
        self,
        domain: str,
        auth_method: str,
        credentials: Dict[str, Any],
        login_url: str,
        browser_type: str,
        headless: bool,
        timeout: int,
    ) -> Dict[str, Any]:
        """Authenticate using Selenium."""
        driver = None
        try:
            if browser_type == "chrome":
                options = ChromeOptions()
                if headless:
                    options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                driver = webdriver.Chrome(options=options)
            elif browser_type == "firefox":
                options = FirefoxOptions()
                if headless:
                    options.add_argument("--headless")
                driver = webdriver.Firefox(options=options)
            else:
                raise ValueError(f"Unsupported browser type: {browser_type}")

            driver.set_page_load_timeout(timeout)
            driver.implicitly_wait(10)
            driver.get(login_url)

            if auth_method == "form":
                success = self._handle_form_auth_selenium(driver, credentials, timeout)
            elif auth_method == "oauth":
                success = self._handle_oauth_selenium(driver, credentials, timeout)
            else:
                raise ValueError(f"Unsupported auth method: {auth_method}")

            if success:
                cookies = {cookie["name"]: cookie["value"] for cookie in driver.get_cookies()}
                session_data = {
                    "cookies": cookies,
                    "user_agent": driver.execute_script("return navigator.userAgent;"),
                    "current_url": driver.current_url,
                }
                session_id = self._create_session(domain, auth_method, session_data)
                return {
                    "success": True,
                    "session_id": session_id,
                    "message": "Authentication successful",
                    "session_data": session_data,
                }

            return {
                "success": False,
                "message": "Authentication failed - invalid credentials or form not found",
            }
        finally:
            if driver:
                driver.quit()

    async def _handle_form_auth_playwright(self, page: Page, credentials: Dict[str, Any], timeout: int) -> bool:
        """Handle form-based authentication with Playwright."""
        try:
            username_selectors = [
                'input[name="username"]',
                'input[name="email"]',
                'input[name="login"]',
                'input[type="email"]',
                'input[id*="username"]',
                'input[id*="email"]',
                'input[placeholder*="username"]',
                'input[placeholder*="email"]',
            ]

            password_selectors = [
                'input[type="password"]',
                'input[name="password"]',
                'input[name="pass"]',
                'input[id*="password"]',
                'input[id*="pass"]',
            ]

            username_filled = False
            for selector in username_selectors:
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                    await page.fill(selector, credentials.get("username", credentials.get("email", "")))
                    username_filled = True
                    break
                except Exception:
                    continue

            if not username_filled:
                logger.warning("Username field not found", selectors=username_selectors)
                return False

            password_filled = False
            for selector in password_selectors:
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                    await page.fill(selector, credentials.get("password", ""))
                    password_filled = True
                    break
                except Exception:
                    continue

            if not password_filled:
                logger.warning("Password field not found", selectors=password_selectors)
                return False

            submit_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:has-text("Login")',
                'button:has-text("Sign in")',
                'button:has-text("Log in")',
                "form button",
            ]

            for selector in submit_selectors:
                try:
                    await page.click(selector)
                    break
                except Exception:
                    continue

            try:
                await page.wait_for_load_state("networkidle", timeout=timeout * 1000)
                error_indicators = await page.query_selector_all(r"text=/error|invalid|incorrect|failed/i")
                if len(error_indicators) == 0:
                    return True
            except TimeoutException:
                pass

            return False
        except Exception as exc:
            logger.error("Form authentication failed", error=str(exc))
            return False

    def _handle_form_auth_selenium(self, driver: Any, credentials: Dict[str, Any], timeout: int) -> bool:
        """Handle form-based authentication with Selenium."""
        try:
            wait = WebDriverWait(driver, timeout)
            username_selectors = [
                (By.NAME, "username"),
                (By.NAME, "email"),
                (By.NAME, "login"),
                (By.CSS_SELECTOR, 'input[type="email"]'),
                (By.ID, "username"),
                (By.ID, "email"),
            ]

            username_element = None
            for by, selector in username_selectors:
                try:
                    username_element = wait.until(EC.presence_of_element_located((by, selector)))
                    break
                except TimeoutException:
                    continue

            if not username_element:
                logger.warning("Username field not found")
                return False

            username_element.clear()
            username_element.send_keys(credentials.get("username", credentials.get("email", "")))

            password_element = None
            password_selectors = [
                (By.CSS_SELECTOR, 'input[type="password"]'),
                (By.NAME, "password"),
                (By.NAME, "pass"),
            ]

            for by, selector in password_selectors:
                try:
                    password_element = driver.find_element(by, selector)
                    break
                except Exception:
                    continue

            if not password_element:
                logger.warning("Password field not found")
                return False

            password_element.clear()
            password_element.send_keys(credentials.get("password", ""))

            submit_selectors = [
                (By.CSS_SELECTOR, 'button[type="submit"]'),
                (By.CSS_SELECTOR, 'input[type="submit"]'),
                (By.XPATH, "//button[contains(text(), 'Login') or contains(text(), 'Sign in')]"),
            ]

            for by, selector in submit_selectors:
                try:
                    submit_element = driver.find_element(by, selector)
                    submit_element.click()
                    break
                except Exception:
                    continue

            time.sleep(3)

            error_selectors = [
                "//div[contains(@class, 'error')]",
                "//span[contains(@class, 'error')]",
                "//div[contains(text(), 'Invalid') or contains(text(), 'Error')]",
            ]

            for selector in error_selectors:
                try:
                    error_element = driver.find_element(By.XPATH, selector)
                    if error_element.is_displayed():
                        return False
                except Exception:
                    continue

            return True
        except Exception as exc:
            logger.error("Selenium form authentication failed", error=str(exc))
            return False

    async def _handle_oauth_playwright(self, page: Page, credentials: Dict[str, Any], timeout: int) -> bool:
        """Handle OAuth authentication with Playwright."""
        try:
            oauth_selectors = [
                'button:has-text("Sign in with Google")',
                'button:has-text("Continue with Google")',
                'button:has-text("Sign in with GitHub")',
                'button:has-text("Continue with GitHub")',
                'a[href*="oauth"]',
                'button[class*="oauth"]',
            ]

            for selector in oauth_selectors:
                try:
                    await page.click(selector)
                    await page.wait_for_load_state("networkidle", timeout=timeout * 1000)
                    return True
                except Exception:
                    continue

            return False
        except Exception as exc:
            logger.error("OAuth authentication failed", error=str(exc))
            return False

    def _handle_oauth_selenium(self, driver: Any, credentials: Dict[str, Any], timeout: int) -> bool:
        """Handle OAuth authentication with Selenium."""
        try:
            oauth_selectors = [
                (By.XPATH, "//button[contains(text(), 'Sign in with')]"),
                (By.CSS_SELECTOR, "button[class*='oauth']"),
                (By.CSS_SELECTOR, "a[href*='oauth']"),
            ]

            for by, selector in oauth_selectors:
                try:
                    oauth_element = driver.find_element(by, selector)
                    oauth_element.click()
                    time.sleep(5)
                    return True
                except Exception:
                    continue

            return False
        except Exception as exc:
            logger.error("Selenium OAuth authentication failed", error=str(exc))
            return False

    def _create_session(self, domain: str, auth_method: str, session_data: Dict[str, Any]) -> str:
        """Create and store an authentication session."""
        session_id = str(uuid.uuid4())

        session = AuthSession(
            session_id=session_id,
            domain=domain,
            auth_method=auth_method,
            cookies=session_data.get("cookies", {}),
            headers={"User-Agent": session_data.get("user_agent", "")},
            expires_at=datetime.now() + timedelta(hours=24),
        )

        self.active_sessions[session_id] = session
        logger.info("Authentication session created", session_id=session_id, domain=domain, auth_method=auth_method)
        return session_id

    def get_session(self, session_id: str) -> Optional[AuthSession]:
        """Retrieve an active session."""
        session = self.active_sessions.get(session_id)
        if session and session.is_active:
            if session.expires_at and datetime.now() > session.expires_at:
                session.is_active = False
                return None
            session.last_used = datetime.now()
            return session
        return None

    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate an authentication session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].is_active = False
            return True
        return False

    def cleanup_expired_sessions(self) -> None:
        """Remove expired sessions."""
        now = datetime.now()
        expired_sessions = [
            sid for sid, session in self.active_sessions.items() if session.expires_at and now > session.expires_at
        ]

        for session_id in expired_sessions:
            del self.active_sessions[session_id]

        logger.info("Cleaned up expired sessions", count=len(expired_sessions))


__all__ = ["InteractiveAuthenticator"]
