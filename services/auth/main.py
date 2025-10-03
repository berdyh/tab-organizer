"""Authentication Service - Handles website authentication requirements."""

import time
import json
import hashlib
import base64
import re
import uuid
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from urllib.parse import urlparse, urljoin, parse_qs
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import queue
import threading

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, HttpUrl
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import structlog

# Import browser automation libraries
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import TimeoutException, WebDriverException
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Data Models
@dataclass
class AuthenticationRequirement:
    """Represents an authentication requirement detected for a URL/domain."""
    url: str
    domain: str
    detected_method: str  # form, oauth, captcha, etc.
    auth_indicators: List[str]  # Login form elements, redirect URLs, etc.
    priority: int = 1  # Higher priority for domains with more URLs
    detection_confidence: float = 0.0  # 0.0-1.0 confidence in auth requirement
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class DomainAuthMapping:
    """Maps domain to authentication requirements and credentials."""
    domain: str
    auth_method: str
    requires_auth: bool
    login_url: Optional[str] = None
    form_selectors: Dict[str, str] = field(default_factory=dict)
    oauth_config: Dict[str, Any] = field(default_factory=dict)
    last_verified: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0

@dataclass
class AuthSession:
    """Represents an active authentication session."""
    session_id: str
    domain: str
    auth_method: str
    cookies: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    tokens: Dict[str, str] = field(default_factory=dict)  # OAuth tokens, JWT, etc.
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    last_used: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    user_agent: Optional[str] = None

@dataclass
class OAuthConfig:
    """OAuth 2.0 configuration for a provider."""
    provider: str
    client_id: str
    client_secret: str
    authorization_url: str
    token_url: str
    redirect_uri: str
    scope: List[str] = field(default_factory=list)
    additional_params: Dict[str, str] = field(default_factory=dict)

@dataclass
class AuthenticationTask:
    """Represents a queued authentication task."""
    task_id: str
    domain: str
    auth_method: str
    credentials: Dict[str, Any]
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, processing, completed, failed
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

# Pydantic Models for API
class URLAnalysisRequest(BaseModel):
    url: HttpUrl
    response_content: Optional[str] = None
    status_code: Optional[int] = None
    headers: Optional[Dict[str, str]] = None

class CredentialStoreRequest(BaseModel):
    domain: str
    auth_method: str
    credentials: Dict[str, Any]
    login_url: Optional[str] = None

class AuthDetectionResponse(BaseModel):
    requires_auth: bool
    detected_method: str
    confidence: float
    indicators: List[str]
    recommended_action: str

class InteractiveAuthRequest(BaseModel):
    domain: str
    auth_method: str
    credentials: Dict[str, Any]
    login_url: str
    browser_type: str = "chrome"  # chrome, firefox, playwright
    headless: bool = True
    timeout: int = 30

class OAuthAuthRequest(BaseModel):
    domain: str
    provider: str
    client_id: str
    client_secret: str
    redirect_uri: str
    scope: Optional[List[str]] = None

class SessionRequest(BaseModel):
    domain: str
    session_data: Optional[Dict[str, Any]] = None

class AuthTaskResponse(BaseModel):
    task_id: str
    status: str
    message: str
    session_id: Optional[str] = None

# Core Classes
class AuthenticationDetector:
    """Detects authentication requirements from HTTP responses and page content."""
    
    def __init__(self):
        self.auth_patterns = {
            'login_forms': [
                r'<form[^>]*(?:login|signin|auth)[^>]*>',
                r'<input[^>]*type=["\']password["\'][^>]*>',
                r'<input[^>]*name=["\'](?:password|pass|pwd)["\'][^>]*>',
            ],
            'login_links': [
                r'<a[^>]*href=["\'][^"\']*(?:login|signin|auth)[^"\']*["\'][^>]*>',
                r'href=["\'][^"\']*(?:login|signin|auth)[^"\']*["\']',
            ],
            'oauth_indicators': [
                r'oauth',
                r'Sign in with Google',
                r'Sign in with GitHub',
                r'Sign in with Microsoft',
                r'Continue with',
            ],
            'auth_redirects': [
                r'window\.location.*(?:login|signin|auth)',
                r'redirect.*(?:login|signin|auth)',
            ],
            'protected_content': [
                r'Please log in',
                r'Access denied',
                r'Authentication required',
                r'You must be logged in',
                r'Sign in to continue',
            ]
        }
        
        self.auth_status_codes = {401, 403, 407}  # Unauthorized, Forbidden, Proxy Auth Required
        
    async def detect_auth_required(self, url: str, response_content: str = None, 
                                 status_code: int = None, headers: Dict[str, str] = None) -> AuthenticationRequirement:
        """Analyze response to determine if authentication is required."""
        domain = urlparse(url).netloc
        indicators = []
        confidence = 0.0
        detected_method = "unknown"
        
        # Check status code indicators
        if status_code in self.auth_status_codes:
            indicators.append(f"HTTP {status_code} status code")
            confidence += 0.8
            detected_method = "form"  # Default assumption
            
        # Check headers for auth requirements
        if headers:
            www_auth = headers.get('www-authenticate', '').lower()
            if www_auth:
                indicators.append(f"WWW-Authenticate header: {www_auth}")
                confidence += 0.9
                if 'basic' in www_auth:
                    detected_method = "basic"
                elif 'bearer' in www_auth:
                    detected_method = "oauth"
                    
        # Analyze page content if available
        if response_content:
            content_indicators, content_method, content_confidence = self._analyze_content(response_content)
            indicators.extend(content_indicators)
            confidence = max(confidence, content_confidence)
            if detected_method == "unknown":
                detected_method = content_method
                
        # Normalize confidence to 0-1 range
        confidence = min(confidence, 1.0)
        
        return AuthenticationRequirement(
            url=url,
            domain=domain,
            detected_method=detected_method,
            auth_indicators=indicators,
            detection_confidence=confidence
        )
    
    def _analyze_content(self, content: str) -> tuple[List[str], str, float]:
        """Analyze page content for authentication indicators."""
        indicators = []
        confidence = 0.0
        detected_method = "form"  # Default
        
        content_lower = content.lower()
        
        # Check for login forms
        for pattern in self.auth_patterns['login_forms']:
            if re.search(pattern, content, re.IGNORECASE):
                indicators.append("Login form detected")
                confidence += 0.7
                detected_method = "form"
                break
                
        # Check for OAuth indicators
        for pattern in self.auth_patterns['oauth_indicators']:
            if pattern.lower() in content_lower:
                indicators.append(f"OAuth indicator: {pattern}")
                confidence += 0.6
                detected_method = "oauth"
                
        # Check for login links
        for pattern in self.auth_patterns['login_links']:
            if re.search(pattern, content, re.IGNORECASE):
                indicators.append("Login link detected")
                confidence += 0.5
                
        # Check for protected content messages
        for pattern in self.auth_patterns['protected_content']:
            if pattern.lower() in content_lower:
                indicators.append(f"Protected content message: {pattern}")
                confidence += 0.8
                
        # Check for auth redirects in JavaScript
        for pattern in self.auth_patterns['auth_redirects']:
            if re.search(pattern, content, re.IGNORECASE):
                indicators.append("Authentication redirect detected")
                confidence += 0.6
                
        return indicators, detected_method, confidence
    
    def classify_auth_method(self, content: str, headers: Dict[str, str] = None) -> str:
        """Classify the type of authentication method required."""
        content_lower = content.lower()
        
        # OAuth detection
        oauth_providers = ['google', 'github', 'microsoft', 'facebook', 'twitter', 'linkedin']
        for provider in oauth_providers:
            if f'sign in with {provider}' in content_lower or f'oauth/{provider}' in content_lower:
                return "oauth"
                
        # Form-based authentication
        if re.search(r'<input[^>]*type=["\']password["\']', content, re.IGNORECASE):
            return "form"
            
        # Basic authentication
        if headers and 'www-authenticate' in headers:
            auth_header = headers['www-authenticate'].lower()
            if 'basic' in auth_header:
                return "basic"
            elif 'bearer' in auth_header:
                return "bearer"
                
        # CAPTCHA detection
        if 'captcha' in content_lower or 'recaptcha' in content_lower:
            return "captcha"
            
        return "unknown"

class SecureCredentialStore:
    """Handles secure storage and retrieval of authentication credentials."""
    
    def __init__(self, master_password: str = "default_master_key"):
        self.master_password = master_password.encode()
        self._fernet = self._create_fernet_key()
        self.credentials_store: Dict[str, bytes] = {}
        
    def _create_fernet_key(self) -> Fernet:
        """Create Fernet encryption key from master password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt_for_demo',  # In production, use random salt per credential
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_password))
        return Fernet(key)
    
    def store_credentials(self, domain: str, credentials: Dict[str, Any]) -> bool:
        """Store encrypted credentials for a domain."""
        try:
            # Serialize and encrypt credentials
            credentials_json = json.dumps(credentials)
            encrypted_credentials = self._fernet.encrypt(credentials_json.encode())
            
            # Store with domain as key
            self.credentials_store[domain] = encrypted_credentials
            
            logger.info("Credentials stored securely", domain=domain)
            return True
            
        except Exception as e:
            logger.error("Failed to store credentials", domain=domain, error=str(e))
            return False
    
    def retrieve_credentials(self, domain: str) -> Optional[Dict[str, Any]]:
        """Retrieve and decrypt credentials for a domain."""
        try:
            if domain not in self.credentials_store:
                return None
                
            # Decrypt and deserialize credentials
            encrypted_credentials = self.credentials_store[domain]
            decrypted_json = self._fernet.decrypt(encrypted_credentials).decode()
            credentials = json.loads(decrypted_json)
            
            logger.info("Credentials retrieved", domain=domain)
            return credentials
            
        except Exception as e:
            logger.error("Failed to retrieve credentials", domain=domain, error=str(e))
            return None
    
    def delete_credentials(self, domain: str) -> bool:
        """Delete stored credentials for a domain."""
        try:
            if domain in self.credentials_store:
                del self.credentials_store[domain]
                logger.info("Credentials deleted", domain=domain)
                return True
            return False
            
        except Exception as e:
            logger.error("Failed to delete credentials", domain=domain, error=str(e))
            return False
    
    def list_stored_domains(self) -> List[str]:
        """List all domains with stored credentials."""
        return list(self.credentials_store.keys())

class DomainAuthMapper:
    """Manages domain authentication mapping and learning."""
    
    def __init__(self):
        self.domain_mappings: Dict[str, DomainAuthMapping] = {}
        
    def learn_domain_auth(self, domain: str, auth_requirement: AuthenticationRequirement) -> None:
        """Learn authentication requirements for a domain."""
        if domain not in self.domain_mappings:
            self.domain_mappings[domain] = DomainAuthMapping(
                domain=domain,
                auth_method=auth_requirement.detected_method,
                requires_auth=auth_requirement.detection_confidence > 0.5
            )
        else:
            # Update existing mapping with new information
            mapping = self.domain_mappings[domain]
            if auth_requirement.detection_confidence > 0.7:
                mapping.auth_method = auth_requirement.detected_method
                mapping.requires_auth = True
                mapping.last_verified = datetime.now()
                
        logger.info("Domain auth mapping updated", domain=domain, 
                   method=auth_requirement.detected_method,
                   confidence=auth_requirement.detection_confidence)
    
    def get_domain_auth_info(self, domain: str) -> Optional[DomainAuthMapping]:
        """Get authentication information for a domain."""
        return self.domain_mappings.get(domain)
    
    def mark_auth_success(self, domain: str) -> None:
        """Mark successful authentication for a domain."""
        if domain in self.domain_mappings:
            self.domain_mappings[domain].success_count += 1
            self.domain_mappings[domain].last_verified = datetime.now()
            
    def mark_auth_failure(self, domain: str) -> None:
        """Mark failed authentication for a domain."""
        if domain in self.domain_mappings:
            self.domain_mappings[domain].failure_count += 1
    
    def get_all_mappings(self) -> Dict[str, DomainAuthMapping]:
        """Get all domain authentication mappings."""
        return self.domain_mappings.copy()

class InteractiveAuthenticator:
    """Handles popup-based authentication using browser automation."""
    
    def __init__(self):
        self.active_sessions: Dict[str, AuthSession] = {}
        self.browser_pool = {}
        self.max_concurrent_browsers = 5
        
    async def authenticate_with_popup(self, domain: str, auth_method: str, 
                                    credentials: Dict[str, Any], login_url: str,
                                    browser_type: str = "chrome", headless: bool = True,
                                    timeout: int = 30) -> Dict[str, Any]:
        """Perform authentication using browser automation."""
        try:
            if browser_type == "playwright":
                return await self._authenticate_with_playwright(
                    domain, auth_method, credentials, login_url, headless, timeout
                )
            else:
                return await self._authenticate_with_selenium(
                    domain, auth_method, credentials, login_url, browser_type, headless, timeout
                )
                
        except Exception as e:
            logger.error("Interactive authentication failed", 
                        domain=domain, error=str(e), auth_method=auth_method)
            raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")
    
    async def _authenticate_with_playwright(self, domain: str, auth_method: str,
                                          credentials: Dict[str, Any], login_url: str,
                                          headless: bool, timeout: int) -> Dict[str, Any]:
        """Authenticate using Playwright."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                # Navigate to login page
                await page.goto(login_url, timeout=timeout * 1000)
                await page.wait_for_load_state('networkidle')
                
                if auth_method == "form":
                    success = await self._handle_form_auth_playwright(page, credentials, timeout)
                elif auth_method == "oauth":
                    success = await self._handle_oauth_playwright(page, credentials, timeout)
                else:
                    raise ValueError(f"Unsupported auth method: {auth_method}")
                
                if success:
                    # Extract session data
                    cookies = await context.cookies()
                    session_data = {
                        'cookies': {cookie['name']: cookie['value'] for cookie in cookies},
                        'user_agent': await page.evaluate('navigator.userAgent'),
                        'current_url': page.url
                    }
                    
                    # Create session
                    session_id = self._create_session(domain, auth_method, session_data)
                    
                    return {
                        'success': True,
                        'session_id': session_id,
                        'message': 'Authentication successful',
                        'session_data': session_data
                    }
                else:
                    return {
                        'success': False,
                        'message': 'Authentication failed - invalid credentials or form not found'
                    }
                    
            finally:
                await browser.close()
    
    async def _authenticate_with_selenium(self, domain: str, auth_method: str,
                                        credentials: Dict[str, Any], login_url: str,
                                        browser_type: str, headless: bool, timeout: int) -> Dict[str, Any]:
        """Authenticate using Selenium."""
        driver = None
        try:
            # Setup browser options
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
            
            # Navigate to login page
            driver.get(login_url)
            
            if auth_method == "form":
                success = self._handle_form_auth_selenium(driver, credentials, timeout)
            elif auth_method == "oauth":
                success = self._handle_oauth_selenium(driver, credentials, timeout)
            else:
                raise ValueError(f"Unsupported auth method: {auth_method}")
            
            if success:
                # Extract session data
                cookies = {cookie['name']: cookie['value'] for cookie in driver.get_cookies()}
                session_data = {
                    'cookies': cookies,
                    'user_agent': driver.execute_script("return navigator.userAgent;"),
                    'current_url': driver.current_url
                }
                
                # Create session
                session_id = self._create_session(domain, auth_method, session_data)
                
                return {
                    'success': True,
                    'session_id': session_id,
                    'message': 'Authentication successful',
                    'session_data': session_data
                }
            else:
                return {
                    'success': False,
                    'message': 'Authentication failed - invalid credentials or form not found'
                }
                
        finally:
            if driver:
                driver.quit()
    
    async def _handle_form_auth_playwright(self, page: Page, credentials: Dict[str, Any], timeout: int) -> bool:
        """Handle form-based authentication with Playwright."""
        try:
            # Common selectors for username/email fields
            username_selectors = [
                'input[name="username"]', 'input[name="email"]', 'input[name="login"]',
                'input[type="email"]', 'input[id*="username"]', 'input[id*="email"]',
                'input[placeholder*="username"]', 'input[placeholder*="email"]'
            ]
            
            # Common selectors for password fields
            password_selectors = [
                'input[type="password"]', 'input[name="password"]', 'input[name="pass"]',
                'input[id*="password"]', 'input[id*="pass"]'
            ]
            
            # Find and fill username/email field
            username_filled = False
            for selector in username_selectors:
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                    await page.fill(selector, credentials.get('username', credentials.get('email', '')))
                    username_filled = True
                    break
                except:
                    continue
            
            if not username_filled:
                logger.warning("Username field not found", selectors=username_selectors)
                return False
            
            # Find and fill password field
            password_filled = False
            for selector in password_selectors:
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                    await page.fill(selector, credentials.get('password', ''))
                    password_filled = True
                    break
                except:
                    continue
            
            if not password_filled:
                logger.warning("Password field not found", selectors=password_selectors)
                return False
            
            # Submit form
            submit_selectors = [
                'button[type="submit"]', 'input[type="submit"]', 'button:has-text("Login")',
                'button:has-text("Sign in")', 'button:has-text("Log in")', 'form button'
            ]
            
            for selector in submit_selectors:
                try:
                    await page.click(selector)
                    break
                except:
                    continue
            
            # Wait for navigation or success indicators
            try:
                await page.wait_for_load_state('networkidle', timeout=timeout * 1000)
                
                # Check for success indicators (absence of error messages, presence of user content)
                error_indicators = await page.query_selector_all('text=/error|invalid|incorrect|failed/i')
                if len(error_indicators) == 0:
                    return True
                    
            except TimeoutException:
                pass
            
            return False
            
        except Exception as e:
            logger.error("Form authentication failed", error=str(e))
            return False
    
    def _handle_form_auth_selenium(self, driver, credentials: Dict[str, Any], timeout: int) -> bool:
        """Handle form-based authentication with Selenium."""
        try:
            wait = WebDriverWait(driver, timeout)
            
            # Common selectors for username/email fields
            username_selectors = [
                (By.NAME, "username"), (By.NAME, "email"), (By.NAME, "login"),
                (By.CSS_SELECTOR, 'input[type="email"]'), (By.ID, "username"), (By.ID, "email")
            ]
            
            # Find and fill username/email field
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
            username_element.send_keys(credentials.get('username', credentials.get('email', '')))
            
            # Find and fill password field
            password_element = None
            password_selectors = [
                (By.CSS_SELECTOR, 'input[type="password"]'), (By.NAME, "password"), (By.NAME, "pass")
            ]
            
            for by, selector in password_selectors:
                try:
                    password_element = driver.find_element(by, selector)
                    break
                except:
                    continue
            
            if not password_element:
                logger.warning("Password field not found")
                return False
            
            password_element.clear()
            password_element.send_keys(credentials.get('password', ''))
            
            # Submit form
            submit_selectors = [
                (By.CSS_SELECTOR, 'button[type="submit"]'), (By.CSS_SELECTOR, 'input[type="submit"]'),
                (By.XPATH, "//button[contains(text(), 'Login') or contains(text(), 'Sign in')]")
            ]
            
            for by, selector in submit_selectors:
                try:
                    submit_element = driver.find_element(by, selector)
                    submit_element.click()
                    break
                except:
                    continue
            
            # Wait for page to load and check for success
            time.sleep(3)  # Give time for redirect
            
            # Check for error messages
            error_selectors = [
                "//div[contains(@class, 'error')]", "//span[contains(@class, 'error')]",
                "//div[contains(text(), 'Invalid') or contains(text(), 'Error')]"
            ]
            
            for selector in error_selectors:
                try:
                    error_element = driver.find_element(By.XPATH, selector)
                    if error_element.is_displayed():
                        return False
                except:
                    continue
            
            return True
            
        except Exception as e:
            logger.error("Selenium form authentication failed", error=str(e))
            return False
    
    async def _handle_oauth_playwright(self, page: Page, credentials: Dict[str, Any], timeout: int) -> bool:
        """Handle OAuth authentication with Playwright."""
        # This is a simplified OAuth handler - in practice, you'd need provider-specific logic
        try:
            # Look for OAuth buttons
            oauth_selectors = [
                'button:has-text("Sign in with Google")', 'button:has-text("Continue with Google")',
                'button:has-text("Sign in with GitHub")', 'button:has-text("Continue with GitHub")',
                'a[href*="oauth"]', 'button[class*="oauth"]'
            ]
            
            for selector in oauth_selectors:
                try:
                    await page.click(selector)
                    await page.wait_for_load_state('networkidle', timeout=timeout * 1000)
                    return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            logger.error("OAuth authentication failed", error=str(e))
            return False
    
    def _handle_oauth_selenium(self, driver, credentials: Dict[str, Any], timeout: int) -> bool:
        """Handle OAuth authentication with Selenium."""
        # Simplified OAuth handler
        try:
            oauth_selectors = [
                (By.XPATH, "//button[contains(text(), 'Sign in with')]"),
                (By.CSS_SELECTOR, "button[class*='oauth']"),
                (By.CSS_SELECTOR, "a[href*='oauth']")
            ]
            
            for by, selector in oauth_selectors:
                try:
                    oauth_element = driver.find_element(by, selector)
                    oauth_element.click()
                    time.sleep(5)  # Wait for OAuth flow
                    return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            logger.error("Selenium OAuth authentication failed", error=str(e))
            return False
    
    def _create_session(self, domain: str, auth_method: str, session_data: Dict[str, Any]) -> str:
        """Create and store an authentication session."""
        session_id = str(uuid.uuid4())
        
        session = AuthSession(
            session_id=session_id,
            domain=domain,
            auth_method=auth_method,
            cookies=session_data.get('cookies', {}),
            headers={'User-Agent': session_data.get('user_agent', '')},
            expires_at=datetime.now() + timedelta(hours=24)  # Default 24h expiry
        )
        
        self.active_sessions[session_id] = session
        
        logger.info("Authentication session created", 
                   session_id=session_id, domain=domain, auth_method=auth_method)
        
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
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions."""
        now = datetime.now()
        expired_sessions = [
            sid for sid, session in self.active_sessions.items()
            if session.expires_at and now > session.expires_at
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            
        logger.info("Cleaned up expired sessions", count=len(expired_sessions))

class OAuthFlowHandler:
    """Handles OAuth 2.0 authentication flows."""
    
    def __init__(self):
        self.oauth_configs: Dict[str, OAuthConfig] = {}
        self.active_flows: Dict[str, Dict[str, Any]] = {}
        
    def register_oauth_provider(self, provider: str, config: OAuthConfig):
        """Register an OAuth provider configuration."""
        self.oauth_configs[provider] = config
        logger.info("OAuth provider registered", provider=provider)
    
    def initiate_oauth_flow(self, provider: str, state: Optional[str] = None) -> Dict[str, Any]:
        """Initiate OAuth 2.0 authorization flow."""
        if provider not in self.oauth_configs:
            raise HTTPException(status_code=400, detail=f"OAuth provider {provider} not configured")
        
        config = self.oauth_configs[provider]
        
        # Generate state parameter for CSRF protection
        if not state:
            state = secrets.token_urlsafe(32)
        
        # Build authorization URL
        auth_params = {
            'client_id': config.client_id,
            'redirect_uri': config.redirect_uri,
            'scope': ' '.join(config.scope) if config.scope else '',
            'state': state,
            'response_type': 'code'
        }
        auth_params.update(config.additional_params)
        
        auth_url = f"{config.authorization_url}?" + "&".join([
            f"{k}={v}" for k, v in auth_params.items() if v
        ])
        
        # Store flow state
        flow_id = str(uuid.uuid4())
        self.active_flows[flow_id] = {
            'provider': provider,
            'state': state,
            'created_at': datetime.now(),
            'config': config
        }
        
        return {
            'flow_id': flow_id,
            'authorization_url': auth_url,
            'state': state
        }
    
    async def complete_oauth_flow(self, flow_id: str, authorization_code: str, 
                                state: str) -> Dict[str, Any]:
        """Complete OAuth 2.0 flow by exchanging code for tokens."""
        if flow_id not in self.active_flows:
            raise HTTPException(status_code=400, detail="Invalid or expired OAuth flow")
        
        flow = self.active_flows[flow_id]
        
        # Verify state parameter
        if flow['state'] != state:
            raise HTTPException(status_code=400, detail="Invalid state parameter")
        
        config = flow['config']
        
        # Exchange authorization code for access token
        token_data = {
            'grant_type': 'authorization_code',
            'client_id': config.client_id,
            'client_secret': config.client_secret,
            'code': authorization_code,
            'redirect_uri': config.redirect_uri
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.token_url, data=token_data) as response:
                if response.status == 200:
                    tokens = await response.json()
                    
                    # Clean up flow
                    del self.active_flows[flow_id]
                    
                    return {
                        'success': True,
                        'tokens': tokens,
                        'provider': flow['provider']
                    }
                else:
                    error_text = await response.text()
                    logger.error("OAuth token exchange failed", 
                               status=response.status, error=error_text)
                    raise HTTPException(status_code=400, detail="Token exchange failed")

class AuthenticationQueue:
    """Manages queued authentication tasks with parallel processing."""
    
    def __init__(self, max_workers: int = 3):
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, AuthenticationTask] = {}
        self.completed_tasks: Dict[str, AuthenticationTask] = {}
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.authenticator = InteractiveAuthenticator()
        self._running = False
        self._worker_threads = []
    
    def start_processing(self):
        """Start the authentication queue processing."""
        if self._running:
            return
        
        self._running = True
        for i in range(self.max_workers):
            thread = threading.Thread(target=self._worker_loop, daemon=True)
            thread.start()
            self._worker_threads.append(thread)
        
        logger.info("Authentication queue processing started", workers=self.max_workers)
    
    def stop_processing(self):
        """Stop the authentication queue processing."""
        self._running = False
        logger.info("Authentication queue processing stopped")
    
    def queue_authentication(self, domain: str, auth_method: str, 
                           credentials: Dict[str, Any], login_url: str,
                           priority: int = 1) -> str:
        """Queue an authentication task."""
        task_id = str(uuid.uuid4())
        
        task = AuthenticationTask(
            task_id=task_id,
            domain=domain,
            auth_method=auth_method,
            credentials=credentials,
            priority=priority
        )
        
        # Add login_url to credentials for processing
        task.credentials['login_url'] = login_url
        
        self.active_tasks[task_id] = task
        self.task_queue.put((-priority, task_id))  # Negative for max-heap behavior
        
        logger.info("Authentication task queued", task_id=task_id, domain=domain)
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[AuthenticationTask]:
        """Get the status of a queued task."""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        return None
    
    def _worker_loop(self):
        """Worker thread loop for processing authentication tasks."""
        while self._running:
            try:
                # Get task from queue with timeout
                try:
                    priority, task_id = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                if task_id not in self.active_tasks:
                    continue
                
                task = self.active_tasks[task_id]
                task.status = "processing"
                
                logger.info("Processing authentication task", task_id=task_id, domain=task.domain)
                
                # Process the authentication task
                try:
                    # Run authentication in thread pool to avoid blocking
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    result = loop.run_until_complete(
                        self.authenticator.authenticate_with_popup(
                            domain=task.domain,
                            auth_method=task.auth_method,
                            credentials=task.credentials,
                            login_url=task.credentials['login_url']
                        )
                    )
                    
                    task.result = result
                    task.status = "completed" if result.get('success') else "failed"
                    
                    if not result.get('success'):
                        task.error_message = result.get('message', 'Authentication failed')
                    
                    logger.info("Authentication task completed", 
                               task_id=task_id, success=result.get('success'))
                    
                except Exception as e:
                    task.status = "failed"
                    task.error_message = str(e)
                    logger.error("Authentication task failed", task_id=task_id, error=str(e))
                
                finally:
                    # Move task to completed
                    self.completed_tasks[task_id] = task
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]
                    
                    self.task_queue.task_done()
                    
            except Exception as e:
                logger.error("Worker thread error", error=str(e))

# Global instances
auth_detector = AuthenticationDetector()
credential_store = SecureCredentialStore()
domain_mapper = DomainAuthMapper()
interactive_auth = InteractiveAuthenticator()
oauth_handler = OAuthFlowHandler()
auth_queue = AuthenticationQueue()

# Start the authentication queue processing
auth_queue.start_processing()

app = FastAPI(
    title="Authentication Service",
    description="Handles website authentication requirements and credential management",
    version="1.0.0"
)

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "auth",
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Authentication Service",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/api/auth/detect", response_model=AuthDetectionResponse)
async def detect_authentication_requirement(request: URLAnalysisRequest):
    """Detect if a URL requires authentication based on response analysis."""
    try:
        auth_req = await auth_detector.detect_auth_required(
            url=str(request.url),
            response_content=request.response_content,
            status_code=request.status_code,
            headers=request.headers
        )
        
        # Learn from this detection
        domain_mapper.learn_domain_auth(auth_req.domain, auth_req)
        
        # Determine recommended action
        if auth_req.detection_confidence > 0.7:
            recommended_action = "setup_authentication"
        elif auth_req.detection_confidence > 0.4:
            recommended_action = "manual_verification_needed"
        else:
            recommended_action = "no_authentication_required"
        
        return AuthDetectionResponse(
            requires_auth=auth_req.detection_confidence > 0.5,
            detected_method=auth_req.detected_method,
            confidence=auth_req.detection_confidence,
            indicators=auth_req.auth_indicators,
            recommended_action=recommended_action
        )
        
    except Exception as e:
        logger.error("Authentication detection failed", url=str(request.url), error=str(e))
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/api/auth/store-credentials")
async def store_domain_credentials(request: CredentialStoreRequest):
    """Store encrypted credentials for a domain."""
    try:
        success = credential_store.store_credentials(request.domain, request.credentials)
        
        if success:
            # Update domain mapping with credential info
            if request.domain in domain_mapper.domain_mappings:
                mapping = domain_mapper.domain_mappings[request.domain]
                mapping.auth_method = request.auth_method
                mapping.login_url = request.login_url
                mapping.requires_auth = True
            
            return {"success": True, "message": "Credentials stored securely"}
        else:
            raise HTTPException(status_code=500, detail="Failed to store credentials")
            
    except Exception as e:
        logger.error("Credential storage failed", domain=request.domain, error=str(e))
        raise HTTPException(status_code=500, detail=f"Storage failed: {str(e)}")

@app.get("/api/auth/credentials/{domain}")
async def get_domain_credentials(domain: str):
    """Retrieve stored credentials for a domain (returns existence only for security)."""
    try:
        has_credentials = domain in credential_store.list_stored_domains()
        domain_info = domain_mapper.get_domain_auth_info(domain)
        
        return {
            "domain": domain,
            "has_stored_credentials": has_credentials,
            "auth_method": domain_info.auth_method if domain_info else None,
            "requires_auth": domain_info.requires_auth if domain_info else False,
            "last_verified": domain_info.last_verified.isoformat() if domain_info and domain_info.last_verified else None
        }
        
    except Exception as e:
        logger.error("Credential retrieval failed", domain=domain, error=str(e))
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

@app.delete("/api/auth/credentials/{domain}")
async def delete_domain_credentials(domain: str):
    """Delete stored credentials for a domain."""
    try:
        success = credential_store.delete_credentials(domain)
        
        if success:
            return {"success": True, "message": f"Credentials deleted for {domain}"}
        else:
            return {"success": False, "message": f"No credentials found for {domain}"}
            
    except Exception as e:
        logger.error("Credential deletion failed", domain=domain, error=str(e))
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@app.get("/api/auth/domains")
async def list_domains_with_auth():
    """List all domains with authentication requirements or stored credentials."""
    try:
        stored_domains = set(credential_store.list_stored_domains())
        mapped_domains = set(domain_mapper.get_all_mappings().keys())
        all_domains = stored_domains.union(mapped_domains)
        
        domain_list = []
        for domain in all_domains:
            domain_info = domain_mapper.get_domain_auth_info(domain)
            domain_list.append({
                "domain": domain,
                "has_credentials": domain in stored_domains,
                "requires_auth": domain_info.requires_auth if domain_info else False,
                "auth_method": domain_info.auth_method if domain_info else "unknown",
                "success_count": domain_info.success_count if domain_info else 0,
                "failure_count": domain_info.failure_count if domain_info else 0,
                "last_verified": domain_info.last_verified.isoformat() if domain_info and domain_info.last_verified else None
            })
        
        return {"domains": domain_list, "total_count": len(domain_list)}
        
    except Exception as e:
        logger.error("Domain listing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Listing failed: {str(e)}")

@app.post("/api/auth/learn-domain")
async def learn_domain_authentication(domain: str, auth_method: str, requires_auth: bool):
    """Manually teach the system about a domain's authentication requirements."""
    try:
        # Create a synthetic auth requirement for learning
        auth_req = AuthenticationRequirement(
            url=f"https://{domain}",
            domain=domain,
            detected_method=auth_method,
            auth_indicators=["Manual learning"],
            detection_confidence=1.0 if requires_auth else 0.0
        )
        
        domain_mapper.learn_domain_auth(domain, auth_req)
        
        return {
            "success": True,
            "message": f"Domain {domain} learned with auth method: {auth_method}",
            "requires_auth": requires_auth
        }
        
    except Exception as e:
        logger.error("Domain learning failed", domain=domain, error=str(e))
        raise HTTPException(status_code=500, detail=f"Learning failed: {str(e)}")

@app.post("/api/auth/mark-success/{domain}")
async def mark_authentication_success(domain: str):
    """Mark successful authentication for a domain."""
    try:
        domain_mapper.mark_auth_success(domain)
        return {"success": True, "message": f"Authentication success recorded for {domain}"}
        
    except Exception as e:
        logger.error("Failed to mark auth success", domain=domain, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to mark success: {str(e)}")

@app.post("/api/auth/mark-failure/{domain}")
async def mark_authentication_failure(domain: str):
    """Mark failed authentication for a domain."""
    try:
        domain_mapper.mark_auth_failure(domain)
        return {"success": True, "message": f"Authentication failure recorded for {domain}"}
        
    except Exception as e:
        logger.error("Failed to mark auth failure", domain=domain, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to mark failure: {str(e)}")

@app.get("/api/auth/domain-mapping/{domain}")
async def get_domain_mapping(domain: str):
    """Get detailed authentication mapping for a specific domain."""
    try:
        mapping = domain_mapper.get_domain_auth_info(domain)
        
        if not mapping:
            raise HTTPException(status_code=404, detail=f"No authentication mapping found for {domain}")
        
        return {
            "domain": mapping.domain,
            "auth_method": mapping.auth_method,
            "requires_auth": mapping.requires_auth,
            "login_url": mapping.login_url,
            "form_selectors": mapping.form_selectors,
            "oauth_config": mapping.oauth_config,
            "last_verified": mapping.last_verified.isoformat() if mapping.last_verified else None,
            "success_count": mapping.success_count,
            "failure_count": mapping.failure_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Domain mapping retrieval failed", domain=domain, error=str(e))
        raise HTTPException(status_code=500, detail=f"Mapping retrieval failed: {str(e)}")

# Interactive Authentication Endpoints

@app.post("/api/auth/interactive", response_model=AuthTaskResponse)
async def authenticate_interactively(request: InteractiveAuthRequest, background_tasks: BackgroundTasks):
    """Perform interactive authentication using browser automation."""
    try:
        # Queue the authentication task for parallel processing
        task_id = auth_queue.queue_authentication(
            domain=request.domain,
            auth_method=request.auth_method,
            credentials=request.credentials,
            login_url=request.login_url,
            priority=1
        )
        
        return AuthTaskResponse(
            task_id=task_id,
            status="queued",
            message=f"Authentication task queued for {request.domain}"
        )
        
    except Exception as e:
        logger.error("Interactive authentication request failed", 
                    domain=request.domain, error=str(e))
        raise HTTPException(status_code=500, detail=f"Authentication request failed: {str(e)}")

@app.get("/api/auth/task/{task_id}")
async def get_authentication_task_status(task_id: str):
    """Get the status of an authentication task."""
    try:
        task = auth_queue.get_task_status(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        response_data = {
            "task_id": task.task_id,
            "domain": task.domain,
            "auth_method": task.auth_method,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
            "error_message": task.error_message
        }
        
        if task.result:
            response_data["result"] = task.result
            if task.result.get('session_id'):
                response_data["session_id"] = task.result['session_id']
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Task status retrieval failed", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

@app.post("/api/auth/oauth/initiate")
async def initiate_oauth_flow(request: OAuthAuthRequest):
    """Initiate OAuth 2.0 authentication flow."""
    try:
        # Register OAuth provider configuration
        oauth_config = OAuthConfig(
            provider=request.provider,
            client_id=request.client_id,
            client_secret=request.client_secret,
            authorization_url=f"https://accounts.{request.provider}.com/oauth/authorize",  # Simplified
            token_url=f"https://accounts.{request.provider}.com/oauth/token",  # Simplified
            redirect_uri=request.redirect_uri,
            scope=request.scope or []
        )
        
        oauth_handler.register_oauth_provider(request.provider, oauth_config)
        
        # Initiate OAuth flow
        flow_data = oauth_handler.initiate_oauth_flow(request.provider)
        
        return {
            "success": True,
            "flow_id": flow_data["flow_id"],
            "authorization_url": flow_data["authorization_url"],
            "state": flow_data["state"],
            "message": f"OAuth flow initiated for {request.provider}"
        }
        
    except Exception as e:
        logger.error("OAuth flow initiation failed", provider=request.provider, error=str(e))
        raise HTTPException(status_code=500, detail=f"OAuth initiation failed: {str(e)}")

@app.post("/api/auth/oauth/callback")
async def oauth_callback(flow_id: str, code: str, state: str):
    """Handle OAuth callback and complete authentication."""
    try:
        result = await oauth_handler.complete_oauth_flow(flow_id, code, state)
        
        if result['success']:
            # Create session with OAuth tokens
            session_data = {
                'tokens': result['tokens'],
                'provider': result['provider'],
                'auth_method': 'oauth'
            }
            
            session_id = interactive_auth._create_session(
                domain=result['provider'],  # Use provider as domain for OAuth
                auth_method='oauth',
                session_data=session_data
            )
            
            return {
                "success": True,
                "session_id": session_id,
                "provider": result['provider'],
                "message": "OAuth authentication completed successfully"
            }
        else:
            return {
                "success": False,
                "message": "OAuth authentication failed"
            }
            
    except Exception as e:
        logger.error("OAuth callback failed", flow_id=flow_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"OAuth callback failed: {str(e)}")

# Session Management Endpoints

@app.get("/api/auth/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about an authentication session."""
    try:
        session = interactive_auth.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        return {
            "session_id": session.session_id,
            "domain": session.domain,
            "auth_method": session.auth_method,
            "created_at": session.created_at.isoformat(),
            "expires_at": session.expires_at.isoformat() if session.expires_at else None,
            "last_used": session.last_used.isoformat(),
            "is_active": session.is_active,
            "has_cookies": len(session.cookies) > 0,
            "has_tokens": len(session.tokens) > 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Session info retrieval failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Session retrieval failed: {str(e)}")

@app.post("/api/auth/session/{session_id}/renew")
async def renew_session(session_id: str):
    """Renew an authentication session."""
    try:
        session = interactive_auth.get_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        
        # Extend session expiry
        session.expires_at = datetime.now() + timedelta(hours=24)
        session.last_used = datetime.now()
        
        return {
            "success": True,
            "session_id": session_id,
            "new_expires_at": session.expires_at.isoformat(),
            "message": "Session renewed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Session renewal failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Session renewal failed: {str(e)}")

@app.delete("/api/auth/session/{session_id}")
async def invalidate_session(session_id: str):
    """Invalidate an authentication session."""
    try:
        success = interactive_auth.invalidate_session(session_id)
        
        if success:
            return {
                "success": True,
                "message": f"Session {session_id} invalidated successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Session invalidation failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Session invalidation failed: {str(e)}")

@app.get("/api/auth/sessions")
async def list_active_sessions():
    """List all active authentication sessions."""
    try:
        sessions = []
        for session_id, session in interactive_auth.active_sessions.items():
            if session.is_active:
                sessions.append({
                    "session_id": session_id,
                    "domain": session.domain,
                    "auth_method": session.auth_method,
                    "created_at": session.created_at.isoformat(),
                    "expires_at": session.expires_at.isoformat() if session.expires_at else None,
                    "last_used": session.last_used.isoformat()
                })
        
        return {
            "sessions": sessions,
            "total_count": len(sessions)
        }
        
    except Exception as e:
        logger.error("Session listing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Session listing failed: {str(e)}")

@app.post("/api/auth/sessions/cleanup")
async def cleanup_expired_sessions():
    """Clean up expired authentication sessions."""
    try:
        initial_count = len(interactive_auth.active_sessions)
        interactive_auth.cleanup_expired_sessions()
        final_count = len(interactive_auth.active_sessions)
        cleaned_count = initial_count - final_count
        
        return {
            "success": True,
            "cleaned_sessions": cleaned_count,
            "remaining_sessions": final_count,
            "message": f"Cleaned up {cleaned_count} expired sessions"
        }
        
    except Exception as e:
        logger.error("Session cleanup failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Session cleanup failed: {str(e)}")

@app.get("/api/auth/queue/status")
async def get_queue_status():
    """Get authentication queue status."""
    try:
        return {
            "active_tasks": len(auth_queue.active_tasks),
            "completed_tasks": len(auth_queue.completed_tasks),
            "queue_size": auth_queue.task_queue.qsize(),
            "max_workers": auth_queue.max_workers,
            "is_running": auth_queue._running
        }
        
    except Exception as e:
        logger.error("Queue status retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Queue status failed: {str(e)}")