"""Authentication Service - Handles website authentication requirements."""

import time
import json
import hashlib
import base64
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from urllib.parse import urlparse, urljoin
import asyncio
import aiohttp

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import structlog

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

# Global instances
auth_detector = AuthenticationDetector()
credential_store = SecureCredentialStore()
domain_mapper = DomainAuthMapper()

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