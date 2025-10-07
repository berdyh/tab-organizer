"""URL Input Service - Handles URL parsing and validation from various sources."""

import time
import json
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from io import StringIO
from urllib.parse import urlparse, parse_qs
from collections import defaultdict, Counter
import hashlib

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse
import structlog

# Setup logging
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

app = FastAPI(
    title="URL Input Service",
    description="Handles URL parsing and validation from various input sources",
    version="1.0.0"
)

# Data Models
@dataclass
class URLMetadata:
    """URL metadata extracted from URL components."""
    domain: str
    subdomain: Optional[str] = None
    path: str = "/"
    parameters: Dict[str, List[str]] = field(default_factory=dict)
    fragment: Optional[str] = None
    port: Optional[int] = None
    scheme: str = "https"
    tld: Optional[str] = None
    path_segments: List[str] = field(default_factory=list)
    parameter_count: int = 0
    path_depth: int = 0
    url_hash: str = ""

@dataclass
class URLEntry:
    url: str
    category: Optional[str] = None
    priority: Optional[str] = None
    notes: Optional[str] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    validated: bool = False
    validation_error: Optional[str] = None
    metadata: Optional[URLMetadata] = None
    enriched: bool = False
    duplicate_of: Optional[str] = None
    similarity_group: Optional[str] = None

@dataclass
class URLInput:
    input_id: str
    urls: List[URLEntry]
    source_type: str  # text, json, csv, excel, form
    source_metadata: Dict[str, Any]
    created_at: datetime
    validated: bool = False

# In-memory storage for demo (in production, use database)
url_inputs: Dict[str, URLInput] = {}

class URLValidator:
    """URL validation utilities."""
    
    URL_PATTERN = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    @classmethod
    def is_valid_url(cls, url: str) -> bool:
        """Check if URL is valid format."""
        if not url or not isinstance(url, str):
            return False
        return bool(cls.URL_PATTERN.match(url.strip()))
    
    @classmethod
    def validate_url(cls, url: str) -> tuple[bool, Optional[str]]:
        """Validate URL and return validation result with error message."""
        url = url.strip() if url else ""
        
        if not url:
            return False, "URL cannot be empty"
        
        if not cls.is_valid_url(url):
            return False, "Invalid URL format"
        
        return True, None

class InputFormatDetector:
    """Auto-detect input file format and extract URLs."""
    
    @classmethod
    def detect_file_type(cls, filename: str, content: str) -> str:
        """Auto-detect file type based on filename and content."""
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.json'):
            return 'json'
        elif filename_lower.endswith(('.csv', '.tsv')):
            return 'csv'
        elif filename_lower.endswith(('.xlsx', '.xls')):
            return 'excel'
        elif filename_lower.endswith('.txt'):
            return 'text'
        
        # Try to detect by content
        try:
            json.loads(content)
            return 'json'
        except:
            pass
        
        # Check if it looks like CSV
        if ',' in content or '\t' in content:
            return 'csv'
        
        # Default to text
        return 'text'
    
    @classmethod
    def extract_url_patterns(cls, text: str) -> List[str]:
        """Find URLs in unstructured text."""
        url_pattern = re.compile(
            r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
            re.IGNORECASE
        )
        return url_pattern.findall(text)

class URLEnricher:
    """URL metadata extraction and enrichment utilities."""
    
    @classmethod
    def extract_metadata(cls, url: str) -> URLMetadata:
        """Extract comprehensive metadata from URL."""
        try:
            parsed = urlparse(url)
            
            # Extract domain components
            # For localhost and IP addresses, keep the port as part of domain
            if 'localhost' in parsed.netloc or parsed.netloc.replace('.', '').replace(':', '').isdigit():
                domain = parsed.netloc
                subdomain = None
                tld = None
            else:
                # For regular domains, remove port for domain extraction
                netloc_without_port = parsed.netloc.split(':')[0] if ':' in parsed.netloc else parsed.netloc
                domain_parts = netloc_without_port.split('.')
                domain = netloc_without_port
                subdomain = None
                tld = None
                
                if len(domain_parts) > 2:
                    subdomain = '.'.join(domain_parts[:-2])
                    domain = '.'.join(domain_parts[-2:])
                    tld = domain_parts[-1]
                elif len(domain_parts) == 2:
                    tld = domain_parts[-1]
            
            # Extract path components
            path = parsed.path or "/"
            path_segments = [seg for seg in path.split('/') if seg]
            path_depth = len(path_segments)
            
            # Extract parameters
            parameters = parse_qs(parsed.query)
            parameter_count = len(parameters)
            
            # Generate URL hash for deduplication
            url_hash = hashlib.md5(url.encode()).hexdigest()
            
            return URLMetadata(
                domain=domain,
                subdomain=subdomain,
                path=path,
                parameters=parameters,
                fragment=parsed.fragment or None,
                port=parsed.port,
                scheme=parsed.scheme,
                tld=tld,
                path_segments=path_segments,
                parameter_count=parameter_count,
                path_depth=path_depth,
                url_hash=url_hash
            )
            
        except Exception as e:
            # Return minimal metadata if parsing fails
            return URLMetadata(
                domain=url,
                url_hash=hashlib.md5(url.encode()).hexdigest()
            )
    
    @classmethod
    def categorize_url(cls, url_entry: URLEntry) -> str:
        """Automatically categorize URL based on domain and path."""
        if not url_entry.metadata:
            return "unknown"
        
        domain = url_entry.metadata.domain.lower()
        subdomain = url_entry.metadata.subdomain.lower() if url_entry.metadata.subdomain else ""
        path = url_entry.metadata.path.lower()
        
        # Combine domain and subdomain for full domain checking
        full_domain = f"{subdomain}.{domain}" if subdomain else domain
        
        # Social media platforms
        social_domains = ['twitter.com', 'facebook.com', 'linkedin.com', 'instagram.com', 
                         'youtube.com', 'tiktok.com', 'reddit.com']
        if any(social in full_domain for social in social_domains):
            return "social_media"
        
        # News and media
        news_indicators = ['news', 'blog', 'article', 'post', 'story']
        if any(indicator in full_domain for indicator in news_indicators) or any(indicator in path for indicator in news_indicators):
            return "news_media"
        
        # Documentation and technical
        tech_indicators = ['docs', 'api', 'github', 'stackoverflow', 'wiki']
        if any(indicator in full_domain or indicator in path for indicator in tech_indicators):
            return "documentation"
        
        # E-commerce
        ecommerce_indicators = ['shop', 'store', 'buy', 'cart', 'product']
        if any(indicator in full_domain or indicator in path for indicator in ecommerce_indicators):
            return "ecommerce"
        
        # Educational
        edu_indicators = ['.edu', 'university', 'college', 'course', 'learn']
        if any(indicator in full_domain or indicator in path for indicator in edu_indicators):
            return "education"
        
        return "general"
    
    @classmethod
    def enrich_url_entry(cls, url_entry: URLEntry) -> URLEntry:
        """Enrich URL entry with metadata and categorization."""
        if not url_entry.validated:
            return url_entry
        
        # Extract metadata
        url_entry.metadata = cls.extract_metadata(url_entry.url)
        
        # Auto-categorize if no category provided
        if not url_entry.category:
            url_entry.category = cls.categorize_url(url_entry)
        
        url_entry.enriched = True
        return url_entry

class URLDeduplicator:
    """URL deduplication and similarity detection."""
    
    @classmethod
    def normalize_url(cls, url: str) -> str:
        """Normalize URL for deduplication comparison."""
        try:
            parsed = urlparse(url.lower().strip())
            
            # Remove common tracking parameters
            tracking_params = {
                'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                'fbclid', 'gclid', 'ref', 'source', 'campaign_id', 'ad_id'
            }
            
            # Parse and filter parameters
            params = parse_qs(parsed.query)
            filtered_params = {k: v for k, v in params.items() 
                             if k.lower() not in tracking_params}
            
            # Rebuild query string
            query_parts = []
            for key, values in sorted(filtered_params.items()):
                for value in sorted(values):
                    query_parts.append(f"{key}={value}")
            
            normalized_query = "&".join(query_parts)
            
            # Remove trailing slashes from path
            path = parsed.path.rstrip('/')
            if not path:
                path = '/'
            
            # Rebuild normalized URL
            normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
            if normalized_query:
                normalized += f"?{normalized_query}"
            
            return normalized
            
        except Exception:
            return url.lower().strip()
    
    @classmethod
    def find_duplicates(cls, url_entries: List[URLEntry]) -> Dict[str, List[URLEntry]]:
        """Find duplicate URLs and group them."""
        normalized_groups = defaultdict(list)
        
        for entry in url_entries:
            if entry.validated:
                normalized = cls.normalize_url(entry.url)
                normalized_groups[normalized].append(entry)
        
        # Return only groups with duplicates
        return {norm_url: entries for norm_url, entries in normalized_groups.items() 
                if len(entries) > 1}
    
    @classmethod
    def mark_duplicates(cls, url_entries: List[URLEntry]) -> List[URLEntry]:
        """Mark duplicate URLs in the list."""
        duplicate_groups = cls.find_duplicates(url_entries)
        
        for normalized_url, entries in duplicate_groups.items():
            # Keep the first entry as primary, mark others as duplicates
            primary = entries[0]
            for duplicate in entries[1:]:
                duplicate.duplicate_of = primary.url
        
        return url_entries
    
    @classmethod
    def get_similarity_groups(cls, url_entries: List[URLEntry]) -> Dict[str, List[URLEntry]]:
        """Group URLs by domain similarity."""
        domain_groups = defaultdict(list)
        
        for entry in url_entries:
            if entry.validated and entry.metadata:
                # Group by base domain
                domain_groups[entry.metadata.domain].append(entry)
        
        return dict(domain_groups)

class BatchProcessor:
    """Handle batch processing of large URL lists."""
    
    @classmethod
    def process_urls_batch(cls, url_entries: List[URLEntry], batch_size: int = 100) -> List[URLEntry]:
        """Process URLs in batches for large datasets."""
        processed_entries = []
        
        for i in range(0, len(url_entries), batch_size):
            batch = url_entries[i:i + batch_size]
            
            # Enrich each URL in the batch
            enriched_batch = []
            for entry in batch:
                enriched_entry = URLEnricher.enrich_url_entry(entry)
                enriched_batch.append(enriched_entry)
            
            processed_entries.extend(enriched_batch)
        
        # Mark duplicates across all processed entries
        processed_entries = URLDeduplicator.mark_duplicates(processed_entries)
        
        return processed_entries
    
    @classmethod
    def get_processing_stats(cls, url_entries: List[URLEntry]) -> Dict[str, Any]:
        """Get comprehensive statistics about processed URLs."""
        total_urls = len(url_entries)
        valid_urls = sum(1 for entry in url_entries if entry.validated)
        enriched_urls = sum(1 for entry in url_entries if entry.enriched)
        duplicate_urls = sum(1 for entry in url_entries if entry.duplicate_of)
        
        # Category distribution
        categories = Counter(entry.category for entry in url_entries if entry.category)
        
        # Domain distribution
        domains = Counter(entry.metadata.domain for entry in url_entries 
                         if entry.metadata and entry.validated)
        
        # Priority distribution
        priorities = Counter(entry.priority for entry in url_entries if entry.priority)
        
        return {
            "total_urls": total_urls,
            "valid_urls": valid_urls,
            "invalid_urls": total_urls - valid_urls,
            "enriched_urls": enriched_urls,
            "duplicate_urls": duplicate_urls,
            "unique_urls": valid_urls - duplicate_urls,
            "categories": dict(categories.most_common()),
            "top_domains": dict(domains.most_common(10)),
            "priorities": dict(priorities),
            "processing_complete": enriched_urls == valid_urls
        }

class URLParser:
    """Parse URLs from various input sources."""
    
    @classmethod
    def parse_text_file(cls, content: str, enrich: bool = True) -> List[URLEntry]:
        """Extract URLs from plain text files."""
        urls = []
        lines = content.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Extract URL (first word/token)
            url = line.split()[0] if line.split() else line
            
            is_valid, error = URLValidator.validate_url(url)
            entry = URLEntry(
                url=url,
                source_metadata={"line_number": line_num, "original_line": line},
                validated=is_valid,
                validation_error=error
            )
            
            # Enrich URL if valid and enrichment is enabled
            if enrich and is_valid:
                entry = URLEnricher.enrich_url_entry(entry)
            
            urls.append(entry)
        
        # Process batch for deduplication if enrichment is enabled
        if enrich:
            urls = URLDeduplicator.mark_duplicates(urls)
        
        return urls
    
    @classmethod
    def parse_json_file(cls, content: str, enrich: bool = True) -> List[URLEntry]:
        """Parse structured JSON with URL lists."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        
        urls = []
        
        # Handle different JSON structures
        if isinstance(data, list):
            # Simple list of URLs
            for i, item in enumerate(data):
                if isinstance(item, str):
                    is_valid, error = URLValidator.validate_url(item)
                    entry = URLEntry(
                        url=item,
                        source_metadata={"index": i},
                        validated=is_valid,
                        validation_error=error
                    )
                    if enrich and is_valid:
                        entry = URLEnricher.enrich_url_entry(entry)
                    urls.append(entry)
                elif isinstance(item, dict) and 'url' in item:
                    url = item['url']
                    is_valid, error = URLValidator.validate_url(url)
                    entry = URLEntry(
                        url=url,
                        category=item.get('category'),
                        priority=item.get('priority'),
                        notes=item.get('notes'),
                        source_metadata={"index": i, **{k: v for k, v in item.items() if k != 'url'}},
                        validated=is_valid,
                        validation_error=error
                    )
                    if enrich and is_valid:
                        entry = URLEnricher.enrich_url_entry(entry)
                    urls.append(entry)
        
        elif isinstance(data, dict):
            # Handle {"urls": [...]} structure
            if 'urls' in data:
                url_list = data['urls']
                metadata = {k: v for k, v in data.items() if k != 'urls'}
                
                for i, item in enumerate(url_list):
                    if isinstance(item, str):
                        is_valid, error = URLValidator.validate_url(item)
                        entry = URLEntry(
                            url=item,
                            source_metadata={"index": i, **metadata},
                            validated=is_valid,
                            validation_error=error
                        )
                        if enrich and is_valid:
                            entry = URLEnricher.enrich_url_entry(entry)
                        urls.append(entry)
                    elif isinstance(item, dict) and 'url' in item:
                        url = item['url']
                        is_valid, error = URLValidator.validate_url(url)
                        entry = URLEntry(
                            url=url,
                            category=item.get('category'),
                            priority=item.get('priority'),
                            notes=item.get('notes'),
                            source_metadata={"index": i, **metadata, **{k: v for k, v in item.items() if k != 'url'}},
                            validated=is_valid,
                            validation_error=error
                        )
                        if enrich and is_valid:
                            entry = URLEnricher.enrich_url_entry(entry)
                        urls.append(entry)
        
        # Process batch for deduplication if enrichment is enabled
        if enrich:
            urls = URLDeduplicator.mark_duplicates(urls)
        
        return urls
    
    @classmethod
    def parse_csv_file(cls, content: str, enrich: bool = True) -> List[URLEntry]:
        """Extract URLs from CSV/Excel columns."""
        try:
            # Try to read as CSV
            df = pd.read_csv(StringIO(content))
        except Exception as e:
            raise ValueError(f"Invalid CSV format: {str(e)}")
        
        urls = []
        
        # Look for URL column (case insensitive)
        url_column = None
        for col in df.columns:
            if col.lower() in ['url', 'urls', 'link', 'links', 'website', 'site']:
                url_column = col
                break
        
        if url_column is None:
            # If no URL column found, assume first column contains URLs
            url_column = df.columns[0]
        
        for index, row in df.iterrows():
            url = str(row[url_column]).strip()
            
            if pd.isna(row[url_column]) or not url or url.lower() == 'nan':
                continue
            
            is_valid, error = URLValidator.validate_url(url)
            
            # Extract other columns as metadata
            metadata = {}
            for col in df.columns:
                if col != url_column and not pd.isna(row[col]):
                    metadata[col.lower()] = row[col]
            
            entry = URLEntry(
                url=url,
                category=metadata.get('category'),
                priority=metadata.get('priority'),
                notes=metadata.get('notes'),
                source_metadata={"row_index": index, **metadata},
                validated=is_valid,
                validation_error=error
            )
            
            # Enrich URL if valid and enrichment is enabled
            if enrich and is_valid:
                entry = URLEnricher.enrich_url_entry(entry)
            
            urls.append(entry)
        
        # Process batch for deduplication if enrichment is enabled
        if enrich:
            urls = URLDeduplicator.mark_duplicates(urls)
        
        return urls

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "url-input",
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "URL Input Service",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/api/input/list")
async def list_url_inputs(skip: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
    """List all URL inputs with pagination."""
    logger.info("Listing URL inputs", skip=skip, limit=limit)
    
    # Convert to list and sort by creation time
    inputs_list = list(url_inputs.values())
    inputs_list.sort(key=lambda x: x.created_at, reverse=True)
    
    # Apply pagination
    paginated_inputs = inputs_list[skip:skip + limit]
    
    # Convert to response format
    response_data = []
    for url_input in paginated_inputs:
        # Get summary stats
        stats = BatchProcessor.get_processing_stats(url_input.urls)
        
        response_data.append({
            "id": url_input.input_id,
            "source_type": url_input.source_type,
            "created_at": url_input.created_at.isoformat(),
            "filename": url_input.source_metadata.get("filename", "Direct input"),
            "total_urls": stats["total_urls"],
            "valid_urls": stats["valid_urls"],
            "unique_urls": stats["unique_urls"],
            "categories": stats["categories"]
        })
    
    return {
        "data": response_data,
        "total": len(inputs_list),
        "skip": skip,
        "limit": limit,
        "has_more": skip + limit < len(inputs_list)
    }

@app.get("/api/input/{input_id}")
async def get_url_input(input_id: str):
    """Get detailed information about a specific URL input."""
    logger.info("Getting URL input details", input_id=input_id)
    
    if input_id not in url_inputs:
        raise HTTPException(status_code=404, detail="URL input not found")
    
    url_input = url_inputs[input_id]
    stats = BatchProcessor.get_processing_stats(url_input.urls)
    
    # Convert URLs to response format
    urls_data = []
    for url_entry in url_input.urls:
        url_data = {
            "url": url_entry.url,
            "validated": url_entry.validated,
            "category": url_entry.category,
            "priority": url_entry.priority,
            "notes": url_entry.notes,
            "duplicate_of": url_entry.duplicate_of,
            "validation_error": url_entry.validation_error
        }
        
        if url_entry.metadata:
            url_data["metadata"] = asdict(url_entry.metadata)
        
        urls_data.append(url_data)
    
    return {
        "input_id": url_input.input_id,
        "source_type": url_input.source_type,
        "source_metadata": url_input.source_metadata,
        "created_at": url_input.created_at.isoformat(),
        "urls": urls_data,
        "stats": stats
    }

@app.delete("/api/input/{input_id}")
async def delete_url_input(input_id: str):
    """Delete a URL input."""
    logger.info("Deleting URL input", input_id=input_id)
    
    if input_id not in url_inputs:
        raise HTTPException(status_code=404, detail="URL input not found")
    
    del url_inputs[input_id]
    
    return {"message": "URL input deleted successfully", "input_id": input_id}

@app.put("/api/input/{input_id}")
async def update_url_input(input_id: str, update_data: Dict[str, Any]):
    """Update URL input metadata."""
    logger.info("Updating URL input", input_id=input_id)
    
    if input_id not in url_inputs:
        raise HTTPException(status_code=404, detail="URL input not found")
    
    url_input = url_inputs[input_id]
    
    # Update allowed fields
    if "source_metadata" in update_data:
        url_input.source_metadata.update(update_data["source_metadata"])
    
    return {"message": "URL input updated successfully", "input_id": input_id}

# File Upload Endpoints

@app.post("/api/input/upload/text")
async def upload_text_file(file: UploadFile = File(...), enrich: bool = Query(True, description="Enable URL enrichment and deduplication")):
    """Upload and parse plain text file with URLs."""
    logger.info("Processing text file upload", filename=file.filename, enrich=enrich)
    
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Parse URLs from text with optional enrichment
        urls = URLParser.parse_text_file(content_str, enrich=enrich)
        
        # Get processing statistics
        stats = BatchProcessor.get_processing_stats(urls)
        
        # Create URL input record
        input_id = str(uuid.uuid4())
        url_input = URLInput(
            input_id=input_id,
            urls=urls,
            source_type="text",
            source_metadata={
                "filename": file.filename,
                "file_size": len(content),
                "total_lines": len(content_str.split('\n')),
                "enriched": enrich,
                "processing_stats": stats
            },
            created_at=datetime.now()
        )
        
        # Store in memory
        url_inputs[input_id] = url_input
        
        logger.info("Text file processed", 
                   input_id=input_id, 
                   **stats)
        
        return {
            "input_id": input_id,
            "source_type": "text",
            "filename": file.filename,
            "enriched": enrich,
            **stats
        }
        
    except Exception as e:
        logger.error("Error processing text file", error=str(e), filename=file.filename)
        raise HTTPException(status_code=400, detail=f"Error processing text file: {str(e)}")

@app.post("/api/input/upload/json")
async def upload_json_file(file: UploadFile = File(...), enrich: bool = Query(True, description="Enable URL enrichment and deduplication")):
    """Upload and parse JSON file with URLs."""
    logger.info("Processing JSON file upload", filename=file.filename, enrich=enrich)
    
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Parse URLs from JSON with optional enrichment
        urls = URLParser.parse_json_file(content_str, enrich=enrich)
        
        # Get processing statistics
        stats = BatchProcessor.get_processing_stats(urls)
        
        # Create URL input record
        input_id = str(uuid.uuid4())
        url_input = URLInput(
            input_id=input_id,
            urls=urls,
            source_type="json",
            source_metadata={
                "filename": file.filename,
                "file_size": len(content),
                "enriched": enrich,
                "processing_stats": stats
            },
            created_at=datetime.now()
        )
        
        # Store in memory
        url_inputs[input_id] = url_input
        
        logger.info("JSON file processed", 
                   input_id=input_id, 
                   **stats)
        
        return {
            "input_id": input_id,
            "source_type": "json",
            "filename": file.filename,
            "enriched": enrich,
            **stats
        }
        
    except Exception as e:
        logger.error("Error processing JSON file", error=str(e), filename=file.filename)
        raise HTTPException(status_code=400, detail=f"Error processing JSON file: {str(e)}")

@app.post("/api/input/upload/csv")
async def upload_csv_file(file: UploadFile = File(...), enrich: bool = Query(True, description="Enable URL enrichment and deduplication")):
    """Upload and parse CSV file with URLs."""
    logger.info("Processing CSV file upload", filename=file.filename, enrich=enrich)
    
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Parse URLs from CSV with optional enrichment
        urls = URLParser.parse_csv_file(content_str, enrich=enrich)
        
        # Get processing statistics
        stats = BatchProcessor.get_processing_stats(urls)
        
        # Create URL input record
        input_id = str(uuid.uuid4())
        url_input = URLInput(
            input_id=input_id,
            urls=urls,
            source_type="csv",
            source_metadata={
                "filename": file.filename,
                "file_size": len(content),
                "enriched": enrich,
                "processing_stats": stats
            },
            created_at=datetime.now()
        )
        
        # Store in memory
        url_inputs[input_id] = url_input
        
        logger.info("CSV file processed", 
                   input_id=input_id, 
                   **stats)
        
        return {
            "input_id": input_id,
            "source_type": "csv",
            "filename": file.filename,
            "enriched": enrich,
            **stats
        }
        
    except Exception as e:
        logger.error("Error processing CSV file", error=str(e), filename=file.filename)
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {str(e)}")

@app.post("/api/input/upload/excel")
async def upload_excel_file(file: UploadFile = File(...), enrich: bool = Query(True, description="Enable URL enrichment and deduplication")):
    """Upload and parse Excel file with URLs."""
    logger.info("Processing Excel file upload", filename=file.filename, enrich=enrich)
    
    try:
        content = await file.read()
        
        # Read Excel file
        df = pd.read_excel(content)
        
        # Convert to CSV string for parsing
        csv_content = df.to_csv(index=False)
        
        # Parse URLs from CSV format with optional enrichment
        urls = URLParser.parse_csv_file(csv_content, enrich=enrich)
        
        # Get processing statistics
        stats = BatchProcessor.get_processing_stats(urls)
        
        # Create URL input record
        input_id = str(uuid.uuid4())
        url_input = URLInput(
            input_id=input_id,
            urls=urls,
            source_type="excel",
            source_metadata={
                "filename": file.filename,
                "file_size": len(content),
                "sheet_shape": df.shape,
                "enriched": enrich,
                "processing_stats": stats
            },
            created_at=datetime.now()
        )
        
        # Store in memory
        url_inputs[input_id] = url_input
        
        logger.info("Excel file processed", 
                   input_id=input_id, 
                   **stats)
        
        return {
            "input_id": input_id,
            "source_type": "excel",
            "filename": file.filename,
            "enriched": enrich,
            **stats
        }
        
    except Exception as e:
        logger.error("Error processing Excel file", error=str(e), filename=file.filename)
        raise HTTPException(status_code=400, detail=f"Error processing Excel file: {str(e)}")

# Direct input endpoints

@app.post("/api/input/urls")
async def input_urls_direct(urls: List[str], enrich: bool = Query(True, description="Enable URL enrichment and deduplication")):
    """Direct URL list input."""
    logger.info("Processing direct URL input", url_count=len(urls), enrich=enrich)
    
    try:
        # Parse and validate URLs
        url_entries = []
        for i, url in enumerate(urls):
            is_valid, error = URLValidator.validate_url(url)
            entry = URLEntry(
                url=url,
                source_metadata={"index": i},
                validated=is_valid,
                validation_error=error
            )
            
            # Enrich URL if valid and enrichment is enabled
            if enrich and is_valid:
                entry = URLEnricher.enrich_url_entry(entry)
            
            url_entries.append(entry)
        
        # Process batch for deduplication if enrichment is enabled
        if enrich:
            url_entries = URLDeduplicator.mark_duplicates(url_entries)
        
        # Get processing statistics
        stats = BatchProcessor.get_processing_stats(url_entries)
        
        # Create URL input record
        input_id = str(uuid.uuid4())
        url_input = URLInput(
            input_id=input_id,
            urls=url_entries,
            source_type="direct",
            source_metadata={
                "input_method": "api_direct",
                "enriched": enrich,
                "processing_stats": stats
            },
            created_at=datetime.now()
        )
        
        # Store in memory
        url_inputs[input_id] = url_input
        
        logger.info("Direct URLs processed", 
                   input_id=input_id, 
                   **stats)
        
        return {
            "input_id": input_id,
            "source_type": "direct",
            "enriched": enrich,
            **stats
        }
        
    except Exception as e:
        logger.error("Error processing direct URL input", error=str(e))
        raise HTTPException(status_code=400, detail=f"Error processing URLs: {str(e)}")

@app.post("/api/input/form")
async def input_urls_form(urls_text: str = Form(...), enrich: bool = Form(True, description="Enable URL enrichment and deduplication")):
    """Web form URL input."""
    logger.info("Processing form URL input", enrich=enrich)
    
    try:
        # Parse URLs from form text with optional enrichment
        urls = URLParser.parse_text_file(urls_text, enrich=enrich)
        
        # Get processing statistics
        stats = BatchProcessor.get_processing_stats(urls)
        
        # Create URL input record
        input_id = str(uuid.uuid4())
        url_input = URLInput(
            input_id=input_id,
            urls=urls,
            source_type="form",
            source_metadata={
                "input_method": "web_form",
                "text_length": len(urls_text),
                "enriched": enrich,
                "processing_stats": stats
            },
            created_at=datetime.now()
        )
        
        # Store in memory
        url_inputs[input_id] = url_input
        
        logger.info("Form URLs processed", 
                   input_id=input_id, 
                   **stats)
        
        return {
            "input_id": input_id,
            "source_type": "form",
            "enriched": enrich,
            **stats
        }
        
    except Exception as e:
        logger.error("Error processing form URL input", error=str(e))
        raise HTTPException(status_code=400, detail=f"Error processing form URLs: {str(e)}")

# URL enrichment and processing endpoints

@app.post("/api/input/enrich/{input_id}")
async def enrich_urls(input_id: str):
    """Enrich existing URL input with metadata and categorization."""
    if input_id not in url_inputs:
        raise HTTPException(status_code=404, detail="Input not found")
    
    url_input = url_inputs[input_id]
    
    try:
        # Enrich all valid URLs
        enriched_urls = []
        for url_entry in url_input.urls:
            if url_entry.validated and not url_entry.enriched:
                enriched_entry = URLEnricher.enrich_url_entry(url_entry)
                enriched_urls.append(enriched_entry)
            else:
                enriched_urls.append(url_entry)
        
        # Mark duplicates
        enriched_urls = URLDeduplicator.mark_duplicates(enriched_urls)
        
        # Update the stored URLs
        url_input.urls = enriched_urls
        
        # Get updated statistics
        stats = BatchProcessor.get_processing_stats(enriched_urls)
        url_input.source_metadata["processing_stats"] = stats
        url_input.source_metadata["enriched"] = True
        
        logger.info("URLs enriched", input_id=input_id, **stats)
        
        return {
            "input_id": input_id,
            "enriched": True,
            **stats
        }
        
    except Exception as e:
        logger.error("Error enriching URLs", error=str(e), input_id=input_id)
        raise HTTPException(status_code=500, detail=f"Error enriching URLs: {str(e)}")

@app.get("/api/input/duplicates/{input_id}")
async def get_duplicates(input_id: str):
    """Get duplicate URLs for an input."""
    if input_id not in url_inputs:
        raise HTTPException(status_code=404, detail="Input not found")
    
    url_input = url_inputs[input_id]
    
    # Find duplicates
    duplicate_groups = URLDeduplicator.find_duplicates(url_input.urls)
    
    # Format response
    duplicates = []
    for normalized_url, entries in duplicate_groups.items():
        duplicates.append({
            "normalized_url": normalized_url,
            "count": len(entries),
            "urls": [
                {
                    "url": entry.url,
                    "category": entry.category,
                    "source_metadata": entry.source_metadata
                }
                for entry in entries
            ]
        })
    
    return {
        "input_id": input_id,
        "duplicate_groups": len(duplicates),
        "total_duplicates": sum(group["count"] - 1 for group in duplicates),
        "duplicates": duplicates
    }

@app.get("/api/input/categories/{input_id}")
async def get_categories(input_id: str):
    """Get URL categories and domain groups for an input."""
    if input_id not in url_inputs:
        raise HTTPException(status_code=404, detail="Input not found")
    
    url_input = url_inputs[input_id]
    
    # Get similarity groups (by domain)
    similarity_groups = URLDeduplicator.get_similarity_groups(url_input.urls)
    
    # Get category distribution
    categories = Counter(entry.category for entry in url_input.urls if entry.category)
    
    # Get domain distribution
    domains = Counter(entry.metadata.domain for entry in url_input.urls 
                     if entry.metadata and entry.validated)
    
    return {
        "input_id": input_id,
        "categories": dict(categories.most_common()),
        "domains": dict(domains.most_common(20)),
        "domain_groups": {
            domain: len(entries) for domain, entries in similarity_groups.items()
        }
    }

@app.post("/api/input/batch-process")
async def batch_process_urls(urls: List[str], batch_size: int = Query(100, description="Batch size for processing")):
    """Process large URL lists in batches."""
    logger.info("Processing batch URL input", url_count=len(urls), batch_size=batch_size)
    
    try:
        # Create URL entries
        url_entries = []
        for i, url in enumerate(urls):
            is_valid, error = URLValidator.validate_url(url)
            url_entries.append(URLEntry(
                url=url,
                source_metadata={"index": i},
                validated=is_valid,
                validation_error=error
            ))
        
        # Process in batches
        processed_urls = BatchProcessor.process_urls_batch(url_entries, batch_size=batch_size)
        
        # Get processing statistics
        stats = BatchProcessor.get_processing_stats(processed_urls)
        
        # Create URL input record
        input_id = str(uuid.uuid4())
        url_input = URLInput(
            input_id=input_id,
            urls=processed_urls,
            source_type="batch",
            source_metadata={
                "input_method": "batch_processing",
                "batch_size": batch_size,
                "enriched": True,
                "processing_stats": stats
            },
            created_at=datetime.now()
        )
        
        # Store in memory
        url_inputs[input_id] = url_input
        
        logger.info("Batch URLs processed", 
                   input_id=input_id, 
                   batch_size=batch_size,
                   **stats)
        
        return {
            "input_id": input_id,
            "source_type": "batch",
            "batch_size": batch_size,
            "enriched": True,
            **stats
        }
        
    except Exception as e:
        logger.error("Error processing batch URLs", error=str(e))
        raise HTTPException(status_code=400, detail=f"Error processing batch URLs: {str(e)}")

# Validation and preview endpoints

@app.get("/api/input/validate/{input_id}")
async def validate_input(input_id: str):
    """Validate parsed URL input."""
    if input_id not in url_inputs:
        raise HTTPException(status_code=404, detail="Input not found")
    
    url_input = url_inputs[input_id]
    
    # Re-validate all URLs
    for url_entry in url_input.urls:
        is_valid, error = URLValidator.validate_url(url_entry.url)
        url_entry.validated = is_valid
        url_entry.validation_error = error
    
    url_input.validated = True
    
    # Count results
    valid_count = sum(1 for url in url_input.urls if url.validated)
    invalid_count = len(url_input.urls) - valid_count
    
    logger.info("Input validated", 
               input_id=input_id,
               total_urls=len(url_input.urls),
               valid_urls=valid_count,
               invalid_urls=invalid_count)
    
    return {
        "input_id": input_id,
        "validated": True,
        "total_urls": len(url_input.urls),
        "valid_urls": valid_count,
        "invalid_urls": invalid_count,
        "validation_errors": [
            {"url": url.url, "error": url.validation_error}
            for url in url_input.urls if not url.validated
        ]
    }

@app.get("/api/input/preview/{input_id}")
async def preview_input(input_id: str, limit: int = Query(10, description="Number of URLs to preview"), 
                       show_duplicates: bool = Query(False, description="Include duplicate URLs in preview"),
                       category_filter: Optional[str] = Query(None, description="Filter by category")):
    """Preview parsed URL list with enriched data."""
    if input_id not in url_inputs:
        raise HTTPException(status_code=404, detail="Input not found")
    
    url_input = url_inputs[input_id]
    
    # Filter URLs based on parameters
    filtered_urls = url_input.urls
    
    # Filter out duplicates if requested
    if not show_duplicates:
        filtered_urls = [url for url in filtered_urls if not url.duplicate_of]
    
    # Filter by category if specified
    if category_filter:
        filtered_urls = [url for url in filtered_urls if url.category == category_filter]
    
    # Get preview of URLs
    preview_urls = filtered_urls[:limit]
    
    # Get processing statistics
    stats = BatchProcessor.get_processing_stats(url_input.urls)
    
    return {
        "input_id": input_id,
        "source_type": url_input.source_type,
        "total_urls": len(url_input.urls),
        "filtered_urls": len(filtered_urls),
        "preview_count": len(preview_urls),
        "created_at": url_input.created_at.isoformat(),
        "source_metadata": url_input.source_metadata,
        "processing_stats": stats,
        "filters": {
            "show_duplicates": show_duplicates,
            "category_filter": category_filter
        },
        "preview_urls": [
            {
                "url": url.url,
                "category": url.category,
                "priority": url.priority,
                "notes": url.notes,
                "validated": url.validated,
                "validation_error": url.validation_error,
                "enriched": url.enriched,
                "duplicate_of": url.duplicate_of,
                "similarity_group": url.similarity_group,
                "metadata": {
                    "domain": url.metadata.domain if url.metadata else None,
                    "subdomain": url.metadata.subdomain if url.metadata else None,
                    "path": url.metadata.path if url.metadata else None,
                    "parameter_count": url.metadata.parameter_count if url.metadata else 0,
                    "path_depth": url.metadata.path_depth if url.metadata else 0,
                    "tld": url.metadata.tld if url.metadata else None
                } if url.metadata else None,
                "source_metadata": url.source_metadata
            }
            for url in preview_urls
        ]
    }

@app.get("/api/input/{input_id}")
async def get_input(input_id: str, include_metadata: bool = Query(True, description="Include URL metadata in response")):
    """Get complete URL input data."""
    if input_id not in url_inputs:
        raise HTTPException(status_code=404, detail="Input not found")
    
    url_input = url_inputs[input_id]
    
    # Get processing statistics
    stats = BatchProcessor.get_processing_stats(url_input.urls)
    
    # Prepare URL data
    urls_data = []
    for url in url_input.urls:
        url_dict = asdict(url)
        if not include_metadata and 'metadata' in url_dict:
            # Remove detailed metadata if not requested
            url_dict['metadata'] = {
                'domain': url.metadata.domain if url.metadata else None,
                'path_depth': url.metadata.path_depth if url.metadata else 0
            }
        urls_data.append(url_dict)
    
    return {
        "input_id": input_id,
        "source_type": url_input.source_type,
        "total_urls": len(url_input.urls),
        "created_at": url_input.created_at.isoformat(),
        "validated": url_input.validated,
        "source_metadata": url_input.source_metadata,
        "processing_stats": stats,
        "urls": urls_data
    }

@app.get("/api/input")
async def list_inputs():
    """List all URL inputs."""
    return {
        "inputs": [
            {
                "input_id": input_id,
                "source_type": url_input.source_type,
                "total_urls": len(url_input.urls),
                "created_at": url_input.created_at.isoformat(),
                "validated": url_input.validated
            }
            for input_id, url_input in url_inputs.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)