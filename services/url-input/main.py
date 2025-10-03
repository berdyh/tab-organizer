"""URL Input Service - Handles URL parsing and validation from various sources."""

import time
import json
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from io import StringIO

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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
class URLEntry:
    url: str
    category: Optional[str] = None
    priority: Optional[str] = None
    notes: Optional[str] = None
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    validated: bool = False
    validation_error: Optional[str] = None

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

class URLParser:
    """Parse URLs from various input sources."""
    
    @classmethod
    def parse_text_file(cls, content: str) -> List[URLEntry]:
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
            urls.append(URLEntry(
                url=url,
                source_metadata={"line_number": line_num, "original_line": line},
                validated=is_valid,
                validation_error=error
            ))
        
        return urls
    
    @classmethod
    def parse_json_file(cls, content: str) -> List[URLEntry]:
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
                    urls.append(URLEntry(
                        url=item,
                        source_metadata={"index": i},
                        validated=is_valid,
                        validation_error=error
                    ))
                elif isinstance(item, dict) and 'url' in item:
                    url = item['url']
                    is_valid, error = URLValidator.validate_url(url)
                    urls.append(URLEntry(
                        url=url,
                        category=item.get('category'),
                        priority=item.get('priority'),
                        notes=item.get('notes'),
                        source_metadata={"index": i, **{k: v for k, v in item.items() if k != 'url'}},
                        validated=is_valid,
                        validation_error=error
                    ))
        
        elif isinstance(data, dict):
            # Handle {"urls": [...]} structure
            if 'urls' in data:
                url_list = data['urls']
                metadata = {k: v for k, v in data.items() if k != 'urls'}
                
                for i, item in enumerate(url_list):
                    if isinstance(item, str):
                        is_valid, error = URLValidator.validate_url(item)
                        urls.append(URLEntry(
                            url=item,
                            source_metadata={"index": i, **metadata},
                            validated=is_valid,
                            validation_error=error
                        ))
                    elif isinstance(item, dict) and 'url' in item:
                        url = item['url']
                        is_valid, error = URLValidator.validate_url(url)
                        urls.append(URLEntry(
                            url=url,
                            category=item.get('category'),
                            priority=item.get('priority'),
                            notes=item.get('notes'),
                            source_metadata={"index": i, **metadata, **{k: v for k, v in item.items() if k != 'url'}},
                            validated=is_valid,
                            validation_error=error
                        ))
        
        return urls
    
    @classmethod
    def parse_csv_file(cls, content: str) -> List[URLEntry]:
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
            
            urls.append(URLEntry(
                url=url,
                category=metadata.get('category'),
                priority=metadata.get('priority'),
                notes=metadata.get('notes'),
                source_metadata={"row_index": index, **metadata},
                validated=is_valid,
                validation_error=error
            ))
        
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

# File Upload Endpoints

@app.post("/api/input/upload/text")
async def upload_text_file(file: UploadFile = File(...)):
    """Upload and parse plain text file with URLs."""
    logger.info("Processing text file upload", filename=file.filename)
    
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Parse URLs from text
        urls = URLParser.parse_text_file(content_str)
        
        # Create URL input record
        input_id = str(uuid.uuid4())
        url_input = URLInput(
            input_id=input_id,
            urls=urls,
            source_type="text",
            source_metadata={
                "filename": file.filename,
                "file_size": len(content),
                "total_lines": len(content_str.split('\n'))
            },
            created_at=datetime.now()
        )
        
        # Store in memory
        url_inputs[input_id] = url_input
        
        # Count valid/invalid URLs
        valid_count = sum(1 for url in urls if url.validated)
        invalid_count = len(urls) - valid_count
        
        logger.info("Text file processed", 
                   input_id=input_id, 
                   total_urls=len(urls),
                   valid_urls=valid_count,
                   invalid_urls=invalid_count)
        
        return {
            "input_id": input_id,
            "source_type": "text",
            "total_urls": len(urls),
            "valid_urls": valid_count,
            "invalid_urls": invalid_count,
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error("Error processing text file", error=str(e), filename=file.filename)
        raise HTTPException(status_code=400, detail=f"Error processing text file: {str(e)}")

@app.post("/api/input/upload/json")
async def upload_json_file(file: UploadFile = File(...)):
    """Upload and parse JSON file with URLs."""
    logger.info("Processing JSON file upload", filename=file.filename)
    
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Parse URLs from JSON
        urls = URLParser.parse_json_file(content_str)
        
        # Create URL input record
        input_id = str(uuid.uuid4())
        url_input = URLInput(
            input_id=input_id,
            urls=urls,
            source_type="json",
            source_metadata={
                "filename": file.filename,
                "file_size": len(content)
            },
            created_at=datetime.now()
        )
        
        # Store in memory
        url_inputs[input_id] = url_input
        
        # Count valid/invalid URLs
        valid_count = sum(1 for url in urls if url.validated)
        invalid_count = len(urls) - valid_count
        
        logger.info("JSON file processed", 
                   input_id=input_id, 
                   total_urls=len(urls),
                   valid_urls=valid_count,
                   invalid_urls=invalid_count)
        
        return {
            "input_id": input_id,
            "source_type": "json",
            "total_urls": len(urls),
            "valid_urls": valid_count,
            "invalid_urls": invalid_count,
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error("Error processing JSON file", error=str(e), filename=file.filename)
        raise HTTPException(status_code=400, detail=f"Error processing JSON file: {str(e)}")

@app.post("/api/input/upload/csv")
async def upload_csv_file(file: UploadFile = File(...)):
    """Upload and parse CSV file with URLs."""
    logger.info("Processing CSV file upload", filename=file.filename)
    
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
        
        # Parse URLs from CSV
        urls = URLParser.parse_csv_file(content_str)
        
        # Create URL input record
        input_id = str(uuid.uuid4())
        url_input = URLInput(
            input_id=input_id,
            urls=urls,
            source_type="csv",
            source_metadata={
                "filename": file.filename,
                "file_size": len(content)
            },
            created_at=datetime.now()
        )
        
        # Store in memory
        url_inputs[input_id] = url_input
        
        # Count valid/invalid URLs
        valid_count = sum(1 for url in urls if url.validated)
        invalid_count = len(urls) - valid_count
        
        logger.info("CSV file processed", 
                   input_id=input_id, 
                   total_urls=len(urls),
                   valid_urls=valid_count,
                   invalid_urls=invalid_count)
        
        return {
            "input_id": input_id,
            "source_type": "csv",
            "total_urls": len(urls),
            "valid_urls": valid_count,
            "invalid_urls": invalid_count,
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error("Error processing CSV file", error=str(e), filename=file.filename)
        raise HTTPException(status_code=400, detail=f"Error processing CSV file: {str(e)}")

@app.post("/api/input/upload/excel")
async def upload_excel_file(file: UploadFile = File(...)):
    """Upload and parse Excel file with URLs."""
    logger.info("Processing Excel file upload", filename=file.filename)
    
    try:
        content = await file.read()
        
        # Read Excel file
        df = pd.read_excel(content)
        
        # Convert to CSV string for parsing
        csv_content = df.to_csv(index=False)
        
        # Parse URLs from CSV format
        urls = URLParser.parse_csv_file(csv_content)
        
        # Create URL input record
        input_id = str(uuid.uuid4())
        url_input = URLInput(
            input_id=input_id,
            urls=urls,
            source_type="excel",
            source_metadata={
                "filename": file.filename,
                "file_size": len(content),
                "sheet_shape": df.shape
            },
            created_at=datetime.now()
        )
        
        # Store in memory
        url_inputs[input_id] = url_input
        
        # Count valid/invalid URLs
        valid_count = sum(1 for url in urls if url.validated)
        invalid_count = len(urls) - valid_count
        
        logger.info("Excel file processed", 
                   input_id=input_id, 
                   total_urls=len(urls),
                   valid_urls=valid_count,
                   invalid_urls=invalid_count)
        
        return {
            "input_id": input_id,
            "source_type": "excel",
            "total_urls": len(urls),
            "valid_urls": valid_count,
            "invalid_urls": invalid_count,
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error("Error processing Excel file", error=str(e), filename=file.filename)
        raise HTTPException(status_code=400, detail=f"Error processing Excel file: {str(e)}")

# Direct input endpoints

@app.post("/api/input/urls")
async def input_urls_direct(urls: List[str]):
    """Direct URL list input."""
    logger.info("Processing direct URL input", url_count=len(urls))
    
    try:
        # Parse and validate URLs
        url_entries = []
        for i, url in enumerate(urls):
            is_valid, error = URLValidator.validate_url(url)
            url_entries.append(URLEntry(
                url=url,
                source_metadata={"index": i},
                validated=is_valid,
                validation_error=error
            ))
        
        # Create URL input record
        input_id = str(uuid.uuid4())
        url_input = URLInput(
            input_id=input_id,
            urls=url_entries,
            source_type="direct",
            source_metadata={
                "input_method": "api_direct"
            },
            created_at=datetime.now()
        )
        
        # Store in memory
        url_inputs[input_id] = url_input
        
        # Count valid/invalid URLs
        valid_count = sum(1 for url in url_entries if url.validated)
        invalid_count = len(url_entries) - valid_count
        
        logger.info("Direct URLs processed", 
                   input_id=input_id, 
                   total_urls=len(url_entries),
                   valid_urls=valid_count,
                   invalid_urls=invalid_count)
        
        return {
            "input_id": input_id,
            "source_type": "direct",
            "total_urls": len(url_entries),
            "valid_urls": valid_count,
            "invalid_urls": invalid_count
        }
        
    except Exception as e:
        logger.error("Error processing direct URL input", error=str(e))
        raise HTTPException(status_code=400, detail=f"Error processing URLs: {str(e)}")

@app.post("/api/input/form")
async def input_urls_form(urls_text: str = Form(...)):
    """Web form URL input."""
    logger.info("Processing form URL input")
    
    try:
        # Parse URLs from form text
        urls = URLParser.parse_text_file(urls_text)
        
        # Create URL input record
        input_id = str(uuid.uuid4())
        url_input = URLInput(
            input_id=input_id,
            urls=urls,
            source_type="form",
            source_metadata={
                "input_method": "web_form",
                "text_length": len(urls_text)
            },
            created_at=datetime.now()
        )
        
        # Store in memory
        url_inputs[input_id] = url_input
        
        # Count valid/invalid URLs
        valid_count = sum(1 for url in urls if url.validated)
        invalid_count = len(urls) - valid_count
        
        logger.info("Form URLs processed", 
                   input_id=input_id, 
                   total_urls=len(urls),
                   valid_urls=valid_count,
                   invalid_urls=invalid_count)
        
        return {
            "input_id": input_id,
            "source_type": "form",
            "total_urls": len(urls),
            "valid_urls": valid_count,
            "invalid_urls": invalid_count
        }
        
    except Exception as e:
        logger.error("Error processing form URL input", error=str(e))
        raise HTTPException(status_code=400, detail=f"Error processing form URLs: {str(e)}")

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
async def preview_input(input_id: str, limit: int = 10):
    """Preview parsed URL list."""
    if input_id not in url_inputs:
        raise HTTPException(status_code=404, detail="Input not found")
    
    url_input = url_inputs[input_id]
    
    # Get preview of URLs
    preview_urls = url_input.urls[:limit]
    
    return {
        "input_id": input_id,
        "source_type": url_input.source_type,
        "total_urls": len(url_input.urls),
        "preview_count": len(preview_urls),
        "created_at": url_input.created_at.isoformat(),
        "source_metadata": url_input.source_metadata,
        "preview_urls": [
            {
                "url": url.url,
                "category": url.category,
                "priority": url.priority,
                "notes": url.notes,
                "validated": url.validated,
                "validation_error": url.validation_error,
                "source_metadata": url.source_metadata
            }
            for url in preview_urls
        ]
    }

@app.get("/api/input/{input_id}")
async def get_input(input_id: str):
    """Get complete URL input data."""
    if input_id not in url_inputs:
        raise HTTPException(status_code=404, detail="Input not found")
    
    url_input = url_inputs[input_id]
    
    return {
        "input_id": input_id,
        "source_type": url_input.source_type,
        "total_urls": len(url_input.urls),
        "created_at": url_input.created_at.isoformat(),
        "validated": url_input.validated,
        "source_metadata": url_input.source_metadata,
        "urls": [asdict(url) for url in url_input.urls]
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