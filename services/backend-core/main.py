import os
import httpx
import structlog
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.logging_config import setup_logging

setup_logging()
logger = structlog.get_logger()

app = FastAPI(
    title="Web Scraping Backend Core",
    description="Unified backend service for Web Scraping Tool",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Internal Services
# URL Input
try:
    from app.services.url_input.main import app as url_input_app
    app.mount("/api/url-input-service", url_input_app)
except ImportError as e:
    logger.error("Failed to mount URL Input", error=str(e))

# Session
try:
    from app.services.session.main import app as session_app
    app.mount("/api/session-service", session_app)
except ImportError as e:
    logger.error("Failed to mount Session", error=str(e))

# Export
try:
    from app.services.export.main import app as export_app
    app.mount("/api/export-service", export_app)
except ImportError as e:
    logger.error("Failed to mount Export", error=str(e))

# Auth (API)
try:
    from app.services.auth.main import app as auth_app
    app.mount("/api/auth-service", auth_app)
except ImportError as e:
    logger.error("Failed to mount Auth", error=str(e))


# Proxy for External Engines
AI_ENGINE_URL = os.getenv("AI_ENGINE_URL", "http://ai-engine:8090")
BROWSER_ENGINE_URL = os.getenv("BROWSER_ENGINE_URL", "http://browser-engine:8083")

async def proxy_request(url: str, request: Request):
    client = httpx.AsyncClient()
    try:
        body = await request.body()
        response = await client.request(
            method=request.method,
            url=url,
            headers=request.headers,
            content=body,
            params=request.query_params
        )
        return JSONResponse(
            content=response.json() if response.headers.get("content-type") == "application/json" else response.text,
            status_code=response.status_code
        )
    except Exception as e:
        logger.error("Proxy error", error=str(e), url=url)
        raise HTTPException(status_code=502, detail="Upstream service failed")

# AI Engine Routes Proxy
@app.api_route("/api/analyzer-service/{path:path}", methods=["GET", "POST"])
async def proxy_analyzer(path: str, request: Request):
    return await proxy_request(f"{AI_ENGINE_URL}/analyzer/{path}", request)

@app.api_route("/api/chatbot-service/{path:path}", methods=["GET", "POST"])
async def proxy_chatbot(path: str, request: Request):
    return await proxy_request(f"{AI_ENGINE_URL}/chat/{path}", request)

@app.api_route("/api/clustering-service/{path:path}", methods=["GET", "POST"])
async def proxy_clustering(path: str, request: Request):
    return await proxy_request(f"{AI_ENGINE_URL}/clustering/{path}", request)

# Browser Engine Routes Proxy
@app.api_route("/api/scraper-service/{path:path}", methods=["GET", "POST"])
async def proxy_scraper(path: str, request: Request):
    # Browser engine mounts scraper at root "/"
    return await proxy_request(f"{BROWSER_ENGINE_URL}/{path}", request)

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "backend-core"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
