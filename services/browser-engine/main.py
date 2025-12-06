from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog
from contextlib import asynccontextmanager

logger = structlog.get_logger()

app = FastAPI(
    title="Browser Engine",
    description="Handles browser-based tasks: Scraping and Authentication",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount sub-apps
try:
    from app.scraper.main import app as scraper_app
    app.mount("/", scraper_app)
except ImportError as e:
    logger.error("Failed to mount Scraper", error=str(e))

try:
    from app.auth.main import app as auth_app
    app.mount("/auth", auth_app)
except ImportError as e:
    logger.error("Failed to mount Auth", error=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "browser-engine"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)
