from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog
from contextlib import asynccontextmanager

logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AI Engine Starting...")
    yield
    logger.info("AI Engine Shutdown")

app = FastAPI(title="AI Engine", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount sub-apps
try:
    from app.services.analyzer.main import app as analyzer_app
    app.mount("/analyzer", analyzer_app)
except ImportError as e:
    logger.error("Failed to mount Analyzer", error=str(e))

try:
    from app.services.chatbot.main import app as chatbot_app
    app.mount("/chat", chatbot_app)
except ImportError as e:
    logger.error("Failed to mount Chatbot", error=str(e))

try:
    from app.services.clustering.main import app as clustering_app
    app.mount("/clustering", clustering_app)
except ImportError as e:
    logger.error("Failed to mount Clustering", error=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "ai-engine"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
