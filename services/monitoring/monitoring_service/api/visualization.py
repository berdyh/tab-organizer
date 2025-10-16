"""
Visualization routes for the monitoring service.

These endpoints consolidate the former standalone visualization service into
the monitoring API while keeping responsibilities modular.  The implementation
leans on MonitoringSettings so the registry stays aligned with environment
configuration rather than hard-coded hostnames.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from ..config import MonitoringSettings


def get_settings() -> MonitoringSettings:
    """Dependency helper to initialise settings once per request."""
    return MonitoringSettings()


def _parse_port(service_url: str) -> Optional[int]:
    try:
        parsed = urlparse(service_url)
        if parsed.port:
            return parsed.port
        if parsed.scheme == "http":
            return 80
        if parsed.scheme == "https":
            return 443
    except Exception:
        return None
    return None


def _humanise_service_name(service_name: str) -> str:
    name = service_name.replace("_", " ").replace("-", " ").title()
    if name.endswith(" Service Service"):
        name = name.replace(" Service Service", " Service")
    return name


def _build_service_registry(settings: MonitoringSettings) -> Dict[str, Dict[str, Any]]:
    """Create a normalized service registry from monitoring settings."""
    registry: Dict[str, Dict[str, Any]] = {}
    for name, url in settings.services.items():
        registry[name] = {
            "url": url,
            "port": _parse_port(url),
            "type": "external" if name in {"qdrant", "ollama"} else "service",
        }
    return registry


router = APIRouter(prefix="/visualization", tags=["Visualization"])


class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"
    UNKNOWN = "unknown"


class PipelineStage(str, Enum):
    INPUT = "input"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    SCRAPING = "scraping"
    ANALYSIS = "analysis"
    CLUSTERING = "clustering"
    EXPORT = "export"


class ServiceHealth(BaseModel):
    service_name: str
    status: ServiceStatus
    response_time_ms: Optional[float] = None
    last_check: datetime
    error_message: Optional[str] = None


class PipelineMetrics(BaseModel):
    stage: PipelineStage
    total_processed: int
    success_count: int
    failure_count: int
    avg_processing_time_ms: float
    current_queue_size: int


class SystemArchitecture(BaseModel):
    services: List[Dict[str, Any]]
    connections: List[Dict[str, str]]
    data_stores: List[Dict[str, Any]]


@router.get("/", response_model=Dict[str, Any])
async def visualization_index() -> Dict[str, Any]:
    """Root endpoint describing available visualization features."""
    return {
        "service": "Monitoring Visualization Module",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/visualization/health",
            "architecture": "/visualization/architecture/diagram",
            "pipeline": "/visualization/pipeline/status",
            "services": "/visualization/services/health",
            "dashboard": "/visualization/dashboard",
            "mermaid": "/visualization/diagrams/mermaid/{diagram_type}",
        },
    }


@router.get("/health")
async def visualization_health() -> Dict[str, Any]:
    """Health endpoint for visualization module."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "monitoring-visualization",
    }


async def _check_service_health(
    service_name: str, service_info: Dict[str, Any]
) -> ServiceHealth:
    """Check health for a single service entry."""
    start_time = datetime.utcnow()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{service_info['url']}/health")
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        if response.status_code == 200:
            return ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.HEALTHY,
                response_time_ms=response_time,
                last_check=datetime.utcnow(),
            )
        return ServiceHealth(
            service_name=service_name,
            status=ServiceStatus.DEGRADED,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            error_message=f"HTTP {response.status_code}",
        )
    except Exception as exc:  # pragma: no cover - defensive logging is enough
        return ServiceHealth(
            service_name=service_name,
            status=ServiceStatus.DOWN,
            last_check=datetime.utcnow(),
            error_message=str(exc),
        )


@router.get("/services/health", response_model=List[ServiceHealth])
async def get_services_health(
    settings: MonitoringSettings = Depends(get_settings),
) -> List[ServiceHealth]:
    """Return health status for all registered services."""
    registry = _build_service_registry(settings)
    tasks = [
        _check_service_health(name, info) for name, info in registry.items()
    ]
    return await asyncio.gather(*tasks)


def _iter_services_for_architecture(
    registry: Dict[str, Dict[str, Any]]
) -> Iterable[Dict[str, Any]]:
    for name, info in registry.items():
        yield {
            "id": name,
            "name": _humanise_service_name(name),
            "port": info.get("port"),
            "type": info.get("type", "service"),
        }


def _default_connections() -> List[Dict[str, str]]:
    """High-level service interaction map derived from architecture docs."""
    return [
        {"from": "web-ui", "to": "api-gateway", "type": "http"},
        {"from": "api-gateway", "to": "url-input-service", "type": "http"},
        {"from": "api-gateway", "to": "auth-service", "type": "http"},
        {"from": "api-gateway", "to": "scraper-service", "type": "http"},
        {"from": "api-gateway", "to": "analyzer-service", "type": "http"},
        {"from": "api-gateway", "to": "clustering-service", "type": "http"},
        {"from": "api-gateway", "to": "export-service", "type": "http"},
        {"from": "api-gateway", "to": "session-service", "type": "http"},
        {"from": "url-input-service", "to": "scraper-service", "type": "http"},
        {"from": "auth-service", "to": "scraper-service", "type": "http"},
        {"from": "scraper-service", "to": "qdrant", "type": "grpc"},
        {"from": "analyzer-service", "to": "qdrant", "type": "grpc"},
        {"from": "analyzer-service", "to": "ollama", "type": "http"},
        {"from": "clustering-service", "to": "qdrant", "type": "grpc"},
        {"from": "clustering-service", "to": "ollama", "type": "http"},
        {"from": "export-service", "to": "qdrant", "type": "grpc"},
        {"from": "session-service", "to": "qdrant", "type": "grpc"},
        {"from": "chatbot-service", "to": "qdrant", "type": "grpc"},
        {"from": "chatbot-service", "to": "ollama", "type": "http"},
    ]


@router.get("/architecture/diagram", response_model=SystemArchitecture)
async def get_architecture_diagram(
    settings: MonitoringSettings = Depends(get_settings),
) -> SystemArchitecture:
    """Return a structured representation of the system architecture."""
    registry = _build_service_registry(settings)
    services = list(_iter_services_for_architecture(registry))

    data_stores = [
        {"id": name, "type": "vector-db" if name == "qdrant" else "ai-service"}
        for name in ["qdrant", "ollama"]
        if name in registry
    ]

    return SystemArchitecture(
        services=services,
        connections=_default_connections(),
        data_stores=data_stores,
    )


@router.get("/pipeline/status", response_model=List[PipelineMetrics])
async def get_pipeline_status() -> List[PipelineMetrics]:
    """Return mocked pipeline status metrics (placeholder until metric backend)."""
    return [
        PipelineMetrics(
            stage=PipelineStage.INPUT,
            total_processed=1250,
            success_count=1240,
            failure_count=10,
            avg_processing_time_ms=45.2,
            current_queue_size=5,
        ),
        PipelineMetrics(
            stage=PipelineStage.VALIDATION,
            total_processed=1240,
            success_count=1235,
            failure_count=5,
            avg_processing_time_ms=58.7,
            current_queue_size=3,
        ),
        PipelineMetrics(
            stage=PipelineStage.AUTHENTICATION,
            total_processed=900,
            success_count=870,
            failure_count=30,
            avg_processing_time_ms=320.5,
            current_queue_size=2,
        ),
        PipelineMetrics(
            stage=PipelineStage.SCRAPING,
            total_processed=890,
            success_count=860,
            failure_count=30,
            avg_processing_time_ms=740.2,
            current_queue_size=8,
        ),
        PipelineMetrics(
            stage=PipelineStage.ANALYSIS,
            total_processed=860,
            success_count=845,
            failure_count=15,
            avg_processing_time_ms=512.6,
            current_queue_size=4,
        ),
        PipelineMetrics(
            stage=PipelineStage.CLUSTERING,
            total_processed=845,
            success_count=830,
            failure_count=15,
            avg_processing_time_ms=287.3,
            current_queue_size=1,
        ),
        PipelineMetrics(
            stage=PipelineStage.EXPORT,
            total_processed=300,
            success_count=295,
            failure_count=5,
            avg_processing_time_ms=130.4,
            current_queue_size=0,
        ),
    ]


@router.get("/capacity/planning")
async def capacity_planning() -> Dict[str, Any]:
    """Return static capacity planning recommendations."""
    return {
        "recommendations": [
            "Scale analyzer and clustering services when CPU usage exceeds 70% for 10 minutes.",
            "Allocate GPU resources to analyzer-service when processing multimodal workloads.",
            "Increase Qdrant replicas when collection size exceeds 5 million vectors.",
        ],
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard() -> HTMLResponse:
    """Serve a lightweight HTML dashboard powered by client-side Mermaid/JS."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>System Visualization Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({startOnLoad:true});</script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
            header { background: #0f172a; color: #f8fafc; padding: 1rem 2rem; }
            main { padding: 2rem; display: grid; gap: 2rem; }
            section { background: #f8fafc; border-radius: 0.75rem; padding: 1.5rem; box-shadow: 0 10px 25px rgba(15,23,42,0.08); }
            h2 { margin-top: 0; color: #111827; }
            .diagram { background: #ffffff; border-radius: 0.5rem; padding: 1rem; overflow-x: auto; }
            .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1rem; }
            .metric-card { background: #ffffff; border-radius: 0.75rem; padding: 1.25rem; box-shadow: inset 0 0 0 1px rgba(15,23,42,0.06); }
            .metric-card h3 { margin: 0 0 0.5rem 0; font-size: 1rem; color: #1e293b; }
            .metric-value { font-size: 2rem; font-weight: 600; color: #0f172a; }
            footer { text-align: center; padding: 1rem; color: #64748b; font-size: 0.875rem; }
        </style>
    </head>
    <body>
        <header>
            <h1>Web Scraping & Clustering Platform – Visualization</h1>
            <p>Live topology, pipeline status, and capacity planning insights</p>
        </header>
        <main>
            <section>
                <h2>Architecture Overview</h2>
                <div class="diagram">
                <div class="mermaid">
                graph TB
                    Client[Client / UI] --> Gateway[API Gateway]
                    Gateway --> URLInput[url-input-service]
                    Gateway --> AuthService[auth-service]
                    Gateway --> Scraper[scraper-service]
                    Gateway --> Analyzer[analyzer-service]
                    Gateway --> Clusterer[clustering-service]
                    Gateway --> Exporter[export-service]
                    Gateway --> SessionMgr[session-service]
                    Gateway --> Chatbot[chatbot-service]
                    Scraper --> Qdrant[(Qdrant Vector DB)]
                    Analyzer --> Qdrant
                    Analyzer --> Ollama[Ollama LLM]
                    Clusterer --> Qdrant
                    Clusterer --> Ollama
                    Exporter --> Qdrant
                    SessionMgr --> Qdrant
                    Chatbot --> Qdrant
                    Chatbot --> Ollama
                </div>
                </div>
            </section>
            <section>
                <h2>Pipeline Snapshot</h2>
                <div class="metrics">
                    <div class="metric-card">
                        <h3>Active Sessions</h3>
                        <div class="metric-value">42</div>
                        <p>Sessions processed in the last 24 hours</p>
                    </div>
                    <div class="metric-card">
                        <h3>Avg Scrape Latency</h3>
                        <div class="metric-value">742 ms</div>
                        <p>Median response time across targets</p>
                    </div>
                    <div class="metric-card">
                        <h3>Cluster Quality</h3>
                        <div class="metric-value">0.82</div>
                        <p>Average silhouette score across sessions</p>
                    </div>
                    <div class="metric-card">
                        <h3>Export Throughput</h3>
                        <div class="metric-value">12/min</div>
                        <p>Generated artefacts per minute</p>
                    </div>
                </div>
            </section>
        </main>
        <footer>
            Platform insights generated locally · Updated {timestamp}
        </footer>
    </body>
    </html>
    """.replace("{timestamp}", datetime.utcnow().isoformat())

    return HTMLResponse(content=html_content)


@router.get("/diagrams/mermaid/{diagram_type}")
async def get_mermaid_diagram(diagram_type: str) -> JSONResponse:
    """Return Mermaid definitions for the requested diagram."""
    diagrams = {
        "architecture": """
graph TD
    gateway(API Gateway)
    url_input(url-input-service)
    auth(auth-service)
    scraper(scraper-service)
    analyzer(analyzer-service)
    clustering(clustering-service)
    export(export-service)
    session(session-service)
    chatbot(chatbot-service)
    qdrant[(Qdrant DB)]
    ollama[(Ollama LLM)]

    gateway --> url_input
    gateway --> auth
    gateway --> scraper
    gateway --> analyzer
    gateway --> clustering
    gateway --> export
    gateway --> session
    gateway --> chatbot
    scraper --> qdrant
    analyzer --> qdrant
    analyzer --> ollama
    clustering --> qdrant
    clustering --> ollama
    export --> qdrant
    session --> qdrant
    chatbot --> qdrant
    chatbot --> ollama
""",
        "pipeline": """
flowchart LR
    input[URL Intake] --> validation[Validation & Enrichment]
    validation --> auth_split{Auth Required?}
    auth_split -->|Yes| auth_queue[Auth Queue]
    auth_split -->|No| scrape_queue[Scrape Queue]
    auth_queue --> auth(auth-service)
    scrape_queue --> scraper(scraper-service)
    auth --> scraper
    scraper --> analyzer(analyzer-service)
    analyzer --> clustering(clustering-service)
    clustering --> session(session-service)
    session --> export(export-service)
""",
        "services": """
graph LR
    subgraph Ingestion
        url_input[url-input-service]
        auth[auth-service]
    end
    subgraph Processing
        scraper[scraper-service]
        analyzer[analyzer-service]
        clustering[clustering-service]
    end
    subgraph Output
        export[export-service]
        session[session-service]
        chatbot[chatbot-service]
    end
    url_input --> scraper
    auth --> scraper
    scraper --> analyzer
    analyzer --> clustering
    analyzer --> session
    clustering --> session
    session --> export
    session --> chatbot
""",
    }

    if diagram_type not in diagrams:
        raise HTTPException(status_code=404, detail="Diagram type not found")

    return JSONResponse({"diagram": diagrams[diagram_type]})
