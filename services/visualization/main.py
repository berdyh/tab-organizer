"""
Visualization Service - Architecture Diagrams and Pipeline Monitoring
Provides comprehensive system visualization, real-time monitoring, and documentation generation
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
from datetime import datetime
import httpx
import asyncio
from enum import Enum

app = FastAPI(title="Visualization Service", version="1.0.0")

# Service registry with health endpoints
SERVICE_REGISTRY = {
    "api-gateway": {"url": "http://api-gateway:8080", "port": 8080},
    "url-input": {"url": "http://url-input:8081", "port": 8081},
    "auth": {"url": "http://auth:8082", "port": 8082},
    "scraper": {"url": "http://scraper:8083", "port": 8083},
    "analyzer": {"url": "http://analyzer:8084", "port": 8084},
    "clustering": {"url": "http://clustering:8085", "port": 8085},
    "export": {"url": "http://export:8086", "port": 8086},
    "session": {"url": "http://session:8087", "port": 8087},
    "model-manager": {"url": "http://model-manager:8088", "port": 8088},
    "web-ui": {"url": "http://web-ui:8089", "port": 8089},
}

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

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Visualization Service",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "architecture": "/architecture/diagram",
            "pipeline": "/pipeline/status",
            "services": "/services/health",
            "dashboard": "/dashboard",
            "mermaid": "/diagrams/mermaid/{diagram_type}",
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "visualization"
    }

async def check_service_health(service_name: str, service_info: Dict) -> ServiceHealth:
    """Check health of a single service"""
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
                    last_check=datetime.utcnow()
                )
            else:
                return ServiceHealth(
                    service_name=service_name,
                    status=ServiceStatus.DEGRADED,
                    response_time_ms=response_time,
                    last_check=datetime.utcnow(),
                    error_message=f"HTTP {response.status_code}"
                )
    except Exception as e:
        return ServiceHealth(
            service_name=service_name,
            status=ServiceStatus.DOWN,
            last_check=datetime.utcnow(),
            error_message=str(e)
        )

@app.get("/services/health", response_model=List[ServiceHealth])
async def get_services_health():
    """Get health status of all services"""
    tasks = [
        check_service_health(name, info)
        for name, info in SERVICE_REGISTRY.items()
    ]
    results = await asyncio.gather(*tasks)
    return results

@app.get("/architecture/diagram")
async def get_architecture_diagram():
    """Get system architecture diagram data"""
    services = [
        {
            "id": name,
            "name": name.replace("-", " ").title(),
            "port": info["port"],
            "type": "service"
        }
        for name, info in SERVICE_REGISTRY.items()
    ]
    
    # Add data stores
    services.extend([
        {"id": "qdrant", "name": "Qdrant Vector DB", "port": 6333, "type": "database"},
        {"id": "ollama", "name": "Ollama LLM", "port": 11434, "type": "ai-service"}
    ])
    
    # Define service connections
    connections = [
        {"from": "web-ui", "to": "api-gateway", "type": "http"},
        {"from": "api-gateway", "to": "url-input", "type": "http"},
        {"from": "api-gateway", "to": "auth", "type": "http"},
        {"from": "api-gateway", "to": "scraper", "type": "http"},
        {"from": "api-gateway", "to": "analyzer", "type": "http"},
        {"from": "api-gateway", "to": "clustering", "type": "http"},
        {"from": "api-gateway", "to": "export", "type": "http"},
        {"from": "api-gateway", "to": "session", "type": "http"},
        {"from": "api-gateway", "to": "model-manager", "type": "http"},
        {"from": "url-input", "to": "scraper", "type": "http"},
        {"from": "auth", "to": "scraper", "type": "http"},
        {"from": "scraper", "to": "qdrant", "type": "grpc"},
        {"from": "analyzer", "to": "qdrant", "type": "grpc"},
        {"from": "analyzer", "to": "ollama", "type": "http"},
        {"from": "clustering", "to": "qdrant", "type": "grpc"},
        {"from": "clustering", "to": "ollama", "type": "http"},
        {"from": "export", "to": "qdrant", "type": "grpc"},
        {"from": "session", "to": "qdrant", "type": "grpc"},
        {"from": "model-manager", "to": "ollama", "type": "http"},
    ]
    
    return SystemArchitecture(
        services=services,
        connections=connections,
        data_stores=[
            {"id": "qdrant", "type": "vector-db", "purpose": "embeddings"},
            {"id": "ollama", "type": "llm-service", "purpose": "ai-models"}
        ]
    )

@app.get("/pipeline/status")
async def get_pipeline_status():
    """Get real-time pipeline processing status"""
    # Mock data - in production, this would query actual service metrics
    stages = [
        PipelineMetrics(
            stage=PipelineStage.INPUT,
            total_processed=1250,
            success_count=1240,
            failure_count=10,
            avg_processing_time_ms=45.2,
            current_queue_size=5
        ),
        PipelineMetrics(
            stage=PipelineStage.VALIDATION,
            total_processed=1240,
            success_count=1235,
            failure_count=5,
            avg_processing_time_ms=120.5,
            current_queue_size=3
        ),
        PipelineMetrics(
            stage=PipelineStage.AUTHENTICATION,
            total_processed=450,
            success_count=445,
            failure_count=5,
            avg_processing_time_ms=2500.0,
            current_queue_size=8
        ),
        PipelineMetrics(
            stage=PipelineStage.SCRAPING,
            total_processed=1235,
            success_count=1200,
            failure_count=35,
            avg_processing_time_ms=3200.0,
            current_queue_size=15
        ),
        PipelineMetrics(
            stage=PipelineStage.ANALYSIS,
            total_processed=1200,
            success_count=1195,
            failure_count=5,
            avg_processing_time_ms=1800.0,
            current_queue_size=10
        ),
        PipelineMetrics(
            stage=PipelineStage.CLUSTERING,
            total_processed=1195,
            success_count=1195,
            failure_count=0,
            avg_processing_time_ms=5000.0,
            current_queue_size=0
        ),
        PipelineMetrics(
            stage=PipelineStage.EXPORT,
            total_processed=250,
            success_count=248,
            failure_count=2,
            avg_processing_time_ms=800.0,
            current_queue_size=2
        ),
    ]
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "stages": stages,
        "overall_health": "healthy",
        "bottlenecks": identify_bottlenecks(stages)
    }

def identify_bottlenecks(stages: List[PipelineMetrics]) -> List[Dict[str, Any]]:
    """Identify pipeline bottlenecks"""
    bottlenecks = []
    
    for stage in stages:
        # High queue size indicates bottleneck
        if stage.current_queue_size > 10:
            bottlenecks.append({
                "stage": stage.stage,
                "issue": "high_queue_size",
                "severity": "warning",
                "queue_size": stage.current_queue_size,
                "recommendation": f"Consider scaling {stage.stage} service"
            })
        
        # High failure rate
        if stage.total_processed > 0:
            failure_rate = stage.failure_count / stage.total_processed
            if failure_rate > 0.05:  # 5% failure rate
                bottlenecks.append({
                    "stage": stage.stage,
                    "issue": "high_failure_rate",
                    "severity": "critical",
                    "failure_rate": f"{failure_rate * 100:.2f}%",
                    "recommendation": f"Investigate failures in {stage.stage} stage"
                })
        
        # Slow processing
        if stage.avg_processing_time_ms > 4000:
            bottlenecks.append({
                "stage": stage.stage,
                "issue": "slow_processing",
                "severity": "info",
                "avg_time_ms": stage.avg_processing_time_ms,
                "recommendation": f"Optimize {stage.stage} processing time"
            })
    
    return bottlenecks

@app.get("/diagrams/mermaid/{diagram_type}")
async def get_mermaid_diagram(diagram_type: str):
    """Generate Mermaid diagram code for various architecture views"""
    diagrams = {
        "architecture": generate_architecture_mermaid(),
        "pipeline": generate_pipeline_mermaid(),
        "services": generate_services_mermaid(),
        "data-flow": generate_dataflow_mermaid(),
        "deployment": generate_deployment_mermaid(),
    }
    
    if diagram_type not in diagrams:
        raise HTTPException(status_code=404, detail=f"Diagram type '{diagram_type}' not found")
    
    return {"diagram_type": diagram_type, "mermaid_code": diagrams[diagram_type]}

def generate_architecture_mermaid() -> str:
    """Generate high-level architecture diagram"""
    return """graph TB
    Client[Client/API Consumer] --> Gateway[API Gateway :8080]
    
    Gateway --> URLInput[URL Input Service :8081]
    Gateway --> AuthService[Authentication Service :8082]
    Gateway --> Scraper[Web Scraper Service :8083]
    Gateway --> Analyzer[Content Analyzer Service :8084]
    Gateway --> Clusterer[Clustering Service :8085]
    Gateway --> Exporter[Export Service :8086]
    Gateway --> SessionMgr[Session Manager :8087]
    Gateway --> ModelMgr[Model Management Service :8088]
    Gateway --> UI[Web UI Service :8089]
    
    URLInput --> Scraper
    AuthService --> Scraper
    Scraper --> Qdrant[(Qdrant Vector DB :6333)]
    Analyzer --> Qdrant
    Analyzer --> Ollama[Ollama LLM :11434]
    Clusterer --> Qdrant
    Clusterer --> Ollama
    Exporter --> Qdrant
    SessionMgr --> Qdrant
    ModelMgr --> Ollama
    UI --> Gateway
    
    style Gateway fill:#4CAF50
    style Qdrant fill:#2196F3
    style Ollama fill:#FF9800"""

def generate_pipeline_mermaid() -> str:
    """Generate data pipeline flow diagram"""
    return """flowchart TD
    A[URL Input] --> B{Authentication Required?}
    B -->|No| C[Direct Scraping Queue]
    B -->|Yes| D[Authentication Queue]
    
    C --> E[Scrapy Framework]
    D --> F[Auth Detection & Login]
    F --> G[Authenticated Scraping]
    
    E --> H[Content Extraction]
    G --> H
    
    H --> I[Content Analysis]
    I --> J[Embedding Generation]
    J --> K[Vector Storage - Qdrant]
    
    K --> L[UMAP Dimensionality Reduction]
    L --> M[HDBSCAN Clustering]
    M --> N[LLM Cluster Labeling]
    
    N --> O[Visualization Generation]
    N --> P[Export Processing]
    
    P --> Q[Notion Export]
    P --> R[Obsidian Export]
    P --> S[Word Export]
    P --> T[Markdown Export]
    
    style A fill:#4CAF50
    style K fill:#2196F3
    style N fill:#FF9800"""

def generate_services_mermaid() -> str:
    """Generate service interaction sequence diagram"""
    return """sequenceDiagram
    participant UI as Web UI
    participant GW as API Gateway
    participant URL as URL Input
    participant AUTH as Auth Service
    participant SCRAPER as Scraper
    participant ANALYZER as Analyzer
    participant CLUSTER as Clustering
    participant Q as Qdrant DB
    participant O as Ollama LLM

    UI->>GW: Submit URLs
    GW->>URL: Validate URLs
    URL->>AUTH: Check auth requirements
    AUTH-->>URL: Auth status
    URL->>SCRAPER: Send URLs
    
    par Parallel Processing
        SCRAPER->>SCRAPER: Process public URLs
    and
        SCRAPER->>AUTH: Request auth
        AUTH->>UI: Prompt credentials
        UI->>AUTH: Provide credentials
        AUTH->>SCRAPER: Auth complete
    end
    
    SCRAPER->>ANALYZER: Send content
    ANALYZER->>O: Generate embeddings
    O-->>ANALYZER: Return embeddings
    ANALYZER->>Q: Store embeddings
    
    ANALYZER->>CLUSTER: Trigger clustering
    CLUSTER->>Q: Retrieve embeddings
    CLUSTER->>O: Generate labels
    O-->>CLUSTER: Return labels
    CLUSTER->>Q: Store results"""

def generate_dataflow_mermaid() -> str:
    """Generate data flow diagram"""
    return """graph LR
    subgraph Input
        A[URLs] --> B[Validation]
        B --> C[Classification]
    end
    
    subgraph Processing
        C --> D[Scraping]
        D --> E[Content Extraction]
        E --> F[Analysis]
    end
    
    subgraph AI Processing
        F --> G[Embedding Generation]
        G --> H[Vector Storage]
        H --> I[Clustering]
        I --> J[Labeling]
    end
    
    subgraph Output
        J --> K[Visualization]
        J --> L[Export]
        L --> M[Multiple Formats]
    end
    
    style Input fill:#E8F5E9
    style Processing fill:#E3F2FD
    style AI Processing fill:#FFF3E0
    style Output fill:#F3E5F5"""

def generate_deployment_mermaid() -> str:
    """Generate deployment architecture diagram"""
    return """graph TB
    subgraph "Docker Host"
        subgraph "Frontend Network"
            UI[Web UI :8089]
            GW[API Gateway :8080]
        end
        
        subgraph "Backend Network"
            URL[URL Input :8081]
            AUTH[Auth Service :8082]
            SCRAPER[Scraper :8083]
            ANALYZER[Analyzer :8084]
            CLUSTER[Clustering :8085]
            EXPORT[Export :8086]
            SESSION[Session :8087]
            MODEL[Model Manager :8088]
        end
        
        subgraph "Data Network"
            QDRANT[(Qdrant :6333)]
            OLLAMA[Ollama :11434]
        end
        
        subgraph "Storage Volumes"
            V1[qdrant_data]
            V2[ollama_data]
            V3[scraped_data]
        end
    end
    
    UI --> GW
    GW --> URL
    GW --> AUTH
    GW --> SCRAPER
    GW --> ANALYZER
    GW --> CLUSTER
    GW --> EXPORT
    GW --> SESSION
    GW --> MODEL
    
    SCRAPER --> QDRANT
    ANALYZER --> QDRANT
    ANALYZER --> OLLAMA
    CLUSTER --> QDRANT
    CLUSTER --> OLLAMA
    
    QDRANT --> V1
    OLLAMA --> V2
    SCRAPER --> V3"""

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Interactive dashboard with real-time visualization"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>System Architecture Dashboard</title>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                border-bottom: 2px solid #ddd;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                background: white;
                border: 1px solid #ddd;
                border-bottom: none;
                border-radius: 5px 5px 0 0;
            }
            .tab.active {
                background: #4CAF50;
                color: white;
            }
            .tab-content {
                display: none;
                background: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .tab-content.active {
                display: block;
            }
            .mermaid {
                text-align: center;
                margin: 20px 0;
            }
            .metrics {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .metric-card {
                background: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-card h3 {
                margin: 0 0 10px 0;
                color: #666;
                font-size: 14px;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #4CAF50;
            }
            .status-healthy { color: #4CAF50; }
            .status-degraded { color: #FF9800; }
            .status-down { color: #F44336; }
            .refresh-btn {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            .refresh-btn:hover {
                background: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏗️ System Architecture & Pipeline Dashboard</h1>
            
            <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>
            
            <div class="tabs">
                <div class="tab active" onclick="showTab('architecture')">Architecture</div>
                <div class="tab" onclick="showTab('pipeline')">Pipeline Flow</div>
                <div class="tab" onclick="showTab('services')">Service Health</div>
                <div class="tab" onclick="showTab('deployment')">Deployment</div>
                <div class="tab" onclick="showTab('metrics')">Metrics</div>
            </div>
            
            <div id="architecture" class="tab-content active">
                <h2>System Architecture</h2>
                <div class="mermaid" id="arch-diagram">
                    Loading...
                </div>
            </div>
            
            <div id="pipeline" class="tab-content">
                <h2>Data Pipeline Flow</h2>
                <div class="mermaid" id="pipeline-diagram">
                    Loading...
                </div>
            </div>
            
            <div id="services" class="tab-content">
                <h2>Service Interaction</h2>
                <div class="mermaid" id="services-diagram">
                    Loading...
                </div>
                <div id="service-health"></div>
            </div>
            
            <div id="deployment" class="tab-content">
                <h2>Deployment Architecture</h2>
                <div class="mermaid" id="deployment-diagram">
                    Loading...
                </div>
            </div>
            
            <div id="metrics" class="tab-content">
                <h2>Pipeline Metrics</h2>
                <div id="pipeline-metrics"></div>
            </div>
        </div>
        
        <script>
            mermaid.initialize({ startOnLoad: false, theme: 'default' });
            
            function showTab(tabName) {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                event.target.classList.add('active');
                document.getElementById(tabName).classList.add('active');
            }
            
            async function loadDiagram(type, elementId) {
                const response = await fetch(`/diagrams/mermaid/${type}`);
                const data = await response.json();
                const element = document.getElementById(elementId);
                element.innerHTML = data.mermaid_code;
                mermaid.init(undefined, element);
            }
            
            async function loadServiceHealth() {
                const response = await fetch('/services/health');
                const services = await response.json();
                const container = document.getElementById('service-health');
                
                container.innerHTML = '<div class="metrics">' + services.map(s => `
                    <div class="metric-card">
                        <h3>${s.service_name}</h3>
                        <div class="metric-value status-${s.status}">${s.status.toUpperCase()}</div>
                        ${s.response_time_ms ? `<p>Response: ${s.response_time_ms.toFixed(2)}ms</p>` : ''}
                        ${s.error_message ? `<p style="color: red; font-size: 12px;">${s.error_message}</p>` : ''}
                    </div>
                `).join('') + '</div>';
            }
            
            async function loadPipelineMetrics() {
                const response = await fetch('/pipeline/status');
                const data = await response.json();
                const container = document.getElementById('pipeline-metrics');
                
                container.innerHTML = '<div class="metrics">' + data.stages.map(s => `
                    <div class="metric-card">
                        <h3>${s.stage.toUpperCase()}</h3>
                        <p>Processed: <strong>${s.total_processed}</strong></p>
                        <p>Success: <strong class="status-healthy">${s.success_count}</strong></p>
                        <p>Failed: <strong class="status-down">${s.failure_count}</strong></p>
                        <p>Avg Time: <strong>${s.avg_processing_time_ms.toFixed(2)}ms</strong></p>
                        <p>Queue: <strong>${s.current_queue_size}</strong></p>
                    </div>
                `).join('') + '</div>';
                
                if (data.bottlenecks.length > 0) {
                    container.innerHTML += '<h3>⚠️ Bottlenecks Detected</h3><div class="metrics">' +
                        data.bottlenecks.map(b => `
                            <div class="metric-card" style="border-left: 4px solid ${b.severity === 'critical' ? '#F44336' : '#FF9800'}">
                                <h3>${b.stage.toUpperCase()}</h3>
                                <p><strong>${b.issue.replace('_', ' ').toUpperCase()}</strong></p>
                                <p>${b.recommendation}</p>
                            </div>
                        `).join('') + '</div>';
                }
            }
            
            async function refreshData() {
                await Promise.all([
                    loadDiagram('architecture', 'arch-diagram'),
                    loadDiagram('pipeline', 'pipeline-diagram'),
                    loadDiagram('services', 'services-diagram'),
                    loadDiagram('deployment', 'deployment-diagram'),
                    loadServiceHealth(),
                    loadPipelineMetrics()
                ]);
            }
            
            // Initial load
            refreshData();
            
            // Auto-refresh every 30 seconds
            setInterval(refreshData, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/capacity/planning")
async def get_capacity_planning():
    """Get capacity planning and resource allocation data"""
    return {
        "current_capacity": {
            "total_services": len(SERVICE_REGISTRY),
            "active_connections": 45,
            "cpu_usage_percent": 65.5,
            "memory_usage_gb": 8.2,
            "storage_usage_gb": 125.5
        },
        "recommendations": [
            {
                "resource": "scraper",
                "current_instances": 1,
                "recommended_instances": 3,
                "reason": "High queue size and processing time",
                "priority": "high"
            },
            {
                "resource": "analyzer",
                "current_instances": 1,
                "recommended_instances": 2,
                "reason": "Embedding generation bottleneck",
                "priority": "medium"
            }
        ],
        "scaling_thresholds": {
            "cpu_threshold_percent": 80,
            "memory_threshold_percent": 85,
            "queue_size_threshold": 50,
            "response_time_threshold_ms": 5000
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
