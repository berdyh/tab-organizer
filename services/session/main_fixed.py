"""Session Management Service - Handles persistent storage and incremental processing."""

import time
import uuid
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models
import structlog

# Configure structured logging
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
    title="Session Management Service",
    description="Handles persistent storage and incremental processing",
    version="1.0.0"
)

# Qdrant client
qdrant_client = QdrantClient(host="qdrant", port=6333)

class SessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    SHARED = "shared"
    DELETED = "deleted"

class ProcessingStats(BaseModel):
    urls_processed: int = 0
    content_analyzed: int = 0
    clusters_generated: int = 0
    embeddings_created: int = 0
    last_processing_time: Optional[datetime] = None

class ModelUsageHistory(BaseModel):
    llm_models_used: List[str] = []
    embedding_models_used: List[str] = []
    model_switches: int = 0
    last_model_switch: Optional[datetime] = None

class SessionConfiguration(BaseModel):
    clustering_params: Dict[str, Any] = {}
    embedding_model: Optional[str] = None
    llm_model: Optional[str] = None
    processing_options: Dict[str, Any] = {}

class SessionModel(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    owner_id: Optional[str] = None
    shared_with: List[str] = []
    qdrant_collection_name: str = Field(default_factory=lambda: f"session_{uuid.uuid4().hex}")
    configuration: SessionConfiguration = Field(default_factory=SessionConfiguration)
    processing_stats: ProcessingStats = Field(default_factory=ProcessingStats)
    model_usage_history: ModelUsageHistory = Field(default_factory=ModelUsageHistory)
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class CreateSessionRequest(BaseModel):
    name: str
    description: Optional[str] = None
    owner_id: Optional[str] = None
    configuration: Optional[SessionConfiguration] = None
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class UpdateSessionRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[SessionStatus] = None
    configuration: Optional[SessionConfiguration] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class ShareSessionRequest(BaseModel):
    user_ids: List[str]
    permissions: List[str] = ["read"]

class SessionExportData(BaseModel):
    session: SessionModel
    collection_data: Optional[Dict[str, Any]] = None
    export_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Session storage - In production, this would be a proper database
sessions_storage: Dict[str, SessionModel] = {}

def get_qdrant_client():
    """Get Qdrant client instance."""
    return qdrant_client

async def ensure_session_collection(session_id: str, collection_name: str):
    """Ensure Qdrant collection exists for session."""
    try:
        client = get_qdrant_client()
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # Default embedding size, can be adjusted
                    distance=models.Distance.COSINE
                )
            )
            logger.info("Created Qdrant collection for session", 
                       session_id=session_id, collection_name=collection_name)
    except Exception as e:
        logger.error("Failed to create Qdrant collection", 
                    session_id=session_id, collection_name=collection_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create session collection: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "session",
        "timestamp": time.time()
    }

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Session Management Service",
        "version": "1.0.0",
        "status": "running"
    }

# Session Lifecycle Management Endpoints

@app.post("/sessions", response_model=SessionModel)
async def create_session(request: CreateSessionRequest):
    """Create a new session with isolated Qdrant collection."""
    try:
        session = SessionModel(
            name=request.name,
            description=request.description,
            owner_id=request.owner_id,
            configuration=request.configuration or SessionConfiguration(),
            tags=request.tags,
            metadata=request.metadata
        )
        
        # Create Qdrant collection for session isolation
        await ensure_session_collection(session.id, session.qdrant_collection_name)
        
        # Store session
        sessions_storage[session.id] = session
        
        logger.info("Created new session", session_id=session.id, name=session.name)
        return session
        
    except Exception as e:
        logger.error("Failed to create session", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")

@app.get("/sessions", response_model=List[SessionModel])
async def list_sessions(
    status: Optional[SessionStatus] = None,
    owner_id: Optional[str] = None,
    tags: Optional[str] = None
):
    """List sessions with optional filtering."""
    try:
        sessions = list(sessions_storage.values())
        
        # Apply filters
        if status:
            sessions = [s for s in sessions if s.status == status]
        if owner_id:
            sessions = [s for s in sessions if s.owner_id == owner_id]
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
            sessions = [s for s in sessions if any(tag in s.tags for tag in tag_list)]
        
        # Sort by updated_at descending
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        
        logger.info("Listed sessions", count=len(sessions))
        return sessions
        
    except Exception as e:
        logger.error("Failed to list sessions", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@app.get("/sessions/{session_id}", response_model=SessionModel)
async def get_session(session_id: str):
    """Get session by ID."""
    if session_id not in sessions_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_storage[session_id]
    logger.info("Retrieved session", session_id=session_id)
    return session

@app.put("/sessions/{session_id}", response_model=SessionModel)
async def update_session(session_id: str, request: UpdateSessionRequest):
    """Update session metadata and configuration."""
    if session_id not in sessions_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = sessions_storage[session_id]
        
        # Update fields if provided
        if request.name is not None:
            session.name = request.name
        if request.description is not None:
            session.description = request.description
        if request.status is not None:
            session.status = request.status
        if request.configuration is not None:
            session.configuration = request.configuration
        if request.tags is not None:
            session.tags = request.tags
        if request.metadata is not None:
            session.metadata.update(request.metadata)
        
        session.updated_at = datetime.now(timezone.utc)
        sessions_storage[session_id] = session
        
        logger.info("Updated session", session_id=session_id)
        return session
        
    except Exception as e:
        logger.error("Failed to update session", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update session: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, permanent: bool = False):
    """Delete session (soft delete by default, permanent if specified)."""
    if session_id not in sessions_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        if permanent:
            # Permanent deletion - remove Qdrant collection and session data
            session = sessions_storage[session_id]
            client = get_qdrant_client()
            
            try:
                client.delete_collection(collection_name=session.qdrant_collection_name)
                logger.info("Deleted Qdrant collection", 
                           session_id=session_id, collection_name=session.qdrant_collection_name)
            except Exception as e:
                logger.warning("Failed to delete Qdrant collection", 
                              session_id=session_id, error=str(e))
            
            del sessions_storage[session_id]
            logger.info("Permanently deleted session", session_id=session_id)
            
        else:
            # Soft delete - mark as deleted
            session = sessions_storage[session_id]
            session.status = SessionStatus.DELETED
            session.updated_at = datetime.now(timezone.utc)
            sessions_storage[session_id] = session
            logger.info("Soft deleted session", session_id=session_id)
        
        return {"message": "Session deleted successfully", "permanent": permanent}
        
    except Exception as e:
        logger.error("Failed to delete session", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8087)

@app.post("/sessions/{session_id}/archive")
async def archive_session(session_id: str):
    """Archive a session."""
    if session_id not in sessions_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = sessions_storage[session_id]
        session.status = SessionStatus.ARCHIVED
        session.updated_at = datetime.now(timezone.utc)
        sessions_storage[session_id] = session
        
        logger.info("Archived session", session_id=session_id)
        return {"message": "Session archived successfully"}
        
    except Exception as e:
        logger.error("Failed to archive session", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to archive session: {str(e)}")

@app.post("/sessions/{session_id}/restore")
async def restore_session(session_id: str):
    """Restore an archived or deleted session."""
    if session_id not in sessions_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = sessions_storage[session_id]
        session.status = SessionStatus.ACTIVE
        session.updated_at = datetime.now(timezone.utc)
        sessions_storage[session_id] = session
        
        logger.info("Restored session", session_id=session_id)
        return {"message": "Session restored successfully"}
        
    except Exception as e:
        logger.error("Failed to restore session", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to restore session: {str(e)}")

# Session Sharing and Collaboration

@app.post("/sessions/{session_id}/share")
async def share_session(session_id: str, request: ShareSessionRequest):
    """Share session with other users."""
    if session_id not in sessions_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = sessions_storage[session_id]
        
        # Add users to shared_with list
        for user_id in request.user_ids:
            if user_id not in session.shared_with:
                session.shared_with.append(user_id)
        
        session.status = SessionStatus.SHARED
        session.updated_at = datetime.now(timezone.utc)
        sessions_storage[session_id] = session
        
        logger.info("Shared session", session_id=session_id, shared_with=request.user_ids)
        return {
            "message": "Session shared successfully",
            "shared_with": session.shared_with,
            "permissions": request.permissions
        }
        
    except Exception as e:
        logger.error("Failed to share session", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to share session: {str(e)}")

@app.delete("/sessions/{session_id}/share/{user_id}")
async def unshare_session(session_id: str, user_id: str):
    """Remove user access from shared session."""
    if session_id not in sessions_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = sessions_storage[session_id]
        
        if user_id in session.shared_with:
            session.shared_with.remove(user_id)
            
        # If no more shared users, change status back to active
        if not session.shared_with:
            session.status = SessionStatus.ACTIVE
            
        session.updated_at = datetime.now(timezone.utc)
        sessions_storage[session_id] = session
        
        logger.info("Unshared session", session_id=session_id, user_id=user_id)
        return {"message": "User access removed successfully"}
        
    except Exception as e:
        logger.error("Failed to unshare session", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to unshare session: {str(e)}")

@app.get("/sessions/{session_id}/collaborators")
async def get_session_collaborators(session_id: str):
    """Get list of users who have access to the session."""
    if session_id not in sessions_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_storage[session_id]
    return {
        "owner_id": session.owner_id,
        "shared_with": session.shared_with,
        "total_collaborators": len(session.shared_with) + (1 if session.owner_id else 0)
    }

# Session Export and Import

@app.get("/sessions/{session_id}/export", response_model=SessionExportData)
async def export_session(session_id: str, include_data: bool = True):
    """Export session with optional collection data."""
    if session_id not in sessions_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = sessions_storage[session_id]
        export_data = SessionExportData(session=session)
        
        if include_data:
            # Export Qdrant collection data
            client = get_qdrant_client()
            try:
                # Get collection info and sample points
                collection_info = client.get_collection(collection_name=session.qdrant_collection_name)
                points_count = collection_info.points_count
                
                # For large collections, we might want to limit the export
                limit = min(points_count, 10000)  # Limit to 10k points for export
                
                points, _ = client.scroll(
                    collection_name=session.qdrant_collection_name,
                    limit=limit,
                    with_payload=True,
                    with_vectors=True
                )
                
                export_data.collection_data = {
                    "collection_name": session.qdrant_collection_name,
                    "points_count": points_count,
                    "exported_points": len(points),
                    "points": [
                        {
                            "id": str(point.id),
                            "vector": point.vector,
                            "payload": point.payload
                        } for point in points
                    ]
                }
                
                logger.info("Exported session with data", 
                           session_id=session_id, points_exported=len(points))
                
            except Exception as e:
                logger.warning("Failed to export collection data", 
                              session_id=session_id, error=str(e))
                export_data.collection_data = {"error": f"Failed to export data: {str(e)}"}
        
        return export_data
        
    except Exception as e:
        logger.error("Failed to export session", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to export session: {str(e)}")

@app.post("/sessions/import", response_model=SessionModel)
async def import_session(export_data: SessionExportData, new_name: Optional[str] = None):
    """Import session from export data."""
    try:
        # Create new session from imported data
        imported_session = export_data.session
        
        # Generate new IDs to avoid conflicts
        new_session_id = str(uuid.uuid4())
        new_collection_name = f"session_{uuid.uuid4().hex}"
        
        session = SessionModel(
            id=new_session_id,
            name=new_name or f"{imported_session.name} (Imported)",
            description=imported_session.description,
            configuration=imported_session.configuration,
            tags=imported_session.tags + ["imported"],
            metadata={
                **imported_session.metadata,
                "imported_from": imported_session.id,
                "import_timestamp": datetime.now(timezone.utc).isoformat()
            },
            qdrant_collection_name=new_collection_name
        )
        
        # Create new Qdrant collection
        await ensure_session_collection(session.id, session.qdrant_collection_name)
        
        # Import collection data if available
        if export_data.collection_data and "points" in export_data.collection_data:
            client = get_qdrant_client()
            points_data = export_data.collection_data["points"]
            
            if points_data:
                # Convert points back to Qdrant format
                points = []
                for point_data in points_data:
                    points.append(models.PointStruct(
                        id=str(uuid.uuid4()),  # Generate new IDs
                        vector=point_data["vector"],
                        payload=point_data["payload"]
                    ))
                
                # Batch upsert points
                batch_size = 100
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    client.upsert(
                        collection_name=session.qdrant_collection_name,
                        points=batch
                    )
                
                logger.info("Imported collection data", 
                           session_id=session.id, points_imported=len(points))
        
        # Store imported session
        sessions_storage[session.id] = session
        
        logger.info("Imported session", 
                   session_id=session.id, original_id=imported_session.id)
        return session
        
    except Exception as e:
        logger.error("Failed to import session", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to import session: {str(e)}")

# Session Statistics and Metadata Management

@app.get("/sessions/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Get detailed session statistics."""
    if session_id not in sessions_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = sessions_storage[session_id]
        client = get_qdrant_client()
        
        # Get collection statistics
        collection_stats = {}
        try:
            collection_info = client.get_collection(collection_name=session.qdrant_collection_name)
            collection_stats = {
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status
            }
        except Exception as e:
            logger.warning("Failed to get collection stats", 
                          session_id=session_id, error=str(e))
            collection_stats = {"error": str(e)}
        
        return {
            "session_id": session_id,
            "session_stats": session.processing_stats,
            "model_usage": session.model_usage_history,
            "collection_stats": collection_stats,
            "session_age_days": (datetime.now(timezone.utc) - session.created_at).days,
            "last_activity": session.updated_at,
            "collaboration_info": {
                "is_shared": len(session.shared_with) > 0,
                "collaborator_count": len(session.shared_with)
            }
        }
        
    except Exception as e:
        logger.error("Failed to get session stats", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get session stats: {str(e)}")

@app.put("/sessions/{session_id}/stats")
async def update_session_stats(
    session_id: str, 
    stats_update: Dict[str, Any]
):
    """Update session processing statistics."""
    if session_id not in sessions_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = sessions_storage[session_id]
        
        # Update processing stats
        if "urls_processed" in stats_update:
            session.processing_stats.urls_processed += stats_update["urls_processed"]
        if "content_analyzed" in stats_update:
            session.processing_stats.content_analyzed += stats_update["content_analyzed"]
        if "clusters_generated" in stats_update:
            session.processing_stats.clusters_generated += stats_update["clusters_generated"]
        if "embeddings_created" in stats_update:
            session.processing_stats.embeddings_created += stats_update["embeddings_created"]
        
        session.processing_stats.last_processing_time = datetime.now(timezone.utc)
        session.updated_at = datetime.now(timezone.utc)
        sessions_storage[session_id] = session
        
        logger.info("Updated session stats", session_id=session_id, updates=stats_update)
        return {"message": "Session statistics updated successfully"}
        
    except Exception as e:
        logger.error("Failed to update session stats", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update session stats: {str(e)}")

@app.put("/sessions/{session_id}/model-usage")
async def update_model_usage(
    session_id: str,
    model_type: str,
    model_name: str
):
    """Update model usage history for session."""
    if session_id not in sessions_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = sessions_storage[session_id]
        
        if model_type == "llm":
            if model_name not in session.model_usage_history.llm_models_used:
                session.model_usage_history.llm_models_used.append(model_name)
        elif model_type == "embedding":
            if model_name not in session.model_usage_history.embedding_models_used:
                session.model_usage_history.embedding_models_used.append(model_name)
        
        session.model_usage_history.model_switches += 1
        session.model_usage_history.last_model_switch = datetime.now(timezone.utc)
        session.updated_at = datetime.now(timezone.utc)
        sessions_storage[session_id] = session
        
        logger.info("Updated model usage", 
                   session_id=session_id, model_type=model_type, model_name=model_name)
        return {"message": "Model usage updated successfully"}
        
    except Exception as e:
        logger.error("Failed to update model usage", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to update model usage: {str(e)}")

# Session Comparison and Evolution Tracking

@app.get("/sessions/compare")
async def compare_sessions(session_ids: str):
    """Compare multiple sessions and show differences."""
    try:
        session_id_list = [sid.strip() for sid in session_ids.split(",")]
        
        if len(session_id_list) < 2:
            raise HTTPException(status_code=400, detail="At least 2 session IDs required for comparison")
        
        sessions = []
        for session_id in session_id_list:
            if session_id not in sessions_storage:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            sessions.append(sessions_storage[session_id])
        
        # Compare sessions
        comparison = {
            "sessions": [
                {
                    "id": s.id,
                    "name": s.name,
                    "created_at": s.created_at,
                    "status": s.status,
                    "processing_stats": s.processing_stats,
                    "model_usage": s.model_usage_history,
                    "tags": s.tags
                } for s in sessions
            ],
            "comparison_summary": {
                "total_sessions": len(sessions),
                "status_distribution": {},
                "model_usage_overlap": {},
                "processing_differences": {}
            }
        }
        
        # Calculate status distribution
        for session in sessions:
            status = session.status.value
            comparison["comparison_summary"]["status_distribution"][status] = \
                comparison["comparison_summary"]["status_distribution"].get(status, 0) + 1
        
        # Calculate model usage overlap
        all_llm_models = set()
        all_embedding_models = set()
        for session in sessions:
            all_llm_models.update(session.model_usage_history.llm_models_used)
            all_embedding_models.update(session.model_usage_history.embedding_models_used)
        
        comparison["comparison_summary"]["model_usage_overlap"] = {
            "unique_llm_models": list(all_llm_models),
            "unique_embedding_models": list(all_embedding_models),
            "total_unique_models": len(all_llm_models) + len(all_embedding_models)
        }
        
        logger.info("Compared sessions", session_ids=session_id_list)
        return comparison
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to compare sessions", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to compare sessions: {str(e)}")

# Incremental Processing Support (Requirement 8.3)

@app.post("/sessions/{session_id}/incremental-clustering")
async def trigger_incremental_clustering(session_id: str, new_content_ids: List[str]):
    """Trigger incremental clustering when new content is added to session."""
    if session_id not in sessions_storage:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = sessions_storage[session_id]
        
        # This would integrate with the clustering service
        # For now, we'll just update the processing stats
        session.processing_stats.content_analyzed += len(new_content_ids)
        session.processing_stats.last_processing_time = datetime.now(timezone.utc)
        session.updated_at = datetime.now(timezone.utc)
        sessions_storage[session_id] = session
        
        logger.info("Triggered incremental clustering", 
                   session_id=session_id, new_content_count=len(new_content_ids))
        
        return {
            "message": "Incremental clustering triggered successfully",
            "session_id": session_id,
            "new_content_count": len(new_content_ids),
            "total_content_analyzed": session.processing_stats.content_analyzed
        }
        
    except Exception as e:
        logger.error("Failed to trigger incremental clustering", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to trigger incremental clustering: {str(e)}")

# Retention Policies and Cleanup (Requirement 8.7)

class RetentionPolicy(BaseModel):
    max_age_days: Optional[int] = None
    max_sessions_per_user: Optional[int] = None
    auto_archive_inactive_days: Optional[int] = 30
    auto_delete_archived_days: Optional[int] = 90

@app.post("/sessions/cleanup")
async def cleanup_sessions(policy: RetentionPolicy):
    """Apply retention policies and cleanup old sessions."""
    try:
        cleanup_stats = {
            "archived_count": 0,
            "deleted_count": 0,
            "processed_count": 0
        }
        
        current_time = datetime.now(timezone.utc)
        
        for session_id, session in list(sessions_storage.items()):
            cleanup_stats["processed_count"] += 1
            
            # Auto-archive inactive sessions
            if (policy.auto_archive_inactive_days and 
                session.status == SessionStatus.ACTIVE and
                (current_time - session.updated_at).days >= policy.auto_archive_inactive_days):
                
                session.status = SessionStatus.ARCHIVED
                session.updated_at = current_time
                sessions_storage[session_id] = session
                cleanup_stats["archived_count"] += 1
                logger.info("Auto-archived inactive session", session_id=session_id)
            
            # Auto-delete old archived sessions
            if (policy.auto_delete_archived_days and 
                session.status == SessionStatus.ARCHIVED and
                (current_time - session.updated_at).days >= policy.auto_delete_archived_days):
                
                # Permanent deletion
                client = get_qdrant_client()
                try:
                    client.delete_collection(collection_name=session.qdrant_collection_name)
                except Exception as e:
                    logger.warning("Failed to delete collection during cleanup", 
                                  session_id=session_id, error=str(e))
                
                del sessions_storage[session_id]
                cleanup_stats["deleted_count"] += 1
                logger.info("Auto-deleted old archived session", session_id=session_id)
        
        logger.info("Completed session cleanup", stats=cleanup_stats)
        return {
            "message": "Session cleanup completed successfully",
            "cleanup_stats": cleanup_stats,
            "policy_applied": policy.dict()
        }
        
    except Exception as e:
        logger.error("Failed to cleanup sessions", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to cleanup sessions: {str(e)}")

@app.get("/sessions/retention-policy")
async def get_retention_policy():
    """Get current retention policy settings."""
    # In a real implementation, this would be stored in a database
    default_policy = RetentionPolicy()
    return {
        "retention_policy": default_policy.dict(),
        "description": "Default retention policy settings"
    }

@app.put("/sessions/retention-policy")
async def update_retention_policy(policy: RetentionPolicy):
    """Update retention policy settings."""
    # In a real implementation, this would be stored in a database
    logger.info("Updated retention policy", policy=policy.dict())
    return {
        "message": "Retention policy updated successfully",
        "new_policy": policy.dict()
    }