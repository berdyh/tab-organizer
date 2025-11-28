"""Business logic for the Session service."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import structlog

from .models import (
    CreateSessionRequest,
    MergeSessionsRequest,
    ModelUsageHistory,
    ProcessingStats,
    RetentionPolicy,
    SessionConfiguration,
    SessionExportData,
    SessionModel,
    SessionStatus,
    ShareSessionRequest,
    SplitSessionRequest,
    SplitSessionPart,
    UpdateSessionRequest,
)
from .qdrant import SessionVectorStore
from .storage import SessionRepository

logger = structlog.get_logger("session.service")


class SessionService:
    """Encapsulates session lifecycle management."""

    def __init__(self, repository: SessionRepository, vector_store: SessionVectorStore) -> None:
        self._repository = repository
        self._vector_store = vector_store
        self._retention_policy = RetentionPolicy()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def create_session(self, request: CreateSessionRequest) -> SessionModel:
        session = SessionModel(
            name=request.name,
            description=request.description,
            owner_id=request.owner_id,
            configuration=request.configuration or SessionConfiguration(),
            tags=request.tags,
            metadata=request.metadata,
        )

        self._vector_store.ensure_collection(session.qdrant_collection_name)
        self._repository.upsert(session)

        logger.info("Created session", session_id=session.id, name=session.name)
        return session

    def list_sessions(
        self,
        status: Optional[SessionStatus] = None,
        owner_id: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> List[SessionModel]:
        sessions = list(self._repository.all())

        if status:
            sessions = [session for session in sessions if session.status == status]
        if owner_id:
            sessions = [session for session in sessions if session.owner_id == owner_id]
        if tags:
            tag_set = {tag.strip() for tag in tags if tag}
            sessions = [
                session for session in sessions if tag_set.intersection(session.tags)
            ]

        sessions.sort(key=lambda session: session.updated_at, reverse=True)
        logger.info("Listed sessions", count=len(sessions))
        return sessions

    def compare_sessions(self, session_ids: List[str]) -> Dict[str, Any]:
        if len(session_ids) < 2:
            raise ValueError("At least two session IDs are required for comparison")

        sessions = [self.require_session(session_id) for session_id in session_ids]
        comparison = {
            "sessions": [
                {
                    "id": session.id,
                    "name": session.name,
                    "created_at": session.created_at,
                    "status": session.status,
                    "processing_stats": session.processing_stats,
                    "model_usage": session.model_usage_history,
                    "tags": session.tags,
                }
                for session in sessions
            ],
            "comparison_summary": {
                "total_sessions": len(sessions),
                "status_distribution": {},
                "model_usage_overlap": {},
                "processing_differences": {},
            },
        }

        status_distribution: Dict[str, int] = {}
        llm_models: set[str] = set()
        embedding_models: set[str] = set()

        for session in sessions:
            status_distribution[session.status.value] = status_distribution.get(session.status.value, 0) + 1
            llm_models.update(session.model_usage_history.llm_models_used)
            embedding_models.update(session.model_usage_history.embedding_models_used)

        comparison["comparison_summary"]["status_distribution"] = status_distribution
        comparison["comparison_summary"]["model_usage_overlap"] = {
            "unique_llm_models": list(llm_models),
            "unique_embedding_models": list(embedding_models),
            "total_unique_models": len(llm_models) + len(embedding_models),
        }

        logger.info("Compared sessions", session_ids=session_ids)
        return comparison

    def require_session(self, session_id: str) -> SessionModel:
        session = self._repository.get(session_id)
        if not session:
            raise KeyError("Session not found")
        return session

    def get_session(self, session_id: str) -> SessionModel:
        session = self.require_session(session_id)
        logger.info("Retrieved session", session_id=session_id)
        return session

    def update_session(self, session_id: str, request: UpdateSessionRequest) -> SessionModel:
        session = self.require_session(session_id)

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

        session.updated_at = self._now()
        self._repository.upsert(session)

        logger.info("Updated session", session_id=session_id)
        return session

    def delete_session(self, session_id: str, *, permanent: bool = False) -> Dict[str, Any]:
        session = self.require_session(session_id)

        if permanent:
            self._vector_store.delete_collection(session.qdrant_collection_name)
            self._repository.delete(session_id)
            logger.info("Deleted session permanently", session_id=session_id)
            return {"message": "Session deleted successfully", "permanent": True}

        session.status = SessionStatus.DELETED
        session.updated_at = self._now()
        self._repository.upsert(session)
        logger.info("Soft deleted session", session_id=session_id)
        return {"message": "Session deleted successfully", "permanent": False}

    def archive_session(self, session_id: str) -> Dict[str, Any]:
        session = self.require_session(session_id)
        session.status = SessionStatus.ARCHIVED
        session.updated_at = self._now()
        self._repository.upsert(session)
        logger.info("Archived session", session_id=session_id)
        return {"message": "Session archived successfully"}

    def restore_session(self, session_id: str) -> Dict[str, Any]:
        session = self.require_session(session_id)
        session.status = SessionStatus.ACTIVE
        session.updated_at = self._now()
        self._repository.upsert(session)
        logger.info("Restored session", session_id=session_id)
        return {"message": "Session restored successfully"}

    def share_session(self, session_id: str, request: ShareSessionRequest) -> SessionModel:
        session = self.require_session(session_id)

        for user_id in request.user_ids:
            if user_id not in session.shared_with:
                session.shared_with.append(user_id)

        session.status = SessionStatus.SHARED
        session.updated_at = self._now()
        self._repository.upsert(session)
        logger.info("Shared session", session_id=session_id, users=request.user_ids)
        return session

    def unshare_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        session = self.require_session(session_id)
        if user_id in session.shared_with:
            session.shared_with.remove(user_id)
            if not session.shared_with:
                session.status = SessionStatus.ACTIVE
        session.updated_at = self._now()
        self._repository.upsert(session)
        logger.info("Unshared session", session_id=session_id, user_id=user_id)
        return {"message": "Session unshared successfully"}

    def get_collaborators(self, session_id: str) -> Dict[str, Any]:
        session = self.require_session(session_id)
        return {
            "session_id": session.id,
            "owner_id": session.owner_id,
            "shared_with": session.shared_with,
            "total_collaborators": 1 + len(session.shared_with) if session.owner_id else len(session.shared_with),
        }

    def trigger_incremental_clustering(self, session_id: str, new_content_ids: List[str]) -> Dict[str, Any]:
        session = self.require_session(session_id)
        session.processing_stats.content_analyzed += len(new_content_ids)
        session.processing_stats.last_processing_time = self._now()
        session.updated_at = self._now()
        self._repository.upsert(session)
        logger.info(
            "Triggered incremental clustering",
            session_id=session_id,
            new_content_count=len(new_content_ids),
        )
        return {
            "message": "Incremental clustering triggered successfully",
            "session_id": session_id,
            "new_content_count": len(new_content_ids),
            "total_content_analyzed": session.processing_stats.content_analyzed,
        }

    def cleanup_sessions(self, policy: RetentionPolicy) -> Dict[str, Any]:
        stats = {"archived_count": 0, "deleted_count": 0, "processed_count": 0}
        current_time = self._now()

        for session_id, session in list(self._repository.items()):
            stats["processed_count"] += 1

            if (
                policy.auto_archive_inactive_days
                and session.status == SessionStatus.ACTIVE
                and (current_time - session.updated_at).days >= policy.auto_archive_inactive_days
            ):
                session.status = SessionStatus.ARCHIVED
                session.updated_at = current_time
                self._repository.upsert(session)
                stats["archived_count"] += 1
                logger.info("Auto-archived session", session_id=session_id)

            if (
                policy.auto_delete_archived_days
                and session.status == SessionStatus.ARCHIVED
                and (current_time - session.updated_at).days >= policy.auto_delete_archived_days
            ):
                self._vector_store.delete_collection(session.qdrant_collection_name)
                self._repository.delete(session_id)
                stats["deleted_count"] += 1
                logger.info("Auto-deleted session", session_id=session_id)

        logger.info("Completed cleanup", stats=stats)
        return {
            "message": "Session cleanup completed successfully",
            "cleanup_stats": stats,
            "policy_applied": policy.dict(),
        }

    def get_retention_policy(self) -> Dict[str, Any]:
        return {
            "retention_policy": self._retention_policy.dict(),
            "description": "Current retention policy settings",
        }

    def update_retention_policy(self, policy: RetentionPolicy) -> Dict[str, Any]:
        self._retention_policy = policy
        logger.info("Updated retention policy", policy=policy.dict())
        return {
            "message": "Retention policy updated successfully",
            "new_policy": policy.dict(),
        }

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        session = self.require_session(session_id)
        collection_stats: Dict[str, Any]
        try:
            collection_stats = self._vector_store.get_collection_info(session.qdrant_collection_name)
        except Exception as exc:
            collection_stats = {"error": str(exc)}
            logger.warning(
                "Failed to fetch collection stats",
                session_id=session_id,
                error=str(exc),
            )

        return {
            "session_id": session_id,
            "session_stats": session.processing_stats,
            "model_usage": session.model_usage_history,
            "collection_stats": collection_stats,
            "session_age_days": (self._now() - session.created_at).days,
            "last_activity": session.updated_at,
            "collaboration_info": {
                "is_shared": bool(session.shared_with),
                "collaborator_count": len(session.shared_with),
            },
        }

    def update_session_stats(self, session_id: str, stats_update: Dict[str, Any]) -> Dict[str, Any]:
        session = self.require_session(session_id)
        stats = session.processing_stats

        stats.urls_processed += stats_update.get("urls_processed", 0)
        stats.content_analyzed += stats_update.get("content_analyzed", 0)
        stats.clusters_generated += stats_update.get("clusters_generated", 0)
        stats.embeddings_created += stats_update.get("embeddings_created", 0)
        stats.last_processing_time = self._now()

        metadata_update = stats_update.get("metadata")
        if isinstance(metadata_update, dict):
            session.metadata.update(metadata_update)
            session.metadata["last_stats_update"] = self._now().isoformat()

        session.updated_at = self._now()
        self._repository.upsert(session)
        logger.info("Updated session stats", session_id=session_id, updates=stats_update)
        return {"message": "Session statistics updated successfully"}

    def update_model_usage(self, session_id: str, model_type: str, model_name: str) -> Dict[str, Any]:
        session = self.require_session(session_id)
        usage = session.model_usage_history
        if model_type == "llm":
            if model_name not in usage.llm_models_used:
                usage.llm_models_used.append(model_name)
        elif model_type == "embedding":
            if model_name not in usage.embedding_models_used:
                usage.embedding_models_used.append(model_name)
        else:
            raise ValueError("Invalid model_type. Expected 'llm' or 'embedding'.")

        usage.model_switches += 1
        usage.last_model_switch = self._now()
        session.updated_at = self._now()
        self._repository.upsert(session)
        logger.info("Updated model usage", session_id=session_id, model_type=model_type, model_name=model_name)
        return {"message": "Session model usage updated successfully"}

    def export_session(self, session_id: str, include_data: bool = True) -> SessionExportData:
        session = self.require_session(session_id)
        collection_data: Optional[Dict[str, Any]] = None

        if include_data:
            try:
                collection_data = self._vector_store.export_points(session.qdrant_collection_name)
            except Exception as exc:
                logger.warning("Failed to export collection data", session_id=session_id, error=str(exc))
                collection_data = {"error": str(exc)}

        export_data = SessionExportData(session=session, collection_data=collection_data)
        logger.info("Exported session", session_id=session_id, include_data=include_data)
        return export_data

    def import_session(
        self,
        export_data: SessionExportData,
        new_name: Optional[str] = None,
    ) -> SessionModel:
        imported_session = export_data.session
        session = SessionModel(
            name=new_name or imported_session.name,
            description=imported_session.description,
            owner_id=imported_session.owner_id,
            configuration=imported_session.configuration,
            tags=list(imported_session.tags) + ["imported"],
            metadata={**imported_session.metadata, "imported_from": imported_session.id},
        )

        self._vector_store.ensure_collection(session.qdrant_collection_name)

        if export_data.collection_data and export_data.collection_data.get("points"):
            points = export_data.collection_data["points"]
            self._vector_store.import_points(session.qdrant_collection_name, points)

        self._repository.upsert(session)
        logger.info("Imported session", new_session_id=session.id, original_id=imported_session.id)
        return session

    def get_auth_stats(self) -> Dict[str, Any]:
        """Compatibility helper for tests mimicking gateway statistics."""
        current_time = time.time()
        active_sessions = len(list(self._repository.all()))
        return {
            "active_sessions": active_sessions,
            "total_sessions": active_sessions,
            "timestamp": current_time,
        }

    def merge_sessions(self, request: MergeSessionsRequest) -> SessionModel:
        if len(request.source_session_ids) < 2:
            raise ValueError("At least two session IDs are required to merge")

        source_sessions = [self.require_session(session_id) for session_id in request.source_session_ids]
        exports = [self.export_session(session_id, include_data=True) for session_id in request.source_session_ids]

        target_name = request.target_name or f"Merged session ({', '.join(session.id[:6] for session in source_sessions)})"
        owner_id = request.owner_id or next((session.owner_id for session in source_sessions if session.owner_id), None)

        new_session = SessionModel(
            name=target_name,
            description=request.target_description or "Merged session",
            owner_id=owner_id,
            configuration=source_sessions[0].configuration,
            tags=list(request.tags or sorted({tag for session in source_sessions for tag in session.tags})),
            metadata={**source_sessions[0].metadata, **request.metadata, "merged_from": request.source_session_ids},
        )

        self._vector_store.ensure_collection(new_session.qdrant_collection_name)
        self._repository.upsert(new_session)

        total_points = 0
        for export_data in exports:
            collection = (export_data.collection_data or {}).get("points", [])
            if collection:
                self._vector_store.import_points(
                    new_session.qdrant_collection_name,
                    collection,
                )
            total_points += len(collection)

        stats = new_session.processing_stats
        stats.urls_processed = sum(session.processing_stats.urls_processed for session in source_sessions)
        stats.content_analyzed = sum(session.processing_stats.content_analyzed for session in source_sessions)
        stats.clusters_generated = sum(session.processing_stats.clusters_generated for session in source_sessions)
        stats.embeddings_created = sum(session.processing_stats.embeddings_created for session in source_sessions)
        stats.last_processing_time = self._now()

        usage = new_session.model_usage_history
        usage.llm_models_used = sorted({model for session in source_sessions for model in session.model_usage_history.llm_models_used})
        usage.embedding_models_used = sorted({model for session in source_sessions for model in session.model_usage_history.embedding_models_used})
        usage.model_switches = sum(session.model_usage_history.model_switches for session in source_sessions)
        usage.last_model_switch = max(
            (session.model_usage_history.last_model_switch for session in source_sessions if session.model_usage_history.last_model_switch),
            default=None,
        )

        new_session.metadata["merged_point_count"] = total_points
        new_session.updated_at = self._now()
        self._repository.upsert(new_session)

        if request.archive_sources:
            for session in source_sessions:
                session.status = SessionStatus.ARCHIVED
                session.updated_at = self._now()
                self._repository.upsert(session)

        logger.info(
            "Merged sessions",
            source_sessions=request.source_session_ids,
            target_session=new_session.id,
            total_points=total_points,
        )

        return new_session

    def split_session(self, session_id: str, request: SplitSessionRequest) -> Dict[str, Any]:
        if not request.parts:
            raise ValueError("At least one split part must be provided")

        source_session = self.require_session(session_id)
        export_data = self.export_session(session_id, include_data=True)
        collection_points = (export_data.collection_data or {}).get("points", [])
        point_map = {point["id"]: point for point in collection_points}

        created_sessions: List[SessionModel] = []
        assigned_point_ids: List[str] = []

        for part in request.parts:
            unique_ids = [pid for pid in part.point_ids if pid in point_map]
            if not unique_ids:
                continue
            points_subset = [point_map[pid] for pid in unique_ids]

            new_session = SessionModel(
                name=part.name,
                description=part.description or f"Split from {session_id}",
                owner_id=source_session.owner_id,
                configuration=source_session.configuration,
                tags=list(part.tags) if part.tags else list(source_session.tags),
                metadata={**source_session.metadata, **part.metadata, "split_from": session_id},
            )

            self._vector_store.ensure_collection(new_session.qdrant_collection_name)
            self._repository.upsert(new_session)
            self._vector_store.import_points(new_session.qdrant_collection_name, points_subset)

            stats = new_session.processing_stats
            subset_len = len(points_subset)
            stats.urls_processed = subset_len
            stats.content_analyzed = subset_len
            stats.embeddings_created = subset_len
            stats.last_processing_time = self._now()

            usage = new_session.model_usage_history
            usage.llm_models_used = list(source_session.model_usage_history.llm_models_used)
            usage.embedding_models_used = list(source_session.model_usage_history.embedding_models_used)
            usage.model_switches = source_session.model_usage_history.model_switches
            usage.last_model_switch = source_session.model_usage_history.last_model_switch

            new_session.updated_at = self._now()
            self._repository.upsert(new_session)
            created_sessions.append(new_session)
            assigned_point_ids.extend(unique_ids)

        if request.remove_points and assigned_point_ids:
            self._vector_store.delete_points(source_session.qdrant_collection_name, assigned_point_ids)
            stats = source_session.processing_stats
            removed_count = len(assigned_point_ids)
            stats.urls_processed = max(0, stats.urls_processed - removed_count)
            stats.content_analyzed = max(0, stats.content_analyzed - removed_count)
            stats.embeddings_created = max(0, stats.embeddings_created - removed_count)
            stats.last_processing_time = self._now()
            source_session.updated_at = self._now()
            self._repository.upsert(source_session)

        if request.archive_original:
            source_session.status = SessionStatus.ARCHIVED
            source_session.updated_at = self._now()
            self._repository.upsert(source_session)

        logger.info(
            "Split session",
            source_session=session_id,
            new_sessions=[session.id for session in created_sessions],
            assigned_points=len(assigned_point_ids),
            removed_points=len(assigned_point_ids) if request.remove_points else 0,
        )

        return {
            "new_sessions": [session.id for session in created_sessions],
            "assigned_points": len(assigned_point_ids),
            "removed_points": len(assigned_point_ids) if request.remove_points else 0,
        }


__all__ = ["SessionService"]
