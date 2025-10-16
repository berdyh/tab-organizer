"""
Distributed Tracer - Implements distributed tracing across microservices for request flow monitoring.
Provides trace collection, correlation, and analysis for debugging and performance optimization.
"""

import asyncio
import time
import uuid
import json
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque

from ..config import MonitoringSettings
from ..logging import get_logger

logger = get_logger("distributed_tracer")


class Span:
    """Represents a single span in a distributed trace."""
    
    def __init__(self, trace_id: str, span_id: str, parent_span_id: str = None,
                 operation_name: str = "", service_name: str = "", **tags):
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.operation_name = operation_name
        self.service_name = service_name
        self.tags = tags
        self.start_time = time.time()
        self.end_time = None
        self.duration = None
        self.status = "active"
        self.logs = []
    
    def finish(self, status: str = "completed"):
        """Finish the span."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status
    
    def add_log(self, message: str, **fields):
        """Add a log entry to the span."""
        self.logs.append({
            "timestamp": time.time(),
            "message": message,
            "fields": fields
        })
    
    def add_tag(self, key: str, value: Any):
        """Add a tag to the span."""
        self.tags[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "tags": self.tags,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status,
            "logs": self.logs
        }


class Trace:
    """Represents a complete distributed trace."""
    
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.spans: Dict[str, Span] = {}
        self.root_span_id = None
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.services: Set[str] = set()
        self.status = "active"
    
    def add_span(self, span: Span):
        """Add a span to the trace."""
        self.spans[span.span_id] = span
        self.services.add(span.service_name)
        
        # Update trace timing
        if self.start_time is None or span.start_time < self.start_time:
            self.start_time = span.start_time
        
        # Determine root span (span with no parent)
        if span.parent_span_id is None:
            self.root_span_id = span.span_id
        
        # Update trace status and timing when spans finish
        if span.status != "active":
            self._update_trace_status()
    
    def _update_trace_status(self):
        """Update trace status based on span statuses."""
        active_spans = [s for s in self.spans.values() if s.status == "active"]
        
        if not active_spans:
            # All spans finished
            self.status = "completed"
            self.end_time = max(s.end_time for s in self.spans.values() if s.end_time)
            if self.start_time and self.end_time:
                self.duration = self.end_time - self.start_time
        
        # Check for errors
        error_spans = [s for s in self.spans.values() if s.status == "error"]
        if error_spans:
            self.status = "error"
    
    def get_root_span(self) -> Optional[Span]:
        """Get the root span of the trace."""
        return self.spans.get(self.root_span_id) if self.root_span_id else None
    
    def get_span_tree(self) -> Dict[str, Any]:
        """Get hierarchical representation of spans."""
        def build_tree(span_id: str) -> Dict[str, Any]:
            span = self.spans[span_id]
            children = [build_tree(child_id) for child_id, child_span in self.spans.items()
                       if child_span.parent_span_id == span_id]
            
            return {
                "span": span.to_dict(),
                "children": children
            }
        
        if self.root_span_id:
            return build_tree(self.root_span_id)
        
        # If no clear root, return all spans
        return {
            "spans": [span.to_dict() for span in self.spans.values()],
            "children": []
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "root_span_id": self.root_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status,
            "services": list(self.services),
            "span_count": len(self.spans),
            "spans": [span.to_dict() for span in self.spans.values()],
            "span_tree": self.get_span_tree()
        }


class DistributedTracer:
    """Manages distributed tracing across microservices."""
    
    def __init__(self, settings: MonitoringSettings):
        self.settings = settings
        self.tracing_enabled = settings.distributed_tracing_enabled
        self.retention_hours = settings.trace_retention_hours
        
        self.traces: Dict[str, Trace] = {}
        self.active_spans: Dict[str, Span] = {}
        self.trace_statistics = defaultdict(int)
        self.service_dependencies = defaultdict(set)
    
    async def start_tracing(self):
        """Start the distributed tracing system."""
        if not self.tracing_enabled:
            logger.info("Distributed tracing disabled")
            return
        
        logger.info("Starting distributed tracing", 
                   retention_hours=self.retention_hours)
        
        while True:
            try:
                # Clean up old traces
                await self._cleanup_old_traces()
                
                # Update statistics
                self._update_statistics()
                
                # Wait before next cleanup
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Distributed tracing error", error=str(e))
                await asyncio.sleep(300)
    
    def create_trace(self, operation_name: str = "", **tags) -> str:
        """Create a new trace and return trace ID."""
        trace_id = str(uuid.uuid4())
        trace = Trace(trace_id)
        self.traces[trace_id] = trace
        
        # Create root span
        root_span = self.start_span(
            trace_id=trace_id,
            operation_name=operation_name or "root",
            service_name="system",
            **tags
        )
        
        logger.log_trace(trace_id, root_span.span_id, operation_name)
        return trace_id
    
    def start_span(self, trace_id: str, operation_name: str, service_name: str,
                   parent_span_id: str = None, **tags) -> Span:
        """Start a new span within a trace."""
        span_id = str(uuid.uuid4())[:8]  # Short span ID
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=service_name,
            **tags
        )
        
        # Add span to trace
        if trace_id in self.traces:
            self.traces[trace_id].add_span(span)
        else:
            # Create trace if it doesn't exist
            trace = Trace(trace_id)
            trace.add_span(span)
            self.traces[trace_id] = trace
        
        # Track active span
        self.active_spans[span_id] = span
        
        # Update service dependencies
        if parent_span_id and parent_span_id in self.active_spans:
            parent_service = self.active_spans[parent_span_id].service_name
            if parent_service != service_name:
                self.service_dependencies[parent_service].add(service_name)
        
        logger.log_trace(trace_id, span_id, f"started {operation_name}", 
                        service=service_name, parent=parent_span_id)
        
        return span
    
    def finish_span(self, span_id: str, status: str = "completed", **tags):
        """Finish a span."""
        if span_id not in self.active_spans:
            logger.warning("Attempted to finish unknown span", span_id=span_id)
            return
        
        span = self.active_spans[span_id]
        span.finish(status)
        
        # Add final tags
        for key, value in tags.items():
            span.add_tag(key, value)
        
        # Remove from active spans
        del self.active_spans[span_id]
        
        # Update trace
        if span.trace_id in self.traces:
            self.traces[span.trace_id]._update_trace_status()
        
        logger.log_trace(span.trace_id, span_id, f"finished {span.operation_name}",
                        duration=span.duration, status=status)
    
    def add_span_log(self, span_id: str, message: str, **fields):
        """Add a log entry to a span."""
        if span_id in self.active_spans:
            self.active_spans[span_id].add_log(message, **fields)
    
    def add_span_tag(self, span_id: str, key: str, value: Any):
        """Add a tag to a span."""
        if span_id in self.active_spans:
            self.active_spans[span_id].add_tag(key, value)
    
    async def _cleanup_old_traces(self):
        """Clean up old traces based on retention policy."""
        current_time = time.time()
        cutoff_time = current_time - (self.retention_hours * 3600)
        
        traces_to_remove = []
        for trace_id, trace in self.traces.items():
            if (trace.end_time and trace.end_time < cutoff_time) or \
               (trace.start_time and trace.start_time < cutoff_time and trace.status != "active"):
                traces_to_remove.append(trace_id)
        
        for trace_id in traces_to_remove:
            del self.traces[trace_id]
        
        if traces_to_remove:
            logger.info("Cleaned up old traces", count=len(traces_to_remove))
    
    def _update_statistics(self):
        """Update tracing statistics."""
        self.trace_statistics.clear()
        
        for trace in self.traces.values():
            self.trace_statistics["total_traces"] += 1
            self.trace_statistics[f"traces_{trace.status}"] += 1
            self.trace_statistics["total_spans"] += len(trace.spans)
            
            for service in trace.services:
                self.trace_statistics[f"service_{service}_traces"] += 1
    
    async def get_traces(self, limit: int = 100, status: str = None,
                        service: str = None) -> List[Dict[str, Any]]:
        """Get traces with optional filtering."""
        traces = list(self.traces.values())
        
        # Apply filters
        if status:
            traces = [t for t in traces if t.status == status]
        
        if service:
            traces = [t for t in traces if service in t.services]
        
        # Sort by start time (newest first)
        traces.sort(key=lambda t: t.start_time or 0, reverse=True)
        
        # Limit results
        traces = traces[:limit]
        
        return [trace.to_dict() for trace in traces]
    
    async def get_trace_details(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific trace."""
        trace = self.traces.get(trace_id)
        return trace.to_dict() if trace else None
    
    async def get_service_map(self) -> Dict[str, Any]:
        """Get service dependency map."""
        # Build service map from dependencies
        service_map = {
            "services": list(set().union(*[deps for deps in self.service_dependencies.values()]) | 
                           set(self.service_dependencies.keys())),
            "dependencies": [
                {"from": service, "to": list(deps)}
                for service, deps in self.service_dependencies.items()
            ]
        }
        
        # Add service statistics
        service_stats = {}
        for trace in self.traces.values():
            for service in trace.services:
                if service not in service_stats:
                    service_stats[service] = {
                        "trace_count": 0,
                        "span_count": 0,
                        "avg_duration": 0,
                        "error_count": 0
                    }
                
                service_stats[service]["trace_count"] += 1
                service_spans = [s for s in trace.spans.values() if s.service_name == service]
                service_stats[service]["span_count"] += len(service_spans)
                
                # Calculate average duration
                durations = [s.duration for s in service_spans if s.duration]
                if durations:
                    service_stats[service]["avg_duration"] = sum(durations) / len(durations)
                
                # Count errors
                error_spans = [s for s in service_spans if s.status == "error"]
                service_stats[service]["error_count"] += len(error_spans)
        
        service_map["service_statistics"] = service_stats
        return service_map
    
    async def get_trace_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics."""
        stats = dict(self.trace_statistics)
        
        # Add real-time statistics
        stats["active_spans"] = len(self.active_spans)
        stats["total_services"] = len(self.service_dependencies)
        
        # Calculate performance statistics
        completed_traces = [t for t in self.traces.values() if t.status == "completed" and t.duration]
        if completed_traces:
            durations = [t.duration for t in completed_traces]
            stats["avg_trace_duration"] = sum(durations) / len(durations)
            stats["min_trace_duration"] = min(durations)
            stats["max_trace_duration"] = max(durations)
        
        return stats
    
    async def search_traces(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search traces by operation name, service, or tags."""
        matching_traces = []
        
        query_lower = query.lower()
        
        for trace in self.traces.values():
            match = False
            
            # Search in spans
            for span in trace.spans.values():
                if (query_lower in span.operation_name.lower() or
                    query_lower in span.service_name.lower() or
                    any(query_lower in str(v).lower() for v in span.tags.values())):
                    match = True
                    break
            
            if match:
                matching_traces.append(trace)
        
        # Sort by relevance (newest first for now)
        matching_traces.sort(key=lambda t: t.start_time or 0, reverse=True)
        
        return [trace.to_dict() for trace in matching_traces[:limit]]
    
    def get_active_spans(self) -> List[Dict[str, Any]]:
        """Get currently active spans."""
        return [span.to_dict() for span in self.active_spans.values()]
    
    async def reload_config(self):
        """Reload configuration."""
        self.settings = MonitoringSettings()
        self.tracing_enabled = self.settings.distributed_tracing_enabled
        self.retention_hours = self.settings.trace_retention_hours
        logger.info("Distributed tracer configuration reloaded")
    
    async def close(self):
        """Clean up resources."""
        # Finish all active spans
        for span_id in list(self.active_spans.keys()):
            self.finish_span(span_id, status="interrupted")
        
        logger.info("Distributed tracer closed")
