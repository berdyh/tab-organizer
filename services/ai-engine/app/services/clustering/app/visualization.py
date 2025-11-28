"""
Cluster visualisation helpers using Plotly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go

from .logging import logger


class VisualizationGenerator:
    """Generate interactive visualisations for dimensionality-reduced embeddings."""

    def __init__(self) -> None:
        self.plot_cache: Dict[str, Dict[str, Any]] = {}

    async def create_cluster_plot(
        self,
        reduced_embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        plot_type: str = "2d",
        color_by: str = "cluster",
        include_metrics: bool = True,
        performance_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create interactive cluster visualisation with model performance metrics.

        Returns:
            Dictionary containing plot HTML and metadata.
        """
        n_samples, n_dims = reduced_embeddings.shape

        if plot_type == "3d" and n_dims < 3:
            raise ValueError("3D plot requires at least 3 dimensions in reduced embeddings")

        # Prepare data for plotting
        plot_data: Dict[str, Any] = {
            "x": reduced_embeddings[:, 0],
            "y": reduced_embeddings[:, 1],
        }

        if plot_type == "3d" and n_dims >= 3:
            plot_data["z"] = reduced_embeddings[:, 2]

        # Add colour information
        if color_by == "cluster":
            plot_data["color"] = [item.get("cluster_id", -1) for item in metadata]
            color_title = "Cluster ID"
            color_discrete = False
        elif color_by == "model":
            # Convert categorical data to numeric for plotly
            unique_models = list({item.get("embedding_model", "unknown") for item in metadata})
            model_to_num = {model: i for i, model in enumerate(unique_models)}
            plot_data["color"] = [
                model_to_num.get(item.get("embedding_model", "unknown"), 0) for item in metadata
            ]
            color_title = "Embedding Model"
            color_discrete = True
        elif color_by == "quality":
            plot_data["color"] = [item.get("quality_score", 0.0) for item in metadata]
            color_title = "Quality Score"
            color_discrete = False
        else:
            plot_data["color"] = [0] * n_samples  # Use numeric values to avoid plotly colour inference
            color_title = "Data Points"
            color_discrete = False

        # Add hover information
        hover_text = []
        for i, item in enumerate(metadata):
            hover_info = [
                f"Point {i}",
                f"Cluster: {item.get('cluster_id', 'N/A')}",
                f"Model: {item.get('embedding_model', 'N/A')}",
                f"Quality: {item.get('quality_score', 'N/A'):.3f}",
            ]
            if "title" in item:
                hover_info.append(f"Title: {item['title'][:50]}...")
            hover_text.append("<br>".join(hover_info))

        plot_data["hover_text"] = hover_text

        # Create plot
        if plot_type == "3d":
            fig = go.Figure(
                data=go.Scatter3d(
                    x=plot_data["x"],
                    y=plot_data["y"],
                    z=plot_data["z"],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=plot_data["color"],
                        colorscale="Viridis",
                        colorbar=dict(title=color_title),
                    ),
                    text=plot_data["hover_text"],
                    hoverinfo="text",
                )
            )
        else:
            fig = go.Figure(
                data=go.Scattergl(
                    x=plot_data["x"],
                    y=plot_data["y"],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=plot_data["color"],
                        colorscale="Viridis",
                        colorbar=dict(title=color_title),
                        showscale=not color_discrete,
                    ),
                    text=plot_data["hover_text"],
                    hoverinfo="text",
                )
            )

        fig.update_layout(
            title="Clustering Visualisation",
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            template="plotly_dark",
            height=700 if plot_type == "3d" else 600,
        )

        if plot_type == "3d":
            fig.update_layout(scene=dict(zaxis_title="UMAP Dimension 3"))

        # Include performance metrics if requested
        metrics_summary: Optional[Dict[str, Any]] = None
        if include_metrics and performance_metrics:
            metrics_summary = {
                "processing_time_seconds": performance_metrics.get("processing_time_seconds"),
                "memory_used_mb": performance_metrics.get("memory_used_mb"),
                "samples_per_second": performance_metrics.get("samples_per_second"),
                "batch_size_used": performance_metrics.get("batch_size_used"),
                "umap_parameters": performance_metrics.get("umap_parameters"),
            }

        metrics_html = ""
        if include_metrics and performance_metrics:
            metrics_lines = [
                f"Processing Time: {performance_metrics.get('processing_time_seconds', 0):.2f}s",
                f"Samples / Second: {performance_metrics.get('samples_per_second', 0):.2f}",
                f"Memory Used: {performance_metrics.get('memory_used_mb', 0):.2f} MB",
            ]
            metrics_html = "<div class='umap-metrics'>" + "<br>".join(metrics_lines) + "</div>"

        plot_html = fig.to_html(full_html=True)
        if metrics_html:
            plot_html = metrics_html + plot_html

        plot_result = {
            "plot_html": plot_html,
            "plot_type": plot_type,
            "n_points": n_samples,
            "color_by": color_by,
            "performance_metrics": performance_metrics if include_metrics else None,
            "plot_config": {"scrollZoom": True, "displayModeBar": True},
            "metadata": {
                "plot_type": plot_type,
                "n_points": n_samples,
                "color_by": color_by,
                "performance_metrics_summary": metrics_summary,
            },
        }

        cache_key = f"{plot_type}_{color_by}_{n_samples}"
        self.plot_cache[cache_key] = plot_result

        logger.debug(
            "Generated clustering visualisation",
            plot_type=plot_type,
            color_by=color_by,
            n_points=n_samples,
        )

        return plot_result


__all__ = ["VisualizationGenerator"]
