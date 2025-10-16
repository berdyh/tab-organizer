"""Embedding model management utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import structlog


class ModelManager:
    """Manage embedding model configurations and recommendations."""

    def __init__(self) -> None:
        self.logger = structlog.get_logger("model_manager")
        self.models_config = self._load_models_config()

    def _load_models_config(self) -> Dict[str, Any]:
        """Load models configuration from JSON file."""
        config_path = Path("/app/config/models.json")
        if not config_path.exists():
            config_path = Path("../../config/models.json")

        try:
            with open(config_path, "r") as file:
                return json.load(file)
        except Exception as exc:
            self.logger.warning("Could not load models config", error=str(exc))
            return {
                "embedding_models": {
                    "all-minilm": {
                        "name": "All-MiniLM",
                        "size": "90MB",
                        "dimensions": 384,
                        "quality": "good",
                        "min_ram_gb": 0.5,
                        "description": "Lightweight embedding model",
                        "model_name": "all-MiniLM-L6-v2",
                        "max_sequence_length": 512,
                        "recommended": False,
                    },
                    "nomic-embed-text": {
                        "name": "Nomic Embed Text",
                        "size": "274MB",
                        "dimensions": 768,
                        "quality": "high",
                        "min_ram_gb": 1.0,
                        "description": "Best general purpose embedding model",
                        "model_name": "nomic-ai/nomic-embed-text-v1",
                        "max_sequence_length": 8192,
                        "recommended": True,
                    },
                }
            }

    def get_available_models(self) -> Dict[str, Any]:
        """Get all available embedding models."""
        return self.models_config.get("embedding_models", {})

    def recommend_model(self, hardware_info: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend best embedding model based on hardware."""
        available_ram = hardware_info.get("available_ram_gb", 4.0)
        has_gpu = hardware_info.get("has_gpu", False)

        suitable_models = []

        for model_id, config in self.models_config.get("embedding_models", {}).items():
            if config["min_ram_gb"] <= available_ram * 0.9:
                score = self._calculate_model_score(config, has_gpu)
                suitable_models.append((model_id, config, score))

        if not suitable_models:
            return {
                "recommended_model": "all-minilm",
                "reason": "Fallback to lightweight model due to resource constraints",
                "alternatives": [],
                "performance_estimate": {"embeddings_per_sec": 10},
            }

        suitable_models.sort(key=lambda item: item[2], reverse=True)
        best_model_id, best_config, _ = suitable_models[0]

        alternatives = [model_id for model_id, *_ in suitable_models[1:3]]

        return {
            "recommended_model": best_model_id,
            "reason": f"Best fit for {available_ram:.1f}GB available RAM",
            "alternatives": alternatives,
            "performance_estimate": self._estimate_performance(best_config, has_gpu),
        }

    def _calculate_model_score(self, config: Dict[str, Any], has_gpu: bool) -> float:
        """Calculate suitability score for a model."""
        score = 0.0
        quality_scores = {"good": 2, "high": 3, "highest": 4}
        score += quality_scores.get(config["quality"], 2) * 10
        score += config["dimensions"] / 100
        if has_gpu:
            score += 5
        if config.get("recommended", False):
            score += 8
        return score

    def _estimate_performance(self, config: Dict[str, Any], has_gpu: bool) -> Dict[str, Any]:
        """Estimate embedding generation performance."""
        base_speed = 100 if has_gpu else 50
        dimension_factor = config["dimensions"] / 384
        embeddings_per_sec = max(5, base_speed / dimension_factor)
        return {
            "embeddings_per_sec": round(embeddings_per_sec, 1),
            "dimensions": config["dimensions"],
            "suitable_for_batch": True,
        }


__all__ = ["ModelManager"]
