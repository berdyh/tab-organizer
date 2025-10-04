"""Unit tests for UMAP dimensionality reduction with model-aware optimization."""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import time

from main import (
    ModelAwareUMAPConfig,
    DimensionalityReducer,
    VisualizationGenerator,
    UMAPRequest,
    VisualizationRequest
)


class TestModelAwareUMAPConfig:
    """Test model-aware UMAP configuration."""
    
    def test_get_config_nomic_embed(self):
        """Test configuration for nomic-embed-text model."""
        config = ModelAwareUMAPConfig.get_config("nomic-embed-text", 1000, 2048)
        
        assert config["n_neighbors"] == 15
        assert config["min_dist"] == 0.1
        assert config["metric"] == "cosine"
        assert config["batch_size"] > 0
        assert "expected_memory_mb" in config
    
    def test_get_config_all_minilm(self):
        """Test configuration for all-minilm model."""
        config = ModelAwareUMAPConfig.get_config("all-minilm", 1000, 2048)
        
        assert config["n_neighbors"] == 12
        assert config["min_dist"] == 0.15
        assert config["metric"] == "cosine"
        assert config["batch_size"] > 0
    
    def test_get_config_mxbai_large(self):
        """Test configuration for mxbai-embed-large model."""
        config = ModelAwareUMAPConfig.get_config("mxbai-embed-large", 1000, 2048)
        
        assert config["n_neighbors"] == 20
        assert config["min_dist"] == 0.05
        assert config["metric"] == "cosine"
        assert config["batch_size"] > 0
    
    def test_get_config_unknown_model_fallback(self):
        """Test fallback to default config for unknown model."""
        config = ModelAwareUMAPConfig.get_config("unknown-model", 1000, 2048)
        
        # Should fallback to nomic-embed-text config
        assert config["n_neighbors"] == 15
        assert config["min_dist"] == 0.1
        assert config["metric"] == "cosine"
    
    def test_config_small_dataset_adjustment(self):
        """Test parameter adjustment for small datasets."""
        config = ModelAwareUMAPConfig.get_config("nomic-embed-text", 50, 2048)
        
        # n_neighbors should be adjusted for small dataset
        assert config["n_neighbors"] <= 15
        assert config["n_neighbors"] >= 5
    
    def test_config_large_dataset_adjustment(self):
        """Test parameter adjustment for large datasets."""
        config = ModelAwareUMAPConfig.get_config("nomic-embed-text", 15000, 2048)
        
        # n_neighbors should be increased for large dataset
        assert config["n_neighbors"] > 15
        assert config["n_neighbors"] <= 30
    
    def test_config_memory_limit_adjustment(self):
        """Test batch size adjustment based on memory limit."""
        # Low memory limit
        config_low = ModelAwareUMAPConfig.get_config("nomic-embed-text", 1000, 512)
        
        # High memory limit
        config_high = ModelAwareUMAPConfig.get_config("nomic-embed-text", 1000, 4096)
        
        # Low memory should have smaller batch size
        assert config_low["batch_size"] <= config_high["batch_size"]
        assert config_low["batch_size"] >= 100  # Minimum batch size


class TestDimensionalityReducer:
    """Test UMAP dimensionality reduction functionality."""
    
    @pytest.fixture
    def reducer(self):
        """Create DimensionalityReducer instance."""
        return DimensionalityReducer()
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        np.random.seed(42)
        return np.random.rand(100, 768)  # 100 samples, 768 dimensions
    
    @pytest.mark.asyncio
    async def test_reduce_embeddings_basic(self, reducer, sample_embeddings):
        """Test basic embedding reduction."""
        reduced, metrics = await reducer.reduce_embeddings(
            embeddings=sample_embeddings,
            embedding_model="nomic-embed-text",
            n_components=2
        )
        
        assert reduced.shape == (100, 2)
        assert "processing_time_seconds" in metrics
        assert "memory_used_mb" in metrics
        assert "original_dimensions" in metrics
        assert "reduced_dimensions" in metrics
        assert metrics["original_dimensions"] == 768
        assert metrics["reduced_dimensions"] == 2
        assert metrics["n_samples"] == 100
    
    @pytest.mark.asyncio
    async def test_reduce_embeddings_3d(self, reducer, sample_embeddings):
        """Test 3D embedding reduction."""
        reduced, metrics = await reducer.reduce_embeddings(
            embeddings=sample_embeddings,
            embedding_model="nomic-embed-text",
            n_components=3
        )
        
        assert reduced.shape == (100, 3)
        assert metrics["reduced_dimensions"] == 3
    
    @pytest.mark.asyncio
    async def test_reduce_embeddings_custom_params(self, reducer, sample_embeddings):
        """Test embedding reduction with custom parameters."""
        custom_params = {
            "n_neighbors": 10,
            "min_dist": 0.2,
            "metric": "euclidean"
        }
        
        reduced, metrics = await reducer.reduce_embeddings(
            embeddings=sample_embeddings,
            embedding_model="nomic-embed-text",
            n_components=2,
            custom_params=custom_params
        )
        
        assert reduced.shape == (100, 2)
        assert metrics["umap_parameters"]["n_neighbors"] == 10
        assert metrics["umap_parameters"]["min_dist"] == 0.2
        assert metrics["umap_parameters"]["metric"] == "euclidean"
    
    @pytest.mark.asyncio
    async def test_reduce_embeddings_different_models(self, reducer, sample_embeddings):
        """Test embedding reduction with different embedding models."""
        models = ["nomic-embed-text", "all-minilm", "mxbai-embed-large"]
        
        for model in models:
            reduced, metrics = await reducer.reduce_embeddings(
                embeddings=sample_embeddings,
                embedding_model=model,
                n_components=2
            )
            
            assert reduced.shape == (100, 2)
            assert metrics["embedding_model"] == model
            assert "umap_parameters" in metrics
    
    @pytest.mark.asyncio
    @patch('main.executor')
    async def test_batch_processing_large_dataset(self, mock_executor, reducer):
        """Test batch processing for large datasets."""
        # Create large dataset
        np.random.seed(42)
        large_embeddings = np.random.rand(2500, 384)  # Large dataset
        
        # Mock the executor to track batch processing
        mock_executor.submit = Mock()
        
        with patch.object(reducer, '_batch_reduce') as mock_batch_reduce:
            mock_batch_reduce.return_value = np.random.rand(2500, 2)
            
            reduced, metrics = await reducer.reduce_embeddings(
                embeddings=large_embeddings,
                embedding_model="all-minilm",
                n_components=2,
                batch_size=1000
            )
            
            # Should use batch processing for large dataset
            mock_batch_reduce.assert_called_once()
            assert reduced.shape == (2500, 2)
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, reducer, sample_embeddings):
        """Test performance metrics tracking and caching."""
        reduced, metrics = await reducer.reduce_embeddings(
            embeddings=sample_embeddings,
            embedding_model="nomic-embed-text",
            n_components=2
        )
        
        # Check metrics content
        assert metrics["processing_time_seconds"] > 0
        assert metrics["memory_used_mb"] > 0
        assert metrics["samples_per_second"] > 0
        assert "umap_parameters" in metrics
        
        # Check model caching
        cached_models = reducer.list_cached_models()
        assert len(cached_models) > 0
        
        # Check performance metrics retrieval
        model_key = cached_models[0]
        cached_metrics = reducer.get_performance_metrics(model_key)
        assert cached_metrics is not None
        assert cached_metrics["embedding_model"] == "nomic-embed-text"
    
    @pytest.mark.asyncio
    async def test_memory_optimization(self, reducer):
        """Test memory optimization for different dataset sizes."""
        # Small dataset
        small_embeddings = np.random.rand(50, 384)
        reduced_small, metrics_small = await reducer.reduce_embeddings(
            embeddings=small_embeddings,
            embedding_model="all-minilm",
            n_components=2,
            memory_limit_mb=512
        )
        
        # Large dataset with memory limit
        large_embeddings = np.random.rand(1000, 384)
        reduced_large, metrics_large = await reducer.reduce_embeddings(
            embeddings=large_embeddings,
            embedding_model="all-minilm", 
            n_components=2,
            memory_limit_mb=512
        )
        
        assert reduced_small.shape == (50, 2)
        assert reduced_large.shape == (1000, 2)
        
        # Memory-limited processing should use smaller batch size
        assert metrics_large["batch_size_used"] <= metrics_small["batch_size_used"] * 2


class TestVisualizationGenerator:
    """Test visualization generation functionality."""
    
    @pytest.fixture
    def generator(self):
        """Create VisualizationGenerator instance."""
        return VisualizationGenerator()
    
    @pytest.fixture
    def sample_reduced_embeddings(self):
        """Create sample reduced embeddings."""
        np.random.seed(42)
        return np.random.rand(50, 2)
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return [
            {
                "cluster_id": i % 5,
                "embedding_model": "nomic-embed-text",
                "quality_score": np.random.rand(),
                "title": f"Document {i}",
                "url": f"https://example.com/doc{i}"
            }
            for i in range(50)
        ]
    
    @pytest.mark.asyncio
    async def test_create_2d_cluster_plot(self, generator, sample_reduced_embeddings, sample_metadata):
        """Test 2D cluster plot creation."""
        plot_result = await generator.create_cluster_plot(
            reduced_embeddings=sample_reduced_embeddings,
            metadata=sample_metadata,
            plot_type="2d",
            color_by="cluster"
        )
        
        assert "plot_html" in plot_result
        assert plot_result["plot_type"] == "2d"
        assert plot_result["n_points"] == 50
        assert plot_result["color_by"] == "cluster"
        assert "<html>" in plot_result["plot_html"]
    
    @pytest.mark.asyncio
    async def test_create_3d_cluster_plot(self, generator, sample_metadata):
        """Test 3D cluster plot creation."""
        # Create 3D embeddings
        np.random.seed(42)
        embeddings_3d = np.random.rand(50, 3)
        
        plot_result = await generator.create_cluster_plot(
            reduced_embeddings=embeddings_3d,
            metadata=sample_metadata,
            plot_type="3d",
            color_by="cluster"
        )
        
        assert "plot_html" in plot_result
        assert plot_result["plot_type"] == "3d"
        assert plot_result["n_points"] == 50
    
    @pytest.mark.asyncio
    async def test_plot_color_by_options(self, generator, sample_reduced_embeddings, sample_metadata):
        """Test different color-by options."""
        color_options = ["cluster", "model", "quality"]
        
        for color_by in color_options:
            plot_result = await generator.create_cluster_plot(
                reduced_embeddings=sample_reduced_embeddings,
                metadata=sample_metadata,
                plot_type="2d",
                color_by=color_by
            )
            
            assert plot_result["color_by"] == color_by
            assert "plot_html" in plot_result
    
    @pytest.mark.asyncio
    async def test_plot_with_performance_metrics(self, generator, sample_reduced_embeddings, sample_metadata):
        """Test plot creation with performance metrics."""
        performance_metrics = {
            "processing_time_seconds": 1.5,
            "samples_per_second": 100.0,
            "memory_used_mb": 50.0,
            "original_dimensions": 768,
            "reduced_dimensions": 2,
            "embedding_model": "nomic-embed-text"
        }
        
        plot_result = await generator.create_cluster_plot(
            reduced_embeddings=sample_reduced_embeddings,
            metadata=sample_metadata,
            plot_type="2d",
            color_by="cluster",
            include_metrics=True,
            performance_metrics=performance_metrics
        )
        
        assert plot_result["performance_metrics"] == performance_metrics
        # Check that metrics are included in the plot
        assert "Processing Time" in plot_result["plot_html"]
        assert "Memory Used" in plot_result["plot_html"]
    
    @pytest.mark.asyncio
    async def test_plot_3d_insufficient_dimensions_error(self, generator, sample_reduced_embeddings, sample_metadata):
        """Test error handling for 3D plot with insufficient dimensions."""
        with pytest.raises(ValueError, match="3D plot requires at least 3 dimensions"):
            await generator.create_cluster_plot(
                reduced_embeddings=sample_reduced_embeddings,  # Only 2D
                metadata=sample_metadata,
                plot_type="3d",
                color_by="cluster"
            )
    
    @pytest.mark.asyncio
    async def test_hover_text_generation(self, generator, sample_reduced_embeddings, sample_metadata):
        """Test hover text generation for interactive plots."""
        plot_result = await generator.create_cluster_plot(
            reduced_embeddings=sample_reduced_embeddings,
            metadata=sample_metadata,
            plot_type="2d",
            color_by="cluster"
        )
        
        # Check that hover information is included
        plot_html = plot_result["plot_html"]
        assert "Cluster:" in plot_html
        assert "Model:" in plot_html
        assert "Quality:" in plot_html


class TestIntegration:
    """Integration tests for UMAP dimensionality reduction."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_umap_workflow(self):
        """Test complete UMAP workflow from embeddings to visualization."""
        # Create sample data
        np.random.seed(42)
        embeddings = np.random.rand(100, 768)
        
        # Initialize components
        reducer = DimensionalityReducer()
        generator = VisualizationGenerator()
        
        # Step 1: Reduce dimensions
        reduced, metrics = await reducer.reduce_embeddings(
            embeddings=embeddings,
            embedding_model="nomic-embed-text",
            n_components=2
        )
        
        assert reduced.shape == (100, 2)
        assert "processing_time_seconds" in metrics
        
        # Step 2: Create metadata
        metadata = [
            {
                "cluster_id": i % 5,
                "embedding_model": "nomic-embed-text",
                "quality_score": np.random.rand(),
                "title": f"Document {i}",
                "url": f"https://example.com/doc{i}"
            }
            for i in range(100)
        ]
        
        # Step 3: Generate visualization
        plot_result = await generator.create_cluster_plot(
            reduced_embeddings=reduced,
            metadata=metadata,
            plot_type="2d",
            color_by="cluster",
            include_metrics=True,
            performance_metrics=metrics
        )
        
        assert "plot_html" in plot_result
        assert plot_result["n_points"] == 100
        assert plot_result["performance_metrics"] == metrics
    
    @pytest.mark.asyncio
    async def test_multi_model_comparison(self):
        """Test UMAP reduction across different embedding models."""
        np.random.seed(42)
        
        # Different embedding dimensions for different models
        test_cases = [
            ("nomic-embed-text", np.random.rand(100, 768)),
            ("all-minilm", np.random.rand(100, 384)),
            ("mxbai-embed-large", np.random.rand(100, 1024))
        ]
        
        reducer = DimensionalityReducer()
        results = {}
        
        for model_name, embeddings in test_cases:
            reduced, metrics = await reducer.reduce_embeddings(
                embeddings=embeddings,
                embedding_model=model_name,
                n_components=2
            )
            
            results[model_name] = {
                "reduced_shape": reduced.shape,
                "processing_time": metrics["processing_time_seconds"],
                "original_dims": metrics["original_dimensions"],
                "samples_per_second": metrics["samples_per_second"]
            }
        
        # Verify all models processed correctly
        for model_name, result in results.items():
            assert result["reduced_shape"] == (100, 2)
            assert result["processing_time"] > 0
            assert result["samples_per_second"] > 0
        
        # Different models should have different original dimensions
        assert results["nomic-embed-text"]["original_dims"] == 768
        assert results["all-minilm"]["original_dims"] == 384
        assert results["mxbai-embed-large"]["original_dims"] == 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v"])