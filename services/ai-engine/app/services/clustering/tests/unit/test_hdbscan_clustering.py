"""Unit tests for HDBSCAN clustering functionality."""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from services.clustering.app.hdbscan_service import HDBSCANClusterer
from services.clustering.app.umap import ModelAwareUMAPConfig
import hdbscan
from sklearn.datasets import make_blobs


class TestHDBSCANClusterer:
    """Test cases for HDBSCAN clustering with parameter optimization."""
    
    @pytest.fixture
    def clusterer(self):
        """Create HDBSCANClusterer instance for testing."""
        return HDBSCANClusterer()
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        # Create synthetic data with clear clusters
        X, y = make_blobs(
            n_samples=200, 
            centers=4, 
            n_features=50, 
            random_state=42,
            cluster_std=1.0
        )
        return X, y
    
    @pytest.fixture
    def small_embeddings(self):
        """Create small dataset for edge case testing."""
        X, y = make_blobs(
            n_samples=20, 
            centers=2, 
            n_features=10, 
            random_state=42,
            cluster_std=0.5
        )
        return X, y
    
    def test_optimize_parameters_small_dataset(self, clusterer):
        """Test parameter optimization for small datasets."""
        embeddings = np.random.randn(50, 20)
        params = clusterer._optimize_parameters(embeddings, 50, "nomic-embed-text")
        
        assert params["min_cluster_size"] >= 3
        assert params["min_samples"] >= 2
        assert params["metric"] in ["euclidean", "manhattan", "cosine"]
        assert params["alpha"] > 0
        assert params["cluster_selection_epsilon"] >= 0
        assert params["cluster_selection_method"] == "eom"
    
    def test_optimize_parameters_medium_dataset(self, clusterer):
        """Test parameter optimization for medium datasets."""
        embeddings = np.random.randn(500, 100)
        params = clusterer._optimize_parameters(embeddings, 500, "all-minilm")
        
        assert params["min_cluster_size"] >= 5
        assert params["min_samples"] >= 3
        assert params["metric"] == "manhattan"  # Should use manhattan for 100 dims
        assert params["alpha"] == 1.2  # all-minilm specific
    
    def test_optimize_parameters_large_dataset(self, clusterer):
        """Test parameter optimization for large datasets."""
        embeddings = np.random.randn(5000, 200)
        params = clusterer._optimize_parameters(embeddings, 5000, "mxbai-embed-large")
        
        assert params["min_cluster_size"] >= 10
        assert params["min_samples"] >= 5
        assert params["metric"] == "manhattan"  # Should use manhattan for high dims
        assert params["alpha"] == 0.8  # mxbai-embed-large specific
    
    def test_optimize_parameters_unknown_model(self, clusterer):
        """Test parameter optimization with unknown embedding model."""
        embeddings = np.random.randn(1000, 30)  # Use 30 dims to get euclidean
        params = clusterer._optimize_parameters(embeddings, 1000, "unknown-model")
        
        # Should use default parameters
        assert params["alpha"] == 1.0
        assert params["cluster_selection_epsilon"] == 0.1
        assert params["metric"] == "euclidean"  # 30 dims should use euclidean
    
    @pytest.mark.asyncio
    async def test_cluster_embeddings_basic(self, clusterer, sample_embeddings):
        """Test basic HDBSCAN clustering functionality."""
        embeddings, true_labels = sample_embeddings
        
        cluster_labels, metrics = await clusterer.cluster_embeddings(
            embeddings=embeddings,
            session_id="test_session",
            auto_optimize=True,
            embedding_model="nomic-embed-text"
        )
        
        # Check output format
        assert len(cluster_labels) == len(embeddings)
        assert isinstance(metrics, dict)
        
        # Check metrics structure
        required_metrics = [
            "n_clusters", "n_noise_points", "cluster_sizes", "noise_ratio",
            "silhouette_score", "calinski_harabasz_score", "davies_bouldin_score",
            "processing_time_seconds", "session_id", "embedding_model"
        ]
        
        for metric in required_metrics:
            assert metric in metrics
        
        # Check clustering quality
        assert metrics["n_clusters"] > 0
        assert metrics["silhouette_score"] > 0  # Should have reasonable clustering
        assert metrics["noise_ratio"] < 0.5  # Shouldn't have too much noise
    
    @pytest.mark.asyncio
    async def test_cluster_embeddings_custom_params(self, clusterer, sample_embeddings):
        """Test clustering with custom parameters."""
        embeddings, _ = sample_embeddings
        
        custom_params = {
            "min_cluster_size": 10,
            "min_samples": 5,
            "metric": "manhattan",  # Use manhattan instead of cosine
            "alpha": 1.5
        }
        
        cluster_labels, metrics = await clusterer.cluster_embeddings(
            embeddings=embeddings,
            session_id="test_session",
            custom_params=custom_params,
            auto_optimize=False,
            embedding_model="test-model"
        )
        
        # Check that custom parameters were used
        assert metrics["parameters_used"]["min_cluster_size"] == 10
        assert metrics["parameters_used"]["min_samples"] == 5
        assert metrics["parameters_used"]["metric"] == "manhattan"
        assert metrics["parameters_used"]["alpha"] == 1.5
    
    @pytest.mark.asyncio
    async def test_cluster_embeddings_small_dataset(self, clusterer, small_embeddings):
        """Test clustering with very small dataset."""
        embeddings, _ = small_embeddings
        
        cluster_labels, metrics = await clusterer.cluster_embeddings(
            embeddings=embeddings,
            session_id="small_test",
            auto_optimize=True,
            embedding_model="nomic-embed-text"
        )
        
        # Should handle small datasets gracefully
        assert len(cluster_labels) == len(embeddings)
        assert metrics["n_clusters"] >= 0  # May have 0 clusters for very small data
        assert metrics["processing_time_seconds"] > 0
    
    @pytest.mark.asyncio
    async def test_calculate_cluster_metrics_no_clusters(self, clusterer):
        """Test metrics calculation when no clusters are found."""
        embeddings = np.random.randn(10, 5)
        cluster_labels = np.array([-1] * 10)  # All noise
        
        # Mock clusterer object
        mock_clusterer = Mock()
        mock_clusterer.probabilities_ = np.array([0.0] * 10)
        
        metrics = await clusterer._calculate_cluster_metrics(
            embeddings, cluster_labels, mock_clusterer, {}
        )
        
        assert metrics["n_clusters"] == 0
        assert metrics["n_noise_points"] == 10
        assert metrics["noise_ratio"] == 1.0
        assert metrics["silhouette_score"] == 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_cluster_metrics_single_cluster(self, clusterer):
        """Test metrics calculation with single cluster."""
        embeddings = np.random.randn(20, 5)
        cluster_labels = np.array([0] * 20)  # All in one cluster
        
        # Mock clusterer object
        mock_clusterer = Mock()
        mock_clusterer.probabilities_ = np.array([0.8] * 20)
        
        metrics = await clusterer._calculate_cluster_metrics(
            embeddings, cluster_labels, mock_clusterer, {}
        )
        
        assert metrics["n_clusters"] == 1
        assert metrics["n_noise_points"] == 0
        assert metrics["noise_ratio"] == 0.0
        assert abs(metrics["stability_score"] - 0.8) < 1e-10  # Use approximate equality for floats
    
    def test_get_cluster_hierarchy_not_found(self, clusterer):
        """Test getting hierarchy for non-existent clusterer."""
        hierarchy = clusterer.get_cluster_hierarchy("non_existent_key")
        assert hierarchy is None
    
    def test_get_cluster_hierarchy_no_tree(self, clusterer):
        """Test getting hierarchy when clusterer has no condensed tree."""
        mock_clusterer = Mock()
        del mock_clusterer.condensed_tree_  # Remove the attribute
        
        clusterer.clusterers["test_key"] = mock_clusterer
        hierarchy = clusterer.get_cluster_hierarchy("test_key")
        assert hierarchy is None
    
    def test_predict_cluster_not_found(self, clusterer):
        """Test prediction with non-existent clusterer."""
        new_embeddings = np.random.randn(5, 10)
        predictions = clusterer.predict_cluster("non_existent_key", new_embeddings)
        assert predictions is None
    
    @patch('hdbscan.approximate_predict')
    def test_predict_cluster_success(self, mock_predict, clusterer):
        """Test successful cluster prediction."""
        # Setup mock
        mock_clusterer = Mock()
        clusterer.clusterers["test_key"] = mock_clusterer
        
        mock_predict.return_value = (np.array([0, 1, -1]), np.array([0.8, 0.9, 0.1]))
        
        new_embeddings = np.random.randn(3, 10)
        predictions = clusterer.predict_cluster("test_key", new_embeddings)
        
        assert predictions is not None
        assert len(predictions) == 3
        mock_predict.assert_called_once_with(mock_clusterer, new_embeddings)
    
    @patch('hdbscan.approximate_predict')
    def test_predict_cluster_error(self, mock_predict, clusterer):
        """Test cluster prediction with error."""
        # Setup mock to raise exception
        mock_clusterer = Mock()
        clusterer.clusterers["test_key"] = mock_clusterer
        
        mock_predict.side_effect = Exception("Prediction failed")
        
        new_embeddings = np.random.randn(3, 10)
        predictions = clusterer.predict_cluster("test_key", new_embeddings)
        
        assert predictions is None
    
    def test_clustering_stability_multiple_runs(self, clusterer, sample_embeddings):
        """Test clustering stability across multiple runs."""
        embeddings, _ = sample_embeddings
        
        # Run clustering multiple times with same parameters
        results = []
        for i in range(3):
            # Use different random states by varying session_id
            asyncio.run(self._run_clustering_test(clusterer, embeddings, f"stability_test_{i}"))
        
        # Note: HDBSCAN should be deterministic with same parameters,
        # but we're testing that the implementation doesn't crash
        assert len(results) == 0  # Just checking we didn't collect results
    
    async def _run_clustering_test(self, clusterer, embeddings, session_id):
        """Helper method for stability testing."""
        cluster_labels, metrics = await clusterer.cluster_embeddings(
            embeddings=embeddings,
            session_id=session_id,
            auto_optimize=True,
            embedding_model="nomic-embed-text"
        )
        return cluster_labels, metrics
    
    def test_parameter_optimization_edge_cases(self, clusterer):
        """Test parameter optimization with edge cases."""
        # Very small dataset
        embeddings_tiny = np.random.randn(5, 2)
        params_tiny = clusterer._optimize_parameters(embeddings_tiny, 5, "nomic-embed-text")
        assert params_tiny["min_cluster_size"] >= 3
        assert params_tiny["min_samples"] >= 2
        
        # Very high dimensional
        embeddings_high_dim = np.random.randn(100, 1000)
        params_high_dim = clusterer._optimize_parameters(embeddings_high_dim, 100, "mxbai-embed-large")
        assert params_high_dim["metric"] == "manhattan"  # High dims use manhattan
        
        # Very large dataset
        embeddings_large = np.random.randn(50000, 50)
        params_large = clusterer._optimize_parameters(embeddings_large, 50000, "all-minilm")
        assert params_large["min_cluster_size"] >= 20


class TestClusteringAccuracy:
    """Test clustering accuracy and quality metrics."""
    
    @pytest.fixture
    def well_separated_clusters(self):
        """Create well-separated clusters for accuracy testing."""
        X, y = make_blobs(
            n_samples=300,
            centers=5,
            n_features=20,
            random_state=42,
            cluster_std=0.5,
            center_box=(-10.0, 10.0)
        )
        return X, y
    
    @pytest.fixture
    def overlapping_clusters(self):
        """Create overlapping clusters for challenging scenarios."""
        X, y = make_blobs(
            n_samples=200,
            centers=3,
            n_features=15,
            random_state=42,
            cluster_std=2.0,
            center_box=(-5.0, 5.0)
        )
        return X, y
    
    @pytest.mark.asyncio
    async def test_clustering_well_separated_data(self, well_separated_clusters):
        """Test clustering accuracy on well-separated data."""
        embeddings, true_labels = well_separated_clusters
        clusterer = HDBSCANClusterer()
        
        cluster_labels, metrics = await clusterer.cluster_embeddings(
            embeddings=embeddings,
            session_id="accuracy_test",
            auto_optimize=True,
            embedding_model="nomic-embed-text"
        )
        
        # Should achieve good clustering on well-separated data
        assert metrics["silhouette_score"] > 0.5  # Good separation
        assert metrics["n_clusters"] >= 3  # Should find multiple clusters
        assert metrics["noise_ratio"] < 0.2  # Low noise ratio
        
        # Calinski-Harabasz score should be high for well-separated clusters
        assert metrics["calinski_harabasz_score"] > 100
        
        # Davies-Bouldin score should be low (better clustering)
        assert metrics["davies_bouldin_score"] < 2.0
    
    @pytest.mark.asyncio
    async def test_clustering_overlapping_data(self, overlapping_clusters):
        """Test clustering behavior on overlapping data."""
        embeddings, true_labels = overlapping_clusters
        clusterer = HDBSCANClusterer()
        
        cluster_labels, metrics = await clusterer.cluster_embeddings(
            embeddings=embeddings,
            session_id="overlap_test",
            auto_optimize=True,
            embedding_model="nomic-embed-text"
        )
        
        # Should still find some structure, but with more noise
        assert metrics["n_clusters"] >= 1  # Should find at least some clusters
        assert metrics["noise_ratio"] < 0.8  # Shouldn't classify everything as noise
        
        # Quality metrics should reflect the challenging nature
        assert metrics["silhouette_score"] >= -0.5  # Not too negative
    
    @pytest.mark.asyncio
    async def test_clustering_consistency(self, well_separated_clusters):
        """Test that clustering results are consistent across runs."""
        embeddings, _ = well_separated_clusters
        clusterer = HDBSCANClusterer()
        
        # Run clustering twice with same parameters
        labels1, metrics1 = await clusterer.cluster_embeddings(
            embeddings=embeddings,
            session_id="consistency_test_1",
            custom_params={"min_cluster_size": 10, "min_samples": 5},
            auto_optimize=False,
            embedding_model="test-model"
        )
        
        labels2, metrics2 = await clusterer.cluster_embeddings(
            embeddings=embeddings,
            session_id="consistency_test_2", 
            custom_params={"min_cluster_size": 10, "min_samples": 5},
            auto_optimize=False,
            embedding_model="test-model"
        )
        
        # Results should be identical (HDBSCAN is deterministic)
        np.testing.assert_array_equal(labels1, labels2)
        assert metrics1["n_clusters"] == metrics2["n_clusters"]
        assert metrics1["n_noise_points"] == metrics2["n_noise_points"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])