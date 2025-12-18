"""Unit tests for Clustering Pipeline."""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/app')

from services.ai_engine.app.clustering.pipeline import TabClusterer, Tab, Cluster


class TestTabClusterer:
    """Tests for TabClusterer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.clusterer = TabClusterer(
            min_cluster_size=2,
            min_samples=1,
        )
    
    def test_extract_domain(self):
        """Test domain extraction."""
        assert self.clusterer.extract_domain("https://example.com/page") == "example.com"
        assert self.clusterer.extract_domain("https://www.example.com/page") == "example.com"
        assert self.clusterer.extract_domain("https://sub.example.com/page") == "sub.example.com"
    
    def test_group_by_domain(self):
        """Test grouping tabs by domain."""
        tabs = [
            Tab(url="https://example.com/page1", title="Page 1"),
            Tab(url="https://example.com/page2", title="Page 2"),
            Tab(url="https://other.com/page", title="Other"),
        ]
        
        groups = self.clusterer.group_by_domain(tabs)
        
        assert len(groups) == 2
        assert len(groups["example.com"]) == 2
        assert len(groups["other.com"]) == 1
    
    def test_reduce_dimensions(self):
        """Test dimension reduction."""
        # Create random embeddings
        embeddings = np.random.rand(10, 100)
        
        reduced = self.clusterer.reduce_dimensions(embeddings)
        
        # Should reduce to fewer dimensions
        assert reduced.shape[0] == 10
        assert reduced.shape[1] <= self.clusterer.umap_n_components
    
    def test_reduce_dimensions_small_sample(self):
        """Test dimension reduction with small sample."""
        embeddings = np.random.rand(3, 100)
        
        reduced = self.clusterer.reduce_dimensions(embeddings)
        
        assert reduced.shape[0] == 3
    
    def test_cluster_embeddings(self):
        """Test clustering embeddings."""
        # Create clustered embeddings
        cluster1 = np.random.rand(5, 10) + np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster2 = np.random.rand(5, 10) + np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        embeddings = np.vstack([cluster1, cluster2])
        
        labels = self.clusterer.cluster_embeddings(embeddings)
        
        assert len(labels) == 10
    
    def test_group_by_labels(self):
        """Test grouping tabs by cluster labels."""
        tabs = [
            Tab(url="https://example.com/1", title="Tab 1"),
            Tab(url="https://example.com/2", title="Tab 2"),
            Tab(url="https://example.com/3", title="Tab 3"),
        ]
        labels = np.array([0, 0, 1])
        
        clusters = self.clusterer._group_by_labels(tabs, labels)
        
        # Should have 2 clusters
        cluster_ids = [c.id for c in clusters]
        assert 0 in cluster_ids
        assert 1 in cluster_ids
    
    def test_group_by_labels_with_noise(self):
        """Test grouping with noise points (label -1)."""
        tabs = [
            Tab(url="https://example.com/1", title="Tab 1"),
            Tab(url="https://example.com/2", title="Tab 2"),
            Tab(url="https://example.com/3", title="Tab 3"),
        ]
        labels = np.array([0, 0, -1])  # -1 is noise
        
        clusters = self.clusterer._group_by_labels(tabs, labels)
        
        # Should have cluster 0 and uncategorized for noise
        assert len(clusters) >= 1
    
    def test_cluster_sync(self):
        """Test synchronous clustering."""
        tabs = [
            Tab(url="https://example.com/1", title="Tab 1"),
            Tab(url="https://example.com/2", title="Tab 2"),
            Tab(url="https://other.com/1", title="Tab 3"),
            Tab(url="https://other.com/2", title="Tab 4"),
        ]
        
        # Create embeddings that should cluster together
        embeddings = np.array([
            [1, 0, 0, 0, 0],
            [1, 0.1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0.1],
        ])
        
        clusters = self.clusterer.cluster_sync(tabs, embeddings)
        
        assert len(clusters) >= 1
    
    def test_to_dict(self):
        """Test converting clusters to dictionary."""
        clusters = [
            Cluster(
                id=0,
                name="Test Cluster",
                tabs=[
                    Tab(url="https://example.com/1", title="Tab 1"),
                    Tab(url="https://example.com/2", title="Tab 2"),
                ],
            ),
        ]
        
        result = self.clusterer.to_dict(clusters)
        
        assert len(result) == 1
        assert result[0]["id"] == 0
        assert result[0]["name"] == "Test Cluster"
        assert result[0]["tab_count"] == 2
        assert len(result[0]["urls"]) == 2
    
    def test_to_dict_with_subclusters(self):
        """Test converting clusters with subclusters to dictionary."""
        clusters = [
            Cluster(
                id=0,
                name="Parent",
                tabs=[Tab(url="https://example.com/1", title="Tab 1")],
                subclusters=[
                    Cluster(
                        id=1,
                        name="Child",
                        tabs=[Tab(url="https://example.com/2", title="Tab 2")],
                    ),
                ],
            ),
        ]
        
        result = self.clusterer.to_dict(clusters)
        
        assert "subclusters" in result[0]
        assert len(result[0]["subclusters"]) == 1


class TestTab:
    """Tests for Tab dataclass."""
    
    def test_tab_creation(self):
        """Test Tab creation."""
        tab = Tab(
            url="https://example.com",
            title="Example",
            content="Content here",
        )
        
        assert tab.url == "https://example.com"
        assert tab.title == "Example"
        assert tab.content == "Content here"
        assert tab.embedding is None
    
    def test_tab_with_embedding(self):
        """Test Tab with embedding."""
        embedding = np.array([1, 2, 3])
        tab = Tab(
            url="https://example.com",
            title="Example",
            embedding=embedding,
        )
        
        assert np.array_equal(tab.embedding, embedding)


class TestCluster:
    """Tests for Cluster dataclass."""
    
    def test_cluster_creation(self):
        """Test Cluster creation."""
        cluster = Cluster(id=0, name="Test")
        
        assert cluster.id == 0
        assert cluster.name == "Test"
        assert cluster.tabs == []
        assert cluster.subclusters == []
    
    def test_cluster_with_tabs(self):
        """Test Cluster with tabs."""
        tabs = [
            Tab(url="https://example.com/1", title="Tab 1"),
            Tab(url="https://example.com/2", title="Tab 2"),
        ]
        
        cluster = Cluster(id=0, name="Test", tabs=tabs)
        
        assert len(cluster.tabs) == 2
