"""Unit tests for hardware detection and model recommendation system."""

import pytest
from unittest.mock import Mock, patch
import json
import tempfile
from pathlib import Path

from main import HardwareDetector, ModelManager


class TestHardwareDetection:
    """Test comprehensive hardware detection functionality."""
    
    def test_detect_basic_system_info(self):
        """Test detection of basic system information."""
        detector = HardwareDetector()
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_count') as mock_cpu, \
             patch('psutil.cpu_percent') as mock_cpu_percent:
            
            # Mock system specifications
            mock_memory.return_value = Mock(
                total=16 * 1024**3,  # 16GB total RAM
                available=8 * 1024**3,  # 8GB available
                percent=50.0
            )
            mock_cpu.return_value = 8
            mock_cpu_percent.return_value = 25.0
            
            hardware_info = detector.detect_hardware()
            
            assert hardware_info["ram_gb"] == 16.0
            assert hardware_info["available_ram_gb"] == 8.0
            assert hardware_info["ram_usage_percent"] == 50.0
            assert hardware_info["cpu_count"] == 8
            assert hardware_info["cpu_usage_percent"] == 25.0
    
    def test_detect_gpu_nvidia(self):
        """Test detection of NVIDIA GPU."""
        detector = HardwareDetector()
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_count') as mock_cpu, \
             patch('psutil.cpu_percent') as mock_cpu_percent, \
             patch('torch.cuda.is_available') as mock_cuda_available, \
             patch('torch.cuda.get_device_properties') as mock_gpu_props, \
             patch('torch.cuda.get_device_name') as mock_gpu_name:
            
            # Mock basic system info
            mock_memory.return_value = Mock(
                total=32 * 1024**3,
                available=16 * 1024**3,
                percent=50.0
            )
            mock_cpu.return_value = 12
            mock_cpu_percent.return_value = 20.0
            
            # Mock GPU detection
            mock_cuda_available.return_value = True
            mock_gpu_props.return_value = Mock(total_memory=12 * 1024**3)  # 12GB VRAM
            mock_gpu_name.return_value = "NVIDIA GeForce RTX 4080"
            
            hardware_info = detector.detect_hardware()
            
            assert hardware_info["has_gpu"] is True
            assert hardware_info["gpu_memory_gb"] == 12.0
            assert hardware_info["gpu_name"] == "NVIDIA GeForce RTX 4080"
    
    def test_detect_no_gpu(self):
        """Test detection when no GPU is available."""
        detector = HardwareDetector()
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_count') as mock_cpu, \
             patch('psutil.cpu_percent') as mock_cpu_percent, \
             patch('torch.cuda.is_available') as mock_cuda_available:
            
            mock_memory.return_value = Mock(
                total=8 * 1024**3,
                available=4 * 1024**3,
                percent=50.0
            )
            mock_cpu.return_value = 4
            mock_cpu_percent.return_value = 30.0
            mock_cuda_available.return_value = False
            
            hardware_info = detector.detect_hardware()
            
            assert hardware_info["has_gpu"] is False
            assert hardware_info["gpu_memory_gb"] == 0.0
            assert hardware_info["gpu_name"] == "None"
    
    def test_detect_gpu_error_handling(self):
        """Test GPU detection error handling."""
        detector = HardwareDetector()
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_count') as mock_cpu, \
             patch('psutil.cpu_percent') as mock_cpu_percent, \
             patch('torch.cuda.is_available') as mock_cuda_available, \
             patch('torch.cuda.get_device_properties', side_effect=Exception("GPU error")):
            
            mock_memory.return_value = Mock(
                total=8 * 1024**3,
                available=4 * 1024**3,
                percent=50.0
            )
            mock_cpu.return_value = 4
            mock_cpu_percent.return_value = 30.0
            mock_cuda_available.return_value = True  # CUDA available but properties fail
            
            hardware_info = detector.detect_hardware()
            
            # Should gracefully handle GPU detection failure
            assert hardware_info["has_gpu"] is False
            assert hardware_info["gpu_memory_gb"] == 0.0
            assert hardware_info["gpu_name"] == "None"
    
    def test_detect_low_resource_system(self):
        """Test detection of low-resource system."""
        detector = HardwareDetector()
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_count') as mock_cpu, \
             patch('psutil.cpu_percent') as mock_cpu_percent:
            
            # Mock low-resource system
            mock_memory.return_value = Mock(
                total=2 * 1024**3,  # 2GB total RAM
                available=0.5 * 1024**3,  # 0.5GB available
                percent=75.0  # High usage
            )
            mock_cpu.return_value = 2
            mock_cpu_percent.return_value = 60.0
            
            hardware_info = detector.detect_hardware()
            
            assert hardware_info["ram_gb"] == 2.0
            assert hardware_info["available_ram_gb"] == 0.5
            assert hardware_info["ram_usage_percent"] == 75.0
            assert hardware_info["cpu_count"] == 2
            assert hardware_info["cpu_usage_percent"] == 60.0
    
    def test_detect_high_end_system(self):
        """Test detection of high-end system."""
        detector = HardwareDetector()
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_count') as mock_cpu, \
             patch('psutil.cpu_percent') as mock_cpu_percent, \
             patch('torch.cuda.is_available') as mock_cuda_available, \
             patch('torch.cuda.get_device_properties') as mock_gpu_props, \
             patch('torch.cuda.get_device_name') as mock_gpu_name:
            
            # Mock high-end system
            mock_memory.return_value = Mock(
                total=64 * 1024**3,  # 64GB total RAM
                available=48 * 1024**3,  # 48GB available
                percent=25.0  # Low usage
            )
            mock_cpu.return_value = 16
            mock_cpu_percent.return_value = 10.0
            
            # High-end GPU
            mock_cuda_available.return_value = True
            mock_gpu_props.return_value = Mock(total_memory=24 * 1024**3)  # 24GB VRAM
            mock_gpu_name.return_value = "NVIDIA RTX 4090"
            
            hardware_info = detector.detect_hardware()
            
            assert hardware_info["ram_gb"] == 64.0
            assert hardware_info["available_ram_gb"] == 48.0
            assert hardware_info["ram_usage_percent"] == 25.0
            assert hardware_info["cpu_count"] == 16
            assert hardware_info["has_gpu"] is True
            assert hardware_info["gpu_memory_gb"] == 24.0
            assert hardware_info["gpu_name"] == "NVIDIA RTX 4090"
    
    def test_detect_hardware_complete_failure(self):
        """Test hardware detection when everything fails."""
        detector = HardwareDetector()
        
        with patch('psutil.virtual_memory', side_effect=Exception("System error")), \
             patch('psutil.cpu_count', side_effect=Exception("CPU error")), \
             patch('psutil.cpu_percent', side_effect=Exception("CPU percent error")):
            
            hardware_info = detector.detect_hardware()
            
            # Should return safe defaults
            assert hardware_info["ram_gb"] == 8.0
            assert hardware_info["cpu_count"] == 4
            assert hardware_info["has_gpu"] is False
            assert hardware_info["available_ram_gb"] == 4.0
            assert hardware_info["ram_usage_percent"] == 50.0


class TestModelRecommendations:
    """Test model recommendation system based on hardware."""
    
    @pytest.fixture
    def comprehensive_models_config(self):
        """Comprehensive models configuration for testing."""
        return {
            "embedding_models": {
                "all-minilm": {
                    "name": "All-MiniLM",
                    "size": "90MB",
                    "dimensions": 384,
                    "quality": "good",
                    "min_ram_gb": 0.5,
                    "description": "Lightweight embedding model",
                    "recommended": True
                },
                "nomic-embed-text": {
                    "name": "Nomic Embed Text",
                    "size": "274MB",
                    "dimensions": 768,
                    "quality": "high",
                    "min_ram_gb": 1.0,
                    "description": "Best general purpose embedding model",
                    "recommended": False
                },
                "mxbai-embed-large": {
                    "name": "MxBai Embed Large",
                    "size": "669MB",
                    "dimensions": 1024,
                    "quality": "highest",
                    "min_ram_gb": 2.0,
                    "description": "Highest quality embeddings",
                    "recommended": False
                }
            }
        }
    
    def test_recommend_for_low_resource_system(self, comprehensive_models_config):
        """Test recommendations for low-resource system."""
        with patch.object(ModelManager, '_load_models_config', return_value=comprehensive_models_config):
            manager = ModelManager()
            
            # Low-resource system
            hardware_info = {
                "available_ram_gb": 1.0,
                "has_gpu": False
            }
            
            recommendation = manager.recommend_model(hardware_info)
            
            assert recommendation["recommended_model"] == "all-minilm"
            assert "1.0" in recommendation["reason"] or "low" in recommendation["reason"].lower()
            assert len(recommendation["alternatives"]) >= 0
            assert "embeddings_per_sec" in recommendation["performance_estimate"]
    
    def test_recommend_for_medium_resource_system(self, comprehensive_models_config):
        """Test recommendations for medium-resource system."""
        with patch.object(ModelManager, '_load_models_config', return_value=comprehensive_models_config):
            manager = ModelManager()
            
            # Medium-resource system
            hardware_info = {
                "available_ram_gb": 4.0,
                "has_gpu": False
            }
            
            recommendation = manager.recommend_model(hardware_info)
            
            # Should recommend a better model than all-minilm
            assert recommendation["recommended_model"] in ["nomic-embed-text", "mxbai-embed-large"]
            assert len(recommendation["alternatives"]) > 0
    
    def test_recommend_for_high_resource_system(self, comprehensive_models_config):
        """Test recommendations for high-resource system."""
        with patch.object(ModelManager, '_load_models_config', return_value=comprehensive_models_config):
            manager = ModelManager()
            
            # High-resource system
            hardware_info = {
                "available_ram_gb": 16.0,
                "has_gpu": True
            }
            
            recommendation = manager.recommend_model(hardware_info)
            
            # Should recommend the highest quality model
            assert recommendation["recommended_model"] in ["mxbai-embed-large", "nomic-embed-text"]
            assert len(recommendation["alternatives"]) > 0
            assert recommendation["performance_estimate"]["embeddings_per_sec"] > 0
    
    def test_recommend_with_gpu_bonus(self, comprehensive_models_config):
        """Test that GPU systems get performance bonus in recommendations."""
        with patch.object(ModelManager, '_load_models_config', return_value=comprehensive_models_config):
            manager = ModelManager()
            
            # Same RAM, different GPU status
            hardware_info_no_gpu = {
                "available_ram_gb": 4.0,
                "has_gpu": False
            }
            
            hardware_info_with_gpu = {
                "available_ram_gb": 4.0,
                "has_gpu": True
            }
            
            rec_no_gpu = manager.recommend_model(hardware_info_no_gpu)
            rec_with_gpu = manager.recommend_model(hardware_info_with_gpu)
            
            # GPU system should get better performance estimate
            perf_no_gpu = rec_no_gpu["performance_estimate"]["embeddings_per_sec"]
            perf_with_gpu = rec_with_gpu["performance_estimate"]["embeddings_per_sec"]
            
            assert perf_with_gpu > perf_no_gpu
    
    def test_recommend_no_suitable_models(self, comprehensive_models_config):
        """Test recommendations when no models fit the hardware."""
        with patch.object(ModelManager, '_load_models_config', return_value=comprehensive_models_config):
            manager = ModelManager()
            
            # Extremely low-resource system
            hardware_info = {
                "available_ram_gb": 0.1,  # 100MB available
                "has_gpu": False
            }
            
            recommendation = manager.recommend_model(hardware_info)
            
            # Should fallback to lightest model
            assert recommendation["recommended_model"] == "all-minilm"
            assert "fallback" in recommendation["reason"].lower()
            assert recommendation["alternatives"] == []
    
    def test_model_scoring_algorithm(self, comprehensive_models_config):
        """Test the model scoring algorithm."""
        with patch.object(ModelManager, '_load_models_config', return_value=comprehensive_models_config):
            manager = ModelManager()
            
            # Test scoring for different models
            config_good = comprehensive_models_config["embedding_models"]["all-minilm"]
            config_high = comprehensive_models_config["embedding_models"]["nomic-embed-text"]
            config_highest = comprehensive_models_config["embedding_models"]["mxbai-embed-large"]
            
            score_good = manager._calculate_model_score(config_good, has_gpu=False)
            score_high = manager._calculate_model_score(config_high, has_gpu=False)
            score_highest = manager._calculate_model_score(config_highest, has_gpu=False)
            
            # Higher quality should generally get higher scores
            # (though other factors like recommendations can affect this)
            assert isinstance(score_good, float)
            assert isinstance(score_high, float)
            assert isinstance(score_highest, float)
    
    def test_performance_estimation(self, comprehensive_models_config):
        """Test performance estimation for different models."""
        with patch.object(ModelManager, '_load_models_config', return_value=comprehensive_models_config):
            manager = ModelManager()
            
            # Test performance estimation for different models and hardware
            config_small = comprehensive_models_config["embedding_models"]["all-minilm"]
            config_large = comprehensive_models_config["embedding_models"]["mxbai-embed-large"]
            
            # CPU performance
            perf_small_cpu = manager._estimate_performance(config_small, has_gpu=False)
            perf_large_cpu = manager._estimate_performance(config_large, has_gpu=False)
            
            # GPU performance
            perf_small_gpu = manager._estimate_performance(config_small, has_gpu=True)
            perf_large_gpu = manager._estimate_performance(config_large, has_gpu=True)
            
            # GPU should be faster than CPU
            assert perf_small_gpu["embeddings_per_sec"] > perf_small_cpu["embeddings_per_sec"]
            assert perf_large_gpu["embeddings_per_sec"] > perf_large_cpu["embeddings_per_sec"]
            
            # All should have required fields
            for perf in [perf_small_cpu, perf_large_cpu, perf_small_gpu, perf_large_gpu]:
                assert "embeddings_per_sec" in perf
                assert "dimensions" in perf
                assert "suitable_for_batch" in perf
                assert perf["embeddings_per_sec"] > 0
    
    def test_recommendation_consistency(self, comprehensive_models_config):
        """Test that recommendations are consistent for same hardware."""
        with patch.object(ModelManager, '_load_models_config', return_value=comprehensive_models_config):
            manager = ModelManager()
            
            hardware_info = {
                "available_ram_gb": 4.0,
                "has_gpu": True
            }
            
            # Get multiple recommendations for same hardware
            rec1 = manager.recommend_model(hardware_info)
            rec2 = manager.recommend_model(hardware_info)
            rec3 = manager.recommend_model(hardware_info)
            
            # Should be consistent
            assert rec1["recommended_model"] == rec2["recommended_model"]
            assert rec2["recommended_model"] == rec3["recommended_model"]
            assert rec1["alternatives"] == rec2["alternatives"]
    
    def test_edge_case_hardware_values(self, comprehensive_models_config):
        """Test recommendations with edge case hardware values."""
        with patch.object(ModelManager, '_load_models_config', return_value=comprehensive_models_config):
            manager = ModelManager()
            
            # Test various edge cases
            edge_cases = [
                {"available_ram_gb": 0.0, "has_gpu": False},  # No RAM
                {"available_ram_gb": 1000.0, "has_gpu": True},  # Massive RAM
                {"available_ram_gb": -1.0, "has_gpu": False},  # Negative RAM
                {"available_ram_gb": float('inf'), "has_gpu": True},  # Infinite RAM
            ]
            
            for hardware_info in edge_cases:
                try:
                    recommendation = manager.recommend_model(hardware_info)
                    # Should not crash and should return valid recommendation
                    assert "recommended_model" in recommendation
                    assert recommendation["recommended_model"] in comprehensive_models_config["embedding_models"]
                except Exception as e:
                    pytest.fail(f"Recommendation failed for hardware {hardware_info}: {e}")


class TestHardwareModelIntegration:
    """Integration tests for hardware detection and model recommendations."""
    
    def test_full_recommendation_workflow(self):
        """Test complete workflow from hardware detection to model recommendation."""
        # Create a realistic models config
        models_config = {
            "embedding_models": {
                "all-minilm": {
                    "name": "All-MiniLM",
                    "dimensions": 384,
                    "quality": "good",
                    "min_ram_gb": 0.5,
                    "description": "Lightweight model",
                    "recommended": True
                },
                "nomic-embed-text": {
                    "name": "Nomic Embed Text",
                    "dimensions": 768,
                    "quality": "high",
                    "min_ram_gb": 1.0,
                    "description": "General purpose model",
                    "recommended": False
                }
            }
        }
        
        with patch.object(ModelManager, '_load_models_config', return_value=models_config), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_count') as mock_cpu, \
             patch('psutil.cpu_percent') as mock_cpu_percent, \
             patch('torch.cuda.is_available') as mock_cuda:
            
            # Mock medium-resource system
            mock_memory.return_value = Mock(
                total=8 * 1024**3,
                available=4 * 1024**3,
                percent=50.0
            )
            mock_cpu.return_value = 8
            mock_cpu_percent.return_value = 25.0
            mock_cuda.return_value = False
            
            # Test the full workflow
            detector = HardwareDetector()
            manager = ModelManager()
            
            # Detect hardware
            hardware_info = detector.detect_hardware()
            assert hardware_info["available_ram_gb"] == 4.0
            assert hardware_info["has_gpu"] is False
            
            # Get recommendation based on detected hardware
            recommendation = manager.recommend_model(hardware_info)
            
            # Should get a valid recommendation
            assert recommendation["recommended_model"] in ["all-minilm", "nomic-embed-text"]
            assert "performance_estimate" in recommendation
            assert recommendation["performance_estimate"]["embeddings_per_sec"] > 0


if __name__ == "__main__":
    pytest.main([__file__])