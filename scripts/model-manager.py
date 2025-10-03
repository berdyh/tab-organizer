#!/usr/bin/env python3
"""
Dynamic Model Manager for Ollama
Supports easy addition, removal, and management of AI models through configuration.
"""

import json
import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import psutil
import platform

class ModelManager:
    def __init__(self, config_path: str = "config/models.json"):
        self.config_path = Path(config_path)
        self.models_config = self.load_config()
        self.hardware_info = self.detect_hardware()
    
    def load_config(self) -> Dict:
        """Load models configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {self.config_path}")
            print("Please ensure config/models.json exists.")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in configuration file: {e}")
            sys.exit(1)
    
    def detect_hardware(self) -> Dict:
        """Detect system hardware capabilities."""
        ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        # Try to detect GPU
        has_gpu = False
        gpu_memory = 0
        gpu_name = "None"
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,name', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                has_gpu = True
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(', ')
                    gpu_memory = int(parts[0]) / 1024  # Convert MB to GB
                    gpu_name = parts[1] if len(parts) > 1 else "Unknown GPU"
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return {
            "ram_gb": ram_gb,
            "cpu_count": cpu_count,
            "has_gpu": has_gpu,
            "gpu_memory_gb": gpu_memory,
            "gpu_name": gpu_name,
            "platform": platform.system()
        }
    
    def get_available_resources(self) -> Dict:
        """Get currently available resources (not just total)."""
        try:
            memory = psutil.virtual_memory()
            available_ram = memory.available / (1024**3)
            
            # Get CPU usage with shorter interval to avoid hanging
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Try to get GPU memory usage with timeout
            gpu_available = 0
            if self.hardware_info["has_gpu"]:
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], 
                                          capture_output=True, text=True, timeout=3)
                    if result.returncode == 0:
                        gpu_available = int(result.stdout.strip()) / 1024  # Convert MB to GB
                except Exception:
                    gpu_available = self.hardware_info["gpu_memory_gb"]  # Fallback to total
            
            return {
                "available_ram_gb": available_ram,
                "ram_usage_percent": memory.percent,
                "cpu_usage_percent": cpu_percent,
                "gpu_available_gb": gpu_available
            }
        except Exception as e:
            print(f"Warning: Could not get resource usage: {e}")
            # Return safe defaults
            return {
                "available_ram_gb": self.hardware_info["ram_gb"] * 0.7,  # Assume 70% available
                "ram_usage_percent": 30.0,
                "cpu_usage_percent": 20.0,
                "gpu_available_gb": self.hardware_info.get("gpu_memory_gb", 0) * 0.8
            }
    
    def estimate_performance(self, model_id: str) -> Dict:
        """Estimate inference performance for given hardware."""
        if model_id not in {**self.models_config["llm_models"], **self.models_config["embedding_models"]}:
            return {"error": "Model not found"}
        
        # Get model info
        if model_id in self.models_config["llm_models"]:
            model_config = self.models_config["llm_models"][model_id]
            model_type = "llm"
        else:
            model_config = self.models_config["embedding_models"][model_id]
            model_type = "embedding"
        
        # Extract parameter count from size (rough estimation)
        size_str = model_config["size"]
        if "GB" in size_str:
            size_gb = float(size_str.replace("GB", "").strip())
            # Rough parameter estimation: 1GB ≈ 0.5B parameters for Q4 quantization
            estimated_params = size_gb * 0.5
        else:
            estimated_params = 0.1  # Small embedding model
        
        # Performance estimation based on hardware
        has_gpu = self.hardware_info["has_gpu"]
        ram_gb = self.hardware_info["ram_gb"]
        
        if model_type == "llm":
            if has_gpu and self.hardware_info["gpu_memory_gb"] >= model_config["min_ram_gb"]:
                # GPU inference
                tokens_per_sec = max(5, 80 / estimated_params)
                time_to_first_token = 200
            else:
                # CPU inference
                tokens_per_sec = max(1, 25 / estimated_params)
                time_to_first_token = 800
            
            return {
                "estimated_tokens_per_sec": round(tokens_per_sec, 1),
                "time_to_first_token_ms": time_to_first_token,
                "suitable_for_realtime": tokens_per_sec > 10,
                "suitable_for_batch": True,
                "estimated_params_b": round(estimated_params, 1)
            }
        else:
            # Embedding model performance
            if has_gpu:
                embeddings_per_sec = max(50, 500 / estimated_params)
            else:
                embeddings_per_sec = max(10, 100 / estimated_params)
            
            return {
                "estimated_embeddings_per_sec": round(embeddings_per_sec, 1),
                "suitable_for_batch": True,
                "dimensions": model_config["dimensions"]
            }
    
    def get_recommended_models(self, use_case: Optional[str] = None, 
                              min_context: int = 8192, 
                              prioritize_speed: bool = False) -> Tuple[str, str]:
        """Get recommended models based on hardware, use case, and requirements."""
        available_resources = self.get_available_resources()
        available_ram = available_resources["available_ram_gb"]
        has_gpu = self.hardware_info["has_gpu"]
        gpu_available = available_resources["gpu_available_gb"]
        
        # Filter LLM models by available RAM (use 80% rule for safety)
        suitable_llms = []
        for model_id, config in self.models_config["llm_models"].items():
            # Check RAM requirements
            if config["min_ram_gb"] <= available_ram * 0.8:
                # Check use case compatibility
                if use_case is None or use_case in config["use_cases"]:
                    # Check context length requirements
                    if config["context_length"] >= min_context:
                        # Calculate suitability score
                        score = self._calculate_model_score(model_id, config, prioritize_speed)
                        suitable_llms.append((model_id, config, score))
        
        # Sort by suitability score
        suitable_llms.sort(key=lambda x: x[2], reverse=True)
        
        # Select best LLM
        best_llm = suitable_llms[0][0] if suitable_llms else "qwen3:1.7b"  # Fallback
        
        # Select best embedding model
        best_embedding = self._select_best_embedding(available_ram)
        
        return best_llm, best_embedding
    
    def _calculate_model_score(self, model_id: str, config: Dict, prioritize_speed: bool) -> float:
        """Calculate suitability score for a model based on hardware and preferences."""
        score = 0.0
        
        # Base score from quality
        quality_scores = {"basic": 1, "good": 2, "high": 3, "highest": 4}
        score += quality_scores.get(config["quality"], 2) * 10
        
        # Speed bonus/penalty
        speed_scores = {"fastest": 4, "fast": 3, "medium": 2, "slow": 1}
        speed_score = speed_scores.get(config["speed"], 2)
        if prioritize_speed:
            score += speed_score * 15
        else:
            score += speed_score * 5
        
        # Hardware compatibility bonus
        if self.hardware_info["has_gpu"] and config["min_ram_gb"] <= 4:
            score += 5  # GPU-friendly models
        
        # Context length bonus for long-context tasks
        if config["context_length"] >= 32768:
            score += 3
        
        # Recommended model bonus
        if config.get("recommended", False):
            score += 8
        
        # Penalize if model is too large for available RAM
        available_ram = self.get_available_resources()["available_ram_gb"]
        if config["min_ram_gb"] > available_ram * 0.6:
            score -= 10
        
        return score
    
    def _select_best_embedding(self, available_ram: float) -> str:
        """Select best embedding model based on available resources."""
        suitable_embeddings = []
        
        for model_id, config in self.models_config["embedding_models"].items():
            if config["min_ram_gb"] <= available_ram * 0.9:  # More lenient for embeddings
                score = 0
                
                # Quality score
                quality_scores = {"good": 2, "high": 3, "highest": 4}
                score += quality_scores.get(config["quality"], 2) * 10
                
                # Dimension bonus (higher dimensions often better)
                score += config["dimensions"] / 100
                
                # Recommended bonus
                if config.get("recommended", False):
                    score += 5
                
                suitable_embeddings.append((model_id, score))
        
        suitable_embeddings.sort(key=lambda x: x[1], reverse=True)
        return suitable_embeddings[0][0] if suitable_embeddings else "all-minilm"
    
    def recommend_for_task(self, task_type: str, min_context: int = 8192) -> List[str]:
        """Recommend models for specific tasks with context requirements."""
        suitable = []
        available_ram = self.get_available_resources()["available_ram_gb"]
        
        for model_id, config in self.models_config["llm_models"].items():
            if (task_type in config["use_cases"] and 
                config["context_length"] >= min_context and
                config["min_ram_gb"] <= available_ram * 0.8):
                
                # Add performance estimate
                perf = self.estimate_performance(model_id)
                suitable.append({
                    "model_id": model_id,
                    "description": config["description"],
                    "performance": perf,
                    "size": config["size"]
                })
        
        return suitable
    
    def get_multi_model_recommendation(self) -> Dict[str, str]:
        """Recommend different models for different tasks."""
        recommendations = {}
        
        # Get general recommendation first as baseline
        try:
            general_llm, _ = self.get_recommended_models()
            recommendations["general"] = general_llm
        except Exception as e:
            print(f"Warning: Could not get general recommendation: {e}")
            recommendations["general"] = "qwen3:1.7b"
        
        # Get best model for each specific use case
        use_cases = ["reasoning", "code", "multilingual"]
        
        for use_case in use_cases:
            try:
                # Find models that support this use case
                suitable_models = []
                for model_id, config in self.models_config["llm_models"].items():
                    if use_case in config.get("use_cases", []):
                        suitable_models.append(model_id)
                
                if suitable_models:
                    # Use the first suitable model or the general recommendation if it's suitable
                    if recommendations["general"] in suitable_models:
                        recommendations[use_case] = recommendations["general"]
                    else:
                        recommendations[use_case] = suitable_models[0]
                else:
                    recommendations[use_case] = recommendations["general"]
                    
            except Exception as e:
                print(f"Warning: Could not get {use_case} recommendation: {e}")
                recommendations[use_case] = recommendations["general"]
        
        # Add speed-optimized option
        try:
            # Find fastest model that fits in available RAM
            available_ram = self.get_available_resources()["available_ram_gb"]
            fastest_model = None
            
            for model_id, config in self.models_config["llm_models"].items():
                if (config["min_ram_gb"] <= available_ram * 0.8 and 
                    config["speed"] in ["fastest", "fast"]):
                    if fastest_model is None or config["speed"] == "fastest":
                        fastest_model = model_id
            
            recommendations["fast_general"] = fastest_model or recommendations["general"]
            
        except Exception as e:
            print(f"Warning: Could not get speed recommendation: {e}")
            recommendations["fast_general"] = recommendations["general"]
        
        return recommendations
    
    def list_available_models(self, category: Optional[str] = None):
        """List all available models or models in a specific category."""
        print("🤖 Available Models")
        print("=" * 50)
        
        if category and category in self.models_config["model_categories"]:
            model_list = self.models_config["model_categories"][category]
            print(f"📂 Category: {category}")
            print()
        else:
            model_list = None
        
        # LLM Models
        print("🧠 LLM Models:")
        for model_id, config in self.models_config["llm_models"].items():
            if model_list is None or model_id in model_list:
                recommended = "⭐" if config.get("recommended", False) else "  "
                print(f"{recommended} {model_id}")
                print(f"    {config['name']} - {config['description']}")
                print(f"    Size: {config['size']}, Speed: {config['speed']}, Quality: {config['quality']}")
                print(f"    Min RAM: {config['min_ram_gb']}GB, Provider: {config['provider']}")
                print(f"    Use cases: {', '.join(config['use_cases'])}")
                print()
        
        # Embedding Models
        print("🔍 Embedding Models:")
        for model_id, config in self.models_config["embedding_models"].items():
            if model_list is None or model_id in model_list:
                recommended = "⭐" if config.get("recommended", False) else "  "
                print(f"{recommended} {model_id}")
                print(f"    {config['name']} - {config['description']}")
                print(f"    Size: {config['size']}, Dimensions: {config['dimensions']}")
                print(f"    Min RAM: {config['min_ram_gb']}GB, Provider: {config['provider']}")
                print()
    
    def show_hardware_info(self):
        """Display comprehensive hardware information and recommendations."""
        print("💻 Hardware Information")
        print("=" * 40)
        
        # Current hardware
        print("🔧 System Specifications:")
        print(f"   RAM: {self.hardware_info['ram_gb']:.1f} GB total")
        print(f"   CPU: {self.hardware_info['cpu_count']} cores")
        print(f"   GPU: {self.hardware_info['gpu_name']}")
        if self.hardware_info['has_gpu']:
            print(f"   GPU Memory: {self.hardware_info['gpu_memory_gb']:.1f} GB")
        print(f"   Platform: {self.hardware_info['platform']}")
        print()
        
        # Current resource usage
        print("📊 Checking current resource usage...")
        try:
            resources = self.get_available_resources()
            print(f"   Available RAM: {resources['available_ram_gb']:.1f} GB ({100-resources['ram_usage_percent']:.1f}% free)")
            print(f"   CPU Usage: {resources['cpu_usage_percent']:.1f}%")
            if self.hardware_info['has_gpu']:
                print(f"   GPU Available: {resources['gpu_available_gb']:.1f} GB")
        except Exception as e:
            print(f"   ⚠️  Could not get resource usage: {e}")
            resources = {
                "available_ram_gb": self.hardware_info['ram_gb'] * 0.7,
                "ram_usage_percent": 30.0
            }
        print()
        
        # General recommendations
        print("🎯 Getting general recommendations...")
        try:
            llm_rec, embed_rec = self.get_recommended_models()
            print("🎯 General Recommendations:")
            print(f"   LLM: {llm_rec}")
            print(f"   Embedding: {embed_rec}")
            
            # Show performance estimates
            try:
                llm_perf = self.estimate_performance(llm_rec)
                if "estimated_tokens_per_sec" in llm_perf:
                    print(f"   Expected LLM Speed: ~{llm_perf['estimated_tokens_per_sec']} tokens/sec")
            except Exception as e:
                print(f"   ⚠️  Could not estimate performance: {e}")
            print()
        except Exception as e:
            print(f"⚠️  Could not get general recommendations: {e}")
            print()
        
        # Task-specific recommendations
        print("🎯 Getting task-specific recommendations...")
        try:
            multi_rec = self.get_multi_model_recommendation()
            if multi_rec:
                print("🎯 Task-Specific Recommendations:")
                for task, model in multi_rec.items():
                    if task != "general":  # Skip general as we already showed it
                        print(f"   {task.replace('_', ' ').title()}: {model}")
                print()
        except Exception as e:
            print(f"⚠️  Could not get task-specific recommendations: {e}")
            print()
        
        # Resource warnings
        try:
            if resources['ram_usage_percent'] > 80:
                print("⚠️  High RAM usage detected. Consider closing other applications.")
            if resources['available_ram_gb'] < 4:
                print("⚠️  Low available RAM. Recommended models may run slowly.")
            if not self.hardware_info['has_gpu']:
                print("💡 No GPU detected. Models will run on CPU (slower but functional).")
        except:
            pass
        print()
    
    def list_installed_models(self):
        """List currently installed and running Ollama models."""
        print("📋 Ollama Model Status")
        print("=" * 25)
        
        try:
            # List installed models
            print("📦 Installed Models:")
            result = subprocess.run(['docker-compose', 'exec', '-T', 'ollama', 'ollama', 'ls'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                if result.stdout.strip():
                    print(result.stdout)
                else:
                    print("No models installed")
            else:
                print("❌ Could not list installed models")
                print("Make sure Ollama service is running: docker-compose up -d ollama")
                return
            
            print()
            
            # List running models
            print("🏃 Running Models:")
            result = subprocess.run(['docker-compose', 'exec', '-T', 'ollama', 'ollama', 'ps'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                if result.stdout.strip():
                    print(result.stdout)
                else:
                    print("No models currently running")
            else:
                print("❌ Could not list running models")
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ Error accessing Ollama: {e}")
            print("Make sure Docker and Ollama service are running:")
    
    def pull_model(self, model_id: str) -> bool:
        """Pull a specific model with resource validation."""
        print(f"📥 Pulling model: {model_id}")
        
        # Check if model exists in config
        all_models = {**self.models_config["llm_models"], **self.models_config["embedding_models"]}
        if model_id not in all_models:
            print(f"⚠️  Model {model_id} not found in configuration")
            print("Available models:")
            for mid in all_models.keys():
                print(f"  - {mid}")
            return False
        
        # Check hardware requirements with current available resources
        model_config = all_models[model_id]
        available_resources = self.get_available_resources()
        
        if available_resources["available_ram_gb"] < model_config["min_ram_gb"]:
            print(f"⚠️  Warning: Model requires {model_config['min_ram_gb']}GB RAM")
            print(f"   Available RAM: {available_resources['available_ram_gb']:.1f}GB")
            print(f"   RAM usage: {available_resources['ram_usage_percent']:.1f}%")
            
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return False
        
        # Show performance estimate
        perf = self.estimate_performance(model_id)
        if "estimated_tokens_per_sec" in perf:
            print(f"📊 Expected performance: ~{perf['estimated_tokens_per_sec']} tokens/sec")
        
        try:
            result = subprocess.run(['docker-compose', 'exec', '-T', 'ollama', 'ollama', 'pull', model_id], 
                                  timeout=600)  # 10 minute timeout for large models
            if result.returncode == 0:
                print(f"✅ Successfully pulled {model_id}")
                return True
            else:
                print(f"❌ Failed to pull {model_id}")
                return False
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout pulling {model_id} (10 minutes)")
            return False
        except Exception as e:
            print(f"❌ Error pulling {model_id}: {e}")
            return False
    
    def pull_with_fallback(self, model_id: str) -> bool:
        """Try to pull model with automatic fallback to smaller variant."""
        if self.pull_model(model_id):
            return True
        
        print(f"🔄 Attempting fallback for {model_id}...")
        
        # Get smaller alternative from same family
        fallback = self.get_smaller_alternative(model_id)
        if fallback:
            print(f"⚠️  Falling back to smaller model: {fallback}")
            return self.pull_model(fallback)
        
        print("❌ No suitable fallback found")
        return False
    
    def get_smaller_alternative(self, model_id: str) -> Optional[str]:
        """Find a smaller model from the same family."""
        # Extract model family (e.g., "qwen3" from "qwen3:4b")
        if ":" in model_id:
            family = model_id.split(":")[0]
        else:
            return None
        
        # Find smaller models from same family
        alternatives = []
        current_model = self.models_config["llm_models"].get(model_id)
        if not current_model:
            return None
        
        current_ram = current_model["min_ram_gb"]
        available_ram = self.get_available_resources()["available_ram_gb"]
        
        for alt_id, alt_config in self.models_config["llm_models"].items():
            if (alt_id.startswith(family) and 
                alt_config["min_ram_gb"] < current_ram and
                alt_config["min_ram_gb"] <= available_ram * 0.8):
                alternatives.append((alt_id, alt_config["min_ram_gb"]))
        
        # Return the largest alternative that fits
        if alternatives:
            alternatives.sort(key=lambda x: x[1], reverse=True)
            return alternatives[0][0]
        
        return None
    
    def interactive_setup(self):
        """Interactive model selection with automatic/manual choice."""
        print("🎯 Interactive Model Setup")
        print("=" * 30)
        
        # Show hardware info first
        self.show_hardware_info()
        
        # Ask user preference for selection method
        print("How would you like to select models?")
        print()
        print("🤖 1. Automatic (Recommended) - Let the system choose optimal models based on your hardware")
        print("👤 2. Manual - Choose models yourself from available options")
        print()
        
        while True:
            try:
                selection_method = input("Choose selection method (1-2): ").strip()
                if selection_method in ['1', '2']:
                    break
                else:
                    print("Please enter 1 or 2.")
            except KeyboardInterrupt:
                print("\nSetup cancelled.")
                return
        
        if selection_method == '1':
            # Automatic selection
            print("\n🤖 Automatic Model Selection")
            print("=" * 30)
            
            llm_rec, embed_rec = self.get_recommended_models()
            
            # Show automatic selection details
            llm_config = self.models_config["llm_models"][llm_rec]
            embed_config = self.models_config["embedding_models"][embed_rec]
            
            print("🎯 Automatically selected models based on your hardware:")
            print(f"   LLM: {llm_rec}")
            print(f"        {llm_config['description']}")
            print(f"        Size: {llm_config['size']}, Provider: {llm_config['provider']}")
            print(f"   Embedding: {embed_rec}")
            print(f"        {embed_config['description']}")
            print(f"        Size: {embed_config['size']}, Provider: {embed_config['provider']}")
            print()
            
            # Confirm automatic selection
            while True:
                try:
                    confirm = input("Proceed with automatic selection? (Y/n): ").strip().lower()
                    if confirm in ['', 'y', 'yes']:
                        selected_llm = llm_rec
                        selected_embedding = embed_rec
                        break
                    elif confirm in ['n', 'no']:
                        print("Switching to manual selection...")
                        selection_method = '2'
                        break
                    else:
                        print("Please enter Y or n.")
                except KeyboardInterrupt:
                    print("\nSetup cancelled.")
                    return
        
        if selection_method == '2':
            # Manual selection
            print("\n👤 Manual Model Selection")
            print("=" * 25)
            
            # Get recommendations for reference
            llm_rec, embed_rec = self.get_recommended_models()
            
            # LLM Selection
            print("Available LLM Models:")
            llm_models = list(self.models_config["llm_models"].keys())
            for i, model_id in enumerate(llm_models, 1):
                config = self.models_config["llm_models"][model_id]
                rec_marker = " ⭐ (RECOMMENDED FOR YOUR HARDWARE)" if model_id == llm_rec else ""
                recommended_marker = " ⭐" if config.get('recommended', False) else ""
                print(f"{i:2d}. {model_id}{rec_marker}{recommended_marker}")
                print(f"     {config['description']}")
                print(f"     Size: {config['size']}, Speed: {config['speed']}, Quality: {config['quality']}")
                print(f"     Min RAM: {config['min_ram_gb']}GB, Provider: {config['provider']}")
                print()
            
            while True:
                try:
                    choice = input(f"Choose LLM model (1-{len(llm_models)}) or 'a' for automatic: ").strip()
                    if choice.lower() == 'a':
                        selected_llm = llm_rec
                        print(f"Selected: {llm_rec} (automatic recommendation)")
                        break
                    else:
                        idx = int(choice) - 1
                        if 0 <= idx < len(llm_models):
                            selected_llm = llm_models[idx]
                            print(f"Selected: {selected_llm}")
                            break
                        else:
                            print("Invalid selection. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number or 'a'.")
                except KeyboardInterrupt:
                    print("\nSetup cancelled.")
                    return
            
            # Embedding Selection
            print("\nAvailable Embedding Models:")
            embed_models = list(self.models_config["embedding_models"].keys())
            for i, model_id in enumerate(embed_models, 1):
                config = self.models_config["embedding_models"][model_id]
                rec_marker = " ⭐ (RECOMMENDED FOR YOUR HARDWARE)" if model_id == embed_rec else ""
                recommended_marker = " ⭐" if config.get('recommended', False) else ""
                print(f"{i}. {model_id}{rec_marker}{recommended_marker}")
                print(f"   {config['description']}")
                print(f"   Size: {config['size']}, Dimensions: {config['dimensions']}, Quality: {config['quality']}")
                print(f"   Min RAM: {config['min_ram_gb']}GB, Provider: {config['provider']}")
                print()
            
            while True:
                try:
                    choice = input(f"Choose embedding model (1-{len(embed_models)}) or 'a' for automatic: ").strip()
                    if choice.lower() == 'a':
                        selected_embedding = embed_rec
                        print(f"Selected: {embed_rec} (automatic recommendation)")
                        break
                    else:
                        idx = int(choice) - 1
                        if 0 <= idx < len(embed_models):
                            selected_embedding = embed_models[idx]
                            print(f"Selected: {selected_embedding}")
                            break
                        else:
                            print("Invalid selection. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number or 'a'.")
                except KeyboardInterrupt:
                    print("\nSetup cancelled.")
                    return
        
        # Pull selected models
        print(f"\n📥 Pulling selected models...")
        print(f"LLM: {selected_llm}")
        print(f"Embedding: {selected_embedding}")
        print()
        
        success = True
        success &= self.pull_model(selected_llm)
        success &= self.pull_model(selected_embedding)
        
        if success:
            # Update .env file
            self.update_env_file(selected_llm, selected_embedding)
            print("\n✅ Model setup complete!")
            print(f"✅ LLM Model: {selected_llm}")
            print(f"✅ Embedding Model: {selected_embedding}")
            print(f"✅ Configuration saved to .env file")
        else:
            print("\n❌ Some models failed to install")
            print("Please check your internet connection and try again.")
    
    def update_env_file(self, llm_model: str, embedding_model: str):
        """Update .env file with selected models."""
        env_file = Path(".env")
        if not env_file.exists():
            print("⚠️  .env file not found. Please run ./scripts/setup.sh first.")
            return
        
        # Read current .env
        with open(env_file, 'r') as f:
            lines = f.readlines()
        
        # Update model lines
        updated_lines = []
        llm_updated = False
        embedding_updated = False
        
        for line in lines:
            if line.startswith('OLLAMA_MODEL='):
                updated_lines.append(f'OLLAMA_MODEL={llm_model}\n')
                llm_updated = True
            elif line.startswith('OLLAMA_EMBEDDING_MODEL='):
                updated_lines.append(f'OLLAMA_EMBEDDING_MODEL={embedding_model}\n')
                embedding_updated = True
            else:
                updated_lines.append(line)
        
        # Add lines if they don't exist
        if not llm_updated:
            updated_lines.append(f'OLLAMA_MODEL={llm_model}\n')
        if not embedding_updated:
            updated_lines.append(f'OLLAMA_EMBEDDING_MODEL={embedding_model}\n')
        
        # Write back to .env
        with open(env_file, 'w') as f:
            f.writelines(updated_lines)
        
        print("📝 Updated .env file with selected models")
    
    def add_model(self, model_data: Dict):
        """Add a new model to the configuration."""
        model_type = model_data.get("type", "llm")  # "llm" or "embedding"
        model_id = model_data["id"]
        
        if model_type == "llm":
            self.models_config["llm_models"][model_id] = {
                "name": model_data["name"],
                "size": model_data["size"],
                "speed": model_data.get("speed", "medium"),
                "quality": model_data.get("quality", "good"),
                "min_ram_gb": model_data.get("min_ram_gb", 4),
                "min_vram_gb": model_data.get("min_vram_gb", 0),
                "description": model_data["description"],
                "provider": model_data.get("provider", "Unknown"),
                "use_cases": model_data.get("use_cases", ["general"]),
                "languages": model_data.get("languages", ["en"]),
                "context_length": model_data.get("context_length", 4096),
                "recommended": model_data.get("recommended", False)
            }
        else:  # embedding
            self.models_config["embedding_models"][model_id] = {
                "name": model_data["name"],
                "size": model_data["size"],
                "dimensions": model_data.get("dimensions", 384),
                "quality": model_data.get("quality", "good"),
                "min_ram_gb": model_data.get("min_ram_gb", 1),
                "description": model_data["description"],
                "provider": model_data.get("provider", "Unknown"),
                "use_cases": model_data.get("use_cases", ["general"]),
                "max_sequence_length": model_data.get("max_sequence_length", 512),
                "recommended": model_data.get("recommended", False)
            }
        
        # Save updated configuration
        self.save_config()
        print(f"✅ Added {model_type} model: {model_id}")
    
    def remove_model(self, model_id: str):
        """Remove a model from the configuration."""
        removed = False
        
        if model_id in self.models_config["llm_models"]:
            del self.models_config["llm_models"][model_id]
            removed = True
        
        if model_id in self.models_config["embedding_models"]:
            del self.models_config["embedding_models"][model_id]
            removed = True
        
        if removed:
            self.save_config()
            print(f"✅ Removed model from configuration: {model_id}")
            print("Note: This only removes from config. To uninstall from Ollama, run:")
            print(f"docker-compose exec ollama ollama rm {model_id}")
        else:
            print(f"❌ Model not found in configuration: {model_id}")
    
    def uninstall_model(self, model_id: str):
        """Uninstall a model from Ollama."""
        print(f"🗑️  Uninstalling model: {model_id}")
        try:
            result = subprocess.run(['docker-compose', 'exec', '-T', 'ollama', 'ollama', 'rm', model_id], 
                                  timeout=30)
            if result.returncode == 0:
                print(f"✅ Successfully uninstalled {model_id}")
                return True
            else:
                print(f"❌ Failed to uninstall {model_id}")
                return False
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout uninstalling {model_id}")
            return False
        except Exception as e:
            print(f"❌ Error uninstalling {model_id}: {e}")
            return False
    
    def save_config(self):
        """Save the current configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.models_config, f, indent=2)
    
    def show_categories(self):
        """Show available model categories."""
        print("📂 Model Categories")
        print("=" * 20)
        for category, models in self.models_config["model_categories"].items():
            print(f"{category}: {', '.join(models)}")
        print()


    def automatic_setup(self):
        """Automatic model setup based on hardware detection."""
        print("🤖 Automatic Model Setup")
        print("=" * 25)
        
        self.show_hardware_info()
        
        llm_rec, embed_rec = self.get_recommended_models()
        
        # Show what will be installed
        llm_config = self.models_config["llm_models"][llm_rec]
        embed_config = self.models_config["embedding_models"][embed_rec]
        
        print("🎯 Automatically selected models:")
        print(f"   LLM: {llm_rec}")
        print(f"        {llm_config['description']}")
        print(f"        Size: {llm_config['size']}, Provider: {llm_config['provider']}")
        print(f"   Embedding: {embed_rec}")
        print(f"        {embed_config['description']}")
        print(f"        Size: {embed_config['size']}, Provider: {embed_config['provider']}")
        print()
        
        # Pull models automatically
        print("📥 Installing recommended models...")
        success = True
        success &= self.pull_model(llm_rec)
        success &= self.pull_model(embed_rec)
        
        if success:
            self.update_env_file(llm_rec, embed_rec)
            print("\n✅ Automatic setup complete!")
            print(f"✅ LLM Model: {llm_rec}")
            print(f"✅ Embedding Model: {embed_rec}")
        else:
            print("\n❌ Automatic setup failed")
            print("You can try manual setup with: python3 scripts/model-manager.py interactive")


def main():
    parser = argparse.ArgumentParser(description="Dynamic Model Manager for Ollama")
    parser.add_argument("command", nargs="?", default="interactive", 
                       choices=["interactive", "auto", "list", "hardware", "installed", "status", "running", 
                               "pull", "recommend", "benchmark", "add", "remove", "uninstall", "categories"],
                       help="Command to execute")
    parser.add_argument("--model", help="Model ID for pull/add/remove commands")
    parser.add_argument("--category", help="Filter models by category")
    parser.add_argument("--task", help="Task type for recommendations (reasoning, code, multilingual, etc.)")
    parser.add_argument("--context", type=int, default=8192, help="Minimum context length required")
    parser.add_argument("--speed", action="store_true", help="Prioritize speed over quality")
    parser.add_argument("--config", default="config/models.json", help="Path to models configuration file")
    
    args = parser.parse_args()
    
    manager = ModelManager(args.config)
    
    if args.command == "interactive":
        manager.interactive_setup()
    elif args.command == "auto":
        manager.automatic_setup()
    elif args.command == "list":
        manager.list_available_models(args.category)
        print()
        manager.list_installed_models()
    elif args.command == "hardware":
        manager.show_hardware_info()
    elif args.command in ["installed", "status"]:
        manager.list_installed_models()
    elif args.command == "running":
        print("🏃 Currently Running Models")
        print("=" * 30)
        try:
            result = subprocess.run(['docker-compose', 'exec', '-T', 'ollama', 'ollama', 'ps'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                if result.stdout.strip():
                    print(result.stdout)
                else:
                    print("No models currently running")
            else:
                print("❌ Could not list running models")
        except Exception as e:
            print(f"❌ Error: {e}")
    elif args.command == "recommend":
        if args.task:
            print(f"🎯 Recommendations for {args.task} tasks:")
            recommendations = manager.recommend_for_task(args.task, args.context)
            for rec in recommendations:
                print(f"   {rec['model_id']} - {rec['description']}")
                if 'performance' in rec and 'estimated_tokens_per_sec' in rec['performance']:
                    print(f"      Performance: ~{rec['performance']['estimated_tokens_per_sec']} tokens/sec")
                print()
        else:
            print("🎯 Multi-Task Recommendations:")
            multi_rec = manager.get_multi_model_recommendation()
            for task, model in multi_rec.items():
                print(f"   {task.title()}: {model}")
    elif args.command == "benchmark":
        if args.model:
            perf = manager.estimate_performance(args.model)
            print(f"📊 Performance Estimate for {args.model}:")
            for key, value in perf.items():
                print(f"   {key.replace('_', ' ').title()}: {value}")
        else:
            print("❌ --model required for benchmark command")
    elif args.command == "pull":
        if not args.model:
            print("❌ --model required for pull command")
            sys.exit(1)
        manager.pull_with_fallback(args.model)
    elif args.command == "categories":
        manager.show_categories()
    elif args.command == "add":
        print("Add model functionality - implement based on your needs")
        print("You can directly edit config/models.json to add new models")
    elif args.command == "remove":
        if not args.model:
            print("❌ --model required for remove command")
            print("Available commands:")
            print("  python3 scripts/model-manager.py remove --model <model_name>  # Remove from config")
            print("  python3 scripts/model-manager.py uninstall --model <model_name>  # Uninstall from Ollama")
            sys.exit(1)
        manager.remove_model(args.model)
    elif args.command == "uninstall":
        if not args.model:
            print("❌ --model required for uninstall command")
            sys.exit(1)
        manager.uninstall_model(args.model)


if __name__ == "__main__":
    main()