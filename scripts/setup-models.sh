#!/bin/bash

# Model setup script for Ollama - Fully JSON-driven configuration
set -e

echo "🤖 Ollama Model Setup (JSON-Driven)"
echo "==================================="

# Check if configuration exists
if [[ -f "config/models.json" ]]; then
    echo "📋 Using models configuration from config/models.json"
    if [[ -f "scripts/model-manager.py" ]] && command -v python3 &> /dev/null; then
        echo "🐍 Advanced Python model manager available"
        echo "   Use: python3 scripts/model-manager.py interactive"
    elif command -v python3 &> /dev/null; then
        echo "� Python asvailable for JSON parsing"
    elif command -v jq &> /dev/null; then
        echo "� jq avlailable for JSON parsing"
    else
        echo "💡 Install Python3 or jq for enhanced features"
    fi
    echo ""
else
    echo "⚠️  config/models.json not found - please ensure it exists"
    echo "   Expected location: config/models.json"
    exit 1
fi

# Function to show available models (reads from models.json)
show_models() {
    if [[ -f "scripts/model-manager.py" ]] && command -v python3 &> /dev/null; then
        echo "📋 Using Python model manager for detailed model information..."
        python3 scripts/model-manager.py list
    elif [[ -f "config/models.json" ]] && command -v python3 &> /dev/null; then
        echo "📋 Reading models from config/models.json..."
        echo ""
        
        python3 -c "
import json
try:
    with open('config/models.json', 'r') as f:
        data = json.load(f)
    
    print('🧠 Available LLM Models:')
    for model_id, config in data['llm_models'].items():
        rec = ' ⭐' if config.get('recommended', False) else ''
        print(f'  {model_id} - {config[\"description\"]} ({config[\"size\"]}){rec}')
    
    print()
    print('🔍 Available Embedding Models:')
    for model_id, config in data['embedding_models'].items():
        rec = ' ⭐' if config.get('recommended', False) else ''
        print(f'  {model_id} - {config[\"description\"]} ({config[\"size\"]}){rec}')
    
    print()
    print('💡 For hardware recommendations: python3 scripts/model-manager.py hardware')
except Exception as e:
    print('Error reading models.json:', e)
"
    elif command -v jq &> /dev/null; then
        echo "📋 Reading models from config/models.json with jq..."
        echo ""
        echo "🧠 Available LLM Models:"
        jq -r '.llm_models | to_entries[] | "  \(.key) - \(.value.description) (\(.value.size))" + (if .value.recommended then " ⭐" else "" end)' config/models.json
        echo ""
        echo "🔍 Available Embedding Models:"
        jq -r '.embedding_models | to_entries[] | "  \(.key) - \(.value.description) (\(.value.size))" + (if .value.recommended then " ⭐" else "" end)' config/models.json
        echo ""
        echo "💡 For hardware recommendations: python3 scripts/model-manager.py hardware"
    else
        echo "�  Install Python3 or jq to read models from config/models.json"
        echo "   Or check config/models.json manually"
        echo ""
        echo "💡 Quick setup options:"
        echo "   ./scripts/setup-models.sh recommended  # Use hardware-optimized models"
        echo "   python3 scripts/model-manager.py interactive  # Full interactive setup"
    fi
}

# Function to pull a model
pull_model() {
    local model=$1
    echo "📥 Pulling model: $model"
    if docker-compose exec -T ollama ollama pull "$model"; then
        echo "✅ Successfully pulled $model"
    else
        echo "❌ Failed to pull $model"
        return 1
    fi
}

# Function to list installed models
list_installed() {
    echo "📋 Currently installed models:"
    if docker-compose exec -T ollama ollama ls 2>/dev/null; then
        echo ""
        echo "🏃 Currently running models:"
        docker-compose exec -T ollama ollama ps 2>/dev/null || echo "No models currently running"
    else
        echo "❌ Could not list models (is Ollama service running?)"
        echo "💡 Try: docker-compose up -d ollama"
    fi
}

# Function to get recommended models from JSON
get_recommended_models() {
    if command -v python3 &> /dev/null; then
        python3 -c "
import json
try:
    with open('config/models.json', 'r') as f:
        data = json.load(f)
    
    # Find recommended LLM
    rec_llm = None
    for model_id, config in data['llm_models'].items():
        if config.get('recommended', False):
            rec_llm = model_id
            break
    
    # Find recommended embedding
    rec_embed = None
    for model_id, config in data['embedding_models'].items():
        if config.get('recommended', False):
            rec_embed = model_id
            break
    
    if rec_llm and rec_embed:
        print(f'{rec_llm}|{rec_embed}')
    else:
        print('gemma3n:e4b|nomic-embed-text')  # fallback
except:
    print('gemma3n:e4b|nomic-embed-text')  # fallback
"
    elif command -v jq &> /dev/null; then
        local rec_llm=$(jq -r '.llm_models | to_entries[] | select(.value.recommended == true) | .key' config/models.json | head -1)
        local rec_embed=$(jq -r '.embedding_models | to_entries[] | select(.value.recommended == true) | .key' config/models.json | head -1)
        echo "${rec_llm:-gemma3n:e4b}|${rec_embed:-nomic-embed-text}"
    else
        echo "gemma3n:e4b|nomic-embed-text"  # fallback
    fi
}

# Function to get hardware profile recommendations
get_hardware_recommendations() {
    local profile=$1
    if command -v python3 &> /dev/null; then
        python3 -c "
import json
try:
    with open('config/models.json', 'r') as f:
        data = json.load(f)
    
    profile = data['hardware_profiles'].get('$profile', {})
    llm = profile.get('recommended_llm', 'gemma3n:e4b')
    embed = profile.get('recommended_embedding', 'nomic-embed-text')
    print(f'{llm}|{embed}')
except:
    print('gemma3n:e4b|nomic-embed-text')
"
    elif command -v jq &> /dev/null; then
        local llm=$(jq -r ".hardware_profiles.$profile.recommended_llm" config/models.json 2>/dev/null || echo "gemma3n:e4b")
        local embed=$(jq -r ".hardware_profiles.$profile.recommended_embedding" config/models.json 2>/dev/null || echo "nomic-embed-text")
        echo "$llm|$embed"
    else
        echo "gemma3n:e4b|nomic-embed-text"
    fi
}

# Function to interactive model selection
interactive_setup() {
    if [[ -f "scripts/model-manager.py" ]] && command -v python3 &> /dev/null; then
        echo "🎯 Using Advanced Python Model Manager"
        echo "======================================"
        python3 scripts/model-manager.py interactive
    else
        echo "🎯 JSON-Driven Interactive Model Setup"
        echo "======================================"
        
        # First, ask user preference for selection method
        echo "How would you like to select models?"
        echo ""
        echo "🤖 1. Automatic (Recommended) - Let the system choose optimal models based on your hardware"
        echo "👤 2. Manual - Choose models yourself from available options"
        echo ""
        read -p "Choose selection method (1-2): " selection_method
        
        if [[ "$selection_method" == "1" ]]; then
            # Automatic hardware-based selection
            echo ""
            echo "🔍 Detecting your system capabilities..."
            
            if command -v python3 &> /dev/null || command -v jq &> /dev/null; then
                # Simple hardware detection in bash
                RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
                CPU_CORES=$(nproc)
                HAS_GPU=false
                
                if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
                    HAS_GPU=true
                    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
                    GPU_MEM_GB=$((GPU_MEM / 1024))
                fi
                
                echo "💻 Detected Hardware:"
                echo "   RAM: ${RAM_GB}GB"
                echo "   CPU Cores: $CPU_CORES"
                echo "   GPU: $(if $HAS_GPU; then echo "Yes (${GPU_MEM_GB}GB VRAM)"; else echo "No"; fi)"
                echo ""
                
                # Determine optimal profile
                if [[ $RAM_GB -lt 4 ]]; then
                    PROFILE="low_resource"
                    echo "📊 Recommendation: Low Resource Profile"
                elif [[ $RAM_GB -lt 8 ]]; then
                    PROFILE="medium_resource"
                    echo "📊 Recommendation: Medium Resource Profile"
                elif $HAS_GPU && [[ ${GPU_MEM_GB:-0} -ge 4 ]]; then
                    PROFILE="gpu_optimized"
                    echo "📊 Recommendation: GPU Optimized Profile"
                else
                    PROFILE="high_resource"
                    echo "📊 Recommendation: High Resource Profile"
                fi
                
                # Get recommended models for the profile
                PROFILE_MODELS=$(get_hardware_recommendations "$PROFILE")
                IFS='|' read -r AUTO_LLM AUTO_EMBED <<< "$PROFILE_MODELS"
                
                echo "🎯 Automatically selected models:"
                echo "   LLM: $AUTO_LLM"
                echo "   Embedding: $AUTO_EMBED"
                echo ""
                
                # Show model details if possible
                if command -v python3 &> /dev/null; then
                    python3 -c "
import json
try:
    with open('config/models.json', 'r') as f:
        data = json.load(f)
    
    llm_config = data['llm_models'].get('$AUTO_LLM', {})
    embed_config = data['embedding_models'].get('$AUTO_EMBED', {})
    
    print('📋 Model Details:')
    print(f'   LLM: {llm_config.get(\"description\", \"N/A\")} ({llm_config.get(\"size\", \"N/A\")})')
    print(f'   Embedding: {embed_config.get(\"description\", \"N/A\")} ({embed_config.get(\"size\", \"N/A\")})')
    print()
except:
    pass
"
                fi
                
                read -p "Proceed with automatic selection? (Y/n): " confirm
                if [[ "$confirm" =~ ^[Nn]$ ]]; then
                    echo "Switching to manual selection..."
                    selection_method="2"
                else
                    LLM_MODEL="$AUTO_LLM"
                    EMBEDDING_MODEL="$AUTO_EMBED"
                fi
            else
                echo "⚠️  Cannot detect hardware without Python3 or jq"
                echo "Switching to manual selection..."
                selection_method="2"
            fi
        fi
        
        if [[ "$selection_method" == "2" ]]; then
            # Manual selection
            echo ""
            echo "👤 Manual Model Selection"
            echo "========================"
            
            if command -v python3 &> /dev/null || command -v jq &> /dev/null; then
                echo "Available hardware profiles from config/models.json:"
                
                # Get hardware profile recommendations
                LOW_MODELS=$(get_hardware_recommendations "low_resource")
                MED_MODELS=$(get_hardware_recommendations "medium_resource")
                HIGH_MODELS=$(get_hardware_recommendations "high_resource")
                GPU_MODELS=$(get_hardware_recommendations "gpu_optimized")
                
                IFS='|' read -r LOW_LLM LOW_EMBED <<< "$LOW_MODELS"
                IFS='|' read -r MED_LLM MED_EMBED <<< "$MED_MODELS"
                IFS='|' read -r HIGH_LLM HIGH_EMBED <<< "$HIGH_MODELS"
                IFS='|' read -r GPU_LLM GPU_EMBED <<< "$GPU_MODELS"
                
                echo "1. Low resource (< 4GB RAM): $LOW_LLM + $LOW_EMBED"
                echo "2. Medium resource (4-8GB RAM): $MED_LLM + $MED_EMBED"
                echo "3. High resource (8GB+ RAM): $HIGH_LLM + $HIGH_EMBED"
                echo "4. GPU optimized (4GB+ VRAM): $GPU_LLM + $GPU_EMBED"
                echo "5. Custom - Choose individual models"
                echo ""
                
                read -p "Choose option (1-5): " choice
            
                case $choice in
                    1)
                        LLM_MODEL="$LOW_LLM"
                        EMBEDDING_MODEL="$LOW_EMBED"
                        echo "Selected: Low resource profile"
                        ;;
                    2)
                        LLM_MODEL="$MED_LLM"
                        EMBEDDING_MODEL="$MED_EMBED"
                        echo "Selected: Medium resource profile"
                        ;;
                    3)
                        LLM_MODEL="$HIGH_LLM"
                        EMBEDDING_MODEL="$HIGH_EMBED"
                        echo "Selected: High resource profile"
                        ;;
                    4)
                        LLM_MODEL="$GPU_LLM"
                        EMBEDDING_MODEL="$GPU_EMBED"
                        echo "Selected: GPU optimized profile"
                        ;;
                    5)
                        echo ""
                        echo "🧠 Available LLM models:"
                        if command -v python3 &> /dev/null; then
                            python3 -c "
import json
with open('config/models.json', 'r') as f:
    data = json.load(f)
for i, (model_id, config) in enumerate(data['llm_models'].items(), 1):
    rec = ' ⭐' if config.get('recommended', False) else ''
    print(f'  {i}. {model_id} - {config[\"description\"]} ({config[\"size\"]}){rec}')
"
                        elif command -v jq &> /dev/null; then
                            jq -r '.llm_models | to_entries[] | "  \(.key) - \(.value.description) (\(.value.size))" + (if .value.recommended then " ⭐" else "" end)' config/models.json
                        fi
                        echo ""
                        read -p "Enter LLM model name: " LLM_MODEL
                        
                        echo ""
                        echo "🔍 Available embedding models:"
                        if command -v python3 &> /dev/null; then
                            python3 -c "
import json
with open('config/models.json', 'r') as f:
    data = json.load(f)
for i, (model_id, config) in enumerate(data['embedding_models'].items(), 1):
    rec = ' ⭐' if config.get('recommended', False) else ''
    print(f'  {i}. {model_id} - {config[\"description\"]} ({config[\"size\"]}){rec}')
"
                        elif command -v jq &> /dev/null; then
                            jq -r '.embedding_models | to_entries[] | "  \(.key) - \(.value.description) (\(.value.size))" + (if .value.recommended then " ⭐" else "" end)' config/models.json
                        fi
                        echo ""
                        read -p "Enter embedding model name: " EMBEDDING_MODEL
                        ;;
                    *)
                        echo "Invalid choice, using medium resource defaults"
                        LLM_MODEL="$MED_LLM"
                        EMBEDDING_MODEL="$MED_EMBED"
                        ;;
                esac
            else
                echo "Install Python3 or jq to read from config/models.json"
                exit 1
            fi
        fi
        
        echo ""
        echo "📥 Pulling selected models..."
        echo "LLM: $LLM_MODEL"
        echo "Embedding: $EMBEDDING_MODEL"
        
        pull_model "$LLM_MODEL"
        pull_model "$EMBEDDING_MODEL"
        
        # Update .env file
        if [[ -f .env ]]; then
            echo ""
            echo "📝 Updating .env file..."
            sed -i.bak "s/^OLLAMA_MODEL=.*/OLLAMA_MODEL=$LLM_MODEL/" .env
            sed -i.bak "s/^OLLAMA_EMBEDDING_MODEL=.*/OLLAMA_EMBEDDING_MODEL=$EMBEDDING_MODEL/" .env
            echo "✅ Updated .env file with selected models"
        else
            echo "⚠️  .env file not found. Please run ./scripts/setup.sh first."
        fi
    fi
}

# Function to pull recommended models
pull_recommended() {
    if [[ -f "scripts/model-manager.py" ]] && command -v python3 &> /dev/null; then
        echo "📥 Using Python model manager for hardware-optimized recommendations..."
        python3 scripts/model-manager.py hardware
        echo ""
        read -p "Pull recommended models for your system? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python3 -c "
import sys
sys.path.append('scripts')
import importlib.util
spec = importlib.util.spec_from_file_location('model_manager', 'scripts/model-manager.py')
model_manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_manager)

manager = model_manager.ModelManager()
llm_rec, embed_rec = manager.get_recommended_models()
print(f'Pulling {llm_rec}...')
manager.pull_model(llm_rec)
print(f'Pulling {embed_rec}...')
manager.pull_model(embed_rec)
manager.update_env_file(llm_rec, embed_rec)
"
        fi
    else
        echo "📥 Reading recommended models from config/models.json..."
        
        REC_MODELS=$(get_recommended_models)
        IFS='|' read -r REC_LLM REC_EMBED <<< "$REC_MODELS"
        
        if command -v python3 &> /dev/null; then
            # Get model descriptions
            python3 -c "
import json
try:
    with open('config/models.json', 'r') as f:
        data = json.load(f)
    
    llm_config = data['llm_models'].get('$REC_LLM', {})
    embed_config = data['embedding_models'].get('$REC_EMBED', {})
    
    print(f'LLM: $REC_LLM ({llm_config.get(\"description\", \"N/A\")}, {llm_config.get(\"size\", \"N/A\")})')
    print(f'Embedding: $REC_EMBED ({embed_config.get(\"description\", \"N/A\")}, {embed_config.get(\"size\", \"N/A\")})')
except:
    print(f'LLM: $REC_LLM')
    print(f'Embedding: $REC_EMBED')
"
        else
            echo "LLM: $REC_LLM"
            echo "Embedding: $REC_EMBED"
        fi
        
        pull_model "$REC_LLM"
        pull_model "$REC_EMBED"
        
        # Update .env file
        if [[ -f .env ]]; then
            sed -i.bak "s/^OLLAMA_MODEL=.*/OLLAMA_MODEL=$REC_LLM/" .env
            sed -i.bak "s/^OLLAMA_EMBEDDING_MODEL=.*/OLLAMA_EMBEDDING_MODEL=$REC_EMBED/" .env
            echo "✅ Updated .env file with recommended models"
        fi
    fi
}

# Function to pull all models
pull_all() {
    echo "📥 Pulling all available models from config/models.json..."
    
    if command -v python3 &> /dev/null; then
        # Calculate total size
        TOTAL_SIZE=$(python3 -c "
import json, re
try:
    with open('config/models.json', 'r') as f:
        data = json.load(f)
    
    total_gb = 0
    for config in data['llm_models'].values():
        size_str = config.get('size', '0GB')
        size_match = re.search(r'(\d+\.?\d*)GB', size_str)
        if size_match:
            total_gb += float(size_match.group(1))
    
    for config in data['embedding_models'].values():
        size_str = config.get('size', '0MB')
        if 'GB' in size_str:
            size_match = re.search(r'(\d+\.?\d*)GB', size_str)
            if size_match:
                total_gb += float(size_match.group(1))
        else:
            size_match = re.search(r'(\d+)MB', size_str)
            if size_match:
                total_gb += float(size_match.group(1)) / 1024
    
    print(f'{total_gb:.1f}')
except:
    print('25')  # fallback estimate
")
        echo "This will download approximately ${TOTAL_SIZE}GB of models."
    else
        echo "This will download approximately 25GB+ of models."
    fi
    
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v python3 &> /dev/null; then
            python3 -c "
import json
try:
    with open('config/models.json', 'r') as f:
        data = json.load(f)
    
    print('Pulling LLM models...')
    for model_id in data['llm_models'].keys():
        print(f'  {model_id}')
    
    print('Pulling embedding models...')
    for model_id in data['embedding_models'].keys():
        print(f'  {model_id}')
except:
    print('Error reading models.json')
"
            # Actually pull the models
            for model in $(python3 -c "
import json
with open('config/models.json', 'r') as f:
    data = json.load(f)
for model_id in list(data['llm_models'].keys()) + list(data['embedding_models'].keys()):
    print(model_id)
"); do
                pull_model "$model"
            done
        else
            echo "Install Python3 to read models from config/models.json"
        fi
    else
        echo "Cancelled."
    fi
}

# Main script logic
case "${1:-}" in
    "list")
        show_models
        echo ""
        list_installed
        ;;
    "status"|"installed")
        list_installed
        ;;
    "interactive"|"")
        interactive_setup
        ;;
    "auto"|"automatic")
        # Automatic setup without user interaction
        echo "🤖 Automatic Model Setup"
        echo "========================"
        
        if [[ -f "scripts/model-manager.py" ]] && command -v python3 &> /dev/null; then
            python3 scripts/model-manager.py auto
        else
            echo "🔍 Detecting hardware and selecting optimal models..."
            
            # Simple hardware detection
            RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
            HAS_GPU=false
            
            if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
                HAS_GPU=true
                GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
                GPU_MEM_GB=$((GPU_MEM / 1024))
            fi
            
            echo "💻 Detected: ${RAM_GB}GB RAM, GPU: $(if $HAS_GPU; then echo "Yes"; else echo "No"; fi)"
            
            # Determine profile
            if [[ $RAM_GB -lt 4 ]]; then
                PROFILE="low_resource"
            elif [[ $RAM_GB -lt 8 ]]; then
                PROFILE="medium_resource"
            elif $HAS_GPU && [[ ${GPU_MEM_GB:-0} -ge 4 ]]; then
                PROFILE="gpu_optimized"
            else
                PROFILE="high_resource"
            fi
            
            echo "📊 Selected profile: $PROFILE"
            
            # Get models for profile
            PROFILE_MODELS=$(get_hardware_recommendations "$PROFILE")
            IFS='|' read -r AUTO_LLM AUTO_EMBED <<< "$PROFILE_MODELS"
            
            echo "🎯 Installing: $AUTO_LLM + $AUTO_EMBED"
            
            pull_model "$AUTO_LLM"
            pull_model "$AUTO_EMBED"
            
            # Update .env
            if [[ -f .env ]]; then
                sed -i.bak "s/^OLLAMA_MODEL=.*/OLLAMA_MODEL=$AUTO_LLM/" .env
                sed -i.bak "s/^OLLAMA_EMBEDDING_MODEL=.*/OLLAMA_EMBEDDING_MODEL=$AUTO_EMBED/" .env
                echo "✅ Updated .env file"
            fi
            
            echo "✅ Automatic setup complete!"
        fi
        ;;
    "recommended")
        pull_recommended
        ;;
    "all")
        pull_all
        ;;
    "pull")
        if [[ -n "${2:-}" ]]; then
            pull_model "$2"
        else
            echo "Usage: $0 pull <model_name>"
            echo "Example: $0 pull qwen3:1.7b"
        fi
        ;;
    "help"|"-h"|"--help")
        echo "Ollama Model Setup Script (JSON-Driven)"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  interactive    Interactive model selection with automatic/manual choice (default)"
        echo "  auto           Automatic model selection based on hardware detection"
        echo "  list          Show available models from config/models.json + installed models"
        echo "  status         Show installed and running models (ollama ls + ollama ps)"
        echo "  recommended   Pull hardware-optimized recommended models from config"
        echo "  all           Pull all available models from config (~25GB+ download)"
        echo "  pull <model>  Pull a specific model"
        echo "  help          Show this help message"
        echo ""
        echo "🐍 Enhanced Python Commands (if available):"
        echo "  python3 scripts/model-manager.py interactive  # Interactive with auto/manual choice"
        echo "  python3 scripts/model-manager.py auto         # Fully automatic setup"
        echo "  python3 scripts/model-manager.py hardware     # Show hardware info"
        echo "  python3 scripts/model-manager.py categories   # Show model categories"
        echo "  python3 scripts/model-manager.py list --category speed_optimized"
        echo ""
        show_models
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for usage information."
        exit 1
        ;;
esac