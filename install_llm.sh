#!/bin/bash
echo "🤖 Installing LLM Analysis Dependencies..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install LLM dependencies
echo "Installing transformers and PyTorch..."
pip install transformers>=4.40.0
pip install torch>=2.1.0
pip install accelerate>=0.20.0

echo ""
echo "🔍 Checking GPU acceleration support..."

# Check for Apple Silicon MPS support
python3 -c "
import torch
print('PyTorch version:', torch.__version__)
if torch.backends.mps.is_available():
    print('✅ M1/M2 GPU (MPS) acceleration: Available')
    print('🚀 Models will run on Apple Silicon GPU for faster inference!')
elif torch.cuda.is_available():
    print('✅ NVIDIA GPU (CUDA) acceleration: Available')
    print('⚡ Models will run on NVIDIA GPU!')
else:
    print('⚠️ GPU acceleration: Not available, using CPU')
    print('💡 Consider upgrading PyTorch for M1 GPU support')
"

echo ""
echo "📥 Pre-downloading models for faster first use..."

# Pre-download models to avoid delays during first analysis
python3 -c "
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    models = [
        'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'microsoft/phi-2',
        'Qwen/Qwen2-0.5B-Instruct'
    ]
    
    for model_name in models:
        print(f'📦 Downloading {model_name}...')
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print(f'✅ {model_name} downloaded successfully')
        except Exception as e:
            print(f'⚠️ Failed to download {model_name}: {e}')
            
    print('🎉 Model pre-download complete!')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
except Exception as e:
    print(f'❌ Error downloading models: {e}')
"

echo ""
echo "✅ LLM dependencies installed!"
echo "🚀 You can now use the AI analysis features with:"
echo "   • Gemma 2B - Fast, good quality analysis"
echo "   • Qwen 1.5B - Faster, lighter analysis"  
echo "   • Gemma 7B - Best quality (requires more memory)"
echo ""
echo "Models are now pre-loaded for instant analysis!"