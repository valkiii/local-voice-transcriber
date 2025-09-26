#!/bin/bash
echo "🎭 Installing Speaker Diarization Dependencies..."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install diarization dependencies
echo "Installing pyannote.audio and related packages..."
pip install pyannote.audio>=3.1.0
pip install pyannote.core>=5.0.0
pip install speechbrain>=0.5.0
pip install torchaudio>=2.1.0

echo ""
echo "🔍 Checking diarization support..."

# Check if diarization is working
python3 -c "
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment
    import torchaudio
    print('✅ Diarization dependencies: Available')
    print('🎭 Speaker diarization is now ready!')
    print('')
    print('📝 Note: First time using diarization will require accepting terms:')
    print('   1. Visit https://hf.co/pyannote/speaker-diarization-3.1')
    print('   2. Accept the terms and conditions')
    print('   3. Create a HuggingFace token at https://hf.co/settings/tokens')
    print('   4. Run: huggingface-cli login')
except ImportError as e:
    print('❌ Import error:', e)
    print('⚠️ Diarization may not work properly')
except Exception as e:
    print('❌ Error:', e)
"

echo ""
echo "✅ Diarization dependencies installed!"
echo "🎭 You can now use speaker diarization features:"
echo "   • Automatic speaker detection"
echo "   • Editable speaker names"
echo "   • Formatted transcription with speaker labels"
echo ""
echo "💡 Enable diarization in the app to separate speakers automatically!"