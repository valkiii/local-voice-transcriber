# 🎤 Local Voice Transcriber

A beautiful, minimalistic desktop application for recording voice and transcribing it using OpenAI Whisper - completely offline and privacy-focused.

![Voice Transcriber Demo](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-blue)
![Python Version](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-purple)

## ✨ Features

- **🎤 Animated Voice Ball**: Click the purple neon ball to start/stop recording
- **🌊 Smooth Animations**: Ball pulses and expands gently when you speak
- **🖤 Dark Neon Theme**: Beautiful purple neon on dark background
- **🔇 Offline Transcription**: Uses OpenAI Whisper locally (no internet required)
- **📝 Selectable Text**: Copy and paste transcriptions easily
- **📅 Auto-Naming**: Saves transcriptions as `YYYYMMDD_HHMMSS.txt`
- **🎯 Minimalistic**: Clean UI with no visible boxes or clutter
- **🔒 Privacy-First**: Everything runs locally, no data sent to servers

## 🖼️ Screenshots

The app features a modern dark theme with a purple neon aesthetic:
- Purple voice ball that pulses when you speak
- Seamless transcription text that blends into the dark background
- Minimal controls for maximum focus

## 🚀 Quick Start

### For Windows Users (No Python Required)
1. Download `VoiceTranscriber_Smooth.zip` from releases
2. Extract the folder
3. Double-click `SETUP.bat` (one-time setup, ~5 minutes)
4. Double-click `START_APP.bat` to run the app

### For Developers (Mac/Linux)
```bash
git clone https://github.com/yourusername/local-voice-transcriber.git
cd local-voice-transcriber
./run_local.sh
```

## 🎯 How to Use

1. **Select Microphone**: Choose your microphone from the dropdown
2. **Choose Output Folder**: Click the 📁 button to select where to save transcriptions
3. **Record**: Click the purple voice ball to start recording
4. **Speak**: The ball will pulse and glow as you speak
5. **Stop**: Click the ball again to stop and transcribe
6. **Copy Text**: Select and copy the transcribed text
7. **Auto-Save**: Transcription automatically saves as `YYYYMMDD_HHMMSS.txt`

## 🛠️ Technology Stack

- **GUI Framework**: PyQt6 (modern, cross-platform interface)
- **Audio Recording**: sounddevice (real-time audio capture)
- **Speech Recognition**: OpenAI Whisper (state-of-the-art offline transcription)
- **Packaging**: PyInstaller (creates standalone executables)

## 📦 Dependencies

```
PyQt6>=6.7.0
sounddevice>=0.5.0
numpy>=1.24.0
openai-whisper>=20240930
scipy>=1.11.0
pyinstaller>=6.0.0
```

## 🔧 Advanced Setup

### Manual Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python main.py
```

### Building Windows Executable
```bash
# After setup, build standalone .exe
pyinstaller --onefile --windowed --name=VoiceTranscriber main.py
```

## 🎨 Design Philosophy

This app embraces minimalism and focuses on the core functionality:
- **One-click recording** with intuitive visual feedback
- **Distraction-free interface** with no unnecessary elements
- **Beautiful animations** that enhance rather than distract
- **Seamless text integration** that feels natural
- **Privacy-focused** with complete offline operation

## 🔒 Privacy & Security

- **100% Offline**: No internet connection required for transcription
- **Local Processing**: All audio processing happens on your device
- **No Data Collection**: No telemetry, analytics, or data transmission
- **Open Source**: Full transparency of all code and functionality

## 🌟 Why This App?

Unlike cloud-based transcription services, this app:
- Works without internet connection
- Keeps your voice data completely private
- Provides beautiful, modern interface
- Offers one-click setup for non-technical users
- Creates standalone executables for easy sharing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Issues & Support

If you encounter any issues or have feature requests, please [open an issue](https://github.com/yourusername/local-voice-transcriber/issues).

---

**Made with 💜 for privacy-conscious voice transcription**