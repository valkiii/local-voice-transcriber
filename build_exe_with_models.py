import os
import sys
import subprocess
import shutil
import tempfile
import whisper as whisper_lib

def download_and_prepare_models():
    """Download Whisper models and prepare them for bundling"""
    print("Preparing Whisper models for bundling...")
    
    # Create a temporary directory for models
    models_dir = os.path.join(os.getcwd(), "temp_models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Download the base model to ensure it's cached
    try:
        print("Downloading Whisper base model...")
        model = whisper_lib.load_model("base")
        
        # Find where the model was cached
        cache_dir = os.path.expanduser("~/.cache/whisper")
        
        if os.path.exists(cache_dir):
            # Copy all model files to our temp directory
            print(f"Copying models from {cache_dir} to {models_dir}")
            for file in os.listdir(cache_dir):
                if file.endswith('.pt'):
                    src = os.path.join(cache_dir, file)
                    dst = os.path.join(models_dir, file)
                    shutil.copy2(src, dst)
                    print(f"Copied: {file}")
        
        return models_dir
        
    except Exception as e:
        print(f"Error preparing models: {e}")
        return None

def build_executable():
    """Build the executable with embedded models"""
    print("Building EXE with embedded Whisper models...")
    
    # Prepare models
    models_dir = download_and_prepare_models()
    if not models_dir:
        print("Failed to prepare models. Aborting build.")
        return False
    
    # Install PyInstaller if needed
    print("Installing PyInstaller...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "pyinstaller"
        ], check=True)
    except subprocess.CalledProcessError:
        print("Failed to install PyInstaller")
        return False
    
    # Build command with model bundling
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed", 
        "--name=VoiceTranscriber",
        f"--add-data={models_dir}{os.pathsep}whisper_cache",
        "--collect-all=whisper",
        "--collect-all=torch", 
        "--collect-all=torchaudio",
        "--collect-all=sounddevice",
        "--collect-all=scipy", 
        "--collect-all=numpy",
        "--collect-all=PyQt6",
        "--hidden-import=whisper",
        "--hidden-import=torch",
        "--hidden-import=torchaudio",
        "--hidden-import=sounddevice", 
        "--hidden-import=scipy",
        "--hidden-import=scipy.signal",
        "--hidden-import=numpy",
        "--add-binary=**/sounddevice/*.dll:sounddevice" if os.name == 'nt' else "",
        "main_fixed_models.py"
    ]
    
    # Remove empty strings from command
    cmd = [c for c in cmd if c]
    
    print("Running PyInstaller with command:")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Check if executable was created
        exe_path = "dist/VoiceTranscriber.exe" if os.name == 'nt' else "dist/VoiceTranscriber"
        
        if os.path.exists(exe_path):
            size = os.path.getsize(exe_path) / (1024 * 1024)
            print(f"\n✓ SUCCESS! VoiceTranscriber executable created ({size:.1f} MB)")
            print(f"File: {exe_path}")
            print("Ready to distribute - includes embedded Whisper models!")
            
            # Cleanup temp models
            shutil.rmtree(models_dir, ignore_errors=True)
            return True
        else:
            print("Build completed but executable not found in expected location")
            print("PyInstaller output:", result.stdout)
            if result.stderr:
                print("PyInstaller errors:", result.stderr)
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"PyInstaller build failed: {e}")
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        return False
    finally:
        # Cleanup temp models
        if os.path.exists(models_dir):
            shutil.rmtree(models_dir, ignore_errors=True)

if __name__ == "__main__":
    print("=== Voice Transcriber Executable Builder ===")
    print("This will create a standalone executable with embedded Whisper models.")
    print()
    
    success = build_executable()
    
    if success:
        print("\n🎉 Build completed successfully!")
        print("The executable should now work on machines without Python installed.")
    else:
        print("\n❌ Build failed. Check the errors above.")
        sys.exit(1)