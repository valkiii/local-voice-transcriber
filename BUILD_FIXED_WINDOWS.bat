@echo off
echo ================================================================
echo Voice Transcriber - Windows Build Script (Fixed Model Bundling)
echo ================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python first.
    pause
    exit /b 1
)

REM Install required packages
echo Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo Building standalone executable with embedded Whisper models...
echo This may take several minutes...
echo.

REM Run the build script
python build_exe_with_models.py
if errorlevel 1 (
    echo ERROR: Build failed
    pause
    exit /b 1
)

echo.
echo ================================================================
echo BUILD COMPLETED!
echo ================================================================
echo.
echo Check the 'dist' folder for VoiceTranscriber.exe
echo This executable includes all dependencies and Whisper models.
echo It should work on any Windows machine without Python installed.
echo.
echo To test: Copy VoiceTranscriber.exe to another machine and run it.
echo.
pause