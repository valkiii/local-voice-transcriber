import sys
import os
import threading
import queue
import math
import tempfile
import shutil
import subprocess
from datetime import datetime
import numpy as np
import sounddevice as sd
import whisper
from scipy.io.wavfile import write
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QWidget, QPushButton, QTextEdit, QLabel, QProgressBar,
                           QFileDialog, QComboBox, QGroupBox, QFrame)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QFont, QPalette, QColor, QPainter, QPen, QBrush, QLinearGradient


def setup_whisper_models():
    """Enhanced model setup that works in both dev and executable environments"""
    print("Setting up Whisper models...")
    
    if hasattr(sys, '_MEIPASS'):
        # Running as PyInstaller executable
        exe_model_dir = os.path.join(sys._MEIPASS, 'whisper_cache')
        print(f"Looking for models in executable: {exe_model_dir}")
        
        if os.path.exists(exe_model_dir):
            model_files = [f for f in os.listdir(exe_model_dir) if f.endswith('.pt')]
            print(f"Found model files: {model_files}")
            
            if model_files:
                # Create a persistent cache directory in user's temp folder
                user_whisper_cache = os.path.join(tempfile.gettempdir(), 'voice_transcriber_cache')
                os.makedirs(user_whisper_cache, exist_ok=True)
                
                # Copy all model files to the cache
                for model_file in model_files:
                    src_path = os.path.join(exe_model_dir, model_file)
                    dst_path = os.path.join(user_whisper_cache, model_file)
                    
                    if not os.path.exists(dst_path):
                        print(f"Copying {model_file} to cache...")
                        shutil.copy2(src_path, dst_path)
                
                # Set environment variables for Whisper
                os.environ['WHISPER_CACHE_DIR'] = user_whisper_cache
                os.environ['XDG_CACHE_HOME'] = os.path.dirname(user_whisper_cache)
                
                print(f"Whisper cache set to: {user_whisper_cache}")
                return user_whisper_cache
        
        print("No models found in executable bundle!")
        return None
    else:
        # Running in development - use default behavior
        print("Running in development mode, using default Whisper cache")
        return None


def load_whisper_model_safely():
    """Safely load Whisper model with better error handling"""
    try:
        # Setup model paths first
        cache_dir = setup_whisper_models()
        
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        print("Whisper model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        
        # Try to download model if running in development
        if not hasattr(sys, '_MEIPASS'):
            try:
                print("Attempting to download Whisper model...")
                model = whisper.load_model("base", download_root=os.path.expanduser("~/.cache/whisper"))
                print("Model downloaded successfully!")
                return model
            except Exception as e2:
                print(f"Failed to download model: {e2}")
        
        return None


class VoiceBall(QWidget):
    def __init__(self):
        super().__init__()
        self.base_size = 70
        self.current_scale = 1.0
        self.audio_level = 0.0
        self.is_recording = False
        self.setFixedSize(200, 200)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Animation for smooth scaling
        self.scale_animation = QPropertyAnimation(self, b"scale")
        self.scale_animation.setDuration(300)
        self.scale_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    
    @pyqtProperty(float)
    def scale(self):
        return self.current_scale
    
    @scale.setter
    def scale(self, value):
        self.current_scale = value
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center_x, center_y = self.width() // 2, self.height() // 2
        
        # Calculate dynamic size based on audio level (smaller expansion)
        audio_boost = 1.0 + (self.audio_level * 0.2) if self.is_recording else 0.0
        radius = (self.base_size // 2) * self.current_scale * (1.0 + audio_boost)
        
        # Purple neon gradient colors
        if self.is_recording:
            # Bright purple neon when recording (pulsing)
            gradient = QLinearGradient(center_x - radius, center_y - radius, 
                                     center_x + radius, center_y + radius)
            gradient.setColorAt(0, QColor(255, 0, 255, 240))
            gradient.setColorAt(0.5, QColor(200, 0, 200, 200))
            gradient.setColorAt(1, QColor(150, 0, 150, 220))
        else:
            # Darker purple when idle
            gradient = QLinearGradient(center_x - radius, center_y - radius,
                                     center_x + radius, center_y + radius)
            gradient.setColorAt(0, QColor(150, 0, 255, 180))
            gradient.setColorAt(0.5, QColor(100, 0, 200, 160))
            gradient.setColorAt(1, QColor(80, 0, 160, 180))
        
        # Draw main circle
        painter.setBrush(QBrush(gradient))
        painter.setPen(QPen(QColor(255, 255, 255, 100), 2))
        painter.drawEllipse(int(center_x - radius), int(center_y - radius), 
                          int(radius * 2), int(radius * 2))
        
        # Draw microphone icon in center
        icon_size = radius * 0.4
        painter.setPen(QPen(QColor(255, 255, 255, 250), 3))
        
        # Microphone body (keep as rounded rectangle for mic shape)
        mic_rect = int(center_x - icon_size//4), int(center_y - icon_size//2), int(icon_size//2), int(icon_size)
        painter.drawRoundedRect(*mic_rect, int(icon_size//6), int(icon_size//6))
        
        # Microphone base
        base_y = int(center_y + icon_size//2 + 5)
        painter.drawLine(int(center_x - icon_size//3), base_y, int(center_x + icon_size//3), base_y)
        painter.drawLine(center_x, base_y, center_x, int(base_y + icon_size//4))
    
    def update_level(self, level):
        self.audio_level = max(0, min(1, level))
        self.update()
    
    def set_recording(self, recording):
        self.is_recording = recording
        target_scale = 1.1 if recording else 1.0
        self.scale_animation.setStartValue(self.current_scale)
        self.scale_animation.setEndValue(target_scale)
        self.scale_animation.start()
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Find the main window and call toggle_recording
            main_window = self
            while main_window.parent():
                main_window = main_window.parent()
            if hasattr(main_window, 'toggle_recording'):
                main_window.toggle_recording()


class TranscriptionWorker(QThread):
    transcription_ready = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    
    def __init__(self, audio_data, sample_rate):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.model = None
    
    def run(self):
        try:
            self.progress_update.emit(20)
            
            # Load model with enhanced error handling
            if self.model is None:
                self.progress_update.emit(30)
                self.model = load_whisper_model_safely()
                
                if self.model is None:
                    self.transcription_ready.emit("❌ Error: Could not load Whisper model. Please ensure the model files are properly bundled or run in development mode to download them.")
                    return
            
            self.progress_update.emit(50)
            
            # Convert audio data to the format Whisper expects
            # Whisper expects float32 audio normalized to [-1, 1]
            if self.audio_data.dtype != np.float32:
                if self.audio_data.dtype == np.int16:
                    audio_float = self.audio_data.astype(np.float32) / 32768.0
                elif self.audio_data.dtype == np.int32:
                    audio_float = self.audio_data.astype(np.float32) / 2147483648.0
                else:
                    audio_float = self.audio_data.astype(np.float32)
            else:
                audio_float = self.audio_data
            
            # Flatten if stereo
            if len(audio_float.shape) > 1:
                audio_float = np.mean(audio_float, axis=1)
            
            # Resample to 16kHz if needed (Whisper's expected sample rate)
            if self.sample_rate != 16000:
                # Simple resampling
                import scipy.signal
                num_samples = int(len(audio_float) * 16000 / self.sample_rate)
                audio_float = scipy.signal.resample(audio_float, num_samples)
            
            self.progress_update.emit(80)
            
            # Transcribe directly from numpy array (no file needed!)
            print("Starting transcription...")
            result = self.model.transcribe(audio_float, fp16=False)
            print(f"Transcription completed: {result['text'][:50]}...")
            
            self.progress_update.emit(100)
            self.transcription_ready.emit(result["text"])
        except Exception as e:
            error_msg = f"Error during transcription: {str(e)}"
            print(error_msg)
            self.transcription_ready.emit(error_msg)


class VoiceTranscriberApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recording = False
        self.recorded_data = []
        self.sample_rate = 44100
        self.output_directory = os.path.expanduser("~/Documents")
        self.audio_queue = queue.Queue()
        self.transcription_worker = None
        
        # Initialize models on startup (with better error handling)
        print("Initializing Voice Transcriber...")
        setup_whisper_models()
        
        self.init_ui()
        self.init_audio()
        
        # Timer for audio level updates (slower for smoother animation)
        self.level_timer = QTimer()
        self.level_timer.timeout.connect(self.update_audio_level)
        
    def init_ui(self):
        self.setWindowTitle("Voice Transcriber")
        self.setGeometry(100, 100, 500, 700)
        
        # Central widget with dark theme
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(30)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Top section - microphone selection (minimal)
        top_layout = QHBoxLayout()
        
        mic_label = QLabel("Microphone:")
        mic_label.setFont(QFont("Segoe UI", 10))
        self.mic_combo = QComboBox()
        self.mic_combo.setFont(QFont("Segoe UI", 10))
        self.populate_microphones()
        
        self.output_button = QPushButton("📁")
        self.output_button.setFixedSize(35, 35)
        self.output_button.setToolTip("Select output folder")
        self.output_button.clicked.connect(self.select_output_directory)
        
        top_layout.addWidget(mic_label)
        top_layout.addWidget(self.mic_combo, 1)
        top_layout.addStretch()
        top_layout.addWidget(self.output_button)
        
        layout.addLayout(top_layout)
        
        # Voice ball (center focus)
        ball_container = QWidget()
        ball_layout = QVBoxLayout(ball_container)
        ball_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.voice_ball = VoiceBall()
        ball_layout.addWidget(self.voice_ball)
        
        # Status text below ball
        self.status_label = QLabel("Click to start recording")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Segoe UI", 12))
        ball_layout.addWidget(self.status_label)
        
        layout.addWidget(ball_container)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(8)
        layout.addWidget(self.progress_bar)
        
        # Transcription display (selectable text, no visible box)
        self.transcription_text = QTextEdit()
        self.transcription_text.setFont(QFont("Segoe UI", 16, QFont.Weight.Normal))
        self.transcription_text.setMinimumHeight(200)
        self.transcription_text.setPlainText("Your transcription will appear here...")
        self.transcription_text.setReadOnly(True)
        self.transcription_text.setStyleSheet("""
            QTextEdit {
                color: #ff00ff;
                background: transparent;
                border: none;
                padding: 20px;
                selection-background-color: rgba(255, 0, 255, 0.3);
                selection-color: #ffffff;
            }
        """)
        layout.addWidget(self.transcription_text)
        
        layout.addStretch()
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0a0a;
                color: #ff00ff;
            }
            QWidget {
                background-color: #0a0a0a;
                color: #ff00ff;
            }
            QLabel {
                color: #ff00ff;
                background: transparent;
            }
            QComboBox {
                background: #3a3a3a;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 8px;
                color: #ffffff;
                font-size: 10pt;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                width: 12px;
                height: 12px;
            }
            QPushButton {
                background: #3a3a3a;
                border: 1px solid #555;
                border-radius: 17px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #4a4a4a;
                border-color: #666;
            }
            QPushButton:pressed {
                background: #2a2a2a;
            }
            QProgressBar {
                border: none;
                border-radius: 4px;
                background: #3a3a3a;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00ff88, stop:1 #00cc66);
                border-radius: 4px;
            }
        """)
    
    def init_audio(self):
        try:
            self.devices = sd.query_devices()
            self.input_devices = [i for i, device in enumerate(self.devices) 
                                if device['max_input_channels'] > 0]
        except Exception as e:
            self.status_label.setText(f"Audio initialization error: {e}")
    
    def populate_microphones(self):
        try:
            devices = sd.query_devices()
            self.mic_combo.clear()
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    self.mic_combo.addItem(f"{device['name']} (Device {i})", i)
        except Exception as e:
            self.mic_combo.addItem("No microphones found", -1)
    
    def select_output_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_directory)
        if directory:
            self.output_directory = directory
            self.status_label.setText("Click to start recording")
    
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        try:
            device_index = self.mic_combo.currentData()
            if device_index == -1:
                self.status_label.setText("No microphone selected")
                return
            
            self.recording = True
            self.recorded_data = []
            self.voice_ball.set_recording(True)
            self.status_label.setText("Recording... Speak now!")
            
            # Start recording stream
            self.stream = sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=self.sample_rate,
                callback=self.audio_callback
            )
            self.stream.start()
            
            # Start level monitoring (slower for smoother animation)
            self.level_timer.start(100)  # Update every 100ms
            
        except Exception as e:
            self.status_label.setText(f"Recording error: {e}")
            self.recording = False
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        
        # Store audio data
        self.recorded_data.append(indata.copy())
        
        # Update voice ball
        volume_norm = np.linalg.norm(indata) * 10
        self.audio_queue.put(volume_norm)
    
    def update_audio_level(self):
        try:
            while True:
                level = self.audio_queue.get_nowait()
                self.voice_ball.update_level(level)
        except queue.Empty:
            pass
    
    def stop_recording(self):
        if not self.recording:
            return
        
        try:
            self.recording = False
            self.level_timer.stop()
            self.stream.stop()
            self.stream.close()
            
            self.voice_ball.set_recording(False)
            self.status_label.setText("Processing audio...")
            
            # Only process audio for transcription - don't save audio files
            if self.recorded_data:
                audio_data = np.concatenate(self.recorded_data, axis=0)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Start transcription using audio data directly (no file needed)
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                
                self.transcription_worker = TranscriptionWorker(audio_data, self.sample_rate)
                self.transcription_worker.transcription_ready.connect(self.on_transcription_ready)
                self.transcription_worker.progress_update.connect(self.progress_bar.setValue)
                self.transcription_worker.start()
            else:
                self.status_label.setText("No audio recorded")
                
        except Exception as e:
            self.status_label.setText(f"Error stopping recording: {e}")
    
    def on_transcription_ready(self, text):
        self.progress_bar.setVisible(False)
        self.transcription_text.setPlainText(text)
        
        # Save transcription to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_filename = f"{timestamp}.txt"
        text_path = os.path.join(self.output_directory, text_filename)
        
        try:
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)
            self.status_label.setText(f"Saved: {text_filename}")
        except Exception as e:
            self.status_label.setText(f"Error saving: {e}")
    
    def closeEvent(self, event):
        if self.recording:
            self.stop_recording()
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Voice Transcriber")
    app.setApplicationVersion("1.0")
    
    window = VoiceTranscriberApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()