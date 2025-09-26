import sys
import os
import threading
import queue
import math
from datetime import datetime
import numpy as np
import sounddevice as sd
import whisper
from scipy.io.wavfile import write
import tempfile
from collections import deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QWidget, QPushButton, QTextEdit, QLabel, QProgressBar,
                           QFileDialog, QComboBox, QGroupBox, QFrame, QMessageBox,
                           QTabWidget, QScrollArea, QDialog, QDialogButtonBox, 
                           QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
                           QSplitter)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("LLM dependencies not available. Install transformers and torch for analysis features.")

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment
    import torchaudio
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    print("Diarization dependencies not available. Install pyannote.audio for speaker identification.")
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QFont, QPalette, QColor, QPainter, QPen, QBrush, QLinearGradient


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
    segments_ready = pyqtSignal(list)  # Emit segments for diarization
    progress_update = pyqtSignal(int)
    
    def __init__(self, audio_data, sample_rate):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.model = None
    
    def run(self):
        try:
            self.progress_update.emit(20)
            if self.model is None:
                try:
                    # Try to load the model with error handling
                    self.model = whisper.load_model("base")
                    if self.model is None:
                        raise Exception("Failed to load Whisper model")
                except Exception as e:
                    self.transcription_ready.emit(f"Error loading Whisper model: {str(e)}. Please ensure the model files are included.")
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
            
            # Transcribe directly from numpy array with word-level timestamps
            result = self.model.transcribe(audio_float, word_timestamps=True)
            
            self.progress_update.emit(100)
            
            # Emit both text and segments
            self.transcription_ready.emit(result["text"])
            
            # Convert segments to our format for diarization
            segments = []
            if "segments" in result:
                for segment in result["segments"]:
                    segments.append({
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'text': segment.get('text', '')
                    })
            
            self.segments_ready.emit(segments)
        except Exception as e:
            self.transcription_ready.emit(f"Error during transcription: {str(e)}")


class LLMAnalysisWorker(QThread):
    analysis_ready = pyqtSignal(str, str)  # analysis_type, result
    analysis_error = pyqtSignal(str)
    
    def __init__(self, text, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        super().__init__()
        self.text = text
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def run(self):
        if not LLM_AVAILABLE:
            self.analysis_error.emit("LLM dependencies not available")
            return
            
        try:
            self.analysis_ready.emit("status", f"Loading {self.model_name}...")
            
            # Check if using keyword-only analysis
            if self.model_name == "keyword-only":
                self.analysis_ready.emit("status", "Using keyword-based analysis")
                self._fallback_analysis()
                return
            
            # Load the selected model
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            # Determine optimal device for Apple Silicon
            device = self._get_optimal_device()
            self.analysis_ready.emit("status", f"Using device: {device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Handle different tokenizer configurations
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                elif self.tokenizer.unk_token:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                else:
                    # Add a new pad token
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            print(f"Tokenizer loaded: pad_token={self.tokenizer.pad_token}, eos_token={self.tokenizer.eos_token}")
            
            # Load model with optimal settings for M1
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self._get_optimal_dtype(device),
                low_cpu_mem_usage=True,
                device_map="auto" if device == "cuda" else None
            )
            
            # Resize model embeddings if we added new tokens
            if len(self.tokenizer) > self.model.config.vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Move model to optimal device
            if device != "cuda":  # cuda uses device_map="auto"
                self.model = self.model.to(device)
            
            self.analysis_ready.emit("status", f"Model loaded on {device}")
            
            self.analysis_ready.emit("status", "Generating summary...")
            
            # Truncate text if too long to avoid context overflow
            text_for_analysis = self.text[:2000] + "..." if len(self.text) > 2000 else self.text
            
            # Generate summary
            summary_prompt = f"""Summarize the following transcript in 2-3 clear sentences. Focus on the main topics discussed.

Transcript: {text_for_analysis}

Summary:"""
            
            print(f"Generating summary with {self.model_name}...")
            summary = self._generate_response(summary_prompt, max_length=150)
            print(f"Summary result: '{summary[:100]}...'")
            
            if not summary or "Generation failed" in summary:
                self.analysis_error.emit(f"Summary generation failed with {self.model_name}")
                return
                
            self.analysis_ready.emit("summary", summary)
            
            self.analysis_ready.emit("status", "Extracting key points...")
            
            # Generate key points
            key_points_prompt = f"""Extract the main key points from this transcript as a bullet list. Include only the most important ideas.

Transcript: {text_for_analysis}

Key Points:
•"""
            
            print(f"Generating key points with {self.model_name}...")
            key_points_response = self._generate_response(key_points_prompt, max_length=200)
            
            if key_points_response and "Generation failed" not in key_points_response:
                key_points = "• " + key_points_response
            else:
                key_points = "• Could not extract key points"
                
            self.analysis_ready.emit("key_points", key_points)
            
            self.analysis_ready.emit("status", "Identifying action items...")
            
            # Generate action items
            action_prompt = f"""Identify any action items, tasks, or follow-up items mentioned in this transcript. List them clearly.

Transcript: {text_for_analysis}

Action Items:
•"""
            
            print(f"Generating actions with {self.model_name}...")
            actions_response = self._generate_response(action_prompt, max_length=200)
            
            if actions_response and "Generation failed" not in actions_response:
                actions = "• " + actions_response
            else:
                actions = "• No specific action items identified"
                
            self.analysis_ready.emit("actions", actions)
            
            self.analysis_ready.emit("status", "Analysis complete")
            
        except Exception as e:
            # Fallback to keyword-based analysis if LLM fails
            self.analysis_ready.emit("status", f"LLM failed, using fallback analysis: {str(e)}")
            self._fallback_analysis()
    
    def _get_optimal_device(self):
        """Determine the best device for inference on this system"""
        import torch
        
        # Check for Apple Silicon MPS (Metal Performance Shaders)
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        # Check for NVIDIA CUDA
        elif torch.cuda.is_available():
            return "cuda"
        # Fallback to CPU
        else:
            return "cpu"
    
    def _get_optimal_dtype(self, device):
        """Get optimal data type for the given device"""
        import torch
        
        if device == "mps":
            # MPS works best with float32 for stability
            return torch.float32
        elif device == "cuda":
            # CUDA can use float16 for faster inference
            return torch.float16
        else:
            # CPU uses float32
            return torch.float32
    
    def _generate_response(self, prompt, max_length=150):
        try:
            import torch
            
            # Model-specific prompt formatting
            formatted_prompt = self._format_prompt_for_model(prompt)
            
            # Tokenize input with attention mask
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=1024  # Increased for better context
            )
            
            # Move inputs to same device as model
            device = next(self.model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            # Model-specific generation parameters
            generation_params = self._get_generation_params(max_length)
            
            # Generate response with optimal settings
            with torch.no_grad():
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        **generation_params
                    )
            
            # Decode only the new tokens (skip input)
            input_length = inputs['input_ids'].shape[1]
            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Post-process response
            response = self._post_process_response(response)
            
            return response[:500]  # Limit response length
            
        except Exception as e:
            print(f"Generation error with {self.model_name}: {str(e)}")
            return f"Generation failed: {str(e)}"
    
    def _format_prompt_for_model(self, prompt):
        """Format prompt according to model requirements"""
        if "phi-2" in self.model_name.lower():
            # Phi-2 works better with instruction format
            return f"Instruction: {prompt}\nResponse:"
        elif "tinyllama" in self.model_name.lower():
            # TinyLlama works well with chat format
            return f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        elif "qwen" in self.model_name.lower():
            # Qwen2 works with simple instruction format
            return f"Human: {prompt}\nAssistant:"
        else:
            # Default format
            return prompt
    
    def _get_generation_params(self, max_length):
        """Get model-specific generation parameters"""
        base_params = {
            "max_new_tokens": max_length,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True
        }
        
        if "phi-2" in self.model_name.lower():
            # Phi-2 specific parameters
            base_params.update({
                "temperature": 0.8,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.2
            })
        elif "tinyllama" in self.model_name.lower():
            # TinyLlama specific parameters
            base_params.update({
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.95,
                "repetition_penalty": 1.1
            })
        elif "qwen" in self.model_name.lower():
            # Qwen2 specific parameters
            base_params.update({
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.8,
                "repetition_penalty": 1.1
            })
        else:
            # Default parameters
            base_params.update({
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.1
            })
        
        return base_params
    
    def _post_process_response(self, response):
        """Clean up model response"""
        response = response.strip()
        
        # Remove common artifacts
        if response.startswith("Response:"):
            response = response[9:].strip()
        if response.startswith("Assistant:"):
            response = response[10:].strip()
        
        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        return response
    
    def _fallback_analysis(self):
        """Fallback keyword-based analysis if LLM fails"""
        try:
            sentences = self.text.split('.')
            
            # Summary (simple extraction)
            summary_sentences = [s.strip() for s in sentences[:3] if len(s.strip()) > 20]
            summary = '. '.join(summary_sentences) + '.' if summary_sentences else "No summary available."
            
            # Key points (extract sentences with keywords)
            key_words = ['important', 'key', 'main', 'significant', 'critical', 'need', 'should', 'must', 'will', 'plan']
            key_sentences = []
            for sentence in sentences:
                if any(word in sentence.lower() for word in key_words) and len(sentence.strip()) > 15:
                    key_sentences.append(sentence.strip())
            
            key_points = '\n• ' + '\n• '.join(key_sentences[:5]) if key_sentences else "No key points identified."
            
            # Action items
            action_words = ['need to', 'should', 'must', 'will', 'plan to', 'going to', 'have to', 'todo', 'action', 'task']
            action_sentences = []
            for sentence in sentences:
                if any(word in sentence.lower() for word in action_words) and len(sentence.strip()) > 15:
                    action_sentences.append(sentence.strip())
            
            actions = '\n• ' + '\n• '.join(action_sentences[:5]) if action_sentences else "No action items identified."
            
            # Emit results
            self.analysis_ready.emit("summary", summary)
            self.analysis_ready.emit("key_points", key_points)
            self.analysis_ready.emit("actions", actions)
            
        except Exception as e:
            self.analysis_error.emit(f"Fallback analysis failed: {str(e)}")


class DiarizationWorker(QThread):
    diarization_ready = pyqtSignal(list)  # List of (start, end, speaker, text) tuples
    diarization_error = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    
    def __init__(self, audio_file_path, transcription_segments=None):
        super().__init__()
        self.audio_file_path = audio_file_path
        self.transcription_segments = transcription_segments or []
        self.pipeline = None
    
    def run(self):
        if not DIARIZATION_AVAILABLE:
            self.diarization_error.emit("Diarization dependencies not available. Install pyannote.audio")
            return
        
        try:
            self.progress_update.emit(10)
            
            # Load the pre-trained speaker diarization pipeline
            # Note: This requires accepting terms of use for pyannote models
            from pyannote.audio import Pipeline
            import warnings
            warnings.filterwarnings("ignore")
            
            self.progress_update.emit(30)
            
            # Use the default speaker diarization pipeline
            try:
                self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
            except Exception as e:
                # Fallback to simple speaker detection if the main model fails
                self.diarization_error.emit(f"Could not load diarization model: {str(e)}")
                return
            
            self.progress_update.emit(50)
            
            # Apply the pipeline to the audio file
            diarization = self.pipeline(self.audio_file_path)
            
            self.progress_update.emit(80)
            
            # Convert diarization results to our format
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_time = turn.start
                end_time = turn.end
                speaker_id = speaker
                
                # Find matching transcription text for this time segment
                matching_text = self._find_matching_text(start_time, end_time)
                
                speaker_segments.append({
                    'start': start_time,
                    'end': end_time,
                    'speaker': speaker_id,
                    'text': matching_text
                })
            
            self.progress_update.emit(100)
            self.diarization_ready.emit(speaker_segments)
            
        except Exception as e:
            self.diarization_error.emit(f"Diarization failed: {str(e)}")
    
    def _find_matching_text(self, start_time, end_time):
        """Find transcription text that matches the speaker segment timing"""
        if not self.transcription_segments:
            return ""
        
        matching_texts = []
        for segment in self.transcription_segments:
            # Check if segment overlaps with speaker time
            seg_start = segment.get('start', 0)
            seg_end = segment.get('end', 0)
            
            # If segments overlap, include the text
            if (seg_start <= end_time and seg_end >= start_time):
                matching_texts.append(segment.get('text', ''))
        
        return ' '.join(matching_texts).strip()


class LiveTranscriptionWorker(QThread):
    live_transcription_ready = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.audio_buffer = deque(maxlen=200)  # Keep last 200 chunks (~10-20 seconds)
        self.model = None
        self.is_running = False
        self.sample_rate = 44100
        self.chunk_duration = 2.0  # Process every 2 seconds
        self.chunk_samples = int(self.chunk_duration * self.sample_rate)
        self.last_processed_time = 0
        self.processing_count = 0
        
    def add_audio_chunk(self, chunk):
        if self.is_running:
            self.audio_buffer.append(chunk)
            # Debug: Print buffer status occasionally
            if len(self.audio_buffer) % 50 == 0:
                print(f"Live transcription buffer: {len(self.audio_buffer)} chunks")
    
    def run(self):
        self.is_running = True
        print("🎤 Starting live transcription worker...")
        try:
            # Load model once
            if self.model is None:
                print("📥 Loading Whisper model for live transcription...")
                self.model = whisper.load_model("base")
                print("✅ Whisper model loaded for live transcription")
                
            while self.is_running:
                if len(self.audio_buffer) > 0:
                    # Get recent audio data - use more chunks for better transcription
                    buffer_size = len(self.audio_buffer)
                    # Use last 2-3 seconds of audio (assuming ~44 chunks per second)
                    recent_chunks = list(self.audio_buffer)[-80:] if buffer_size > 80 else list(self.audio_buffer)
                    
                    # Lower threshold - process with fewer chunks
                    if len(recent_chunks) >= 20:  # About 0.5 seconds of audio
                        self.processing_count += 1
                        print(f"🔄 Processing live transcription #{self.processing_count} with {len(recent_chunks)} chunks")
                        
                        try:
                            audio_data = np.concatenate(recent_chunks)
                            
                            # Convert to format Whisper expects
                            if audio_data.dtype != np.float32:
                                if audio_data.dtype == np.int16:
                                    audio_float = audio_data.astype(np.float32) / 32768.0
                                else:
                                    audio_float = audio_data.astype(np.float32)
                            else:
                                audio_float = audio_data
                            
                            # Flatten if needed
                            if len(audio_float.shape) > 1:
                                audio_float = np.mean(audio_float, axis=1)
                            
                            # Resample to 16kHz if needed
                            if self.sample_rate != 16000:
                                import scipy.signal
                                num_samples = int(len(audio_float) * 16000 / self.sample_rate)
                                audio_float = scipy.signal.resample(audio_float, num_samples)
                            
                            # Check audio length
                            audio_duration = len(audio_float) / 16000
                            print(f"🎵 Audio duration: {audio_duration:.2f}s")
                            
                            # Only transcribe if we have meaningful audio (at least 0.5 seconds)
                            if audio_duration >= 0.5:
                                # Transcribe with optimized settings for live transcription
                                result = self.model.transcribe(
                                    audio_float, 
                                    fp16=False,
                                    language="en",  # Force English for faster processing
                                    task="transcribe",
                                    temperature=0.0,  # More deterministic
                                    no_speech_threshold=0.3  # Lower threshold for detecting speech
                                )
                                text = result["text"].strip()
                                print(f"📝 Transcribed: '{text}'")
                                
                                if text and len(text) > 1:  # Only emit if there's meaningful text
                                    self.live_transcription_ready.emit(text)
                            else:
                                print(f"⏩ Skipping short audio: {audio_duration:.2f}s")
                                
                        except Exception as e:
                            print(f"❌ Live transcription error: {e}")
                    else:
                        print(f"⏳ Buffer too small: {len(recent_chunks)} chunks (need at least 20)")
                else:
                    print("📭 No audio in buffer")
                
                # Wait before next processing - longer interval for stability
                self.msleep(2000)  # Process every 2 seconds
                
        except Exception as e:
            print(f"Live transcription worker error: {e}")
    
    def stop_transcription(self):
        self.is_running = False
        self.wait()


class SpeakerEditDialog(QDialog):
    def __init__(self, speaker_segments, parent=None):
        super().__init__(parent)
        self.speaker_segments = speaker_segments
        self.speaker_names = {}
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Edit Speaker Names")
        self.setModal(True)
        self.resize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Edit speaker names below. Changes will be applied to the transcription.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #ff00ff; margin-bottom: 10px; font-size: 11pt;")
        layout.addWidget(instructions)
        
        # Table for speaker editing
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Time", "Original ID", "Speaker Name", "Preview"])
        
        # Populate table with unique speakers
        unique_speakers = {}
        for segment in self.speaker_segments:
            speaker_id = segment['speaker']
            if speaker_id not in unique_speakers:
                unique_speakers[speaker_id] = {
                    'first_appearance': segment['start'],
                    'sample_text': segment['text'][:50] + "..." if len(segment['text']) > 50 else segment['text']
                }
        
        self.table.setRowCount(len(unique_speakers))
        
        row = 0
        for speaker_id, info in unique_speakers.items():
            # Time column
            time_item = QTableWidgetItem(f"{info['first_appearance']:.1f}s")
            time_item.setFlags(time_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, time_item)
            
            # Original ID column
            id_item = QTableWidgetItem(speaker_id)
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, id_item)
            
            # Speaker name column (editable)
            name_item = QTableWidgetItem(f"Speaker {row + 1}")
            self.table.setItem(row, 2, name_item)
            self.speaker_names[speaker_id] = f"Speaker {row + 1}"
            
            # Preview column
            preview_item = QTableWidgetItem(info['sample_text'])
            preview_item.setFlags(preview_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 3, preview_item)
            
            row += 1
        
        # Auto-resize columns
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        # Connect cell changes
        self.table.cellChanged.connect(self.on_cell_changed)
        
        # Style the table
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #555;
                border-radius: 6px;
                selection-background-color: rgba(255, 0, 255, 0.3);
            }
            QTableWidget::item {
                padding: 8px;
                border: none;
            }
            QTableWidget::item:selected {
                background-color: rgba(255, 0, 255, 0.2);
            }
            QHeaderView::section {
                background-color: #2a2a2a;
                color: #ff00ff;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
        
        layout.addWidget(self.table)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        button_box.setStyleSheet("""
            QPushButton {
                background: #3a3a3a;
                border: 1px solid #555;
                border-radius: 6px;
                color: #ffffff;
                font-weight: bold;
                padding: 8px 16px;
                margin: 2px;
            }
            QPushButton:hover {
                background: #4a4a4a;
                border-color: #666;
            }
        """)
        
        layout.addWidget(button_box)
        
        # Set dialog style
        self.setStyleSheet("""
            QDialog {
                background-color: #0a0a0a;
                color: #ff00ff;
            }
            QLabel {
                color: #ff00ff;
            }
        """)
    
    def on_cell_changed(self, row, column):
        if column == 2:  # Speaker name column
            original_id = self.table.item(row, 1).text()
            new_name = self.table.item(row, 2).text()
            self.speaker_names[original_id] = new_name
    
    def get_speaker_names(self):
        return self.speaker_names


class VoiceTranscriberApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recording = False
        self.transcription_in_progress = False
        self.recorded_data = []
        self.sample_rate = 44100
        self.output_directory = os.path.expanduser("~/Documents")
        self.audio_queue = queue.Queue()
        self.transcription_worker = None
        self.live_transcription_worker = None
        self.live_transcription_enabled = False  # Disabled due to stability issues
        self.accumulated_live_text = ""
        self.last_recorded_audio = None  # Backup of last recording
        self.llm_worker = None
        self.current_transcription_text = ""
        self.selected_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Default model
        self.live_text_sentences = []  # Store live transcription sentences
        self.diarization_worker = None
        self.speaker_segments = []  # Store speaker diarization results
        self.transcription_segments = []  # Store transcription segments with timing
        self.diarization_enabled = True
        self.last_audio_file = None  # Path to last saved audio file
        
        self.init_ui()
        self.init_audio()
        
        # Timer for audio level updates (slower for smoother animation)
        self.level_timer = QTimer()
        self.level_timer.timeout.connect(self.update_audio_level)
        
    def init_ui(self):
        self.setWindowTitle("Voice Transcriber with AI Analysis")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget with dark theme
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout - split screen
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Left panel - Recording and Transcription
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        left_layout.setContentsMargins(20, 20, 20, 20)
        
        # Right panel - AI Analysis
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(20)
        right_layout.setContentsMargins(20, 20, 20, 20)
        
        # Add panels to main layout with 60/40 split
        main_layout.addWidget(left_panel, 6)
        main_layout.addWidget(right_panel, 4)
        
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
        
        left_layout.addLayout(top_layout)
        
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
        
        # Processing indicator
        self.processing_label = QLabel("")
        self.processing_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processing_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.processing_label.setStyleSheet("color: #ffaa00;")
        self.processing_label.setVisible(False)
        ball_layout.addWidget(self.processing_label)
        
        left_layout.addWidget(ball_container)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(8)
        left_layout.addWidget(self.progress_bar)
        
        # Live transcription display (disabled for stability)
        live_label = QLabel("Live Transcription (Disabled for Stability):")
        live_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        live_label.setVisible(False)  # Hide the label
        left_layout.addWidget(live_label)
        
        self.live_transcription_text = QTextEdit()
        self.live_transcription_text.setFont(QFont("Segoe UI", 12, QFont.Weight.Normal))
        self.live_transcription_text.setMaximumHeight(120)
        self.live_transcription_text.setPlainText("Live transcription disabled for stability. Final transcription will appear below after recording.")
        self.live_transcription_text.setReadOnly(True)
        self.live_transcription_text.setVisible(False)  # Hide the text area
        self.live_transcription_text.setStyleSheet("""
            QTextEdit {
                color: #888;
                background: rgba(136, 136, 136, 0.1);
                border: 1px solid rgba(136, 136, 136, 0.3);
                border-radius: 8px;
                padding: 15px;
                selection-background-color: rgba(136, 136, 136, 0.3);
                selection-color: #ffffff;
            }
        """)
        left_layout.addWidget(self.live_transcription_text)
        
        # Final transcription display (after recording stops)
        final_label = QLabel("Final Transcription:")
        final_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        left_layout.addWidget(final_label)
        
        self.transcription_text = QTextEdit()
        self.transcription_text.setFont(QFont("Segoe UI", 14, QFont.Weight.Normal))
        self.transcription_text.setMinimumHeight(250)
        self.transcription_text.setPlainText("Final transcription will appear here...")
        self.transcription_text.setReadOnly(True)
        self.transcription_text.setStyleSheet("""
            QTextEdit {
                color: #ff00ff;
                background: rgba(255, 0, 255, 0.05);
                border: 1px solid rgba(255, 0, 255, 0.3);
                border-radius: 8px;
                padding: 15px;
                selection-background-color: rgba(255, 0, 255, 0.3);
                selection-color: #ffffff;
            }
        """)
        left_layout.addWidget(self.transcription_text)
        
        left_layout.addStretch()
        
        # === RIGHT PANEL - AI Analysis ===
        analysis_title = QLabel("🤖 AI Analysis")
        analysis_title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        analysis_title.setStyleSheet("color: #ff00ff; margin-bottom: 10px;")
        right_layout.addWidget(analysis_title)
        
        # Analysis controls
        analysis_controls = QVBoxLayout()
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_label.setFont(QFont("Segoe UI", 10))
        
        self.model_combo = QComboBox()
        self.model_combo.addItem("TinyLlama 1.1B (Fastest)", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.model_combo.addItem("Phi-2 2.7B (Fast)", "microsoft/phi-2")
        self.model_combo.addItem("Qwen2 0.5B (Ultra-Fast)", "Qwen/Qwen2-0.5B-Instruct")
        self.model_combo.addItem("Keyword Analysis (No Download)", "keyword-only")
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.model_combo.setStyleSheet("""
            QComboBox {
                background: #2a2a2a;
                border: 1px solid #555;
                border-radius: 6px;
                padding: 8px;
                color: #ffffff;
                font-size: 10pt;
            }
        """)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        analysis_controls.addLayout(model_layout)
        
        # Analyze button
        self.analyze_button = QPushButton("🧠 Analyze Content")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.analyze_transcription)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 11pt;
                margin: 5px 0px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5CBF60, stop:1 #55b059);
            }
            QPushButton:disabled {
                background: #666;
                color: #999;
            }
        """)
        analysis_controls.addWidget(self.analyze_button)
        
        # Load transcription button
        self.load_button = QPushButton("📁 Load Transcription")
        self.load_button.clicked.connect(self.load_transcription_file)
        self.load_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF6B35, stop:1 #E55A2B);
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10pt;
                margin: 2px 0px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FF7B45, stop:1 #F56A3B);
            }
        """)
        analysis_controls.addWidget(self.load_button)
        
        # Load audio file button
        self.load_audio_button = QPushButton("🎵 Load Audio File")
        self.load_audio_button.clicked.connect(self.load_audio_file)
        self.load_audio_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4CAF50, stop:1 #45a049);
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 10pt;
                margin: 2px 0px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5CBF60, stop:1 #55b059);
            }
        """)
        analysis_controls.addWidget(self.load_audio_button)
        
        # Diarization controls
        diarization_layout = QHBoxLayout()
        
        self.diarization_checkbox = QCheckBox("Enable Speaker Diarization")
        self.diarization_checkbox.setChecked(DIARIZATION_AVAILABLE and self.diarization_enabled)
        self.diarization_checkbox.setEnabled(DIARIZATION_AVAILABLE)
        self.diarization_checkbox.stateChanged.connect(self.on_diarization_toggle)
        self.diarization_checkbox.setStyleSheet("""
            QCheckBox {
                color: #ffffff;
                font-size: 10pt;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #555;
                border-radius: 3px;
                background: #2a2a2a;
            }
            QCheckBox::indicator:checked {
                background: #ff00ff;
                border-color: #ff00ff;
            }
            QCheckBox::indicator:checked:disabled {
                background: #666;
                border-color: #666;
            }
        """)
        
        self.edit_speakers_button = QPushButton("👥 Edit Speakers")
        self.edit_speakers_button.setEnabled(False)
        self.edit_speakers_button.clicked.connect(self.edit_speakers)
        self.edit_speakers_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #9C27B0, stop:1 #7B1FA2);
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 9pt;
                margin: 2px 0px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #AB47BC, stop:1 #8E24AA);
            }
            QPushButton:disabled {
                background: #666;
                color: #999;
            }
        """)
        
        diarization_layout.addWidget(self.diarization_checkbox)
        diarization_layout.addStretch()
        diarization_layout.addWidget(self.edit_speakers_button)
        analysis_controls.addLayout(diarization_layout)
        
        # Status label
        status_text = "Ready for analysis"
        if not DIARIZATION_AVAILABLE:
            status_text += " • Speaker diarization unavailable (install pyannote.audio)"
        self.analysis_status = QLabel(status_text)
        self.analysis_status.setFont(QFont("Segoe UI", 9))
        self.analysis_status.setStyleSheet("color: #888; margin-bottom: 10px;")
        self.analysis_status.setWordWrap(True)
        analysis_controls.addWidget(self.analysis_status)
        
        right_layout.addLayout(analysis_controls)
        
        # Analysis results tabs
        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #555;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #2a2a2a;
                color: #fff;
                padding: 6px 12px;
                margin-right: 1px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-size: 9pt;
            }
            QTabBar::tab:selected {
                background: #3a3a3a;
                color: #ff00ff;
            }
        """)
        
        # Summary tab
        self.summary_text = QTextEdit()
        self.summary_text.setPlainText("Summary will appear here after analysis...")
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet("""
            QTextEdit {
                background: rgba(76, 175, 80, 0.1);
                border: 1px solid rgba(76, 175, 80, 0.3);
                border-radius: 6px;
                padding: 10px;
                color: #4CAF50;
                font-size: 10pt;
            }
        """)
        self.analysis_tabs.addTab(self.summary_text, "📋 Summary")
        
        # Key points tab
        self.key_points_text = QTextEdit()
        self.key_points_text.setPlainText("Key points will appear here after analysis...")
        self.key_points_text.setReadOnly(True)
        self.key_points_text.setStyleSheet("""
            QTextEdit {
                background: rgba(255, 193, 7, 0.1);
                border: 1px solid rgba(255, 193, 7, 0.3);
                border-radius: 6px;
                padding: 10px;
                color: #FFC107;
                font-size: 10pt;
            }
        """)
        self.analysis_tabs.addTab(self.key_points_text, "🔑 Key Points")
        
        # Action items tab
        self.action_items_text = QTextEdit()
        self.action_items_text.setPlainText("Action items will appear here after analysis...")
        self.action_items_text.setReadOnly(True)
        self.action_items_text.setStyleSheet("""
            QTextEdit {
                background: rgba(244, 67, 54, 0.1);
                border: 1px solid rgba(244, 67, 54, 0.3);
                border-radius: 6px;
                padding: 10px;
                color: #F44336;
                font-size: 10pt;
            }
        """)
        self.analysis_tabs.addTab(self.action_items_text, "✅ Actions")
        
        right_layout.addWidget(self.analysis_tabs)
        right_layout.addStretch()
        
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
        # Check if transcription is in progress
        if self.transcription_in_progress and not self.recording:
            reply = QMessageBox.question(self, 
                'Transcription in Progress', 
                'A transcription is currently being processed. Starting a new recording will not affect it.\\n\\nDo you want to start a new recording?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            
            if reply != QMessageBox.StandardButton.Yes:
                return
        
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
            self.accumulated_live_text = ""
            self.voice_ball.set_recording(True)
            self.status_label.setText("Recording... Speak now!")
            
            # Clear live transcription display and reset accumulator
            # Live transcription disabled for stability
            self.live_text_sentences = []
            
            # Live transcription worker disabled to prevent crashes
            self.live_transcription_worker = None
            
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
        
        # Live transcription disabled - no audio chunks sent to prevent crashes
        
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
            
            # Live transcription worker is disabled - nothing to stop
            
            self.voice_ball.set_recording(False)
            self.status_label.setText("Recording stopped")
            
            # Save audio file
            if self.recorded_data:
                audio_data = np.concatenate(self.recorded_data, axis=0)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Store backup of recorded audio
                self.last_recorded_audio = audio_data.copy()
                
                # Auto-save audio as backup WAV file
                try:
                    backup_audio_path = os.path.join(self.output_directory, f"backup_audio_{timestamp}.wav")
                    # Convert to int16 for WAV format
                    if audio_data.dtype == np.float32:
                        audio_int16 = (audio_data * 32767).astype(np.int16)
                    else:
                        audio_int16 = audio_data.astype(np.int16)
                    write(backup_audio_path, self.sample_rate, audio_int16)
                    print(f"Audio backup saved: {backup_audio_path}")
                    
                    # Store audio file path for diarization
                    self.last_audio_file = backup_audio_path
                except Exception as e:
                    print(f"Failed to save audio backup: {e}")
                    self.last_audio_file = None
                
                # Set transcription in progress flag
                self.transcription_in_progress = True
                self.processing_label.setText("⚙️ Transcription in progress...")
                self.processing_label.setVisible(True)
                self.status_label.setText("Processing final transcription...")
                
                # Start final transcription using audio data directly
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                
                self.transcription_worker = TranscriptionWorker(audio_data, self.sample_rate)
                self.transcription_worker.transcription_ready.connect(self.on_transcription_ready)
                self.transcription_worker.segments_ready.connect(self.on_segments_ready)
                self.transcription_worker.progress_update.connect(self.progress_bar.setValue)
                self.transcription_worker.start()
            else:
                self.status_label.setText("No audio recorded")
                
        except Exception as e:
            self.status_label.setText(f"Error stopping recording: {e}")
    
    def on_live_transcription(self, text):
        # Accumulate live transcription text
        if text and text.strip():
            # Add new text to accumulated sentences
            self.live_text_sentences.append(text.strip())
            
            # Keep only last 10 sentences to avoid UI overflow
            if len(self.live_text_sentences) > 10:
                self.live_text_sentences = self.live_text_sentences[-10:]
            
            # Display accumulated text
            accumulated_text = ' '.join(self.live_text_sentences)
            self.live_transcription_text.setPlainText(accumulated_text)
            
            # Auto-scroll to bottom to show latest text
            cursor = self.live_transcription_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            self.live_transcription_text.setTextCursor(cursor)
    
    def on_model_changed(self, index):
        model_name = self.model_combo.itemData(index)
        if model_name:
            self.selected_model = model_name
            self.analysis_status.setText(f"Ready for analysis with {model_name.split('/')[-1]}")
    
    def analyze_transcription(self):
        if not self.current_transcription_text.strip():
            self.analysis_status.setText("No transcription to analyze")
            return
        
        if not LLM_AVAILABLE:
            self.analysis_status.setText("Run ./install_llm.sh to install LLM dependencies")
            return
            
        self.analyze_button.setEnabled(False)
        self.analysis_status.setText(f"🔄 Loading {self.selected_model.split('/')[-1]}...")
        
        # Clear previous results
        self.summary_text.setPlainText("Loading model and analyzing...")
        self.key_points_text.setPlainText("Loading model and analyzing...")
        self.action_items_text.setPlainText("Loading model and analyzing...")
        
        self.llm_worker = LLMAnalysisWorker(self.current_transcription_text, self.selected_model)
        self.llm_worker.analysis_ready.connect(self.on_analysis_ready)
        self.llm_worker.analysis_error.connect(self.on_analysis_error)
        self.llm_worker.start()
    
    def load_transcription_file(self):
        """Load a previous transcription file for analysis"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Transcription File",
            self.output_directory,
            "Text files (*.txt);;All files (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content:
                    # Load into final transcription area
                    self.transcription_text.setPlainText(content)
                    self.current_transcription_text = content
                    
                    # Enable analyze button
                    self.analyze_button.setEnabled(True)
                    
                    # Update status
                    filename = os.path.basename(file_path)
                    self.analysis_status.setText(f"Loaded: {filename} - Ready for analysis")
                    
                    # Clear previous analysis results
                    self.summary_text.setPlainText("Analysis will appear here after clicking 'Analyze Content'...")
                    self.key_points_text.setPlainText("Key points will appear here after analysis...")
                    self.action_items_text.setPlainText("Action items will appear here after analysis...")
                else:
                    self.analysis_status.setText("Error: File is empty")
                    
            except Exception as e:
                self.analysis_status.setText(f"Error loading file: {str(e)}")
    
    def load_audio_file(self):
        """Load an audio file and transcribe it"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Audio File",
            self.output_directory,
            "Audio files (*.wav *.mp3 *.m4a *.flac *.ogg);;WAV files (*.wav);;All files (*.*)"
        )
        
        if file_path:
            try:
                self.analysis_status.setText("Loading audio file...")
                
                # Import librosa for loading different audio formats
                try:
                    import librosa
                    import soundfile as sf
                    LIBROSA_AVAILABLE = True
                except ImportError:
                    LIBROSA_AVAILABLE = False
                
                # Load audio file
                if LIBROSA_AVAILABLE:
                    # Use librosa for better format support
                    audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
                    # Convert to the format expected by our transcription worker
                    audio_data = audio_data.astype(np.float32)
                else:
                    # Fallback: try to load with scipy for WAV files
                    from scipy.io import wavfile
                    sample_rate, audio_data = wavfile.read(file_path)
                    
                    # Convert to float32 and normalize
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    elif audio_data.dtype == np.int32:
                        audio_data = audio_data.astype(np.float32) / 2147483648.0
                    else:
                        audio_data = audio_data.astype(np.float32)
                    
                    # Handle stereo -> mono conversion
                    if len(audio_data.shape) > 1:
                        audio_data = np.mean(audio_data, axis=1)
                
                # Store the audio file path for diarization
                self.last_audio_file = file_path
                
                # Set up UI for transcription
                filename = os.path.basename(file_path)
                self.analysis_status.setText(f"Loaded: {filename} - Starting transcription...")
                self.transcription_text.setPlainText("Transcribing audio file...")
                
                # Start transcription process
                self.transcription_in_progress = True
                self.processing_label.setText("⚙️ Transcribing audio file...")
                self.processing_label.setVisible(True)
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                
                # Create and start transcription worker
                self.transcription_worker = TranscriptionWorker(audio_data, sample_rate)
                self.transcription_worker.transcription_ready.connect(self.on_transcription_ready)
                self.transcription_worker.segments_ready.connect(self.on_segments_ready)
                self.transcription_worker.progress_update.connect(self.progress_bar.setValue)
                self.transcription_worker.start()
                
                # Clear previous analysis results
                self.summary_text.setPlainText("Analysis will appear here after transcription completes...")
                self.key_points_text.setPlainText("Key points will appear here after analysis...")
                self.action_items_text.setPlainText("Action items will appear here after analysis...")
                
            except ImportError:
                self.analysis_status.setText("Install librosa for better audio format support: pip install librosa soundfile")
            except Exception as e:
                self.analysis_status.setText(f"Error loading audio file: {str(e)}")
                self.processing_label.setVisible(False)
                self.progress_bar.setVisible(False)
    
    def on_analysis_ready(self, analysis_type, result):
        if analysis_type == "summary":
            self.summary_text.setPlainText(result)
        elif analysis_type == "key_points":
            self.key_points_text.setPlainText(result)
        elif analysis_type == "actions":
            self.action_items_text.setPlainText(result)
        elif analysis_type == "status":
            self.analysis_status.setText(result)
            if "complete" in result.lower():
                self.analyze_button.setEnabled(True)
    
    def on_analysis_error(self, error_message):
        self.analysis_status.setText(f"Analysis failed: {error_message}")
        self.analyze_button.setEnabled(True)
        self.summary_text.setPlainText(f"Error: {error_message}")
        self.key_points_text.setPlainText(f"Error: {error_message}")
        self.action_items_text.setPlainText(f"Error: {error_message}")
    
    def on_diarization_toggle(self, state):
        self.diarization_enabled = state == Qt.CheckState.Checked.value
    
    def on_segments_ready(self, segments):
        """Store transcription segments for diarization"""
        self.transcription_segments = segments
        
        # Start diarization if enabled and we have an audio file
        if (self.diarization_enabled and DIARIZATION_AVAILABLE and 
            self.last_audio_file and os.path.exists(self.last_audio_file)):
            self.start_diarization()
    
    def start_diarization(self):
        """Start speaker diarization process"""
        if not self.last_audio_file:
            self.analysis_status.setText("No audio file available for diarization")
            return
            
        self.analysis_status.setText("Starting speaker diarization...")
        self.diarization_worker = DiarizationWorker(
            self.last_audio_file, 
            self.transcription_segments
        )
        self.diarization_worker.diarization_ready.connect(self.on_diarization_ready)
        self.diarization_worker.diarization_error.connect(self.on_diarization_error)
        self.diarization_worker.progress_update.connect(self.on_diarization_progress)
        self.diarization_worker.start()
    
    def on_diarization_progress(self, progress):
        """Update status during diarization"""
        self.analysis_status.setText(f"Diarization progress: {progress}%")
    
    def on_diarization_ready(self, speaker_segments):
        """Handle completed diarization results"""
        self.speaker_segments = speaker_segments
        self.edit_speakers_button.setEnabled(True)
        
        # Format transcription with speaker labels
        formatted_text = self.format_transcription_with_speakers(speaker_segments)
        self.transcription_text.setPlainText(formatted_text)
        self.current_transcription_text = formatted_text
        
        self.analysis_status.setText(f"Diarization complete. Found {len(set(s['speaker'] for s in speaker_segments))} speakers.")
    
    def on_diarization_error(self, error_message):
        """Handle diarization errors"""
        self.analysis_status.setText(f"Diarization failed: {error_message}")
        self.edit_speakers_button.setEnabled(False)
    
    def format_transcription_with_speakers(self, speaker_segments, speaker_names=None):
        """Format transcription text with speaker labels"""
        if not speaker_segments:
            return self.current_transcription_text
        
        formatted_lines = []
        current_speaker = None
        current_text = []
        
        # Sort segments by start time
        sorted_segments = sorted(speaker_segments, key=lambda x: x['start'])
        
        for segment in sorted_segments:
            speaker_id = segment['speaker']
            text = segment['text'].strip()
            
            if not text:
                continue
            
            # Get speaker name
            if speaker_names and speaker_id in speaker_names:
                speaker_name = speaker_names[speaker_id]
            else:
                # Use a more readable default format
                speaker_name = f"Speaker {speaker_id.split('_')[-1] if '_' in speaker_id else speaker_id}"
            
            if speaker_id != current_speaker:
                # New speaker, save previous and start new
                if current_speaker and current_text:
                    formatted_lines.append(f"{current_speaker}: {' '.join(current_text)}")
                current_speaker = speaker_name
                current_text = [text]
            else:
                # Same speaker, continue
                current_text.append(text)
        
        # Add the last speaker's text
        if current_speaker and current_text:
            formatted_lines.append(f"{current_speaker}: {' '.join(current_text)}")
        
        return '\n\n'.join(formatted_lines)
    
    def edit_speakers(self):
        """Open dialog to edit speaker names"""
        if not self.speaker_segments:
            QMessageBox.information(self, "No Speakers", "No speaker information available to edit.")
            return
        
        dialog = SpeakerEditDialog(self.speaker_segments, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            speaker_names = dialog.get_speaker_names()
            
            # Update the transcription with new speaker names
            formatted_text = self.format_transcription_with_speakers(self.speaker_segments, speaker_names)
            self.transcription_text.setPlainText(formatted_text)
            self.current_transcription_text = formatted_text
            
            self.analysis_status.setText("Speaker names updated successfully.")
    
    def on_transcription_ready(self, text):
        self.progress_bar.setVisible(False)
        self.transcription_in_progress = False
        self.processing_label.setVisible(False)
        self.transcription_text.setPlainText(text)
        
        # Store current transcription for analysis
        self.current_transcription_text = text
        self.analyze_button.setEnabled(True)
        self.analysis_status.setText("Ready for analysis")
        
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
        
        # Clear the backup audio data after successful transcription
        self.last_recorded_audio = None
    
    def closeEvent(self, event):
        if self.recording:
            self.stop_recording()
        # Live transcription worker is disabled
        if self.llm_worker and self.llm_worker.isRunning():
            self.llm_worker.terminate()
            self.llm_worker.wait()
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