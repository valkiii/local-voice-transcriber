import sys
import os
import threading
import queue
import math
import gc
from datetime import datetime
import numpy as np
import sounddevice as sd
import whisper
from scipy.io.wavfile import write
import tempfile
from collections import deque
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QGridLayout,
                           QWidget, QPushButton, QTextEdit, QLabel, QProgressBar,
                           QFileDialog, QComboBox, QGroupBox, QFrame, QMessageBox,
                           QTabWidget, QScrollArea, QDialog, QDialogButtonBox, 
                           QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
                           QSplitter)
# Ollama is now used for LLM functionality - no transformers needed

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment
    import torchaudio
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    print("Diarization dependencies not available. Install pyannote.audio for speaker identification.")
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QFont, QPalette, QColor, QPainter, QPen, QBrush, QLinearGradient, QIcon


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
    
    def __init__(self, audio_data, sample_rate, model=None):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.model = model
    
    def run(self):
        try:
            self.progress_update.emit(20)
            if self.model is None:
                try:
                    # Load model with safer settings to prevent crashes
                    import torch
                    # Force CPU usage to avoid GPU memory issues
                    self.model = whisper.load_model("base", device="cpu")
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
            
            # Transcribe directly from numpy array with safe settings
            # Always use safe settings to prevent segmentation faults
            try:
                # Limit audio length to prevent memory issues
                max_duration = 300  # 5 minutes max to prevent crashes
                max_samples = max_duration * 16000
                if len(audio_float) > max_samples:
                    audio_float = audio_float[:max_samples]
                    
                result = self.model.transcribe(
                    audio_float,
                    fp16=False,  # Always use FP32 for stability
                    verbose=False,  # Reduce output
                    word_timestamps=False,  # Disable for stability
                    temperature=0.0,  # Deterministic
                    compression_ratio_threshold=2.0,  # Lower threshold to catch repetition earlier
                    logprob_threshold=-0.5,  # Higher threshold to filter low-confidence
                    no_speech_threshold=0.8,  # Much higher threshold to avoid hallucination
                    condition_on_previous_text=False,  # Prevent context issues
                    initial_prompt=""  # No initial prompt to avoid bias
                )
            except Exception as transcribe_error:
                self.transcription_ready.emit(f"Transcription failed: {str(transcribe_error)}")
                return
            
            self.progress_update.emit(100)
            
            # Clean the transcription text for repetitions
            clean_text = self._clean_repetitive_text(result["text"])
            
            # Emit both text and segments
            self.transcription_ready.emit(clean_text)
            
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
    
    def _clean_repetitive_text(self, text):
        """Remove repetitive patterns that indicate Whisper hallucination"""
        if not text or len(text) < 20:
            return text
        
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return text
        
        # Detect repetitive patterns
        cleaned_sentences = []
        repetition_threshold = 3  # Number of repetitions to trigger cleaning
        
        for i, sentence in enumerate(sentences):
            # Check if this sentence repeats the previous ones
            if len(cleaned_sentences) >= repetition_threshold:
                recent_sentences = cleaned_sentences[-repetition_threshold:]
                # If current sentence is very similar to recent ones, likely repetition
                if any(self._similarity_ratio(sentence, prev) > 0.8 for prev in recent_sentences):
                    print(f"Detected repetitive sentence: '{sentence[:50]}...'")
                    continue
            
            # Check for repetitive phrases within the sentence
            words = sentence.split()
            if len(words) > 10:
                # Look for repeated phrases
                cleaned_words = self._remove_repeated_phrases(words)
                sentence = ' '.join(cleaned_words)
            
            cleaned_sentences.append(sentence)
        
        return '. '.join(cleaned_sentences) + ('.' if cleaned_sentences else '')
    
    def _similarity_ratio(self, text1, text2):
        """Calculate similarity ratio between two texts"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _remove_repeated_phrases(self, words):
        """Remove repeated phrases within a list of words"""
        if len(words) < 6:
            return words
        
        # Look for patterns of 2-4 words that repeat
        for phrase_len in range(2, 5):
            for start in range(len(words) - phrase_len * 2):
                phrase = words[start:start + phrase_len]
                
                # Count how many times this phrase repeats consecutively
                repeats = 1
                pos = start + phrase_len
                
                while pos + phrase_len <= len(words) and words[pos:pos + phrase_len] == phrase:
                    repeats += 1
                    pos += phrase_len
                
                # If phrase repeats 3+ times, keep only one instance
                if repeats >= 3:
                    # Remove the repeated portions
                    end_remove = start + phrase_len * repeats
                    return words[:start + phrase_len] + words[end_remove:]
        
        return words


class ChunkedTranscriptionWorker(QThread):
    transcription_ready = pyqtSignal(str)
    segments_ready = pyqtSignal(list)  
    progress_update = pyqtSignal(int)
    
    def __init__(self, audio_data, sample_rate, chunk_duration=300, model=None):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.model = model
        
    def run(self):
        try:
            self.progress_update.emit(5)
            
            # Use provided model or load new one
            if self.model is None:
                try:
                    import torch
                    # Force CPU usage to prevent crashes
                    self.model = whisper.load_model("base", device="cpu")
                    if self.model is None:
                        raise Exception("Failed to load Whisper model")
                except Exception as e:
                    self.transcription_ready.emit(f"Error loading Whisper model: {str(e)}")
                    return
            
            self.progress_update.emit(10)
            
            # Calculate chunks with overlap to preserve context
            chunk_samples = int(self.chunk_duration * self.sample_rate)
            overlap_samples = int(30 * self.sample_rate)  # 30 second overlap
            total_samples = len(self.audio_data)
            
            # Calculate chunks with overlap
            chunks = []
            start = 0
            chunk_id = 0
            
            while start < total_samples:
                end = min(start + chunk_samples, total_samples)
                chunks.append({
                    'id': chunk_id,
                    'start_sample': start,
                    'end_sample': end,
                    'start_time': start / self.sample_rate,
                    'end_time': end / self.sample_rate
                })
                
                # Move start forward, but with overlap
                if end >= total_samples:
                    break
                start = end - overlap_samples
                chunk_id += 1
            
            print(f"Processing {len(chunks)} overlapping chunks of {self.chunk_duration/60:.1f} minutes each...")
            
            # Process each chunk
            all_text = []
            all_segments = []
            
            for chunk_info in chunks:
                start_sample = chunk_info['start_sample']
                end_sample = chunk_info['end_sample']
                chunk_audio = self.audio_data[start_sample:end_sample]
                chunk_id = chunk_info['id']
                
                # Calculate time offset for this chunk
                time_offset = chunk_info['start_time']
                
                print(f"Processing chunk {chunk_id+1}/{len(chunks)} ({time_offset/60:.1f}-{chunk_info['end_time']/60:.1f}min)...")
                
                # Resample chunk to 16kHz if needed
                if self.sample_rate != 16000:
                    import scipy.signal
                    num_samples = int(len(chunk_audio) * 16000 / self.sample_rate)
                    chunk_audio = scipy.signal.resample(chunk_audio, num_samples)
                
                # Transcribe chunk with settings to prevent hallucination and repetition
                try:
                    result = self.model.transcribe(
                        chunk_audio,
                        fp16=False,  # Always use FP32 for stability
                        verbose=False,  # Reduce output
                        word_timestamps=False,  # Disable for stability
                        temperature=0.0,  # Deterministic
                        compression_ratio_threshold=2.0,  # Lower threshold to catch repetition earlier
                        logprob_threshold=-0.5,  # Higher threshold to filter low-confidence
                        no_speech_threshold=0.6,  # Moderate threshold
                        condition_on_previous_text=False,  # Prevent context issues
                        language="en",  # Force English to prevent language switching
                        initial_prompt="This is a technical presentation in English."  # Bias toward English
                    )
                    
                    # Post-process to detect and remove repetitive text
                    chunk_text = result["text"].strip()
                    chunk_text = self._clean_repetitive_text(chunk_text)
                    result["text"] = chunk_text
                    
                except Exception as chunk_error:
                    print(f"Chunk transcription error: {chunk_error}")
                    # Create empty result for failed chunk
                    result = {"text": "", "segments": []}
                
                chunk_text = result["text"].strip()
                
                # For overlapping chunks, only keep the first half of overlapped chunks
                # (except for the last chunk)
                if chunk_id < len(chunks) - 1 and chunk_text:
                    # This chunk overlaps with the next one, so trim the end
                    sentences = chunk_text.split('.')
                    if len(sentences) > 2:
                        # Keep first 75% of sentences to avoid overlap duplication
                        keep_count = int(len(sentences) * 0.75)
                        chunk_text = '.'.join(sentences[:keep_count]) + '.'
                
                if chunk_text and chunk_text not in all_text:  # Avoid exact duplicates
                    all_text.append(chunk_text)
                
                # Process segments with time offset
                if "segments" in result:
                    for segment in result["segments"]:
                        seg_start = segment.get('start', 0) + time_offset
                        # Only include segments from the non-overlapping part
                        if chunk_id == len(chunks) - 1 or seg_start < time_offset + (self.chunk_duration * 0.75):
                            adjusted_segment = {
                                'start': seg_start,
                                'end': segment.get('end', 0) + time_offset,
                                'text': segment.get('text', '')
                            }
                            all_segments.append(adjusted_segment)
                
                # Update progress
                chunk_progress = 10 + (chunk_id + 1) * 85 // len(chunks)
                self.progress_update.emit(chunk_progress)
            
            # Stitch all transcriptions together with intelligent merging
            full_text = self._merge_overlapping_text(all_text)
            
            self.progress_update.emit(100)
            
            # Emit results
            self.transcription_ready.emit(full_text)
            self.segments_ready.emit(all_segments)
            
            print(f"Chunked transcription complete: {len(full_text)} characters, {len(all_segments)} segments")
            
        except Exception as e:
            print(f"Chunked transcription error: {str(e)}")
            self.transcription_ready.emit(f"Error during chunked transcription: {str(e)}")
    
    def _merge_overlapping_text(self, text_chunks):
        """Intelligently merge overlapping text chunks"""
        if not text_chunks:
            return ""
        
        if len(text_chunks) == 1:
            return text_chunks[0]
        
        merged = text_chunks[0]
        
        for i in range(1, len(text_chunks)):
            current_chunk = text_chunks[i]
            
            # Find the best merge point by looking for common endings/beginnings
            merged_words = merged.split()
            current_words = current_chunk.split()
            
            if len(merged_words) < 5 or len(current_words) < 5:
                # Too short to find overlap, just append
                merged += " " + current_chunk
                continue
            
            # Look for overlap in the last part of merged and first part of current
            best_overlap = 0
            merge_point = len(merged_words)
            
            # Check last 20 words of merged against first 20 words of current
            search_length = min(20, len(merged_words), len(current_words))
            
            for overlap_len in range(1, search_length + 1):
                merged_end = merged_words[-overlap_len:]
                current_start = current_words[:overlap_len]
                
                # Calculate similarity
                similarity = self._calculate_word_similarity(merged_end, current_start)
                
                if similarity > 0.7 and overlap_len > best_overlap:
                    best_overlap = overlap_len
                    merge_point = len(merged_words) - overlap_len
            
            if best_overlap > 0:
                # Merge at the overlap point
                merged = ' '.join(merged_words[:merge_point]) + " " + current_chunk
            else:
                # No good overlap found, just append
                merged += " " + current_chunk
        
        return merged
    
    def _calculate_word_similarity(self, words1, words2):
        """Calculate similarity between two word lists"""
        if len(words1) != len(words2):
            return 0.0
        
        matches = sum(1 for w1, w2 in zip(words1, words2) if w1.lower() == w2.lower())
        return matches / len(words1) if words1 else 0.0
    
    def _clean_repetitive_text(self, text):
        """Remove repetitive patterns that indicate Whisper hallucination"""
        if not text or len(text) < 20:
            return text
        
        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return text
        
        # Detect repetitive patterns
        cleaned_sentences = []
        repetition_threshold = 3  # Number of repetitions to trigger cleaning
        
        for i, sentence in enumerate(sentences):
            # Check if this sentence repeats the previous ones
            if len(cleaned_sentences) >= repetition_threshold:
                recent_sentences = cleaned_sentences[-repetition_threshold:]
                # If current sentence is very similar to recent ones, likely repetition
                if any(self._similarity_ratio(sentence, prev) > 0.8 for prev in recent_sentences):
                    print(f"Detected repetitive sentence: '{sentence[:50]}...'")
                    continue
            
            # Check for repetitive phrases within the sentence
            words = sentence.split()
            if len(words) > 10:
                # Look for repeated phrases
                cleaned_words = self._remove_repeated_phrases(words)
                sentence = ' '.join(cleaned_words)
            
            cleaned_sentences.append(sentence)
        
        return '. '.join(cleaned_sentences) + ('.' if cleaned_sentences else '')
    
    def _similarity_ratio(self, text1, text2):
        """Calculate similarity ratio between two texts"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _remove_repeated_phrases(self, words):
        """Remove repeated phrases within a list of words"""
        if len(words) < 6:
            return words
        
        # Look for patterns of 2-4 words that repeat
        for phrase_len in range(2, 5):
            for start in range(len(words) - phrase_len * 2):
                phrase = words[start:start + phrase_len]
                
                # Count how many times this phrase repeats consecutively
                repeats = 1
                pos = start + phrase_len
                
                while pos + phrase_len <= len(words) and words[pos:pos + phrase_len] == phrase:
                    repeats += 1
                    pos += phrase_len
                
                # If phrase repeats 3+ times, keep only one instance
                if repeats >= 3:
                    # Remove the repeated portions
                    end_remove = start + phrase_len * repeats
                    return words[:start + phrase_len] + words[end_remove:]
        
        return words


class LLMAnalysisWorker(QThread):
    analysis_ready = pyqtSignal(str, str)  # analysis_type, result
    analysis_error = pyqtSignal(str)
    
    def __init__(self, text, model_name="gemma2:2b"):
        super().__init__()
        self.text = text
        self.model_name = model_name
        
    def run(self):
        try:
            self.analysis_ready.emit("status", f"Using Ollama model: {self.model_name}")
            
            # Check if using keyword-only analysis
            if self.model_name == "keyword-only":
                self.analysis_ready.emit("status", "Using keyword-based analysis")
                self._fallback_analysis()
                return
            
            # Check if Ollama is available
            import subprocess
            try:
                subprocess.run(["ollama", "list"], capture_output=True, check=True)
            except (FileNotFoundError, subprocess.CalledProcessError):
                self.analysis_error.emit("Ollama is not available. Please install Ollama or use keyword-only analysis.")
                return
            
            # Truncate text if too long to avoid context overflow
            text_for_analysis = self.text[:4000] + "..." if len(self.text) > 4000 else self.text
            
            self.analysis_ready.emit("status", "Generating summary...")
            
            # Generate summary
            summary_prompt = f"""Summarize the following transcript in 2-3 clear sentences. Focus on the main topics discussed.

Transcript: {text_for_analysis}

Summary:"""
            
            print(f"Generating summary with {self.model_name}...")
            summary = self._generate_response(summary_prompt)
            print(f"Summary result: '{summary[:100]}...'")
            
            if not summary or "error" in summary.lower():
                self.analysis_error.emit(f"Summary generation failed with {self.model_name}")
                return
                
            self.analysis_ready.emit("summary", summary)
            
            self.analysis_ready.emit("status", "Extracting key points...")
            
            # Generate key points
            key_points_prompt = f"""Extract the main key points from this transcript as a bullet list. Include only the most important ideas. Format as bullet points with • symbols.

Transcript: {text_for_analysis}

Key Points:"""
            
            print(f"Generating key points with {self.model_name}...")
            key_points_response = self._generate_response(key_points_prompt)
            
            if key_points_response and "error" not in key_points_response.lower():
                # Ensure proper bullet formatting
                if not key_points_response.strip().startswith("•"):
                    key_points = "• " + key_points_response.replace("\n", "\n• ")
                else:
                    key_points = key_points_response
            else:
                key_points = "• Could not extract key points"
                
            self.analysis_ready.emit("key_points", key_points)
            
            self.analysis_ready.emit("status", "Identifying action items...")
            
            # Generate action items
            action_prompt = f"""Identify any action items, tasks, or follow-up items mentioned in this transcript. List them clearly with • bullet points.

Transcript: {text_for_analysis}

Action Items:"""
            
            print(f"Generating actions with {self.model_name}...")
            actions_response = self._generate_response(action_prompt)
            
            if actions_response and "error" not in actions_response.lower():
                # Ensure proper bullet formatting
                if not actions_response.strip().startswith("•"):
                    actions = "• " + actions_response.replace("\n", "\n• ")
                else:
                    actions = actions_response
            else:
                actions = "• No specific action items identified"
                
            self.analysis_ready.emit("actions", actions)
            
            self.analysis_ready.emit("status", "Analysis complete")
            
        except Exception as e:
            # Fallback to keyword-based analysis if Ollama fails
            self.analysis_ready.emit("status", f"Ollama failed, using fallback analysis: {str(e)}")
            self._fallback_analysis()
    
    def _generate_response(self, prompt):
        try:
            import subprocess
            import json
            
            # Use Ollama API to generate response
            cmd = [
                "ollama", "run", self.model_name,
                prompt
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )
            
            if result.returncode != 0:
                print(f"Ollama error: {result.stderr}")
                return f"Generation failed: {result.stderr}"
            
            response = result.stdout.strip()
            
            # Post-process response
            response = self._post_process_response(response)
            
            return response[:800]  # Limit response length
            
        except subprocess.TimeoutExpired:
            return "Generation failed: Timeout"
        except Exception as e:
            print(f"Generation error with {self.model_name}: {str(e)}")
            return f"Generation failed: {str(e)}"
    
    def _post_process_response(self, response):
        """Clean up model response"""
        response = response.strip()
        
        # Remove common artifacts and prompts echoed back
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and prompt repetitions
            if (line and 
                not line.startswith("Transcript:") and 
                not line.startswith("Summary:") and
                not line.startswith("Key Points:") and
                not line.startswith("Action Items:")):
                cleaned_lines.append(line)
        
        response = '\n'.join(cleaned_lines)
        
        # Remove incomplete sentences at the end
        if '.' in response:
            sentences = response.split('.')
            if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
                response = '.'.join(sentences[:-1]) + '.'
        
        return response
    
    def _fallback_analysis(self):
        """Fallback keyword-based analysis if Ollama fails"""
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
            
            # Try local-first diarization approaches
            try:
                # First try older, more accessible model
                try:
                    self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1")
                except:
                    # If that fails, try even simpler approach with local VAD + clustering
                    self.diarization_error.emit("Advanced diarization unavailable. Using simple voice activity detection.")
                    self._use_simple_diarization()
                    return
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
    
    def _use_simple_diarization(self):
        """Simple local diarization using basic audio features and clustering"""
        try:
            import librosa
            from sklearn.cluster import KMeans
            from scipy.signal import find_peaks
            
            # Load audio file
            audio, sr = librosa.load(self.audio_file_path, sr=16000)
            
            # Simple voice activity detection using RMS energy
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            # Compute features for clustering
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, 
                                       hop_length=hop_length, n_fft=frame_length*2)
            
            # Simple energy-based VAD
            rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
            rms_threshold = np.percentile(rms, 30)  # Bottom 30% is likely silence
            
            # Find speech segments
            speech_frames = rms > rms_threshold
            
            # Extract features only from speech frames
            speech_features = mfccs[:, speech_frames].T
            
            if len(speech_features) < 10:
                # Not enough speech detected
                self.diarization_error.emit("Not enough speech detected for diarization")
                return
            
            # Cluster speech segments (assume 2-4 speakers)
            n_speakers = min(4, max(2, len(speech_features) // 100))  # Heuristic
            kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
            speaker_labels = kmeans.fit_predict(speech_features)
            
            # Map back to time segments
            frame_times = librosa.frames_to_time(np.arange(len(speech_frames)), 
                                                sr=sr, hop_length=hop_length)
            speech_times = frame_times[speech_frames]
            
            # Create speaker segments
            segments = []
            current_speaker = None
            segment_start = None
            
            for i, (time, speaker) in enumerate(zip(speech_times, speaker_labels)):
                if speaker != current_speaker:
                    # End previous segment
                    if current_speaker is not None and segment_start is not None:
                        text = self._find_matching_text(segment_start, time)
                        if text.strip():
                            segments.append({
                                'start': segment_start,
                                'end': time,
                                'speaker': f'SPEAKER_{current_speaker:02d}',
                                'text': text
                            })
                    
                    # Start new segment
                    current_speaker = speaker
                    segment_start = time
            
            # Add final segment
            if current_speaker is not None and segment_start is not None:
                text = self._find_matching_text(segment_start, speech_times[-1] + 1.0)
                if text.strip():
                    segments.append({
                        'start': segment_start,
                        'end': speech_times[-1] + 1.0,
                        'speaker': f'SPEAKER_{current_speaker:02d}',
                        'text': text
                    })
            
            self.progress_update.emit(100)
            self.diarization_ready.emit(segments)
            
        except ImportError:
            self.diarization_error.emit("Install scikit-learn and librosa for local diarization: pip install scikit-learn librosa")
        except Exception as e:
            self.diarization_error.emit(f"Simple diarization failed: {str(e)}")


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
        self.selected_model = "qwen2.5:1.5b"  # Default model
        self.live_text_sentences = []  # Store live transcription sentences
        self.diarization_worker = None
        self.speaker_segments = []  # Store speaker diarization results
        self.transcription_segments = []  # Store transcription segments with timing
        self.diarization_enabled = True
        self.last_audio_file = None  # Path to last saved audio file
        
        # Shared Whisper model to prevent multiple loading (causes crashes)
        self.shared_whisper_model = None
        
        self.init_ui()
        self.init_audio()
        
        # Timer for audio level updates (slower for smoother animation)
        self.level_timer = QTimer()
        self.level_timer.timeout.connect(self.update_audio_level)
    
    def load_shared_whisper_model(self):
        """Load the shared Whisper model to prevent multiple model loading crashes"""
        try:
            import torch
            print("Loading Whisper model (this may take a moment)...")
            # Force CPU usage and safe settings to prevent crashes
            self.shared_whisper_model = whisper.load_model("base", device="cpu")
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            self.shared_whisper_model = None
        
    def create_custom_dropdown(self):
        """Create a custom dropdown widget that mimics the React component"""
        dropdown_widget = QWidget()
        dropdown_layout = QVBoxLayout(dropdown_widget)
        dropdown_layout.setContentsMargins(0, 0, 0, 0)
        dropdown_layout.setSpacing(4)
        
        # Create the main button with arrow
        self.dropdown_button = QPushButton("Qwen2.5 1.5B (Fast)  ▼")
        self.dropdown_button.setCheckable(True)
        self.dropdown_button.clicked.connect(self.toggle_dropdown)
        
        # Style the main button
        self.dropdown_button.setStyleSheet("""
            QPushButton {
                background: #2a2a2a;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 8px 12px;
                color: #ffffff;
                font-size: 10pt;
                font-family: "Circular", "SF Pro Display", "Segoe UI", sans-serif;
                text-align: left;
                min-height: 16px;
            }
            QPushButton:hover {
                background: #3a3a3a;
                border: 1px solid #777;
            }
            QPushButton:checked {
                background: #3a3a3a;
                border: 1px solid #777;
            }
        """)
        
        # Create dropdown options container as overlay
        self.dropdown_options = QWidget(dropdown_widget)
        self.dropdown_options.setVisible(False)
        self.dropdown_options.setStyleSheet("""
            QWidget {
                background: #2a2a2a;
                border: 1px solid #555;
                border-radius: 8px;
            }
        """)
        
        # Position it as an overlay (absolute positioning)
        self.dropdown_options.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint)
        self.dropdown_options.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        
        options_layout = QVBoxLayout(self.dropdown_options)
        options_layout.setContentsMargins(4, 4, 4, 4)
        options_layout.setSpacing(2)
        
        # Create option buttons
        self.model_options = [
            ("Qwen2.5 1.5B (Fast)", "qwen2.5:1.5b"),
            ("Gemma2 2B (Fastest)", "gemma2:2b"),
            ("Keyword Analysis (No Download)", "keyword-only")
        ]
        
        self.option_buttons = []
        for option_text, option_value in self.model_options:
            option_btn = QPushButton(option_text)
            option_btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 12px;
                    color: #ffffff;
                    font-size: 10pt;
                    font-family: "Circular", "SF Pro Display", "Segoe UI", sans-serif;
                    text-align: left;
                }
                QPushButton:hover {
                    background: #3a3a3a;
                }
            """)
            option_btn.clicked.connect(lambda checked, text=option_text, value=option_value: self.select_model_option(text, value))
            options_layout.addWidget(option_btn)
            self.option_buttons.append(option_btn)
        
        dropdown_layout.addWidget(self.dropdown_button)
        # Don't add dropdown_options to layout - it's positioned as overlay
        
        # Store current selection
        self.current_model = "qwen2.5:1.5b"
        
        return dropdown_widget
    
    def toggle_dropdown(self):
        """Toggle the dropdown visibility"""
        is_visible = self.dropdown_options.isVisible()
        
        if not is_visible:
            # Position the dropdown below the button
            button_pos = self.dropdown_button.mapToGlobal(self.dropdown_button.rect().bottomLeft())
            self.dropdown_options.move(button_pos.x(), button_pos.y() + 4)
            self.dropdown_options.resize(self.dropdown_button.width(), self.dropdown_options.sizeHint().height())
            self.dropdown_options.show()
            self.dropdown_button.setChecked(True)
        else:
            self.dropdown_options.hide()
            self.dropdown_button.setChecked(False)
    
    def select_model_option(self, text, value):
        """Handle model selection"""
        self.dropdown_button.setText(f"{text}  ▼")
        self.current_model = value
        self.dropdown_options.hide()
        self.dropdown_button.setChecked(False)
        
        # Call the original model change handler
        self.on_model_changed_custom(value)
    
    def on_model_changed_custom(self, model_value):
        """Handle model change for custom dropdown"""
        self.selected_model = model_value
        self.analysis_status.setText(f"Ready for analysis with {model_value.split('/')[-1]}")
    
    def show_analysis_content(self, content_type):
        """Show the selected analysis content and update button states"""
        # Uncheck all buttons first
        self.summary_button.setChecked(False)
        self.key_points_button.setChecked(False)
        self.actions_button.setChecked(False)
        
        # Hide all content areas
        self.summary_text.setVisible(False)
        self.key_points_text.setVisible(False)
        self.action_items_text.setVisible(False)
        
        # Show selected content and check corresponding button
        if content_type == 'summary':
            self.summary_button.setChecked(True)
            self.summary_text.setVisible(True)
        elif content_type == 'key_points':
            self.key_points_button.setChecked(True)
            self.key_points_text.setVisible(True)
        elif content_type == 'actions':
            self.actions_button.setChecked(True)
            self.action_items_text.setVisible(True)
        
    def init_ui(self):
        self.setWindowTitle("Voice Transcriber with AI Analysis")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set application icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
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
        analysis_title.setFont(QFont("Circular", 16, QFont.Weight.Bold))
        analysis_title.setStyleSheet("color: #ff00ff; margin-bottom: 15px; font-family: 'Circular', 'SF Pro Display', 'Segoe UI', sans-serif;")
        right_layout.addWidget(analysis_title)
        
        # Grid layout for the action buttons and model selection
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(12)  # Increased spacing for better card separation
        grid_layout.setContentsMargins(0, 0, 0, 0)
        
        # Position 1: Load Transcription (top-left) - Green accent for loading
        self.load_button = QPushButton("📁 Load Transcription")
        self.load_button.clicked.connect(self.load_transcription_file)
        self.load_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(45, 85, 45, 0.9), stop:1 rgba(35, 65, 35, 0.9));
                color: #e8f5e8;
                border: 1px solid rgba(76, 175, 80, 0.3);
                padding: 15px;
                border-radius: 16px;
                font-weight: bold;
                font-size: 10pt;
                font-family: "Circular", "SF Pro Display", "Segoe UI", sans-serif;
                min-height: 50px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(55, 95, 55, 0.95), stop:1 rgba(45, 75, 45, 0.95));
                border: 1px solid rgba(76, 175, 80, 0.5);
            }
        """)
        grid_layout.addWidget(self.load_button, 0, 0)
        
        # Position 2: Analyze Content (top-center) - Purple accent for analysis
        self.analyze_button = QPushButton("🧠 Analyze Content")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.analyze_transcription)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(85, 45, 95, 0.9), stop:1 rgba(65, 35, 75, 0.9));
                color: #f0e8f5;
                border: 1px solid rgba(156, 39, 176, 0.3);
                padding: 15px;
                border-radius: 16px;
                font-weight: bold;
                font-size: 10pt;
                font-family: "Circular", "SF Pro Display", "Segoe UI", sans-serif;
                min-height: 50px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(95, 55, 105, 0.95), stop:1 rgba(75, 45, 85, 0.95));
                border: 1px solid rgba(156, 39, 176, 0.5);
            }
            QPushButton:disabled {
                background: rgba(45, 45, 45, 0.7);
                color: rgba(120, 120, 120, 0.8);
                border: 1px solid rgba(80, 80, 80, 0.3);
            }
        """)
        grid_layout.addWidget(self.analyze_button, 0, 1)
        
        # Position 3: Model Selection (top-right) - 1.5x larger with neutral tinting
        model_container = QWidget()
        model_container.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(55, 50, 45, 0.9), stop:1 rgba(45, 40, 35, 0.9));
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 16px;
                padding: 15px;
            }
            QWidget:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(65, 60, 55, 0.95), stop:1 rgba(55, 50, 45, 0.95));
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
        """)
        model_layout = QVBoxLayout(model_container)
        model_layout.setContentsMargins(8, 8, 8, 8)
        model_layout.setSpacing(5)
        
        model_label = QLabel("Model:")
        model_label.setFont(QFont("Circular", 9))
        model_label.setStyleSheet("color: #ff69b4; font-family: 'Circular', 'SF Pro Display', 'Segoe UI', sans-serif; background: transparent; border: none; margin-bottom: 4px;")
        model_layout.addWidget(model_label)
        
        # Create custom dropdown widget
        self.custom_dropdown = self.create_custom_dropdown()
        model_layout.addWidget(self.custom_dropdown)
        
        # Add model container spanning 1.5x width (using column span)
        grid_layout.addWidget(model_container, 0, 2, 1, 2)  # spans 2 columns for 1.5x effect
        
        # Position 4: Load Audio File (bottom-left) - Green accent for loading
        self.load_audio_button = QPushButton("🎵 Load Audio File")
        self.load_audio_button.clicked.connect(self.load_audio_file)
        self.load_audio_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(45, 85, 45, 0.9), stop:1 rgba(35, 65, 35, 0.9));
                color: #e8f5e8;
                border: 1px solid rgba(76, 175, 80, 0.3);
                padding: 15px;
                border-radius: 16px;
                font-weight: bold;
                font-size: 10pt;
                font-family: "Circular", "SF Pro Display", "Segoe UI", sans-serif;
                min-height: 50px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(55, 95, 55, 0.95), stop:1 rgba(45, 75, 45, 0.95));
                border: 1px solid rgba(76, 175, 80, 0.5);
            }
        """)
        grid_layout.addWidget(self.load_audio_button, 1, 0)
        
        # Position 5: Clean Backup Audio (bottom-center) - Orange accent for deleting
        self.clean_backup_button = QPushButton("🗑️ Clean Backup Audio")
        self.clean_backup_button.clicked.connect(self.clean_backup_audio)
        self.clean_backup_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(95, 55, 35, 0.9), stop:1 rgba(75, 45, 25, 0.9));
                color: #fff5e8;
                border: 1px solid rgba(255, 152, 0, 0.3);
                padding: 15px;
                border-radius: 16px;
                font-weight: bold;
                font-size: 10pt;
                font-family: "Circular", "SF Pro Display", "Segoe UI", sans-serif;
                min-height: 50px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(105, 65, 45, 0.95), stop:1 rgba(85, 55, 35, 0.95));
                border: 1px solid rgba(255, 152, 0, 0.5);
            }
        """)
        grid_layout.addWidget(self.clean_backup_button, 1, 1)
        
        # Set column stretch ratios to accommodate 1.5x model selection
        grid_layout.setColumnStretch(0, 2)  # Load Transcription
        grid_layout.setColumnStretch(1, 2)  # Analyze Content
        grid_layout.setColumnStretch(2, 2)  # Model Selection part 1
        grid_layout.setColumnStretch(3, 1)  # Model Selection part 2 (total 3 for 1.5x effect)
        
        right_layout.addWidget(grid_widget)
        
        # Settings/Diarization card
        settings_card = QWidget()
        settings_card.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(55, 50, 45, 0.9), stop:1 rgba(45, 40, 35, 0.9));
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 16px;
                padding: 18px;
                margin-bottom: 15px;
            }
            QWidget:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(65, 60, 55, 0.95), stop:1 rgba(55, 50, 45, 0.95));
                border: 1px solid rgba(255, 255, 255, 0.3);
            }
        """)
        settings_layout = QVBoxLayout(settings_card)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(10)
        
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
                padding: 8px 12px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 9pt;
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
        settings_layout.addLayout(diarization_layout)
        
        # Status label
        status_text = "Ready for analysis"
        if not DIARIZATION_AVAILABLE:
            status_text += " • Speaker diarization unavailable (requires Hugging Face authentication)"
        self.analysis_status = QLabel(status_text)
        self.analysis_status.setFont(QFont("Segoe UI", 9))
        self.analysis_status.setStyleSheet("color: #888; margin-top: 5px;")
        self.analysis_status.setWordWrap(True)
        settings_layout.addWidget(self.analysis_status)
        
        right_layout.addWidget(settings_card)
        
        # Analysis results with oval buttons
        analysis_container = QWidget()
        analysis_layout = QVBoxLayout(analysis_container)
        analysis_layout.setContentsMargins(0, 0, 0, 0)
        analysis_layout.setSpacing(12)
        
        # Oval button controls
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)
        
        # Create oval buttons
        self.summary_button = QPushButton("📊 Summary")
        self.key_points_button = QPushButton("🔑 Key Points")
        self.actions_button = QPushButton("✅ Actions")
        
        # Style oval buttons
        oval_button_style = """
            QPushButton {
                background: #3a3a3a;
                border: 1px solid #555;
                border-radius: 20px;
                padding: 8px 16px;
                color: #ffffff;
                font-size: 9pt;
                font-family: "Circular", "SF Pro Display", "Segoe UI", sans-serif;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background: #4a4a4a;
                border: 1px solid #777;
            }
            QPushButton:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 105, 180, 0.8), stop:1 rgba(255, 69, 180, 0.9));
                border: 1px solid rgba(255, 105, 180, 0.6);
                color: #ffffff;
            }
        """
        
        # Make buttons checkable and connect handlers
        self.summary_button.setCheckable(True)
        self.summary_button.setChecked(True)  # Default selected
        self.summary_button.clicked.connect(lambda: self.show_analysis_content('summary'))
        self.summary_button.setStyleSheet(oval_button_style)
        
        self.key_points_button.setCheckable(True)
        self.key_points_button.clicked.connect(lambda: self.show_analysis_content('key_points'))
        self.key_points_button.setStyleSheet(oval_button_style)
        
        self.actions_button.setCheckable(True)
        self.actions_button.clicked.connect(lambda: self.show_analysis_content('actions'))
        self.actions_button.setStyleSheet(oval_button_style)
        
        # Add buttons to layout
        button_layout.addWidget(self.summary_button)
        button_layout.addWidget(self.key_points_button)
        button_layout.addWidget(self.actions_button)
        button_layout.addStretch()
        
        analysis_layout.addWidget(button_container)
        
        # Content area
        self.content_container = QWidget()
        self.content_layout = QVBoxLayout(self.content_container)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        
        # Summary content
        self.summary_text = QTextEdit()
        self.summary_text.setPlainText("Summary will appear here after analysis...")
        self.summary_text.setReadOnly(True)
        self.summary_text.setStyleSheet("""
            QTextEdit {
                background: rgba(76, 175, 80, 0.1);
                border: 1px solid rgba(76, 175, 80, 0.3);
                border-radius: 8px;
                padding: 15px;
                color: #4CAF50;
                font-size: 10pt;
            }
        """)
        
        # Key points content
        self.key_points_text = QTextEdit()
        self.key_points_text.setPlainText("Key points will appear here after analysis...")
        self.key_points_text.setReadOnly(True)
        self.key_points_text.setVisible(False)
        self.key_points_text.setStyleSheet("""
            QTextEdit {
                background: rgba(255, 193, 7, 0.1);
                border: 1px solid rgba(255, 193, 7, 0.3);
                border-radius: 8px;
                padding: 15px;
                color: #FFC107;
                font-size: 10pt;
            }
        """)
        
        # Action items content
        self.action_items_text = QTextEdit()
        self.action_items_text.setPlainText("Action items will appear here after analysis...")
        self.action_items_text.setReadOnly(True)
        self.action_items_text.setVisible(False)
        self.action_items_text.setStyleSheet("""
            QTextEdit {
                background: rgba(244, 67, 54, 0.1);
                border: 1px solid rgba(244, 67, 54, 0.3);
                border-radius: 8px;
                padding: 15px;
                color: #F44336;
                font-size: 10pt;
            }
        """)
        
        # Add all content areas to the container
        self.content_layout.addWidget(self.summary_text)
        self.content_layout.addWidget(self.key_points_text)
        self.content_layout.addWidget(self.action_items_text)
        
        analysis_layout.addWidget(self.content_container)
        
        right_layout.addWidget(analysis_container)
        right_layout.addStretch()
        
        # Set dark theme with Spotify-inspired gradient
        self.setStyleSheet("""
            /* Spotify-Inspired Dynamic Gradient Background */
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #1a0f0d, stop:0.3 #2d1b16, stop:0.7 #3d261f, stop:1 #4a3026);
                color: #ffffff;
            }
            QMainWindow > QWidget {
                background: transparent;
                color: #ffffff;
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
                
                # Initialize shared model if not already loaded
                if self.shared_whisper_model is None:
                    self.load_shared_whisper_model()
                
                self.transcription_worker = TranscriptionWorker(audio_data, self.sample_rate, self.shared_whisper_model)
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
        
        # LLM functionality is now provided by Ollama
            
        self.analyze_button.setEnabled(False)
        self.analysis_status.setText(f"🔄 Using Ollama model {self.selected_model}...")
        
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
        """Load an audio file and transcribe it with chunking for stability"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Audio File",
            self.output_directory,
            "Audio files (*.wav *.mp3 *.m4a *.flac *.ogg);;WAV files (*.wav);;All files (*.*)"
        )
        
        if file_path:
            try:
                # Check file size first (limit to ~100MB to prevent crashes)
                file_size = os.path.getsize(file_path)
                max_size = 100 * 1024 * 1024  # 100MB limit
                
                if file_size > max_size:
                    reply = QMessageBox.question(
                        self,
                        'Large Audio File',
                        f'Audio file is {file_size / (1024*1024):.1f}MB. Large files may cause crashes.\n\nContinue anyway?',
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return
                
                self.analysis_status.setText("Loading audio file...")
                
                # Import librosa for loading different audio formats
                try:
                    import librosa
                    import soundfile as sf
                    LIBROSA_AVAILABLE = True
                except ImportError:
                    LIBROSA_AVAILABLE = False
                
                # Load audio file with chunked processing for long files
                chunk_duration = 300  # 5 minutes per chunk for processing
                
                if LIBROSA_AVAILABLE:
                    # Get full duration first
                    duration = librosa.get_duration(path=file_path)
                    
                    if duration > chunk_duration:
                        # Inform user about chunked processing
                        QMessageBox.information(
                            self,
                            'Long Audio File',
                            f'Audio is {duration/60:.1f} minutes long. It will be processed in {chunk_duration//60}-minute chunks and stitched together.\n\nThis may take a while but will transcribe the full audio.',
                            QMessageBox.StandardButton.Ok
                        )
                        
                        # Load full file for chunked processing
                        audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
                        self.process_chunked_audio = True
                        self.chunk_duration = chunk_duration
                    else:
                        # Load full file normally
                        audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
                        self.process_chunked_audio = False
                    
                    # Convert to the format expected by our transcription worker
                    audio_data = audio_data.astype(np.float32)
                else:
                    # Fallback: try to load with scipy for WAV files
                    from scipy.io import wavfile
                    sample_rate, audio_data = wavfile.read(file_path)
                    
                    # Check duration for chunked processing
                    duration = len(audio_data) / sample_rate
                    if duration > chunk_duration:
                        QMessageBox.information(
                            self,
                            'Long Audio File',
                            f'Audio is {duration/60:.1f} minutes long. It will be processed in {chunk_duration//60}-minute chunks and stitched together.\n\nThis may take a while but will transcribe the full audio.',
                            QMessageBox.StandardButton.Ok
                        )
                        self.process_chunked_audio = True
                        self.chunk_duration = chunk_duration
                    else:
                        self.process_chunked_audio = False
                    
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
                
                # Calculate final duration for display
                duration = len(audio_data) / sample_rate
                
                # Store the audio file path for diarization
                self.last_audio_file = file_path
                
                # Set up UI for transcription
                filename = os.path.basename(file_path)
                duration_min = len(audio_data) / sample_rate / 60
                self.analysis_status.setText(f"Loaded: {filename} ({duration_min:.1f}min) - Starting transcription...")
                self.transcription_text.setPlainText("Transcribing audio file...")
                
                # Start transcription process
                self.transcription_in_progress = True
                self.processing_label.setText("⚙️ Transcribing audio file...")
                self.processing_label.setVisible(True)
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)
                
                # Initialize shared model if not already loaded
                if self.shared_whisper_model is None:
                    self.load_shared_whisper_model()
                
                # Create and start appropriate transcription worker
                if hasattr(self, 'process_chunked_audio') and self.process_chunked_audio:
                    # Use chunked processing for long audio
                    self.transcription_worker = ChunkedTranscriptionWorker(
                        audio_data, 
                        sample_rate, 
                        self.chunk_duration,
                        self.shared_whisper_model
                    )
                    print(f"Using chunked transcription for {duration/60:.1f}min audio file")
                else:
                    # Use regular processing for short audio
                    self.transcription_worker = TranscriptionWorker(audio_data, sample_rate, self.shared_whisper_model)
                    print(f"Using regular transcription for {duration/60:.1f}min audio file")
                
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
    
    def clean_backup_audio(self):
        """Clean up backup audio files to free disk space"""
        try:
            # Find backup audio files
            backup_files = []
            for file in os.listdir(self.output_directory):
                if file.startswith("backup_audio_") and file.endswith(".wav"):
                    backup_files.append(os.path.join(self.output_directory, file))
            
            if not backup_files:
                QMessageBox.information(
                    self,
                    "No Backup Files",
                    "No backup audio files found to delete."
                )
                return
            
            # Calculate total size
            total_size = sum(os.path.getsize(f) for f in backup_files if os.path.exists(f))
            size_mb = total_size / (1024 * 1024)
            
            # Confirm deletion
            reply = QMessageBox.question(
                self,
                "Delete Backup Audio Files",
                f"Found {len(backup_files)} backup audio files ({size_mb:.1f} MB).\n\n"
                f"These are automatically created copies of your recordings.\n"
                f"Deleting them will free up disk space but you'll lose the audio backups.\n\n"
                f"Do you want to delete all backup audio files?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                deleted_count = 0
                deleted_size = 0
                
                for file_path in backup_files:
                    try:
                        if os.path.exists(file_path):
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            deleted_count += 1
                            deleted_size += file_size
                            print(f"Deleted: {os.path.basename(file_path)}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
                
                deleted_mb = deleted_size / (1024 * 1024)
                QMessageBox.information(
                    self,
                    "Cleanup Complete",
                    f"Successfully deleted {deleted_count} backup audio files.\n"
                    f"Freed {deleted_mb:.1f} MB of disk space."
                )
                
                self.analysis_status.setText(f"Deleted {deleted_count} backup files ({deleted_mb:.1f} MB freed)")
                
        except Exception as e:
            QMessageBox.critical(
                self,
                "Cleanup Error",
                f"Error during backup cleanup: {str(e)}"
            )
            self.analysis_status.setText(f"Cleanup error: {str(e)}")
    
    def closeEvent(self, event):
        if self.recording:
            self.stop_recording()
        
        # Clean up workers
        if self.transcription_worker and self.transcription_worker.isRunning():
            self.transcription_worker.terminate()
            self.transcription_worker.wait()
        
        if self.llm_worker and self.llm_worker.isRunning():
            self.llm_worker.terminate()
            self.llm_worker.wait()
        
        if self.diarization_worker and self.diarization_worker.isRunning():
            self.diarization_worker.terminate()
            self.diarization_worker.wait()
        
        # Clean up shared model
        if self.shared_whisper_model is not None:
            del self.shared_whisper_model
            self.shared_whisper_model = None
        
        # Force garbage collection to prevent memory leaks
        gc.collect()
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Voice Transcriber")
    app.setApplicationVersion("1.0")
    
    # Set application icon
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    window = VoiceTranscriberApp()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()