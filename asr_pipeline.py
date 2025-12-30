"""
ASR Pipeline: Speech-to-text using faster-whisper (GPU) with Vosk fallback.
Provides incremental transcript updates for real-time display.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Callable
import time
import threading
import queue
import logging
import os

from config import config

logger = logging.getLogger(__name__)

# Try to import faster-whisper
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False
    logger.warning("faster-whisper not installed")

# Try to import Vosk as fallback
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    import json
    HAS_VOSK = True
except ImportError:
    HAS_VOSK = False
    logger.warning("Vosk not installed (optional fallback)")

@dataclass
class TranscriptSegment:
    """A segment of transcribed text."""
    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    is_final: bool = True
    words: List[dict] = field(default_factory=list)

@dataclass
class ASRResult:
    """Result from ASR processing."""
    segments: List[TranscriptSegment] = field(default_factory=list)
    full_text: str = ""
    is_partial: bool = False
    processing_time_ms: float = 0.0

class ASRPipeline:
    """
    Streaming ASR using faster-whisper with Vosk fallback.
    Buffers audio and processes in chunks for low-latency transcription.
    """
    
    def __init__(self, word_callback: Optional[Callable[[str, float], None]] = None):
        """
        Initialize ASR pipeline.
        
        Args:
            word_callback: Called for each transcribed word (word, timestamp)
        """
        self.word_callback = word_callback
        self.sample_rate = config.asr.SAMPLE_RATE if hasattr(config.asr, 'SAMPLE_RATE') else 16000
        
        # Audio buffer
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        # Transcript accumulator
        self.transcript_segments: List[TranscriptSegment] = []
        self.full_transcript = ""
        
        # Processing state
        self.last_process_time = time.time()
        self.processing = False
        
        # Initialize model
        self.model = None
        self.vosk_model = None
        self.vosk_recognizer = None
        self.using_vosk = False
        
        self._init_model()
        
    def _init_model(self):
        """Initialize ASR model (faster-whisper preferred, Vosk fallback)."""
        
        if HAS_FASTER_WHISPER:
            try:
                logger.info(f"Loading faster-whisper model: {config.asr.MODEL_SIZE}")
                logger.info(f"Device: {config.asr.DEVICE}, Compute: {config.asr.COMPUTE_TYPE}")
                
                self.model = WhisperModel(
                    config.asr.MODEL_SIZE,
                    device=config.asr.DEVICE,
                    compute_type=config.asr.COMPUTE_TYPE
                )
                
                logger.info("faster-whisper model loaded successfully")
                return
                
            except Exception as e:
                logger.warning(f"Failed to load faster-whisper: {e}")
                logger.info("Falling back to CPU mode...")
                
                try:
                    self.model = WhisperModel(
                        config.asr.MODEL_SIZE,
                        device="cpu",
                        compute_type="int8"
                    )
                    logger.info("faster-whisper loaded in CPU mode")
                    return
                except Exception as e2:
                    logger.warning(f"CPU fallback also failed: {e2}")
        
        # Try Vosk fallback
        if HAS_VOSK and config.asr.USE_VOSK_FALLBACK:
            try:
                model_path = config.asr.VOSK_MODEL_PATH
                if os.path.exists(model_path):
                    logger.info(f"Loading Vosk model from {model_path}")
                    self.vosk_model = VoskModel(model_path)
                    self.vosk_recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
                    self.vosk_recognizer.SetWords(True)
                    self.using_vosk = True
                    logger.info("Vosk model loaded successfully")
                    return
                else:
                    logger.warning(f"Vosk model not found at {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load Vosk: {e}")
        
        logger.error("No ASR model available! Transcription will be disabled.")
    
    def add_audio(self, audio_bytes: bytes):
        """
        Add audio chunk to buffer.
        Audio should be 16-bit PCM, 16kHz, mono.
        """
        try:
            # Convert to float32
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0  # Normalize to [-1, 1]
            
            with self.buffer_lock:
                self.audio_buffer = np.concatenate([self.audio_buffer, audio])
                
        except Exception as e:
            logger.error(f"Failed to add audio: {e}")
    
    def process(self) -> Optional[ASRResult]:
        """
        Process buffered audio if enough has accumulated.
        Returns transcription result or None if not enough audio.
        """
        with self.buffer_lock:
            buffer_duration = len(self.audio_buffer) / self.sample_rate
            
            # Check if we have enough audio
            if buffer_duration < config.asr.MIN_AUDIO_LENGTH_SEC:
                return None
            
            # Force process if buffer is too long
            should_process = buffer_duration >= config.asr.MAX_AUDIO_LENGTH_SEC
            
            # Also process if we've waited long enough
            time_since_last = time.time() - self.last_process_time
            if time_since_last >= 1.0 and buffer_duration >= config.asr.MIN_AUDIO_LENGTH_SEC:
                should_process = True
            
            if not should_process:
                return None
            
            # Copy buffer and clear
            audio_to_process = self.audio_buffer.copy()
            self.audio_buffer = np.array([], dtype=np.float32)
        
        # Process audio
        self.processing = True
        start_time = time.time()
        
        try:
            if self.using_vosk:
                result = self._process_vosk(audio_to_process)
            elif self.model is not None:
                result = self._process_whisper(audio_to_process)
            else:
                result = ASRResult()  # No model available
            
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            # Update transcript
            for segment in result.segments:
                self.transcript_segments.append(segment)
                self.full_transcript += " " + segment.text
                
                # Notify word callback
                if self.word_callback:
                    for word_info in segment.words:
                        self.word_callback(word_info.get('word', ''), time.time())
            
            self.last_process_time = time.time()
            return result
            
        except Exception as e:
            logger.error(f"ASR processing error: {e}")
            return ASRResult()
        finally:
            self.processing = False
    
    def _process_whisper(self, audio: np.ndarray) -> ASRResult:
        """Process audio with faster-whisper."""
        result = ASRResult()
        
        try:
            segments, info = self.model.transcribe(
                audio,
                language=config.asr.LANGUAGE,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=200
                )
            )
            
            for segment in segments:
                words = []
                if segment.words:
                    for word in segment.words:
                        words.append({
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': word.probability
                        })
                
                result.segments.append(TranscriptSegment(
                    text=segment.text.strip(),
                    start_time=segment.start,
                    end_time=segment.end,
                    confidence=segment.avg_logprob if hasattr(segment, 'avg_logprob') else 1.0,
                    is_final=True,
                    words=words
                ))
                
                result.full_text += " " + segment.text.strip()
            
            result.full_text = result.full_text.strip()
            
        except Exception as e:
            logger.error(f"Whisper processing error: {e}")
        
        return result
    
    def _process_vosk(self, audio: np.ndarray) -> ASRResult:
        """Process audio with Vosk."""
        result = ASRResult()
        
        try:
            # Convert to int16 for Vosk
            audio_int16 = (audio * 32768).astype(np.int16).tobytes()
            
            if self.vosk_recognizer.AcceptWaveform(audio_int16):
                vosk_result = json.loads(self.vosk_recognizer.Result())
                
                if 'text' in vosk_result and vosk_result['text']:
                    words = []
                    if 'result' in vosk_result:
                        for word_info in vosk_result['result']:
                            words.append({
                                'word': word_info.get('word', ''),
                                'start': word_info.get('start', 0),
                                'end': word_info.get('end', 0),
                                'probability': word_info.get('conf', 1.0)
                            })
                    
                    result.segments.append(TranscriptSegment(
                        text=vosk_result['text'],
                        start_time=time.time(),
                        end_time=time.time(),
                        words=words,
                        is_final=True
                    ))
                    result.full_text = vosk_result['text']
            else:
                # Partial result
                partial = json.loads(self.vosk_recognizer.PartialResult())
                if 'partial' in partial and partial['partial']:
                    result.is_partial = True
                    result.full_text = partial['partial']
                    
        except Exception as e:
            logger.error(f"Vosk processing error: {e}")
        
        return result
    
    def force_process(self) -> Optional[ASRResult]:
        """Force processing of any buffered audio."""
        with self.buffer_lock:
            if len(self.audio_buffer) == 0:
                return None
            audio_to_process = self.audio_buffer.copy()
            self.audio_buffer = np.array([], dtype=np.float32)
        
        if self.using_vosk:
            return self._process_vosk(audio_to_process)
        elif self.model is not None:
            return self._process_whisper(audio_to_process)
        return None
    
    def get_transcript(self) -> str:
        """Get full transcript so far."""
        return self.full_transcript.strip()
    
    def get_recent_transcript(self, num_words: int = 50) -> str:
        """Get recent transcript (last N words)."""
        words = self.full_transcript.split()
        if len(words) <= num_words:
            return self.full_transcript
        return " ".join(words[-num_words:])
    
    def reset(self):
        """Reset for new session."""
        with self.buffer_lock:
            self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_segments = []
        self.full_transcript = ""
        self.last_process_time = time.time()
        
        # Reset Vosk recognizer if using
        if self.using_vosk and self.vosk_recognizer:
            self.vosk_recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)
            self.vosk_recognizer.SetWords(True)
        
        logger.info("ASRPipeline reset")
    
    def get_session_stats(self) -> dict:
        """Get session summary statistics."""
        word_count = len(self.full_transcript.split())
        return {
            "total_segments": len(self.transcript_segments),
            "total_words": word_count,
            "full_transcript": self.full_transcript,
            "model_type": "vosk" if self.using_vosk else "whisper",
            "model_size": config.asr.MODEL_SIZE if not self.using_vosk else "small-en-us"
        }

