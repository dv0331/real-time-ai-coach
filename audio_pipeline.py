"""
Audio Pipeline: VAD, energy, pitch, pause detection, filler word tracking.
Processes 16kHz PCM audio chunks and extracts speech features.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Deque
import time
import logging

try:
    import webrtcvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False
    logging.warning("webrtcvad not installed, VAD disabled")

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logging.warning("librosa not installed, pitch analysis disabled")

from config import config

logger = logging.getLogger(__name__)

@dataclass
class AudioFeatures:
    """Features extracted from audio."""
    timestamp: float = 0.0
    
    # VAD
    is_speech: bool = False
    speech_ratio: float = 0.0         # Ratio of speech in recent window
    
    # Energy
    energy_db: float = -60.0          # Current energy in dB
    energy_normalized: float = 0.0    # 0-1 normalized
    
    # Pitch
    pitch_hz: float = 0.0             # Fundamental frequency
    pitch_std: float = 0.0            # Pitch variability (low = monotone)
    pitch_normalized: float = 0.5     # 0-1, 0.5 = average
    
    # Pace
    estimated_wpm: float = 0.0
    pause_ratio: float = 0.0          # Ratio of pauses to speech
    
    # Scores (0-1, higher is better)
    pace_score: float = 0.5
    energy_score: float = 0.5
    pitch_variety_score: float = 0.5

class AudioPipeline:
    """
    Processes audio chunks and maintains running statistics.
    Thread-safe for use in async context.
    """
    
    def __init__(self):
        self.sample_rate = config.audio.SAMPLE_RATE
        self.chunk_duration = config.audio.CHUNK_DURATION_MS / 1000.0
        self.samples_per_chunk = int(self.sample_rate * self.chunk_duration)
        
        # VAD setup
        if HAS_VAD:
            self.vad = webrtcvad.Vad(config.audio.VAD_AGGRESSIVENESS)
        else:
            self.vad = None
        
        # Ring buffers for running statistics
        buffer_size = int(config.audio.WPM_WINDOW_SEC / self.chunk_duration)
        self.vad_history: Deque[bool] = deque(maxlen=buffer_size)
        self.energy_history: Deque[float] = deque(maxlen=buffer_size)
        self.pitch_history: Deque[float] = deque(maxlen=buffer_size)
        
        # Word count tracking (updated by ASR)
        self.word_timestamps: Deque[float] = deque(maxlen=1000)
        
        # Filler word tracking
        self.filler_count: int = 0
        self.total_words: int = 0
        
        # Session stats
        self.session_start: float = time.time()
        self.total_speech_frames: int = 0
        self.total_frames: int = 0
        
        logger.info(f"AudioPipeline initialized: {self.sample_rate}Hz, {self.samples_per_chunk} samples/chunk")
    
    def process_chunk(self, audio_bytes: bytes) -> AudioFeatures:
        """
        Process a single audio chunk (16-bit PCM, 16kHz, mono).
        Returns extracted features.
        """
        features = AudioFeatures(timestamp=time.time())
        
        # Convert bytes to numpy array
        try:
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0  # Normalize to [-1, 1]
        except Exception as e:
            logger.error(f"Failed to parse audio: {e}")
            return features
        
        if len(audio) == 0:
            return features
        
        # VAD
        features.is_speech = self._detect_speech(audio_bytes)
        self.vad_history.append(features.is_speech)
        features.speech_ratio = sum(self.vad_history) / max(len(self.vad_history), 1)
        
        # Track speech/pause ratio
        self.total_frames += 1
        if features.is_speech:
            self.total_speech_frames += 1
        
        # Energy
        features.energy_db = self._compute_energy_db(audio)
        self.energy_history.append(features.energy_db)
        features.energy_normalized = self._normalize_energy(features.energy_db)
        
        # Pitch (only if speech detected and librosa available)
        if features.is_speech and HAS_LIBROSA and len(audio) >= 512:
            features.pitch_hz = self._estimate_pitch(audio)
            if features.pitch_hz > 0:
                self.pitch_history.append(features.pitch_hz)
        
        # Compute pitch variability
        if len(self.pitch_history) >= 5:
            pitches = list(self.pitch_history)
            features.pitch_std = float(np.std(pitches))
            mean_pitch = np.mean(pitches)
            features.pitch_normalized = min(1.0, features.pitch_std / (mean_pitch * 0.2 + 1))
        
        # Compute WPM from word timestamps
        features.estimated_wpm = self._compute_wpm()
        
        # Pause ratio
        if self.total_frames > 0:
            features.pause_ratio = 1.0 - (self.total_speech_frames / self.total_frames)
        
        # Compute scores
        features.pace_score = self._score_pace(features.estimated_wpm)
        features.energy_score = self._score_energy(features.energy_normalized, features.is_speech)
        features.pitch_variety_score = self._score_pitch_variety(features.pitch_std)
        
        return features
    
    def _detect_speech(self, audio_bytes: bytes) -> bool:
        """Use WebRTC VAD to detect speech."""
        if self.vad is None:
            return True  # Assume speech if no VAD
        
        try:
            # VAD expects 10, 20, or 30ms frames
            frame_duration = 30  # ms
            frame_size = int(self.sample_rate * frame_duration / 1000) * 2  # bytes
            
            if len(audio_bytes) < frame_size:
                return False
            
            # Check first frame
            return self.vad.is_speech(audio_bytes[:frame_size], self.sample_rate)
        except Exception as e:
            logger.debug(f"VAD error: {e}")
            return True
    
    def _compute_energy_db(self, audio: np.ndarray) -> float:
        """Compute RMS energy in dB."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return -60.0
        return float(20 * np.log10(rms))
    
    def _normalize_energy(self, db: float) -> float:
        """Normalize energy to 0-1 range."""
        # Typical speech: -40dB to -10dB
        min_db, max_db = -50.0, -10.0
        normalized = (db - min_db) / (max_db - min_db)
        return float(np.clip(normalized, 0.0, 1.0))
    
    def _estimate_pitch(self, audio: np.ndarray) -> float:
        """Estimate fundamental frequency using librosa."""
        try:
            # Use pyin for robust pitch estimation
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=50,
                fmax=500,
                sr=self.sample_rate,
                frame_length=1024
            )
            # Get median of voiced frames
            voiced_f0 = f0[voiced_flag]
            if len(voiced_f0) > 0:
                return float(np.median(voiced_f0))
        except Exception as e:
            logger.debug(f"Pitch estimation error: {e}")
        return 0.0
    
    def _compute_wpm(self) -> float:
        """Compute words per minute from recent word timestamps."""
        now = time.time()
        window_start = now - config.audio.WPM_WINDOW_SEC
        
        # Count words in window
        words_in_window = sum(1 for t in self.word_timestamps if t >= window_start)
        
        # Calculate WPM
        window_duration = min(now - self.session_start, config.audio.WPM_WINDOW_SEC)
        if window_duration > 0:
            return (words_in_window / window_duration) * 60.0
        return 0.0
    
    def _score_pace(self, wpm: float) -> float:
        """Score pace (WPM) on 0-1 scale."""
        target_min = config.audio.TARGET_WPM_MIN
        target_max = config.audio.TARGET_WPM_MAX
        target_mid = (target_min + target_max) / 2
        
        if wpm == 0:
            return 0.5  # No data
        
        if target_min <= wpm <= target_max:
            # Perfect range
            return 1.0
        elif wpm < target_min:
            # Too slow
            return max(0.0, 1.0 - (target_min - wpm) / target_min)
        else:
            # Too fast
            return max(0.0, 1.0 - (wpm - target_max) / target_max)
    
    def _score_energy(self, normalized_energy: float, is_speech: bool) -> float:
        """Score energy level."""
        if not is_speech:
            return 0.5  # Neutral during pauses
        
        # Ideal energy is around 0.5-0.8 normalized
        if 0.4 <= normalized_energy <= 0.9:
            return 1.0
        elif normalized_energy < 0.4:
            return 0.5 + normalized_energy  # Low energy penalty
        else:
            return 0.9  # Slightly high is OK
    
    def _score_pitch_variety(self, pitch_std: float) -> float:
        """Score pitch variety (penalize monotone)."""
        # Typical pitch std for expressive speech: 20-50 Hz
        if pitch_std < 5:
            return 0.3  # Very monotone
        elif pitch_std < 15:
            return 0.5 + (pitch_std / 30)
        elif pitch_std < 50:
            return 1.0  # Good variety
        else:
            return 0.8  # Too variable (nervous?)
    
    def add_word(self, word: str, timestamp: float = None):
        """Called by ASR to register a new word."""
        if timestamp is None:
            timestamp = time.time()
        
        self.word_timestamps.append(timestamp)
        self.total_words += 1
        
        # Check for filler words
        word_lower = word.lower().strip()
        if word_lower in config.audio.FILLER_WORDS:
            self.filler_count += 1
    
    def get_filler_ratio(self) -> float:
        """Get ratio of filler words to total words."""
        if self.total_words == 0:
            return 0.0
        return self.filler_count / self.total_words
    
    def reset(self):
        """Reset for new session."""
        self.vad_history.clear()
        self.energy_history.clear()
        self.pitch_history.clear()
        self.word_timestamps.clear()
        self.filler_count = 0
        self.total_words = 0
        self.session_start = time.time()
        self.total_speech_frames = 0
        self.total_frames = 0
        logger.info("AudioPipeline reset")
    
    def get_session_stats(self) -> dict:
        """Get session summary statistics."""
        duration = time.time() - self.session_start
        return {
            "duration_seconds": duration,
            "total_words": self.total_words,
            "filler_words": self.filler_count,
            "filler_ratio": self.get_filler_ratio(),
            "average_wpm": (self.total_words / duration * 60) if duration > 0 else 0,
            "speech_ratio": self.total_speech_frames / max(self.total_frames, 1),
            "average_energy_db": float(np.mean(list(self.energy_history))) if self.energy_history else -40,
            "pitch_variability": float(np.std(list(self.pitch_history))) if len(self.pitch_history) > 1 else 0,
        }

