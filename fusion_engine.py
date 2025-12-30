"""
Fusion Engine: Combines audio, vision, and ASR features into stable scores.
Implements EMA smoothing and hysteresis to prevent metric flicker.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import time
import logging

from config import config
from audio_pipeline import AudioFeatures
from vision_pipeline import VisionFeatures

logger = logging.getLogger(__name__)

@dataclass
class FusedMetrics:
    """Combined and smoothed metrics from all pipelines."""
    timestamp: float = 0.0
    
    # Individual scores (0-1, higher is better)
    pace_score: float = 0.5
    energy_score: float = 0.5
    pitch_variety_score: float = 0.5
    filler_word_score: float = 1.0   # 1.0 = no fillers
    eye_contact_score: float = 0.5
    presence_score: float = 0.5
    stability_score: float = 0.5
    
    # Overall score
    overall_score: float = 0.5
    
    # Raw values for display
    wpm: float = 0.0
    energy_db: float = -40.0
    pitch_variety_hz: float = 0.0
    filler_ratio: float = 0.0
    
    # Flags for coaching
    is_speaking: bool = False
    face_detected: bool = False
    looking_at_camera: bool = False
    
    # Trend indicators (-1 = declining, 0 = stable, 1 = improving)
    overall_trend: int = 0

class FusionEngine:
    """
    Fuses features from audio and vision pipelines into stable, smoothed metrics.
    Uses EMA smoothing and hysteresis to prevent tip flicker.
    """
    
    def __init__(self):
        # EMA state for each metric
        self.ema_state: Dict[str, float] = {
            'pace': 0.5,
            'energy': 0.5,
            'pitch_variety': 0.5,
            'filler_words': 1.0,
            'eye_contact': 0.5,
            'presence': 0.5,
            'stability': 0.5,
            'overall': 0.5,
        }
        
        # Previous values for hysteresis
        self.prev_values: Dict[str, float] = self.ema_state.copy()
        
        # Trend tracking
        self.score_history: list = []
        self.max_history = 20
        
        # Filler word tracking
        self.filler_count = 0
        self.word_count = 0
        
        self.alpha = config.fusion.EMA_ALPHA
        self.hysteresis = config.fusion.SCORE_HYSTERESIS
        
        logger.info(f"FusionEngine initialized: alpha={self.alpha}, hysteresis={self.hysteresis}")
    
    def update(
        self,
        audio_features: Optional[AudioFeatures] = None,
        vision_features: Optional[VisionFeatures] = None,
        filler_count: int = 0,
        word_count: int = 0
    ) -> FusedMetrics:
        """
        Update fused metrics with new features.
        Returns smoothed, stable metrics.
        """
        metrics = FusedMetrics(timestamp=time.time())
        
        # Update filler tracking
        self.filler_count = filler_count
        self.word_count = word_count
        
        # Extract raw scores from audio
        if audio_features:
            raw_pace = audio_features.pace_score
            raw_energy = audio_features.energy_score
            raw_pitch = audio_features.pitch_variety_score
            metrics.wpm = audio_features.estimated_wpm
            metrics.energy_db = audio_features.energy_db
            metrics.pitch_variety_hz = audio_features.pitch_std
            metrics.is_speaking = audio_features.is_speech
        else:
            raw_pace = self.ema_state['pace']
            raw_energy = self.ema_state['energy']
            raw_pitch = self.ema_state['pitch_variety']
        
        # Filler word score
        if self.word_count > 0:
            filler_ratio = self.filler_count / self.word_count
            raw_filler = max(0, 1.0 - filler_ratio * 10)  # Penalize heavily
            metrics.filler_ratio = filler_ratio
        else:
            raw_filler = 1.0
        
        # Extract raw scores from vision
        if vision_features:
            raw_eye_contact = vision_features.eye_contact_score
            raw_presence = vision_features.presence_score
            raw_stability = vision_features.stability_score
            metrics.face_detected = vision_features.face_detected
            metrics.looking_at_camera = vision_features.looking_at_camera
        else:
            raw_eye_contact = self.ema_state['eye_contact']
            raw_presence = self.ema_state['presence']
            raw_stability = self.ema_state['stability']
        
        # Apply EMA smoothing
        metrics.pace_score = self._ema_update('pace', raw_pace)
        metrics.energy_score = self._ema_update('energy', raw_energy)
        metrics.pitch_variety_score = self._ema_update('pitch_variety', raw_pitch)
        metrics.filler_word_score = self._ema_update('filler_words', raw_filler)
        metrics.eye_contact_score = self._ema_update('eye_contact', raw_eye_contact)
        metrics.presence_score = self._ema_update('presence', raw_presence)
        metrics.stability_score = self._ema_update('stability', raw_stability)
        
        # Compute weighted overall score
        weights = config.fusion.WEIGHTS
        weighted_sum = (
            weights.get('pace', 0.2) * metrics.pace_score +
            weights.get('energy', 0.15) * metrics.energy_score +
            weights.get('pitch_variety', 0.1) * metrics.pitch_variety_score +
            weights.get('filler_words', 0.15) * metrics.filler_word_score +
            weights.get('eye_contact', 0.2) * metrics.eye_contact_score +
            weights.get('presence', 0.1) * metrics.presence_score +
            weights.get('stability', 0.1) * metrics.stability_score
        )
        weight_total = sum(weights.values())
        raw_overall = weighted_sum / weight_total if weight_total > 0 else 0.5
        
        metrics.overall_score = self._ema_update('overall', raw_overall)
        
        # Compute trend
        self.score_history.append(metrics.overall_score)
        if len(self.score_history) > self.max_history:
            self.score_history.pop(0)
        
        metrics.overall_trend = self._compute_trend()
        
        return metrics
    
    def _ema_update(self, key: str, new_value: float) -> float:
        """
        Update EMA with hysteresis.
        Only updates if change is significant (beyond hysteresis threshold).
        """
        current = self.ema_state.get(key, 0.5)
        prev = self.prev_values.get(key, 0.5)
        
        # EMA update
        smoothed = self.alpha * new_value + (1 - self.alpha) * current
        
        # Hysteresis: only report change if significant
        if abs(smoothed - prev) > self.hysteresis:
            self.prev_values[key] = smoothed
        else:
            smoothed = prev
        
        self.ema_state[key] = smoothed
        return smoothed
    
    def _compute_trend(self) -> int:
        """Compute overall score trend."""
        if len(self.score_history) < 5:
            return 0
        
        recent = self.score_history[-5:]
        older = self.score_history[-10:-5] if len(self.score_history) >= 10 else self.score_history[:5]
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        diff = recent_avg - older_avg
        
        if diff > 0.05:
            return 1   # Improving
        elif diff < -0.05:
            return -1  # Declining
        return 0       # Stable
    
    def get_problem_areas(self, threshold: float = 0.5) -> list:
        """
        Get list of metrics below threshold.
        Returns list of (metric_name, score) tuples sorted by score (lowest first).
        """
        metrics = {
            'pace': self.ema_state['pace'],
            'energy': self.ema_state['energy'],
            'pitch_variety': self.ema_state['pitch_variety'],
            'filler_words': self.ema_state['filler_words'],
            'eye_contact': self.ema_state['eye_contact'],
            'presence': self.ema_state['presence'],
            'stability': self.ema_state['stability'],
        }
        
        problems = [(k, v) for k, v in metrics.items() if v < threshold]
        problems.sort(key=lambda x: x[1])
        return problems
    
    def reset(self):
        """Reset for new session."""
        self.ema_state = {k: 0.5 for k in self.ema_state}
        self.ema_state['filler_words'] = 1.0
        self.prev_values = self.ema_state.copy()
        self.score_history = []
        self.filler_count = 0
        self.word_count = 0
        logger.info("FusionEngine reset")
    
    def get_session_stats(self) -> dict:
        """Get session summary statistics."""
        return {
            "final_scores": self.ema_state.copy(),
            "trend_history": self.score_history[-20:] if self.score_history else [],
            "problem_areas": self.get_problem_areas(0.6),
        }

