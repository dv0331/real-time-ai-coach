"""
Emotion Engine - Multimodal Emotion Detection for Actors

Combines:
1. Audio Emotion (SpeechBrain wav2vec2) - delivery energy/arousal
2. Face Emotion (BEiT/ResNet) - expressiveness trends  
3. Text Emotion (GoEmotions) - script intent analysis

All models run locally on GPU (RTX 4060)
"""

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from collections import deque
import threading
import io

logger = logging.getLogger(__name__)

# ============================================================================
# Try importing emotion models
# ============================================================================

# Audio emotion (SpeechBrain)
HAS_SPEECHBRAIN = False
audio_emotion_model = None
try:
    import torch
    from speechbrain.inference.classifiers import EncoderClassifier
    HAS_SPEECHBRAIN = True
    logger.info("SpeechBrain available for audio emotion")
except ImportError:
    logger.warning("SpeechBrain not installed - audio emotion disabled")

# Text emotion (transformers)
HAS_TEXT_EMOTION = False
text_emotion_pipeline = None
try:
    from transformers import pipeline
    HAS_TEXT_EMOTION = True
    logger.info("Transformers available for text emotion")
except ImportError:
    logger.warning("Transformers not installed - text emotion disabled")

# Face emotion (transformers vision)
HAS_FACE_EMOTION = False
face_emotion_pipeline = None

# OpenCV for face detection
HAS_CV2 = False
try:
    import cv2
    from PIL import Image
    HAS_CV2 = True
except ImportError:
    logger.warning("OpenCV not installed - face emotion disabled")


@dataclass
class EmotionResult:
    """Combined emotion analysis result."""
    timestamp: float = 0.0
    
    # Audio emotion (delivery/arousal)
    audio_emotion: str = "neutral"
    audio_emotion_scores: Dict[str, float] = field(default_factory=dict)
    audio_arousal: float = 0.5  # 0=calm, 1=activated
    audio_valence: float = 0.5  # 0=negative, 1=positive
    
    # Face emotion (expressiveness)
    face_emotion: str = "neutral"
    face_emotion_scores: Dict[str, float] = field(default_factory=dict)
    face_expressiveness: float = 0.5  # How expressive (0=flat, 1=very expressive)
    
    # Text emotion (script intent)
    text_emotions: List[Dict[str, Any]] = field(default_factory=list)
    text_sentiment: str = "neutral"
    
    # Combined metrics
    emotional_range: float = 0.5  # Variety of emotions shown
    emotion_intensity: float = 0.5  # Overall intensity
    emotion_authenticity: float = 0.5  # Proxy for genuine expression


class EmotionEngine:
    """
    Multimodal emotion detection engine.
    Runs locally on GPU for real-time feedback.
    """
    
    def __init__(self, 
                 use_audio_emotion: bool = True,
                 use_face_emotion: bool = True,
                 use_text_emotion: bool = True,
                 device: str = "cuda"):
        """
        Initialize emotion detection models.
        
        Args:
            use_audio_emotion: Enable SpeechBrain SER
            use_face_emotion: Enable face emotion classifier
            use_text_emotion: Enable GoEmotions text classifier
            device: "cuda" or "cpu"
        """
        self.device = device
        self.use_audio = use_audio_emotion and HAS_SPEECHBRAIN
        self.use_face = use_face_emotion and HAS_FACE_EMOTION and HAS_CV2
        self.use_text = use_text_emotion and HAS_TEXT_EMOTION
        
        # Models (lazy loaded)
        self._audio_model = None
        self._face_model = None
        self._text_model = None
        self._face_cascade = None
        
        # History for trend analysis
        self.audio_history: deque = deque(maxlen=30)
        self.face_history: deque = deque(maxlen=30)
        self.text_history: deque = deque(maxlen=20)
        
        # Emotion label mappings (for arousal/valence estimation)
        self.arousal_map = {
            # High arousal
            "angry": 0.9, "fear": 0.85, "excited": 0.9, "surprised": 0.8,
            "happy": 0.7, "disgust": 0.6,
            # Low arousal
            "sad": 0.3, "neutral": 0.4, "calm": 0.2, "bored": 0.2
        }
        
        self.valence_map = {
            # Positive
            "happy": 0.9, "excited": 0.85, "joy": 0.9, "love": 0.9,
            "amusement": 0.8, "optimism": 0.75, "pride": 0.8,
            # Negative
            "angry": 0.2, "sad": 0.2, "fear": 0.25, "disgust": 0.2,
            "disappointment": 0.3, "grief": 0.1, "remorse": 0.25,
            # Neutral
            "neutral": 0.5, "surprise": 0.5, "confusion": 0.4
        }
        
        logger.info(f"EmotionEngine: audio={self.use_audio}, face={self.use_face}, text={self.use_text}")
    
    def _load_audio_model(self):
        """Lazy load audio emotion model."""
        if self._audio_model is not None:
            return self._audio_model
        
        if not HAS_SPEECHBRAIN:
            return None
            
        try:
            logger.info("Loading SpeechBrain emotion model...")
            self._audio_model = EncoderClassifier.from_hparams(
                source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                savedir="models/speechbrain_ser",
                run_opts={"device": self.device}
            )
            logger.info("SpeechBrain model loaded")
            return self._audio_model
        except Exception as e:
            logger.error(f"Failed to load audio emotion model: {e}")
            self.use_audio = False
            return None
    
    def _load_text_model(self):
        """Lazy load text emotion model."""
        if self._text_model is not None:
            return self._text_model
            
        if not HAS_TEXT_EMOTION:
            return None
            
        try:
            logger.info("Loading GoEmotions text model...")
            self._text_model = pipeline(
                "text-classification",
                model="SamLowe/roberta-base-go_emotions",
                top_k=5,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("GoEmotions model loaded")
            return self._text_model
        except Exception as e:
            logger.error(f"Failed to load text emotion model: {e}")
            self.use_text = False
            return None
    
    def _load_face_model(self):
        """Lazy load face emotion model."""
        if self._face_model is not None:
            return self._face_model
            
        if not HAS_TEXT_EMOTION or not HAS_CV2:
            return None
            
        try:
            logger.info("Loading face emotion model...")
            # Use a lighter model for real-time
            self._face_model = pipeline(
                "image-classification",
                model="trpakov/vit-face-expression",  # Lighter than BEiT
                top_k=5,
                device=0 if self.device == "cuda" else -1
            )
            
            # Face detector
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            
            logger.info("Face emotion model loaded")
            return self._face_model
        except Exception as e:
            logger.error(f"Failed to load face emotion model: {e}")
            self.use_face = False
            return None
    
    def analyze_audio(self, audio_samples: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Analyze audio for emotion/arousal.
        
        Args:
            audio_samples: Float32 audio samples [-1, 1]
            sample_rate: Audio sample rate
            
        Returns:
            Dict with emotion label and scores
        """
        if not self.use_audio:
            return {"emotion": "neutral", "scores": {}, "arousal": 0.5, "valence": 0.5}
        
        model = self._load_audio_model()
        if model is None:
            return {"emotion": "neutral", "scores": {}, "arousal": 0.5, "valence": 0.5}
        
        try:
            import torch
            
            # Ensure correct shape [batch, time]
            if audio_samples.ndim == 1:
                audio_samples = audio_samples.reshape(1, -1)
            
            # Convert to tensor
            wav_tensor = torch.tensor(audio_samples, dtype=torch.float32)
            
            # Get prediction
            out_prob, score, index, label = model.classify_batch(wav_tensor)
            
            emotion = str(label[0]) if label else "neutral"
            
            # Get scores for all classes
            scores = {}
            if hasattr(model, 'hparams') and hasattr(model.hparams, 'label_encoder'):
                labels = model.hparams.label_encoder.lab2ind
                probs = out_prob[0].cpu().numpy()
                for lab, idx in labels.items():
                    if idx < len(probs):
                        scores[lab] = float(probs[idx])
            
            # Estimate arousal/valence
            arousal = self.arousal_map.get(emotion.lower(), 0.5)
            valence = self.valence_map.get(emotion.lower(), 0.5)
            
            result = {
                "emotion": emotion,
                "scores": scores,
                "arousal": arousal,
                "valence": valence
            }
            
            self.audio_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Audio emotion analysis failed: {e}")
            return {"emotion": "neutral", "scores": {}, "arousal": 0.5, "valence": 0.5}
    
    def analyze_face(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Analyze face for emotion/expressiveness.
        
        Args:
            image_bytes: JPEG image bytes
            
        Returns:
            Dict with emotion label and scores
        """
        if not self.use_face or not HAS_CV2:
            return {"emotion": "neutral", "scores": {}, "expressiveness": 0.5}
        
        model = self._load_face_model()
        if model is None:
            return {"emotion": "neutral", "scores": {}, "expressiveness": 0.5}
        
        try:
            # Decode image
            img_array = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                return {"emotion": "neutral", "scores": {}, "expressiveness": 0.5}
            
            # Detect face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return {"emotion": "neutral", "scores": {}, "expressiveness": 0.5}
            
            # Get largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Add padding
            pad = int(w * 0.15)
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(frame.shape[1], x + w + pad)
            y1 = min(frame.shape[0], y + h + pad)
            
            # Crop and convert to PIL
            face_crop = frame[y0:y1, x0:x1]
            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            
            # Run classifier
            results = model(face_pil)
            
            # Parse results
            emotion = results[0]["label"] if results else "neutral"
            scores = {r["label"]: r["score"] for r in results}
            
            # Calculate expressiveness (how confident/intense the emotion)
            max_score = max(scores.values()) if scores else 0.5
            expressiveness = max_score * 0.7 + 0.3  # Scale to avoid extremes
            
            result = {
                "emotion": emotion,
                "scores": scores,
                "expressiveness": expressiveness
            }
            
            self.face_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Face emotion analysis failed: {e}")
            return {"emotion": "neutral", "scores": {}, "expressiveness": 0.5}
    
    def analyze_text(self, transcript: str) -> Dict[str, Any]:
        """
        Analyze transcript text for emotional content.
        
        Args:
            transcript: Text to analyze
            
        Returns:
            Dict with emotion labels and sentiment
        """
        if not self.use_text or not transcript.strip():
            return {"emotions": [], "sentiment": "neutral"}
        
        model = self._load_text_model()
        if model is None:
            return {"emotions": [], "sentiment": "neutral"}
        
        try:
            # Get recent text (last ~100 words)
            words = transcript.split()[-100:]
            text = " ".join(words)
            
            if len(text) < 10:
                return {"emotions": [], "sentiment": "neutral"}
            
            # Run classifier
            results = model(text)
            
            # Handle nested list from top_k
            if results and isinstance(results[0], list):
                results = results[0]
            
            emotions = [{"label": r["label"], "score": round(r["score"], 3)} for r in results[:5]]
            
            # Determine overall sentiment
            positive = ["joy", "love", "admiration", "amusement", "approval", 
                       "caring", "desire", "excitement", "gratitude", "optimism", "pride", "relief"]
            negative = ["anger", "annoyance", "disappointment", "disapproval", "disgust",
                       "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness"]
            
            pos_score = sum(r["score"] for r in results if r["label"] in positive)
            neg_score = sum(r["score"] for r in results if r["label"] in negative)
            
            if pos_score > neg_score + 0.2:
                sentiment = "positive"
            elif neg_score > pos_score + 0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            result = {"emotions": emotions, "sentiment": sentiment}
            self.text_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Text emotion analysis failed: {e}")
            return {"emotions": [], "sentiment": "neutral"}
    
    def get_combined_analysis(self,
                             audio_samples: Optional[np.ndarray] = None,
                             image_bytes: Optional[bytes] = None,
                             transcript: str = "") -> EmotionResult:
        """
        Run combined emotion analysis on all modalities.
        
        Returns:
            EmotionResult with all emotion data
        """
        result = EmotionResult(timestamp=time.time())
        
        # Audio emotion
        if audio_samples is not None and len(audio_samples) > 0:
            audio_result = self.analyze_audio(audio_samples)
            result.audio_emotion = audio_result["emotion"]
            result.audio_emotion_scores = audio_result["scores"]
            result.audio_arousal = audio_result["arousal"]
            result.audio_valence = audio_result["valence"]
        
        # Face emotion
        if image_bytes is not None:
            face_result = self.analyze_face(image_bytes)
            result.face_emotion = face_result["emotion"]
            result.face_emotion_scores = face_result["scores"]
            result.face_expressiveness = face_result["expressiveness"]
        
        # Text emotion
        if transcript:
            text_result = self.analyze_text(transcript)
            result.text_emotions = text_result["emotions"]
            result.text_sentiment = text_result["sentiment"]
        
        # Calculate combined metrics
        result.emotional_range = self._calculate_emotional_range()
        result.emotion_intensity = self._calculate_intensity(result)
        result.emotion_authenticity = self._calculate_authenticity(result)
        
        return result
    
    def _calculate_emotional_range(self) -> float:
        """Calculate variety of emotions expressed."""
        all_emotions = set()
        
        for r in self.audio_history:
            all_emotions.add(r.get("emotion", "neutral"))
        for r in self.face_history:
            all_emotions.add(r.get("emotion", "neutral"))
        
        # More unique emotions = higher range
        range_score = min(1.0, len(all_emotions) / 5.0)
        return range_score
    
    def _calculate_intensity(self, result: EmotionResult) -> float:
        """Calculate overall emotional intensity."""
        intensity = 0.5
        
        # Audio arousal contributes most
        if result.audio_arousal:
            intensity = result.audio_arousal * 0.5 + intensity * 0.5
        
        # Face expressiveness
        if result.face_expressiveness:
            intensity = result.face_expressiveness * 0.3 + intensity * 0.7
        
        return intensity
    
    def _calculate_authenticity(self, result: EmotionResult) -> float:
        """
        Estimate emotion authenticity (alignment between modalities).
        Higher = audio/face/text emotions align.
        """
        # Simple proxy: if audio and face emotions are in same valence category
        audio_valence = result.audio_valence
        face_emotion = result.face_emotion.lower()
        face_valence = self.valence_map.get(face_emotion, 0.5)
        
        # Close valence = more "authentic"
        valence_diff = abs(audio_valence - face_valence)
        authenticity = 1.0 - valence_diff
        
        return max(0.3, authenticity)  # Floor at 0.3
    
    def get_emotion_summary(self) -> Dict[str, Any]:
        """Get summary of emotional trends for session."""
        audio_emotions = [r["emotion"] for r in self.audio_history]
        face_emotions = [r["emotion"] for r in self.face_history]
        
        return {
            "audio_emotion_distribution": self._count_emotions(audio_emotions),
            "face_emotion_distribution": self._count_emotions(face_emotions),
            "emotional_range": self._calculate_emotional_range(),
            "dominant_audio_emotion": max(set(audio_emotions), key=audio_emotions.count) if audio_emotions else "neutral",
            "dominant_face_emotion": max(set(face_emotions), key=face_emotions.count) if face_emotions else "neutral"
        }
    
    def _count_emotions(self, emotions: List[str]) -> Dict[str, int]:
        """Count emotion occurrences."""
        counts = {}
        for e in emotions:
            counts[e] = counts.get(e, 0) + 1
        return counts
    
    def reset(self):
        """Reset for new session."""
        self.audio_history.clear()
        self.face_history.clear()
        self.text_history.clear()
        logger.info("EmotionEngine reset")


# ============================================================================
# Initialization helper
# ============================================================================

def create_emotion_engine(device: str = "cuda") -> EmotionEngine:
    """Create and initialize emotion engine."""
    return EmotionEngine(
        use_audio_emotion=True,
        use_face_emotion=True,
        use_text_emotion=True,
        device=device
    )

