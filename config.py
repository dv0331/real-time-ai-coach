"""
Configuration and tuning knobs for the Real-Time AI Coach.
Adjust these parameters based on your hardware and latency requirements.
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class AudioConfig:
    """Audio pipeline configuration."""
    SAMPLE_RATE: int = 16000          # 16kHz for speech
    CHUNK_DURATION_MS: int = 100      # Audio chunk size in ms
    VAD_AGGRESSIVENESS: int = 2       # 0-3, higher = more aggressive filtering
    
    # Analysis windows
    ENERGY_WINDOW_SEC: float = 0.5    # Window for energy calculation
    PITCH_WINDOW_SEC: float = 1.0     # Window for pitch analysis
    WPM_WINDOW_SEC: float = 10.0      # Window for WPM calculation
    
    # Thresholds
    SILENCE_THRESHOLD_DB: float = -40.0    # Below this = silence
    FILLER_WORDS: List[str] = field(default_factory=lambda: [
        "um", "uh", "like", "you know", "so", "actually", 
        "basically", "literally", "right", "okay", "well"
    ])
    
    # Targets for scoring
    TARGET_WPM_MIN: int = 120
    TARGET_WPM_MAX: int = 160
    TARGET_PAUSE_RATIO: float = 0.15  # 15% pauses is good
    
@dataclass
class VisionConfig:
    """Vision pipeline configuration."""
    TARGET_FPS: int = 8               # Process frames at this rate
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    JPEG_QUALITY: int = 70            # Lower = smaller, faster
    
    # MediaPipe settings
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5
    
    # Gaze thresholds (normalized coords, 0.5 = center)
    GAZE_CENTER_TOLERANCE: float = 0.15  # How far from center is OK
    
    # Head pose thresholds (degrees)
    HEAD_YAW_TOLERANCE: float = 20.0     # Left/right rotation
    HEAD_PITCH_TOLERANCE: float = 15.0   # Up/down tilt
    
    # Presence scoring
    FACE_SIZE_MIN: float = 0.1        # Min face size as fraction of frame
    FACE_SIZE_MAX: float = 0.6        # Max face size (too close)

@dataclass
class ASRConfig:
    """ASR (Speech-to-Text) configuration."""
    MODEL_SIZE: str = "base"          # tiny, base, small, medium, large-v3
    DEVICE: str = "cuda"              # cuda or cpu
    COMPUTE_TYPE: str = "float16"     # float16 for GPU, int8 for CPU
    LANGUAGE: str = "en"
    
    # Streaming settings
    MIN_AUDIO_LENGTH_SEC: float = 0.5     # Min audio before processing
    MAX_AUDIO_LENGTH_SEC: float = 30.0    # Force process at this length
    SILENCE_DURATION_SEC: float = 0.5     # Silence before segment break
    
    # Fallback to Vosk if faster-whisper fails
    USE_VOSK_FALLBACK: bool = True
    VOSK_MODEL_PATH: str = "models/vosk-model-small-en-us-0.15"

@dataclass
class FusionConfig:
    """Score fusion and smoothing configuration."""
    # Exponential moving average alpha (0-1, higher = more responsive)
    EMA_ALPHA: float = 0.3
    
    # Hysteresis for tip triggering (prevents flicker)
    SCORE_HYSTERESIS: float = 0.1     # 10% hysteresis band
    
    # Score weights for overall score
    WEIGHTS: dict = field(default_factory=lambda: {
        "pace": 0.20,
        "energy": 0.15,
        "pitch_variety": 0.10,
        "filler_words": 0.15,
        "eye_contact": 0.20,
        "presence": 0.10,
        "stability": 0.10,
    })

@dataclass
class CoachConfig:
    """Coaching tip generation configuration."""
    TIP_COOLDOWN_SEC: float = 3.0     # Min seconds between same tip type
    MAX_TIPS_PER_UPDATE: int = 2      # Max tips to show at once
    
    # LLM settings (Ollama)
    USE_LLM: bool = True              # Enable Ollama for rich coaching tips
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"    # Fast model, good for real-time
    LLM_PROMPT_INTERVAL_SEC: float = 8.0  # How often to query LLM
    
    # Tip priority (lower = higher priority)
    TIP_PRIORITIES: dict = field(default_factory=lambda: {
        "eye_contact": 1,
        "pace_slow": 2,
        "pace_fast": 2,
        "filler_words": 3,
        "energy_low": 4,
        "monotone": 5,
        "presence": 6,
    })

@dataclass
class EmotionConfig:
    """Emotion detection configuration."""
    # Enable/disable emotion modalities
    USE_AUDIO_EMOTION: bool = True    # SpeechBrain wav2vec2-IEMOCAP
    USE_FACE_EMOTION: bool = False    # BEiT/ViT face emotion (slower)
    USE_TEXT_EMOTION: bool = True     # GoEmotions text classifier
    
    # Analysis frequency
    ANALYSIS_INTERVAL_SEC: float = 2.0  # Run emotion analysis every N seconds
    
    # Audio emotion settings
    AUDIO_BUFFER_SEC: float = 4.0     # Audio window for emotion analysis
    
    # Model settings
    AUDIO_MODEL: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
    TEXT_MODEL: str = "SamLowe/roberta-base-go_emotions"
    FACE_MODEL: str = "trpakov/vit-face-expression"  # Lightweight
    
    # Device settings
    DEVICE: str = "cuda"              # cuda or cpu

@dataclass  
class SessionConfig:
    """Session recording configuration."""
    ENABLE_RECORDING: bool = True
    RECORDING_DIR: str = "sessions"
    SAVE_TRANSCRIPT: bool = True
    SAVE_METRICS: bool = True
    SAVE_AUDIO: bool = False          # Large files, disabled by default
    SAVE_VIDEO: bool = False          # Very large, disabled by default

@dataclass
class ServerConfig:
    """Server configuration."""
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # WebSocket settings
    MAX_MESSAGE_SIZE: int = 10 * 1024 * 1024  # 10MB max message
    HEARTBEAT_INTERVAL_SEC: float = 30.0

# Global config instance
class Config:
    audio = AudioConfig()
    vision = VisionConfig()
    asr = ASRConfig()
    fusion = FusionConfig()
    coach = CoachConfig()
    emotion = EmotionConfig()
    session = SessionConfig()
    server = ServerConfig()

config = Config()

