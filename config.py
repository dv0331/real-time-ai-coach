"""
Configuration and tuning knobs for the Real-Time AI Coach.
Adjust these parameters based on your hardware and latency requirements.

DEPLOYMENT MODES:
1. LOCAL (default): Uses GPU for all AI models - free, private, fast
2. CLOUD: Uses OpenAI APIs - works online, no GPU needed, costs money

To switch modes:
- Set DEPLOYMENT_MODE = "cloud" 
- Set OPENAI_API_KEY environment variable
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================================
# DEPLOYMENT CONFIGURATION
# ============================================================================

@dataclass
class DeploymentConfig:
    """
    Deployment mode configuration.
    
    LOCAL mode (default):
    - Uses Ollama for LLM tips
    - Uses faster-whisper for ASR (GPU)
    - Uses SpeechBrain for emotion (GPU)
    - Free, private, requires GPU
    
    CLOUD mode:
    - Uses OpenAI GPT-4o-mini for LLM tips
    - Uses OpenAI Whisper API for ASR
    - Uses GPT-4 Vision for face analysis (optional)
    - Paid per usage, no GPU needed, works anywhere
    """
    
    # "local" or "cloud" - can also be set via DEPLOYMENT_MODE env var
    MODE: str = os.environ.get("DEPLOYMENT_MODE", "local")
    
    # OpenAI API key (required for cloud mode)
    # Can also be set via OPENAI_API_KEY environment variable
    OPENAI_API_KEY: Optional[str] = os.environ.get("OPENAI_API_KEY")
    
    # Cloud model settings
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"      # For coaching tips
    OPENAI_ASR_MODEL: str = "whisper-1"        # For transcription
    OPENAI_VISION_MODEL: str = "gpt-4o"        # For face analysis
    
    # Use GPT-4 Vision for face emotion (more accurate than MediaPipe)
    USE_VISION_FOR_EMOTION: bool = False
    
    # ASR buffer duration for cloud (longer = better accuracy, more latency)
    CLOUD_ASR_BUFFER_SEC: float = 3.0
    
    def __post_init__(self):
        """Load API key from environment if not set."""
        if not self.OPENAI_API_KEY:
            self.OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    
    @property
    def is_cloud(self) -> bool:
        """Check if running in cloud mode."""
        return self.MODE.lower() == "cloud"
    
    @property
    def is_local(self) -> bool:
        """Check if running in local mode."""
        return self.MODE.lower() == "local"
    
    @property
    def has_api_key(self) -> bool:
        """Check if OpenAI API key is available."""
        return bool(self.OPENAI_API_KEY)

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
    PORT: int = int(os.environ.get("PORT", 8000))  # Use PORT env var for cloud platforms
    DEBUG: bool = os.environ.get("DEBUG", "false").lower() != "false"
    
    # WebSocket settings
    MAX_MESSAGE_SIZE: int = 10 * 1024 * 1024  # 10MB max message
    HEARTBEAT_INTERVAL_SEC: float = 30.0

# ============================================================================
# GLOBAL CONFIG INSTANCE
# ============================================================================

class Config:
    """Global configuration singleton."""
    
    # Deployment mode (local vs cloud)
    deployment = DeploymentConfig()
    
    # Individual pipeline configs
    audio = AudioConfig()
    vision = VisionConfig()
    asr = ASRConfig()
    fusion = FusionConfig()
    coach = CoachConfig()
    emotion = EmotionConfig()
    session = SessionConfig()
    server = ServerConfig()
    
    @classmethod
    def set_cloud_mode(cls, api_key: str = None):
        """
        Switch to cloud deployment mode.
        
        Args:
            api_key: OpenAI API key (optional if OPENAI_API_KEY env var is set)
        """
        cls.deployment.MODE = "cloud"
        if api_key:
            cls.deployment.OPENAI_API_KEY = api_key
        
        # Adjust settings for cloud
        cls.coach.USE_LLM = True
        cls.asr.DEVICE = "cpu"  # No GPU needed
        cls.emotion.DEVICE = "cpu"
        
        print(f"✅ Switched to CLOUD mode")
        print(f"   LLM: OpenAI {cls.deployment.OPENAI_LLM_MODEL}")
        print(f"   ASR: OpenAI {cls.deployment.OPENAI_ASR_MODEL}")
    
    @classmethod
    def set_local_mode(cls):
        """Switch to local deployment mode (default)."""
        cls.deployment.MODE = "local"
        cls.asr.DEVICE = "cuda"
        cls.emotion.DEVICE = "cuda"
        
        print(f"✅ Switched to LOCAL mode")
        print(f"   LLM: Ollama {cls.coach.OLLAMA_MODEL}")
        print(f"   ASR: faster-whisper (GPU)")


config = Config()


# ============================================================================
# QUICK SETUP HELPERS
# ============================================================================

def setup_for_cloud(api_key: str = None):
    """
    Quick setup for cloud deployment.
    
    Usage:
        from config import setup_for_cloud
        setup_for_cloud("sk-your-api-key")
    
    Or set OPENAI_API_KEY environment variable first:
        export OPENAI_API_KEY="sk-your-api-key"
        
        from config import setup_for_cloud
        setup_for_cloud()
    """
    config.set_cloud_mode(api_key)


def setup_for_local():
    """
    Quick setup for local deployment (default).
    
    Requires:
    - CUDA-enabled GPU
    - Ollama running with llama3.2
    """
    config.set_local_mode()

