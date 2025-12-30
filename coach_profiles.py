"""
Coach Profiles: Modular coaching configurations for different domains.

Inspired by:
- Valkyrie (Personal Trainer): Goal-based coaching with structured conversations
- Witmo (Gaming Coach): Context-aware, spoiler-controlled feedback  
- Fitness AI Trainer: Pose-based exercise tracking
- SwimCoach AI: Video analysis with AI feedback

This module enables the same core engine to support:
- Presentation/Speech Coach (default)
- Life Coach
- Swim Coach  
- Gym/Fitness Coach
- Dance/Posture Coach
- Music/Performance Coach
- Interview Coach
- And more...
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum

class CoachType(Enum):
    """Available coach types."""
    PRESENTATION = "presentation"
    LIFE_COACH = "life_coach"
    SWIM_COACH = "swim_coach"
    GYM_COACH = "gym_coach"
    DANCE_COACH = "dance_coach"
    MUSIC_COACH = "music_coach"
    INTERVIEW_COACH = "interview_coach"
    GAMING_COACH = "gaming_coach"

@dataclass
class MetricWeights:
    """Customizable metric weights for different coach types."""
    pace: float = 0.20
    energy: float = 0.15
    pitch_variety: float = 0.10
    filler_words: float = 0.15
    eye_contact: float = 0.20
    presence: float = 0.10
    stability: float = 0.10
    
    # Domain-specific weights (optional)
    posture: float = 0.0
    movement_quality: float = 0.0
    rhythm: float = 0.0
    form_accuracy: float = 0.0

@dataclass
class CoachProfile:
    """Complete configuration for a specific coach type."""
    
    # Basic info
    coach_type: CoachType
    name: str
    description: str
    icon: str
    tagline: str
    
    # Metric configuration
    weights: MetricWeights
    enabled_metrics: List[str]
    
    # Scoring targets
    target_wpm_min: int = 120
    target_wpm_max: int = 160
    target_pause_ratio: float = 0.15
    
    # Tip templates (category -> list of tips)
    tip_templates: Dict[str, List[dict]] = field(default_factory=dict)
    
    # LLM prompt customization
    system_prompt_prefix: str = ""
    coaching_style: str = "supportive"  # supportive, direct, motivational, technical
    
    # Feature flags
    track_pose: bool = False
    track_movement: bool = False
    track_rhythm: bool = False
    count_reps: bool = False
    
    # Session configuration
    warmup_duration_sec: int = 10
    min_session_duration_sec: int = 30
    recommended_session_duration_sec: int = 300

# ============================================================================
# COACH PROFILES
# ============================================================================

PRESENTATION_COACH = CoachProfile(
    coach_type=CoachType.PRESENTATION,
    name="Presentation Coach",
    description="Master your public speaking and presentation skills",
    icon="🎤",
    tagline="Speak with confidence, connect with your audience",
    weights=MetricWeights(
        pace=0.20,
        energy=0.15,
        pitch_variety=0.15,
        filler_words=0.15,
        eye_contact=0.20,
        presence=0.10,
        stability=0.05
    ),
    enabled_metrics=["pace", "energy", "pitch_variety", "filler_words", "eye_contact", "presence", "stability"],
    target_wpm_min=120,
    target_wpm_max=160,
    tip_templates={
        "eye_contact": [
            {"id": "eye_low", "message": "Look at the camera - imagine your audience there", "severity": "warning"},
            {"id": "eye_good", "message": "Great eye contact! Your audience feels connected", "severity": "info"},
        ],
        "pace": [
            {"id": "pace_slow", "message": "Pick up the pace slightly - aim for 130-150 WPM", "severity": "warning"},
            {"id": "pace_fast", "message": "Slow down - let your key points land", "severity": "warning"},
            {"id": "pace_good", "message": "Perfect pacing! Keep it up", "severity": "info"},
        ],
        "energy": [
            {"id": "energy_low", "message": "Project your voice more - speak from the diaphragm", "severity": "warning"},
            {"id": "energy_good", "message": "Great energy! Your passion is showing", "severity": "info"},
        ],
        "fillers": [
            {"id": "filler_high", "message": "Pause instead of saying 'um' or 'like'", "severity": "warning"},
        ],
    },
    system_prompt_prefix="You are a professional presentation coach helping someone prepare for an important speech or presentation.",
    coaching_style="supportive"
)

LIFE_COACH = CoachProfile(
    coach_type=CoachType.LIFE_COACH,
    name="Life Coach",
    description="Develop confidence, communication, and personal presence",
    icon="🌟",
    tagline="Unlock your potential, one conversation at a time",
    weights=MetricWeights(
        pace=0.15,
        energy=0.20,
        pitch_variety=0.20,
        filler_words=0.10,
        eye_contact=0.15,
        presence=0.15,
        stability=0.05
    ),
    enabled_metrics=["pace", "energy", "pitch_variety", "eye_contact", "presence"],
    target_wpm_min=100,
    target_wpm_max=140,
    tip_templates={
        "energy": [
            {"id": "energy_low", "message": "Speak with more conviction - own your words", "severity": "warning"},
            {"id": "energy_good", "message": "Your confidence is shining through!", "severity": "info"},
        ],
        "pitch_variety": [
            {"id": "monotone", "message": "Add emotion to your voice - let your feelings show", "severity": "warning"},
            {"id": "expressive", "message": "Beautiful vocal expression!", "severity": "info"},
        ],
    },
    system_prompt_prefix="You are a supportive life coach helping someone develop their personal communication and presence.",
    coaching_style="motivational"
)

SWIM_COACH = CoachProfile(
    coach_type=CoachType.SWIM_COACH,
    name="Swim Coach",
    description="Analyze swimming technique and get personalized feedback",
    icon="🏊",
    tagline="Perfect your stroke, conquer the water",
    weights=MetricWeights(
        posture=0.30,
        movement_quality=0.30,
        rhythm=0.20,
        stability=0.20
    ),
    enabled_metrics=["posture", "movement_quality", "rhythm", "stability"],
    track_pose=True,
    track_movement=True,
    track_rhythm=True,
    tip_templates={
        "posture": [
            {"id": "body_position", "message": "Keep your body horizontal - head in neutral position", "severity": "warning"},
            {"id": "streamline", "message": "Excellent streamlined position!", "severity": "info"},
        ],
        "movement": [
            {"id": "stroke_rhythm", "message": "Maintain consistent stroke rhythm", "severity": "warning"},
            {"id": "arm_entry", "message": "Watch your hand entry angle - fingertips first", "severity": "warning"},
        ],
    },
    system_prompt_prefix="You are an expert swim coach analyzing technique and providing feedback to improve efficiency and speed.",
    coaching_style="technical",
    recommended_session_duration_sec=600
)

GYM_COACH = CoachProfile(
    coach_type=CoachType.GYM_COACH,
    name="Gym Coach",
    description="Perfect your exercise form and track your reps",
    icon="💪",
    tagline="Train smarter, not just harder",
    weights=MetricWeights(
        posture=0.30,
        form_accuracy=0.35,
        movement_quality=0.20,
        stability=0.15
    ),
    enabled_metrics=["posture", "form_accuracy", "movement_quality", "stability"],
    track_pose=True,
    track_movement=True,
    count_reps=True,
    tip_templates={
        "form": [
            {"id": "form_check", "message": "Check your form - maintain neutral spine", "severity": "warning"},
            {"id": "form_good", "message": "Perfect form! Keep that controlled movement", "severity": "info"},
        ],
        "posture": [
            {"id": "core_engaged", "message": "Engage your core throughout the movement", "severity": "warning"},
            {"id": "alignment", "message": "Great alignment - protecting your joints", "severity": "info"},
        ],
        "tempo": [
            {"id": "too_fast", "message": "Slow down - control the eccentric phase", "severity": "warning"},
            {"id": "good_tempo", "message": "Excellent controlled tempo!", "severity": "info"},
        ],
    },
    system_prompt_prefix="You are a certified personal trainer helping someone perfect their exercise form and technique.",
    coaching_style="direct",
    recommended_session_duration_sec=1800
)

DANCE_COACH = CoachProfile(
    coach_type=CoachType.DANCE_COACH,
    name="Dance & Posture Coach",
    description="Improve your posture, movement quality, and body awareness",
    icon="💃",
    tagline="Move with grace, stand with confidence",
    weights=MetricWeights(
        posture=0.30,
        movement_quality=0.25,
        rhythm=0.20,
        stability=0.15,
        presence=0.10
    ),
    enabled_metrics=["posture", "movement_quality", "rhythm", "presence", "stability"],
    track_pose=True,
    track_movement=True,
    track_rhythm=True,
    tip_templates={
        "posture": [
            {"id": "shoulders", "message": "Roll your shoulders back - open your chest", "severity": "warning"},
            {"id": "spine", "message": "Lengthen your spine - imagine a string pulling you up", "severity": "warning"},
            {"id": "posture_good", "message": "Beautiful posture!", "severity": "info"},
        ],
        "movement": [
            {"id": "flow", "message": "Connect your movements - let them flow together", "severity": "warning"},
            {"id": "extension", "message": "Extend through your fingertips and toes", "severity": "info"},
        ],
    },
    system_prompt_prefix="You are an experienced dance and movement coach helping someone improve their body awareness and movement quality.",
    coaching_style="supportive",
    recommended_session_duration_sec=600
)

MUSIC_COACH = CoachProfile(
    coach_type=CoachType.MUSIC_COACH,
    name="Music Performance Coach",
    description="Enhance your stage presence and musical delivery",
    icon="🎵",
    tagline="Connect with your audience through music",
    weights=MetricWeights(
        energy=0.25,
        pitch_variety=0.20,
        rhythm=0.20,
        presence=0.20,
        eye_contact=0.15
    ),
    enabled_metrics=["energy", "pitch_variety", "rhythm", "presence", "eye_contact"],
    track_rhythm=True,
    tip_templates={
        "performance": [
            {"id": "stage_presence", "message": "Own the space - use your body to express the music", "severity": "warning"},
            {"id": "connection", "message": "Connect with your audience - make eye contact", "severity": "warning"},
        ],
        "energy": [
            {"id": "dynamics", "message": "Vary your dynamics - soft moments make loud ones powerful", "severity": "info"},
        ],
    },
    system_prompt_prefix="You are a music performance coach helping someone improve their stage presence and musical expression.",
    coaching_style="motivational",
    recommended_session_duration_sec=900
)

INTERVIEW_COACH = CoachProfile(
    coach_type=CoachType.INTERVIEW_COACH,
    name="Interview Coach",
    description="Ace your next job interview with confidence",
    icon="👔",
    tagline="Make a great impression, land the job",
    weights=MetricWeights(
        pace=0.15,
        energy=0.15,
        pitch_variety=0.10,
        filler_words=0.20,
        eye_contact=0.25,
        presence=0.10,
        stability=0.05
    ),
    enabled_metrics=["pace", "energy", "filler_words", "eye_contact", "presence", "stability"],
    target_wpm_min=110,
    target_wpm_max=145,
    tip_templates={
        "eye_contact": [
            {"id": "eye_low", "message": "Maintain steady eye contact - it shows confidence", "severity": "warning"},
            {"id": "eye_good", "message": "Great eye contact - you're making a connection", "severity": "info"},
        ],
        "fillers": [
            {"id": "filler_high", "message": "Take a breath instead of using filler words", "severity": "warning"},
            {"id": "filler_good", "message": "Clear and articulate - well done!", "severity": "info"},
        ],
        "pace": [
            {"id": "pace_fast", "message": "Slow down - give the interviewer time to absorb your answers", "severity": "warning"},
        ],
    },
    system_prompt_prefix="You are an interview coach helping someone prepare for a job interview. Focus on professionalism and clear communication.",
    coaching_style="direct",
    warmup_duration_sec=5
)

GAMING_COACH = CoachProfile(
    coach_type=CoachType.GAMING_COACH,
    name="Gaming Coach",
    description="Improve your gaming communication and streaming presence",
    icon="🎮",
    tagline="Level up your content creation",
    weights=MetricWeights(
        energy=0.25,
        pitch_variety=0.20,
        pace=0.15,
        presence=0.20,
        eye_contact=0.20
    ),
    enabled_metrics=["energy", "pitch_variety", "pace", "presence", "eye_contact"],
    target_wpm_min=140,
    target_wpm_max=180,
    tip_templates={
        "energy": [
            {"id": "energy_low", "message": "Bring more hype! Your audience feeds off your energy", "severity": "warning"},
            {"id": "energy_good", "message": "Great energy! Chat is loving it", "severity": "info"},
        ],
        "engagement": [
            {"id": "camera_check", "message": "Look at the camera when addressing your audience", "severity": "warning"},
        ],
    },
    system_prompt_prefix="You are a streaming and content creation coach helping someone improve their on-camera presence and audience engagement.",
    coaching_style="motivational"
)

# ============================================================================
# PROFILE REGISTRY
# ============================================================================

COACH_PROFILES: Dict[CoachType, CoachProfile] = {
    CoachType.PRESENTATION: PRESENTATION_COACH,
    CoachType.LIFE_COACH: LIFE_COACH,
    CoachType.SWIM_COACH: SWIM_COACH,
    CoachType.GYM_COACH: GYM_COACH,
    CoachType.DANCE_COACH: DANCE_COACH,
    CoachType.MUSIC_COACH: MUSIC_COACH,
    CoachType.INTERVIEW_COACH: INTERVIEW_COACH,
    CoachType.GAMING_COACH: GAMING_COACH,
}

def get_coach_profile(coach_type: CoachType) -> CoachProfile:
    """Get the profile for a specific coach type."""
    return COACH_PROFILES.get(coach_type, PRESENTATION_COACH)

def get_all_profiles() -> List[CoachProfile]:
    """Get all available coach profiles."""
    return list(COACH_PROFILES.values())

def get_profile_summary() -> List[dict]:
    """Get summary info for all profiles (for UI display)."""
    return [
        {
            "type": profile.coach_type.value,
            "name": profile.name,
            "icon": profile.icon,
            "description": profile.description,
            "tagline": profile.tagline,
            "features": {
                "track_pose": profile.track_pose,
                "track_movement": profile.track_movement,
                "track_rhythm": profile.track_rhythm,
                "count_reps": profile.count_reps,
            }
        }
        for profile in COACH_PROFILES.values()
    ]

