"""
Coach Engine: Generates actionable coaching tips based on current metrics.
Implements rule-based tips with optional LLM enhancement.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import time
import logging
import asyncio

from config import config
from fusion_engine import FusedMetrics

logger = logging.getLogger(__name__)

# Try to import httpx for Ollama
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

@dataclass
class CoachingTip:
    """A coaching tip with metadata."""
    id: str                           # Unique identifier (e.g., "pace_slow")
    category: str                     # Category (pace, energy, eye_contact, etc.)
    message: str                      # The tip text
    priority: int = 5                 # Lower = higher priority
    severity: str = "info"            # info, warning, critical
    triggered_at: float = 0.0

# Rule-based tip templates
RULE_TIPS = {
    # Pace issues
    "pace_slow": CoachingTip(
        id="pace_slow",
        category="pace",
        message="Try speaking a bit faster - aim for 130-150 words per minute",
        priority=2,
        severity="warning"
    ),
    "pace_fast": CoachingTip(
        id="pace_fast",
        category="pace",
        message="Slow down slightly - you're speaking quite fast",
        priority=2,
        severity="warning"
    ),
    "pace_good": CoachingTip(
        id="pace_good",
        category="pace",
        message="Great pacing! Keep it up",
        priority=10,
        severity="info"
    ),
    
    # Energy issues
    "energy_low": CoachingTip(
        id="energy_low",
        category="energy",
        message="Project your voice more - speak with more energy",
        priority=4,
        severity="warning"
    ),
    "energy_good": CoachingTip(
        id="energy_good",
        category="energy",
        message="Good vocal energy!",
        priority=10,
        severity="info"
    ),
    
    # Pitch/monotone issues
    "monotone": CoachingTip(
        id="monotone",
        category="pitch",
        message="Add more vocal variety - emphasize key words",
        priority=5,
        severity="info"
    ),
    "expressive": CoachingTip(
        id="expressive",
        category="pitch",
        message="Great vocal variety!",
        priority=10,
        severity="info"
    ),
    
    # Filler words
    "filler_words": CoachingTip(
        id="filler_words",
        category="fillers",
        message="Watch the filler words (um, uh, like) - pause instead",
        priority=3,
        severity="warning"
    ),
    "filler_good": CoachingTip(
        id="filler_good",
        category="fillers",
        message="Great job avoiding filler words!",
        priority=10,
        severity="info"
    ),
    
    # Eye contact
    "eye_contact_low": CoachingTip(
        id="eye_contact_low",
        category="eye_contact",
        message="Look at the camera more - imagine your audience there",
        priority=1,
        severity="critical"
    ),
    "eye_contact_good": CoachingTip(
        id="eye_contact_good",
        category="eye_contact",
        message="Excellent eye contact!",
        priority=10,
        severity="info"
    ),
    
    # Presence/framing
    "presence_low": CoachingTip(
        id="presence_low",
        category="presence",
        message="Check your framing - center yourself in the camera",
        priority=6,
        severity="info"
    ),
    "no_face": CoachingTip(
        id="no_face",
        category="presence",
        message="Face not detected - check your camera position",
        priority=0,
        severity="critical"
    ),
    
    # Stability
    "too_fidgety": CoachingTip(
        id="too_fidgety",
        category="stability",
        message="Try to stay a bit more still - minimize distracting movements",
        priority=7,
        severity="info"
    ),
    
    # Positive reinforcement
    "doing_great": CoachingTip(
        id="doing_great",
        category="overall",
        message="You're doing great! Keep this energy!",
        priority=10,
        severity="info"
    ),
    "improving": CoachingTip(
        id="improving",
        category="overall",
        message="Nice improvement! You're getting better",
        priority=10,
        severity="info"
    ),
}

class CoachEngine:
    """
    Generates coaching tips based on current metrics.
    Uses rule-based logic with optional LLM enhancement.
    """
    
    def __init__(self):
        # Tip cooldown tracking
        self.tip_cooldowns: Dict[str, float] = {}
        self.cooldown_duration = config.coach.TIP_COOLDOWN_SEC
        
        # LLM state
        self.use_llm = config.coach.USE_LLM and HAS_HTTPX
        self.last_llm_time = 0.0
        self.llm_interval = config.coach.LLM_PROMPT_INTERVAL_SEC
        self.last_llm_tip: Optional[str] = None
        
        # Session tracking
        self.tips_given: List[CoachingTip] = []
        self.positive_tips = 0
        self.corrective_tips = 0
        
        logger.info(f"CoachEngine initialized: LLM enabled={self.use_llm}")
    
    def generate_tips(
        self,
        metrics: FusedMetrics,
        recent_transcript: str = ""
    ) -> List[CoachingTip]:
        """
        Generate coaching tips based on current metrics.
        Returns list of tips sorted by priority.
        """
        now = time.time()
        tips = []
        
        # Rule-based tip generation
        tips.extend(self._apply_rules(metrics))
        
        # Filter by cooldown
        active_tips = []
        for tip in tips:
            last_triggered = self.tip_cooldowns.get(tip.id, 0)
            if now - last_triggered >= self.cooldown_duration:
                tip.triggered_at = now
                active_tips.append(tip)
                self.tip_cooldowns[tip.id] = now
        
        # Sort by priority
        active_tips.sort(key=lambda t: t.priority)
        
        # Limit to max tips
        max_tips = config.coach.MAX_TIPS_PER_UPDATE
        final_tips = active_tips[:max_tips]
        
        # Track tips
        for tip in final_tips:
            self.tips_given.append(tip)
            if tip.severity == "info" and "good" in tip.id.lower() or "great" in tip.id.lower():
                self.positive_tips += 1
            else:
                self.corrective_tips += 1
        
        return final_tips
    
    def _apply_rules(self, metrics: FusedMetrics) -> List[CoachingTip]:
        """Apply rule-based logic to generate tips."""
        tips = []
        
        # Check if face is detected
        if not metrics.face_detected:
            tips.append(RULE_TIPS["no_face"])
            return tips  # Critical - skip other checks
        
        # Eye contact
        if metrics.eye_contact_score < 0.4:
            tips.append(RULE_TIPS["eye_contact_low"])
        elif metrics.eye_contact_score > 0.8:
            tips.append(RULE_TIPS["eye_contact_good"])
        
        # Only check speech-related metrics if speaking
        if metrics.is_speaking:
            # Pace
            if metrics.wpm > 0:
                if metrics.wpm < config.audio.TARGET_WPM_MIN:
                    tips.append(RULE_TIPS["pace_slow"])
                elif metrics.wpm > config.audio.TARGET_WPM_MAX:
                    tips.append(RULE_TIPS["pace_fast"])
                elif metrics.pace_score > 0.8:
                    tips.append(RULE_TIPS["pace_good"])
            
            # Energy
            if metrics.energy_score < 0.4:
                tips.append(RULE_TIPS["energy_low"])
            elif metrics.energy_score > 0.8:
                tips.append(RULE_TIPS["energy_good"])
            
            # Pitch variety (monotone)
            if metrics.pitch_variety_score < 0.4:
                tips.append(RULE_TIPS["monotone"])
            elif metrics.pitch_variety_score > 0.7:
                tips.append(RULE_TIPS["expressive"])
            
            # Filler words
            if metrics.filler_ratio > 0.05:  # More than 5% fillers
                tips.append(RULE_TIPS["filler_words"])
            elif metrics.filler_word_score > 0.9:
                tips.append(RULE_TIPS["filler_good"])
        
        # Presence/framing
        if metrics.presence_score < 0.5:
            tips.append(RULE_TIPS["presence_low"])
        
        # Stability
        if metrics.stability_score < 0.4:
            tips.append(RULE_TIPS["too_fidgety"])
        
        # Overall encouragement
        if metrics.overall_score > 0.8:
            tips.append(RULE_TIPS["doing_great"])
        elif metrics.overall_trend == 1:
            tips.append(RULE_TIPS["improving"])
        
        return tips
    
    async def get_llm_tip(
        self,
        metrics: FusedMetrics,
        recent_transcript: str
    ) -> Optional[str]:
        """
        Get enhanced tip from local LLM (Ollama).
        Only called periodically to avoid latency.
        """
        if not self.use_llm:
            return None
        
        now = time.time()
        if now - self.last_llm_time < self.llm_interval:
            return self.last_llm_tip
        
        self.last_llm_time = now
        
        try:
            # Build prompt with current metrics
            prompt = self._build_llm_prompt(metrics, recent_transcript)
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{config.coach.OLLAMA_URL}/api/generate",
                    json={
                        "model": config.coach.OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 100,
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    tip = result.get("response", "").strip()
                    # Clean up the tip
                    tip = tip.split('\n')[0]  # First line only
                    if len(tip) > 10 and len(tip) < 200:
                        self.last_llm_tip = tip
                        return tip
                        
        except Exception as e:
            logger.debug(f"LLM tip generation failed: {e}")
        
        return None
    
    def _build_llm_prompt(self, metrics: FusedMetrics, transcript: str) -> str:
        """Build prompt for LLM coaching tip."""
        return f"""You are a presentation coach giving brief, actionable feedback.

Current performance metrics (0-1 scale, higher is better):
- Pace score: {metrics.pace_score:.2f} (WPM: {metrics.wpm:.0f})
- Energy score: {metrics.energy_score:.2f}
- Vocal variety: {metrics.pitch_variety_score:.2f}
- Eye contact: {metrics.eye_contact_score:.2f}
- Overall score: {metrics.overall_score:.2f}

Recent transcript: "{transcript[-200:] if transcript else 'No speech yet'}"

Give ONE short, specific, encouraging coaching tip (max 20 words). Focus on the lowest scoring area. Be constructive and supportive."""
    
    def reset(self):
        """Reset for new session."""
        self.tip_cooldowns = {}
        self.tips_given = []
        self.positive_tips = 0
        self.corrective_tips = 0
        self.last_llm_time = 0.0
        self.last_llm_tip = None
        logger.info("CoachEngine reset")
    
    def get_session_stats(self) -> dict:
        """Get session summary statistics."""
        return {
            "total_tips_given": len(self.tips_given),
            "positive_tips": self.positive_tips,
            "corrective_tips": self.corrective_tips,
            "tip_breakdown": self._get_tip_breakdown(),
        }
    
    def _get_tip_breakdown(self) -> Dict[str, int]:
        """Get count of tips by category."""
        breakdown: Dict[str, int] = {}
        for tip in self.tips_given:
            breakdown[tip.category] = breakdown.get(tip.category, 0) + 1
        return breakdown

