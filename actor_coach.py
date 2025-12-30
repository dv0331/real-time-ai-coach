"""
Actor's AI Coach - Specialized Coaching Rules for Actors

Provides context-aware feedback tailored for:
- Monologue practice
- Scene rehearsal  
- Audition preparation
- Self-tape sessions

The coaching focuses on:
1. Delivery (pace, pauses, rhythm)
2. Vocal Performance (energy, variety, projection)
3. Physical Presence (eye contact, framing, stillness)
4. Technical Skills (filler words, breath control)
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
import time
import logging
import random

logger = logging.getLogger(__name__)


class SceneIntensity(Enum):
    """Scene emotional intensity levels."""
    INTIMATE = "intimate"      # Quiet, close moments
    NEUTRAL = "neutral"        # Normal conversation
    HEIGHTENED = "heightened"  # Emotional scenes
    INTENSE = "intense"        # Arguments, confrontations


class PerformanceMode(Enum):
    """Different performance contexts."""
    FREE_PRACTICE = "free_practice"    # Open rehearsal
    MONOLOGUE = "monologue"            # Solo piece
    SCENE = "scene"                     # With partner/reader
    SELF_TAPE = "self_tape"            # Audition recording
    COLD_READ = "cold_read"            # First read of material


@dataclass
class ActorTip:
    """A coaching tip for the actor."""
    id: str
    category: str  # delivery, vocal, presence, technique
    message: str
    severity: str  # success, info, warning, error
    priority: int = 5  # 1-10, higher = more important
    subcategory: str = ""


class ActorCoach:
    """
    Specialized coaching engine for actors.
    Generates contextual feedback based on performance metrics.
    """
    
    def __init__(self):
        # Performance mode
        self.mode = PerformanceMode.FREE_PRACTICE
        self.scene_intensity = SceneIntensity.NEUTRAL
        
        # Thresholds (adjusted based on mode)
        self.thresholds = self._get_thresholds()
        
        # Tip cooldowns (prevent spam)
        self.tip_cooldowns: Dict[str, float] = {}
        self.cooldown_duration = 4.0  # seconds
        
        # Session stats
        self.tips_given: List[ActorTip] = []
        self.positive_count = 0
        self.corrective_count = 0
        
        # Encouragement messages (actors need positivity!)
        self.encouragements = [
            "Great work! Keep that energy",
            "Beautiful moment there",
            "That's the commitment we need",
            "You're in the zone",
            "Excellent instincts",
            "Trust your choices",
            "That landed perfectly",
            "Strong presence!",
        ]
        
        logger.info("ActorCoach initialized")
    
    def _get_thresholds(self) -> Dict[str, Any]:
        """Get thresholds based on current mode and intensity."""
        
        # Base thresholds
        thresholds = {
            # Pace (WPM)
            "wpm_slow": 100,
            "wpm_ideal_min": 120,
            "wpm_ideal_max": 160,
            "wpm_fast": 180,
            
            # Energy (dB)
            "energy_low": -40,
            "energy_good": -30,
            "energy_high": -15,
            
            # Pitch variety (score 0-1)
            "variety_monotone": 0.3,
            "variety_good": 0.5,
            
            # Eye contact (score 0-1)
            "eye_contact_poor": 0.3,
            "eye_contact_good": 0.6,
            
            # Presence (score 0-1)
            "presence_weak": 0.4,
            "presence_strong": 0.7,
            
            # Fillers (ratio)
            "filler_acceptable": 0.02,
            "filler_problem": 0.05,
            
            # Pause ratio
            "pause_too_little": 0.1,
            "pause_ideal": 0.2,
            "pause_too_much": 0.4,
        }
        
        # Adjust for scene intensity
        if self.scene_intensity == SceneIntensity.INTIMATE:
            thresholds["wpm_ideal_max"] = 140
            thresholds["energy_good"] = -35
            thresholds["pause_ideal"] = 0.25
            
        elif self.scene_intensity == SceneIntensity.INTENSE:
            thresholds["wpm_ideal_max"] = 180
            thresholds["energy_good"] = -20
            thresholds["pause_ideal"] = 0.15
        
        return thresholds
    
    def set_mode(self, mode: PerformanceMode, intensity: SceneIntensity = SceneIntensity.NEUTRAL):
        """Set the performance mode and scene intensity."""
        self.mode = mode
        self.scene_intensity = intensity
        self.thresholds = self._get_thresholds()
        logger.info(f"Mode set to {mode.value}, intensity {intensity.value}")
    
    def can_give_tip(self, tip_id: str) -> bool:
        """Check if enough time has passed since last tip of this type."""
        now = time.time()
        if tip_id in self.tip_cooldowns:
            if now - self.tip_cooldowns[tip_id] < self.cooldown_duration:
                return False
        return True
    
    def record_tip(self, tip: ActorTip):
        """Record that a tip was given."""
        self.tip_cooldowns[tip.id] = time.time()
        self.tips_given.append(tip)
        
        if tip.severity == "success":
            self.positive_count += 1
        elif tip.severity in ("warning", "error"):
            self.corrective_count += 1
    
    def generate_tips(self, metrics: Any, transcript: str = "") -> List[ActorTip]:
        """
        Generate actor-specific coaching tips based on current metrics.
        
        Args:
            metrics: FusedMetrics object with current performance data
            transcript: Recent transcript text
            
        Returns:
            List of ActorTip objects to display
        """
        tips = []
        
        # === DELIVERY FEEDBACK ===
        tips.extend(self._analyze_pace(metrics))
        tips.extend(self._analyze_pauses(metrics))
        
        # === VOCAL FEEDBACK ===
        tips.extend(self._analyze_energy(metrics))
        tips.extend(self._analyze_variety(metrics))
        
        # === PRESENCE FEEDBACK ===
        tips.extend(self._analyze_eye_contact(metrics))
        tips.extend(self._analyze_presence(metrics))
        
        # === TECHNIQUE FEEDBACK ===
        tips.extend(self._analyze_fillers(metrics, transcript))
        
        # === POSITIVE REINFORCEMENT ===
        # Occasionally give encouragement if doing well
        if len(tips) == 0 and random.random() < 0.1:  # 10% chance
            tips.append(self._get_encouragement(metrics))
        
        # Record tips and filter by cooldown
        filtered_tips = []
        for tip in tips:
            if self.can_give_tip(tip.id):
                self.record_tip(tip)
                filtered_tips.append(tip)
        
        return filtered_tips
    
    def _analyze_pace(self, metrics) -> List[ActorTip]:
        """Analyze speaking pace/tempo."""
        tips = []
        wpm = metrics.wpm
        
        if wpm < self.thresholds["wpm_slow"]:
            tips.append(ActorTip(
                id="pace_very_slow",
                category="delivery",
                subcategory="pace",
                message="Pick up the pace - don't lose momentum",
                severity="warning",
                priority=7
            ))
        elif wpm < self.thresholds["wpm_ideal_min"]:
            tips.append(ActorTip(
                id="pace_slow",
                category="delivery",
                subcategory="pace",
                message="Slightly slow - keep the energy moving",
                severity="info",
                priority=5
            ))
        elif wpm > self.thresholds["wpm_fast"]:
            tips.append(ActorTip(
                id="pace_very_fast",
                category="delivery",
                subcategory="pace",
                message="Slow down! Let the words land",
                severity="warning",
                priority=8
            ))
        elif wpm > self.thresholds["wpm_ideal_max"]:
            tips.append(ActorTip(
                id="pace_fast",
                category="delivery",
                subcategory="pace",
                message="Slightly rushed - breathe and pace yourself",
                severity="info",
                priority=5
            ))
        elif self.thresholds["wpm_ideal_min"] <= wpm <= self.thresholds["wpm_ideal_max"]:
            # Good pace - occasionally acknowledge
            if random.random() < 0.05:
                tips.append(ActorTip(
                    id="pace_good",
                    category="delivery",
                    subcategory="pace",
                    message="Great tempo! You're in control",
                    severity="success",
                    priority=3
                ))
        
        return tips
    
    def _analyze_pauses(self, metrics) -> List[ActorTip]:
        """Analyze use of pauses and silence."""
        tips = []
        pause_ratio = getattr(metrics, 'pause_ratio', 0.2)
        
        if pause_ratio < self.thresholds["pause_too_little"]:
            tips.append(ActorTip(
                id="pause_none",
                category="delivery",
                subcategory="pauses",
                message="Use pauses - silence is powerful",
                severity="info",
                priority=6
            ))
        elif pause_ratio > self.thresholds["pause_too_much"]:
            tips.append(ActorTip(
                id="pause_excessive",
                category="delivery",
                subcategory="pauses",
                message="Too many pauses - keep the scene moving",
                severity="warning",
                priority=6
            ))
        
        return tips
    
    def _analyze_energy(self, metrics) -> List[ActorTip]:
        """Analyze vocal energy and projection."""
        tips = []
        energy = metrics.energy_db
        
        if energy < self.thresholds["energy_low"]:
            tips.append(ActorTip(
                id="energy_very_low",
                category="vocal",
                subcategory="energy",
                message="Project more! Fill the space with your voice",
                severity="warning",
                priority=8
            ))
        elif energy < self.thresholds["energy_good"]:
            # Depends on scene intensity
            if self.scene_intensity != SceneIntensity.INTIMATE:
                tips.append(ActorTip(
                    id="energy_low",
                    category="vocal",
                    subcategory="energy",
                    message="Bring more energy - commit to the moment",
                    severity="info",
                    priority=5
                ))
        elif energy > self.thresholds["energy_high"]:
            if self.scene_intensity != SceneIntensity.INTENSE:
                tips.append(ActorTip(
                    id="energy_high",
                    category="vocal",
                    subcategory="energy",
                    message="Pull back slightly - save it for the big moments",
                    severity="info",
                    priority=4
                ))
        
        return tips
    
    def _analyze_variety(self, metrics) -> List[ActorTip]:
        """Analyze vocal variety (avoid monotone)."""
        tips = []
        variety = metrics.pitch_variety_score
        
        if variety < self.thresholds["variety_monotone"]:
            tips.append(ActorTip(
                id="variety_monotone",
                category="vocal",
                subcategory="variety",
                message="Vary your pitch - avoid monotone delivery",
                severity="warning",
                priority=7
            ))
        elif variety < self.thresholds["variety_good"]:
            tips.append(ActorTip(
                id="variety_low",
                category="vocal",
                subcategory="variety",
                message="More vocal variety - find the music in the text",
                severity="info",
                priority=5
            ))
        elif variety > 0.7 and random.random() < 0.1:
            tips.append(ActorTip(
                id="variety_good",
                category="vocal",
                subcategory="variety",
                message="Beautiful vocal range! Keep it up",
                severity="success",
                priority=3
            ))
        
        return tips
    
    def _analyze_eye_contact(self, metrics) -> List[ActorTip]:
        """Analyze eye contact and gaze."""
        tips = []
        eye_contact = metrics.eye_contact_score
        looking = metrics.looking_at_camera
        
        if not metrics.face_detected:
            tips.append(ActorTip(
                id="face_not_detected",
                category="presence",
                subcategory="framing",
                message="Find your light - stay in frame",
                severity="warning",
                priority=9
            ))
        elif eye_contact < self.thresholds["eye_contact_poor"]:
            tips.append(ActorTip(
                id="eye_contact_poor",
                category="presence",
                subcategory="eye_contact",
                message="Connect with the camera - eye contact matters",
                severity="warning",
                priority=8
            ))
        elif eye_contact < self.thresholds["eye_contact_good"]:
            tips.append(ActorTip(
                id="eye_contact_moderate",
                category="presence",
                subcategory="eye_contact",
                message="More eye contact - draw us in",
                severity="info",
                priority=5
            ))
        elif eye_contact > 0.8 and random.random() < 0.1:
            tips.append(ActorTip(
                id="eye_contact_great",
                category="presence",
                subcategory="eye_contact",
                message="Strong connection! We believe you",
                severity="success",
                priority=3
            ))
        
        return tips
    
    def _analyze_presence(self, metrics) -> List[ActorTip]:
        """Analyze physical presence and command."""
        tips = []
        presence = metrics.presence_score
        stability = metrics.stability_score
        
        if presence < self.thresholds["presence_weak"]:
            tips.append(ActorTip(
                id="presence_weak",
                category="presence",
                subcategory="command",
                message="Own the space - you belong here",
                severity="warning",
                priority=7
            ))
        elif presence > self.thresholds["presence_strong"] and random.random() < 0.1:
            tips.append(ActorTip(
                id="presence_strong",
                category="presence",
                subcategory="command",
                message="Commanding presence! You've got the room",
                severity="success",
                priority=3
            ))
        
        # Stability (too much movement)
        if stability < 0.3:
            tips.append(ActorTip(
                id="stability_low",
                category="presence",
                subcategory="movement",
                message="Too much movement - find stillness when needed",
                severity="info",
                priority=5
            ))
        
        return tips
    
    def _analyze_fillers(self, metrics, transcript: str) -> List[ActorTip]:
        """Analyze filler word usage."""
        tips = []
        filler_ratio = metrics.filler_ratio
        
        if filler_ratio > self.thresholds["filler_problem"]:
            tips.append(ActorTip(
                id="filler_problem",
                category="technique",
                subcategory="fillers",
                message="Too many 'um's and 'uh's - stay in character",
                severity="warning",
                priority=7
            ))
        elif filler_ratio > self.thresholds["filler_acceptable"]:
            tips.append(ActorTip(
                id="filler_notice",
                category="technique",
                subcategory="fillers",
                message="Watch the filler words - replace with silence",
                severity="info",
                priority=4
            ))
        
        return tips
    
    def _get_encouragement(self, metrics) -> ActorTip:
        """Get a random encouragement message."""
        message = random.choice(self.encouragements)
        return ActorTip(
            id="encouragement",
            category="general",
            subcategory="positive",
            message=message,
            severity="success",
            priority=2
        )
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get end-of-session summary for actors."""
        total_tips = len(self.tips_given)
        
        # Analyze tip categories
        category_counts = {}
        for tip in self.tips_given:
            cat = tip.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Identify main areas for improvement
        areas_to_work_on = []
        if category_counts.get("delivery", 0) > 3:
            areas_to_work_on.append("Pace and timing")
        if category_counts.get("vocal", 0) > 3:
            areas_to_work_on.append("Vocal variety and energy")
        if category_counts.get("presence", 0) > 3:
            areas_to_work_on.append("Physical presence and eye contact")
        if category_counts.get("technique", 0) > 2:
            areas_to_work_on.append("Technical habits (fillers, breathing)")
        
        # Generate summary feedback
        if self.positive_count > self.corrective_count:
            overall_assessment = "Strong session! Your instincts are good."
        elif total_tips < 5:
            overall_assessment = "Solid work! Keep refining."
        else:
            overall_assessment = "Good practice! Focus on the areas highlighted."
        
        return {
            "total_tips_given": total_tips,
            "positive_tips": self.positive_count,
            "corrective_tips": self.corrective_count,
            "category_breakdown": category_counts,
            "areas_to_work_on": areas_to_work_on,
            "overall_assessment": overall_assessment,
            "mode": self.mode.value,
            "intensity": self.scene_intensity.value
        }
    
    def reset(self):
        """Reset for new session."""
        self.tip_cooldowns.clear()
        self.tips_given.clear()
        self.positive_count = 0
        self.corrective_count = 0
        logger.info("ActorCoach reset")


# ============================================================================
# SCRIPT MODE (for line practice)
# ============================================================================

@dataclass
class ScriptLine:
    """A line from a script."""
    character: str
    text: str
    line_number: int
    cue: str = ""  # Previous character's line (cue line)


class ScriptManager:
    """
    Manages script/sides for practice.
    Tracks current line and provides prompts.
    """
    
    def __init__(self):
        self.script: List[ScriptLine] = []
        self.current_line_index = 0
        self.character_name = ""
        
    def load_script(self, text: str, character: str = ""):
        """
        Load a script from text.
        
        Format:
        CHARACTER: Line of dialogue
        OTHER: Their line
        CHARACTER: Another line
        """
        self.script = []
        self.character_name = character
        
        lines = text.strip().split("\n")
        cue = ""
        
        for i, line in enumerate(lines):
            if ":" in line:
                parts = line.split(":", 1)
                char = parts[0].strip()
                text = parts[1].strip()
                
                self.script.append(ScriptLine(
                    character=char,
                    text=text,
                    line_number=i + 1,
                    cue=cue
                ))
                
                cue = text  # This line becomes cue for next
            else:
                # Stage direction or continuation
                if self.script:
                    self.script[-1].text += " " + line.strip()
        
        self.current_line_index = 0
        logger.info(f"Loaded script with {len(self.script)} lines")
    
    def get_current_line(self) -> Optional[ScriptLine]:
        """Get the current line to practice."""
        if 0 <= self.current_line_index < len(self.script):
            return self.script[self.current_line_index]
        return None
    
    def get_my_lines(self) -> List[ScriptLine]:
        """Get all lines for the specified character."""
        if not self.character_name:
            return self.script
        return [l for l in self.script if l.character.upper() == self.character_name.upper()]
    
    def advance_line(self):
        """Move to next line."""
        self.current_line_index += 1
        if self.current_line_index >= len(self.script):
            self.current_line_index = 0  # Loop back
    
    def compare_delivery(self, spoken_text: str, expected: ScriptLine) -> Dict[str, Any]:
        """
        Compare spoken text to expected script line.
        Returns accuracy metrics.
        """
        from difflib import SequenceMatcher
        
        expected_text = expected.text.lower()
        spoken_lower = spoken_text.lower()
        
        # Calculate similarity
        matcher = SequenceMatcher(None, expected_text, spoken_lower)
        similarity = matcher.ratio()
        
        # Find missing/added words
        expected_words = set(expected_text.split())
        spoken_words = set(spoken_lower.split())
        
        missing = expected_words - spoken_words
        added = spoken_words - expected_words
        
        return {
            "accuracy": similarity,
            "missing_words": list(missing),
            "added_words": list(added),
            "expected": expected.text,
            "spoken": spoken_text
        }
    
    def reset(self):
        """Reset to beginning."""
        self.current_line_index = 0


# ============================================================================
# LLM INTEGRATION (Optional - Ollama)
# ============================================================================

ACTOR_LLM_PROMPT = """You are an experienced acting coach providing brief, actionable feedback to an actor during rehearsal.

Current performance metrics:
- Speaking pace: {wpm} WPM (ideal: 120-160)
- Energy level: {energy} dB
- Vocal variety: {variety}%  
- Eye contact: {eye_contact}%
- Presence score: {presence}%
- Filler words: {fillers}%

Recent transcript:
"{transcript}"

Performance mode: {mode}
Scene intensity: {intensity}

Give ONE brief, specific, encouraging piece of direction (max 2 sentences). Focus on what will most improve the performance right now. Speak like a supportive director, not a critic."""


async def get_llm_actor_feedback(
    metrics: Any,
    transcript: str,
    mode: str = "free_practice",
    intensity: str = "neutral",
    ollama_model: str = "llama3.2"
) -> Optional[str]:
    """
    Get rich feedback from local LLM (Ollama).
    
    Returns None if LLM is unavailable or times out.
    """
    try:
        import httpx
        
        prompt = ACTOR_LLM_PROMPT.format(
            wpm=round(metrics.wpm),
            energy=round(metrics.energy_db),
            variety=round(metrics.pitch_variety_score * 100),
            eye_contact=round(metrics.eye_contact_score * 100),
            presence=round(metrics.presence_score * 100),
            fillers=round(metrics.filler_ratio * 100, 1),
            transcript=transcript[-500:] if len(transcript) > 500 else transcript,
            mode=mode,
            intensity=intensity
        )
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
                
    except Exception as e:
        logger.debug(f"LLM feedback unavailable: {e}")
    
    return None

