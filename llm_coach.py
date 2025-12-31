"""
LLM Coach - Ollama-powered coaching tips for actors

Uses local Llama model via Ollama to generate rich, contextual coaching feedback
based on multimodal signals (audio emotion, face expression, presence, transcript).

Run Ollama server:
    ollama serve
    ollama pull llama3.2  # or llama3.1:8b
"""

import json
import logging
import time
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

# Default Ollama configuration
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"  # Fast, good for real-time


@dataclass
class LLMCoachingTip:
    """Structured coaching tip from LLM."""
    tip: str
    strength: str  # What actor is doing well
    improve: str   # What to focus on
    emotion_note: str = ""  # Note about emotional delivery
    raw_response: str = ""


class LLMCoach:
    """
    Local LLM coach using Ollama.
    Generates contextual coaching tips based on performance signals.
    """
    
    def __init__(self, 
                 model: str = DEFAULT_MODEL,
                 ollama_url: str = OLLAMA_URL,
                 timeout: float = 10.0):
        """
        Initialize LLM coach.
        
        Args:
            model: Ollama model name (e.g., "llama3.2", "llama3.1:8b")
            ollama_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.is_available = False
        self.last_tip_time = 0
        self.tip_cooldown = 5.0  # Minimum seconds between LLM tips
        
        # Track LLM usage
        self.tips_generated = 0
        self.total_latency_ms = 0
        
        logger.info(f"LLMCoach initialized with model: {model}")
    
    async def check_availability(self) -> bool:
        """Check if Ollama server is running."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    self.is_available = any(self.model in n for n in model_names)
                    if not self.is_available:
                        logger.warning(f"Model {self.model} not found. Available: {model_names}")
                    return self.is_available
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            self.is_available = False
        return False
    
    async def generate_tip(self, 
                          signals: Dict[str, Any],
                          transcript: str = "",
                          context: str = "acting rehearsal") -> Optional[LLMCoachingTip]:
        """
        Generate a coaching tip using the LLM.
        
        Args:
            signals: Dict with performance signals (emotion, presence, metrics)
            transcript: Recent transcript text
            context: Performance context (e.g., "acting rehearsal", "monologue")
            
        Returns:
            LLMCoachingTip or None if unavailable/cooldown
        """
        # Check cooldown
        now = time.time()
        if now - self.last_tip_time < self.tip_cooldown:
            return None
        
        if not self.is_available:
            # Try to check availability
            await self.check_availability()
            if not self.is_available:
                return None
        
        try:
            start_time = time.time()
            
            # Build prompt
            prompt = self._build_prompt(signals, transcript, context)
            
            # Call Ollama
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.4,
                            "num_predict": 200,  # Keep responses short
                            "top_p": 0.9
                        }
                    }
                )
                
                if response.status_code != 200:
                    logger.warning(f"Ollama error: {response.status_code}")
                    return None
                
                result = response.json()
                content = result.get("message", {}).get("content", "")
                
                # Parse response
                tip = self._parse_response(content)
                
                # Track metrics
                latency = (time.time() - start_time) * 1000
                self.total_latency_ms += latency
                self.tips_generated += 1
                self.last_tip_time = now
                
                logger.info(f"LLM tip generated in {latency:.0f}ms")
                
                return tip
                
        except asyncio.TimeoutError:
            logger.warning("LLM request timed out")
            return None
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return None
    
    def _build_prompt(self, signals: Dict[str, Any], transcript: str, context: str) -> List[Dict]:
        """Build chat prompt for Ollama."""
        
        system_prompt = """You are a supportive acting coach providing real-time feedback during rehearsal.
        
Your job is to give BRIEF, SPECIFIC, ACTIONABLE coaching based on the performance signals.

Rules:
- Be encouraging but honest
- Focus on ONE thing to improve
- Keep each field under 15 words
- Speak like a director, not a critic
- Reference specific signals when relevant

Output JSON with these fields:
{
  "tip": "Main coaching direction (action to take NOW)",
  "strength": "What the actor is doing well",
  "improve": "Specific thing to work on",
  "emotion_note": "Comment on emotional delivery if relevant"
}"""

        # Format signals for the prompt
        signal_summary = self._format_signals(signals)
        
        user_prompt = f"""PERFORMANCE CONTEXT: {context}

SIGNALS:
{signal_summary}

RECENT TRANSCRIPT (last ~50 words):
"{transcript[-300:] if len(transcript) > 300 else transcript}"

Based on these signals, give ONE brief coaching tip. Output valid JSON only."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def _format_signals(self, signals: Dict[str, Any]) -> str:
        """Format signals dict into readable summary."""
        lines = []
        
        # Audio emotion
        if "audio_emotion" in signals:
            lines.append(f"- Voice emotion: {signals['audio_emotion']}")
        if "audio_arousal" in signals:
            arousal = signals['audio_arousal']
            level = "high" if arousal > 0.7 else "low" if arousal < 0.3 else "moderate"
            lines.append(f"- Voice energy/arousal: {level} ({arousal:.2f})")
        
        # Face emotion
        if "face_emotion" in signals:
            lines.append(f"- Facial expression: {signals['face_emotion']}")
        if "face_expressiveness" in signals:
            expr = signals['face_expressiveness']
            level = "very expressive" if expr > 0.7 else "flat" if expr < 0.3 else "moderate"
            lines.append(f"- Face expressiveness: {level} ({expr:.2f})")
        
        # Text emotion
        if "text_sentiment" in signals:
            lines.append(f"- Script/text sentiment: {signals['text_sentiment']}")
        if "text_emotions" in signals and signals["text_emotions"]:
            top_emotions = ", ".join([e["label"] for e in signals["text_emotions"][:3]])
            lines.append(f"- Text emotions detected: {top_emotions}")
        
        # Presence metrics
        if "eye_contact" in signals:
            lines.append(f"- Eye contact: {signals['eye_contact']:.0%}")
        if "presence" in signals:
            lines.append(f"- Stage presence: {signals['presence']:.0%}")
        if "pace_wpm" in signals:
            lines.append(f"- Speaking pace: {signals['pace_wpm']:.0f} WPM")
        if "energy_db" in signals:
            lines.append(f"- Volume: {signals['energy_db']:.0f} dB")
        if "pitch_variety" in signals:
            lines.append(f"- Vocal variety: {signals['pitch_variety']:.0%}")
        if "filler_ratio" in signals:
            lines.append(f"- Filler words: {signals['filler_ratio']:.1%}")
        
        # Trends
        if "emotion_intensity" in signals:
            lines.append(f"- Overall emotional intensity: {signals['emotion_intensity']:.0%}")
        if "emotional_range" in signals:
            lines.append(f"- Emotional range shown: {signals['emotional_range']:.0%}")
        
        return "\n".join(lines) if lines else "No signals available"
    
    def _parse_response(self, content: str) -> LLMCoachingTip:
        """Parse LLM response into structured tip."""
        try:
            # Try to extract JSON from response
            content = content.strip()
            
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            # Parse JSON
            data = json.loads(content)
            
            return LLMCoachingTip(
                tip=data.get("tip", "Keep going!"),
                strength=data.get("strength", "Good work"),
                improve=data.get("improve", "Stay focused"),
                emotion_note=data.get("emotion_note", ""),
                raw_response=content
            )
            
        except json.JSONDecodeError:
            # If JSON parsing fails, extract text directly
            logger.debug(f"JSON parse failed, using raw: {content[:100]}")
            return LLMCoachingTip(
                tip=content[:100] if content else "Keep going!",
                strength="",
                improve="",
                raw_response=content
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        avg_latency = self.total_latency_ms / max(1, self.tips_generated)
        return {
            "tips_generated": self.tips_generated,
            "average_latency_ms": round(avg_latency, 1),
            "model": self.model,
            "is_available": self.is_available
        }
    
    def reset(self):
        """Reset for new session."""
        self.last_tip_time = 0
        logger.info("LLMCoach reset")


# ============================================================================
# Simplified sync wrapper for non-async contexts
# ============================================================================

class LLMCoachSync:
    """Synchronous wrapper for LLMCoach."""
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.coach = LLMCoach(model=model)
        self._loop = None
    
    def _get_loop(self):
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop
    
    def check_availability(self) -> bool:
        loop = self._get_loop()
        return loop.run_until_complete(self.coach.check_availability())
    
    def generate_tip(self, signals: Dict[str, Any], transcript: str = "") -> Optional[LLMCoachingTip]:
        loop = self._get_loop()
        return loop.run_until_complete(
            self.coach.generate_tip(signals, transcript)
        )


# ============================================================================
# Quick test
# ============================================================================

async def test_llm_coach():
    """Test LLM coach with sample signals."""
    coach = LLMCoach(model="llama3.2")
    
    available = await coach.check_availability()
    print(f"Ollama available: {available}")
    
    if available:
        signals = {
            "audio_emotion": "neutral",
            "audio_arousal": 0.4,
            "face_emotion": "neutral",
            "face_expressiveness": 0.3,
            "eye_contact": 0.6,
            "presence": 0.7,
            "pace_wpm": 145,
            "pitch_variety": 0.35
        }
        
        tip = await coach.generate_tip(
            signals=signals,
            transcript="To be or not to be, that is the question.",
            context="Shakespeare monologue rehearsal"
        )
        
        if tip:
            print(f"\nTip: {tip.tip}")
            print(f"Strength: {tip.strength}")
            print(f"Improve: {tip.improve}")
            print(f"Emotion note: {tip.emotion_note}")
        else:
            print("No tip generated (cooldown or error)")


if __name__ == "__main__":
    asyncio.run(test_llm_coach())

