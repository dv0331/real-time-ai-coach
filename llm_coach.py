"""
LLM Coach - Multi-backend coaching tips for actors

Supports both:
1. LOCAL: Ollama (Llama 3.2) - Free, runs on your GPU
2. CLOUD: OpenAI API (GPT-4o-mini/GPT-4o) - Paid, works online

Usage:
    # Local (Ollama) - Default
    coach = LLMCoach(backend="ollama", model="llama3.2")
    
    # Cloud (OpenAI) - For online deployment
    coach = LLMCoach(backend="openai", model="gpt-4o-mini", api_key="sk-...")
    
    # Or set environment variable OPENAI_API_KEY
"""

import json
import logging
import time
import asyncio
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

# Configuration
OLLAMA_URL = "http://localhost:11434"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Default models
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"  # Fast and cheap, good for real-time


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
    Multi-backend LLM coach for actors.
    Supports both local (Ollama) and cloud (OpenAI) backends.
    """
    
    def __init__(self, 
                 backend: str = "ollama",  # "ollama" or "openai"
                 model: str = None,
                 api_key: str = None,
                 ollama_url: str = OLLAMA_URL,
                 timeout: float = 10.0):
        """
        Initialize LLM coach.
        
        Args:
            backend: "ollama" (local) or "openai" (cloud)
            model: Model name. Defaults based on backend.
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            ollama_url: Ollama server URL (for local backend)
            timeout: Request timeout in seconds
        """
        self.backend = backend.lower()
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.is_available = False
        self.last_tip_time = 0
        self.tip_cooldown = 5.0
        
        # Set model based on backend
        if model:
            self.model = model
        elif self.backend == "openai":
            self.model = DEFAULT_OPENAI_MODEL
        else:
            self.model = DEFAULT_OLLAMA_MODEL
        
        # Get API key from param or environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Validate OpenAI setup
        if self.backend == "openai" and not self.api_key:
            logger.warning("OpenAI backend selected but no API key provided!")
            logger.info("Set OPENAI_API_KEY environment variable or pass api_key parameter")
        
        # Track usage
        self.tips_generated = 0
        self.total_latency_ms = 0
        
        logger.info(f"LLMCoach initialized: backend={self.backend}, model={self.model}")
    
    async def check_availability(self) -> bool:
        """Check if the selected backend is available."""
        if self.backend == "openai":
            return await self._check_openai_availability()
        else:
            return await self._check_ollama_availability()
    
    async def _check_openai_availability(self) -> bool:
        """Check if OpenAI API is accessible."""
        if not self.api_key:
            logger.warning("OpenAI API key not set")
            self.is_available = False
            return False
        
        try:
            # Simple test request
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                self.is_available = response.status_code == 200
                if not self.is_available:
                    logger.warning(f"OpenAI API error: {response.status_code}")
                return self.is_available
        except Exception as e:
            logger.debug(f"OpenAI not available: {e}")
            self.is_available = False
            return False
    
    async def _check_ollama_availability(self) -> bool:
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
            await self.check_availability()
            if not self.is_available:
                return None
        
        try:
            start_time = time.time()
            
            # Build prompt
            messages = self._build_prompt(signals, transcript, context)
            
            # Call appropriate backend
            if self.backend == "openai":
                content = await self._call_openai(messages)
            else:
                content = await self._call_ollama(messages)
            
            if not content:
                return None
            
            # Parse response
            tip = self._parse_response(content)
            
            # Track metrics
            latency = (time.time() - start_time) * 1000
            self.total_latency_ms += latency
            self.tips_generated += 1
            self.last_tip_time = now
            
            logger.info(f"LLM tip generated in {latency:.0f}ms ({self.backend})")
            
            return tip
            
        except asyncio.TimeoutError:
            logger.warning("LLM request timed out")
            return None
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return None
    
    async def _call_openai(self, messages: List[Dict]) -> Optional[str]:
        """Call OpenAI API."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                OPENAI_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.4,
                    "max_tokens": 200,
                    "response_format": {"type": "json_object"}  # Force JSON output
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"OpenAI error: {response.status_code} - {response.text}")
                return None
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
    
    async def _call_ollama(self, messages: List[Dict]) -> Optional[str]:
        """Call Ollama API."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 200,
                        "top_p": 0.9
                    }
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Ollama error: {response.status_code}")
                return None
            
            result = response.json()
            return result.get("message", {}).get("content", "")
    
    def _build_prompt(self, signals: Dict[str, Any], transcript: str, context: str) -> List[Dict]:
        """Build chat prompt for LLM."""
        
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
            content = content.strip()
            
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            
            return LLMCoachingTip(
                tip=data.get("tip", "Keep going!"),
                strength=data.get("strength", "Good work"),
                improve=data.get("improve", "Stay focused"),
                emotion_note=data.get("emotion_note", ""),
                raw_response=content
            )
            
        except json.JSONDecodeError:
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
            "backend": self.backend,
            "model": self.model,
            "is_available": self.is_available
        }
    
    def reset(self):
        """Reset for new session."""
        self.last_tip_time = 0
        logger.info("LLMCoach reset")


# ============================================================================
# Vision-enabled coach (uses GPT-4 Vision for face analysis)
# ============================================================================

class VisionLLMCoach(LLMCoach):
    """
    Extended LLM coach that can analyze video frames using GPT-4 Vision.
    
    Use this for cloud deployment when you don't have local GPU for
    face emotion detection.
    """
    
    def __init__(self, 
                 api_key: str = None,
                 model: str = "gpt-4o",  # GPT-4o has vision
                 **kwargs):
        super().__init__(backend="openai", model=model, api_key=api_key, **kwargs)
        self.last_frame_time = 0
        self.frame_analysis_cooldown = 3.0  # Analyze frame every 3 seconds
    
    async def analyze_frame_with_signals(self,
                                          image_base64: str,
                                          signals: Dict[str, Any],
                                          transcript: str = "") -> Optional[LLMCoachingTip]:
        """
        Analyze a video frame along with performance signals.
        
        Args:
            image_base64: Base64 encoded JPEG image
            signals: Performance metrics
            transcript: Recent transcript
            
        Returns:
            LLMCoachingTip with visual analysis
        """
        now = time.time()
        
        # Check cooldowns
        if now - self.last_tip_time < self.tip_cooldown:
            return None
        if now - self.last_frame_time < self.frame_analysis_cooldown:
            return None
        
        if not self.api_key:
            logger.warning("OpenAI API key required for vision analysis")
            return None
        
        try:
            start_time = time.time()
            
            # Build vision prompt
            messages = self._build_vision_prompt(image_base64, signals, transcript)
            
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    OPENAI_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": 300,
                        "temperature": 0.4
                    }
                )
                
                if response.status_code != 200:
                    logger.warning(f"Vision API error: {response.status_code}")
                    return None
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
            
            tip = self._parse_response(content)
            
            latency = (time.time() - start_time) * 1000
            self.total_latency_ms += latency
            self.tips_generated += 1
            self.last_tip_time = now
            self.last_frame_time = now
            
            logger.info(f"Vision tip generated in {latency:.0f}ms")
            
            return tip
            
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return None
    
    def _build_vision_prompt(self, image_base64: str, signals: Dict[str, Any], transcript: str) -> List[Dict]:
        """Build vision-enabled prompt."""
        
        signal_summary = self._format_signals(signals)
        
        return [
            {
                "role": "system",
                "content": """You are an expert acting coach analyzing a performer's video feed.

Analyze the image for:
1. Facial expression and emotional authenticity
2. Eye contact with camera (are they looking at lens?)
3. Body language and presence
4. Framing and positioning

Combine visual analysis with the performance signals to give ONE actionable tip.

Output JSON:
{
  "tip": "Main direction (what to do NOW)",
  "strength": "Visual strength observed",
  "improve": "Visual area to work on",
  "emotion_note": "Emotional authenticity observation"
}"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "low"  # Use "low" for faster processing
                        }
                    },
                    {
                        "type": "text",
                        "text": f"""PERFORMANCE SIGNALS:
{signal_summary}

RECENT TRANSCRIPT:
"{transcript[-200:] if len(transcript) > 200 else transcript}"

Analyze the frame and signals. Give ONE brief coaching tip as JSON."""
                    }
                ]
            }
        ]


# ============================================================================
# Synchronous wrapper
# ============================================================================

class LLMCoachSync:
    """Synchronous wrapper for LLMCoach."""
    
    def __init__(self, backend: str = "ollama", model: str = None, api_key: str = None):
        self.coach = LLMCoach(backend=backend, model=model, api_key=api_key)
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
# Factory function
# ============================================================================

def create_llm_coach(use_openai: bool = False, api_key: str = None, model: str = None) -> LLMCoach:
    """
    Factory function to create appropriate LLM coach.
    
    Args:
        use_openai: If True, use OpenAI API. If False, use local Ollama.
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        model: Model override
        
    Returns:
        LLMCoach instance
    """
    if use_openai:
        return LLMCoach(
            backend="openai",
            model=model or DEFAULT_OPENAI_MODEL,
            api_key=api_key
        )
    else:
        return LLMCoach(
            backend="ollama", 
            model=model or DEFAULT_OLLAMA_MODEL
        )


# ============================================================================
# Quick test
# ============================================================================

async def test_llm_coach():
    """Test LLM coach with sample signals."""
    
    # Test with environment variable or default to Ollama
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key:
        print("Testing with OpenAI API...")
        coach = LLMCoach(backend="openai", model="gpt-4o-mini", api_key=api_key)
    else:
        print("Testing with local Ollama...")
        coach = LLMCoach(backend="ollama", model="llama3.2")
    
    available = await coach.check_availability()
    print(f"Backend available: {available}")
    
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
            print(f"\n[TIP] {tip.tip}")
            print(f"[STRENGTH] {tip.strength}")
            print(f"[IMPROVE] {tip.improve}")
            print(f"[EMOTION] {tip.emotion_note}")
        else:
            print("No tip generated (cooldown or error)")
    
    print(f"\nStats: {coach.get_stats()}")


if __name__ == "__main__":
    asyncio.run(test_llm_coach())
