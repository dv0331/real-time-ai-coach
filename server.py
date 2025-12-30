"""
Real-Time AI Coach - FastAPI Backend Server

Main WebSocket server that coordinates all pipelines:
- Receives audio/video chunks from browser
- Processes through audio, vision, and ASR pipelines
- Fuses features and generates coaching tips
- Sends metrics and tips back to client

Run with: python server.py
"""

import asyncio
import json
import logging
import time
import base64
from typing import Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import config
from audio_pipeline import AudioPipeline, AudioFeatures
from vision_pipeline import VisionPipeline, VisionFeatures
from asr_pipeline import ASRPipeline
from fusion_engine import FusionEngine, FusedMetrics
from coach_engine import CoachEngine, CoachingTip
from session_recorder import SessionRecorder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# JSON CONTRACTS
# ============================================================================
"""
CLIENT → SERVER MESSAGE FORMAT:
{
    "type": "audio" | "video" | "control",
    "data": "<base64 encoded data>",     // For audio/video
    "action": "start" | "stop" | "reset", // For control
    "timestamp": 1234567890.123
}

Audio chunk: 16-bit PCM, 16kHz, mono, ~100ms per chunk
Video frame: JPEG encoded, 640x480, sent at 5-10 FPS

SERVER → CLIENT MESSAGE FORMAT:
{
    "type": "metrics" | "transcript" | "tip" | "status" | "summary",
    "data": { ... },
    "timestamp": 1234567890.123
}

METRICS UPDATE:
{
    "type": "metrics",
    "data": {
        "scores": {
            "overall": 0.75,
            "pace": 0.8,
            "energy": 0.7,
            "pitch_variety": 0.6,
            "filler_words": 0.9,
            "eye_contact": 0.8,
            "presence": 0.7,
            "stability": 0.75
        },
        "raw": {
            "wpm": 135,
            "energy_db": -25.5,
            "filler_ratio": 0.02
        },
        "flags": {
            "is_speaking": true,
            "face_detected": true,
            "looking_at_camera": true
        },
        "trend": 1  // -1=declining, 0=stable, 1=improving
    }
}

TRANSCRIPT UPDATE:
{
    "type": "transcript",
    "data": {
        "text": "Hello everyone, today I want to...",
        "is_partial": false,
        "words": ["Hello", "everyone", "today", "I", "want", "to"]
    }
}

COACHING TIP:
{
    "type": "tip",
    "data": {
        "tips": [
            {
                "id": "eye_contact_low",
                "category": "eye_contact",
                "message": "Look at the camera more",
                "severity": "warning"
            }
        ]
    }
}

END-OF-SESSION SUMMARY:
{
    "type": "summary",
    "data": {
        "session_id": "20241230_143022",
        "duration_seconds": 180.5,
        "final_scores": { ... },
        "audio_stats": {
            "total_words": 450,
            "filler_words": 12,
            "average_wpm": 142
        },
        "vision_stats": { ... },
        "coach_stats": {
            "total_tips": 15,
            "positive_tips": 8,
            "corrective_tips": 7
        },
        "transcript": "Full session transcript..."
    }
}
"""

# ============================================================================
# GLOBAL PIPELINES (loaded once at startup)
# ============================================================================

# These are loaded once to avoid slow initialization on each connection
_global_asr: Optional[ASRPipeline] = None
_global_vision: Optional[VisionPipeline] = None
_models_loaded = False

def load_models():
    """Pre-load heavy models at startup."""
    global _global_asr, _global_vision, _models_loaded
    if _models_loaded:
        return
    
    logger.info("Loading AI models (this may take a moment)...")
    
    try:
        _global_vision = VisionPipeline()
        logger.info("Vision pipeline loaded")
    except Exception as e:
        logger.warning(f"Vision pipeline failed: {e}")
        _global_vision = None
    
    try:
        _global_asr = ASRPipeline()
        logger.info("ASR pipeline loaded")
    except Exception as e:
        logger.warning(f"ASR pipeline failed: {e}")
        _global_asr = None
    
    _models_loaded = True
    logger.info("Model loading complete!")

# ============================================================================
# SESSION STATE
# ============================================================================

class SessionState:
    """Holds all pipeline instances for a session."""
    
    def __init__(self):
        self.audio = AudioPipeline()
        self.vision = _global_vision
        self.asr = _global_asr
        if self.asr:
            self.asr.word_callback = self._on_word
        self.fusion = FusionEngine()
        self.coach = CoachEngine()
        self.recorder = SessionRecorder()
        
        self.active = False
        self.start_time: float = 0
        self.last_metrics_time: float = 0
        self.metrics_interval: float = 0.1  # Send metrics every 100ms
        
        # Latest features
        self.latest_audio: Optional[AudioFeatures] = None
        self.latest_vision: Optional[VisionFeatures] = None
        self.latest_transcript: str = ""
        
    def _on_word(self, word: str, timestamp: float):
        """Callback when ASR detects a word."""
        self.audio.add_word(word, timestamp)
    
    def start(self):
        """Start a new session."""
        self.audio.reset()
        if self.vision:
            self.vision.reset()
        if self.asr:
            self.asr.reset()
        self.fusion.reset()
        self.coach.reset()
        self.recorder.start_session()
        
        self.active = True
        self.start_time = time.time()
        logger.info("Session started")
    
    def stop(self):
        """Stop current session."""
        self.active = False
        logger.info("Session stopped")
    
    def get_duration(self) -> float:
        """Get session duration in seconds."""
        if not self.active:
            return 0
        return time.time() - self.start_time

# Store sessions by WebSocket connection
sessions: Dict[WebSocket, SessionState] = {}

# ============================================================================
# FASTAPI APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup and shutdown."""
    logger.info("=" * 60)
    logger.info("Real-Time AI Coach Server Starting...")
    logger.info("=" * 60)
    
    # Pre-load heavy models at startup
    load_models()
    
    logger.info(f"Server: http://{config.server.HOST}:{config.server.PORT}")
    logger.info(f"Open http://localhost:{config.server.PORT} in your browser")
    logger.info("=" * 60)
    
    yield
    
    logger.info("Server shutting down...")

app = FastAPI(
    title="Real-Time AI Coach",
    description="Real-time presentation coaching with audio and vision analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse("frontend/index.html")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

# ============================================================================
# WEBSOCKET HANDLER
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time communication.
    Handles audio/video streaming and sends back metrics.
    """
    await websocket.accept()
    logger.info(f"Client connected: {websocket.client}")
    
    # Create session state
    session = SessionState()
    sessions[websocket] = session
    
    # Send initial status
    await send_status(websocket, "connected", "Ready to start")
    
    try:
        # Start background task for processing
        process_task = asyncio.create_task(process_loop(websocket, session))
        
        # Main receive loop
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await handle_message(websocket, session, message)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON: {e}")
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup
        process_task.cancel()
        if websocket in sessions:
            del sessions[websocket]

async def handle_message(websocket: WebSocket, session: SessionState, message: dict):
    """Handle incoming WebSocket message."""
    msg_type = message.get("type", "")
    
    if msg_type == "control":
        await handle_control(websocket, session, message)
        
    elif msg_type == "audio":
        if session.active:
            # Decode and process audio
            audio_b64 = message.get("data", "")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                session.latest_audio = session.audio.process_chunk(audio_bytes)
                if session.asr:
                    session.asr.add_audio(audio_bytes)
                session.recorder.add_audio_chunk(audio_bytes)
                
    elif msg_type == "video":
        if session.active and session.vision:
            # Decode and process video
            video_b64 = message.get("data", "")
            if video_b64:
                video_bytes = base64.b64decode(video_b64)
                vision_result = session.vision.process_frame(video_bytes)
                if vision_result:
                    session.latest_vision = vision_result

async def handle_control(websocket: WebSocket, session: SessionState, message: dict):
    """Handle control messages (start/stop/reset)."""
    action = message.get("action", "")
    
    if action == "start":
        session.start()
        await send_status(websocket, "started", "Session started")
        
    elif action == "stop":
        session.stop()
        
        # Generate and send summary
        summary = await generate_summary(session)
        await websocket.send_text(json.dumps({
            "type": "summary",
            "data": summary,
            "timestamp": time.time()
        }))
        
        await send_status(websocket, "stopped", "Session ended")
        
    elif action == "reset":
        session.start()  # Reset is just a new start
        await send_status(websocket, "reset", "Session reset")

async def process_loop(websocket: WebSocket, session: SessionState):
    """
    Background processing loop.
    Runs ASR, fusion, and coaching, then sends updates to client.
    """
    while True:
        try:
            if session.active:
                now = time.time()
                
                # Process ASR
                if session.asr:
                    asr_result = session.asr.process()
                    if asr_result and asr_result.full_text:
                        session.latest_transcript = session.asr.get_transcript()
                        await send_transcript(websocket, asr_result)
                
                # Update fusion and generate tips periodically
                if now - session.last_metrics_time >= session.metrics_interval:
                    session.last_metrics_time = now
                    
                    # Fuse features
                    metrics = session.fusion.update(
                        audio_features=session.latest_audio,
                        vision_features=session.latest_vision,
                        filler_count=session.audio.filler_count,
                        word_count=session.audio.total_words
                    )
                    
                    # Record metrics
                    session.recorder.sample_metrics({
                        "overall": metrics.overall_score,
                        "pace": metrics.pace_score,
                        "energy": metrics.energy_score,
                        "eye_contact": metrics.eye_contact_score
                    })
                    
                    # Send metrics update
                    await send_metrics(websocket, metrics)
                    
                    # Generate coaching tips (less frequently)
                    tips = session.coach.generate_tips(
                        metrics,
                        session.latest_transcript
                    )
                    if tips:
                        await send_tips(websocket, tips)
                    
                    # Try LLM tip if enabled
                    if config.coach.USE_LLM:
                        llm_tip = await session.coach.get_llm_tip(
                            metrics,
                            session.latest_transcript
                        )
                        if llm_tip:
                            await websocket.send_text(json.dumps({
                                "type": "llm_tip",
                                "data": {"message": llm_tip},
                                "timestamp": time.time()
                            }))
            
            # Small sleep to prevent busy loop
            await asyncio.sleep(0.05)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Processing error: {e}")
            await asyncio.sleep(0.1)

# ============================================================================
# MESSAGE SENDERS
# ============================================================================

async def send_status(websocket: WebSocket, status: str, message: str):
    """Send status update to client."""
    await websocket.send_text(json.dumps({
        "type": "status",
        "data": {"status": status, "message": message},
        "timestamp": time.time()
    }))

async def send_metrics(websocket: WebSocket, metrics: FusedMetrics):
    """Send metrics update to client."""
    await websocket.send_text(json.dumps({
        "type": "metrics",
        "data": {
            "scores": {
                "overall": round(metrics.overall_score, 3),
                "pace": round(metrics.pace_score, 3),
                "energy": round(metrics.energy_score, 3),
                "pitch_variety": round(metrics.pitch_variety_score, 3),
                "filler_words": round(metrics.filler_word_score, 3),
                "eye_contact": round(metrics.eye_contact_score, 3),
                "presence": round(metrics.presence_score, 3),
                "stability": round(metrics.stability_score, 3)
            },
            "raw": {
                "wpm": round(metrics.wpm, 1),
                "energy_db": round(metrics.energy_db, 1),
                "filler_ratio": round(metrics.filler_ratio, 3)
            },
            "flags": {
                "is_speaking": metrics.is_speaking,
                "face_detected": metrics.face_detected,
                "looking_at_camera": metrics.looking_at_camera
            },
            "trend": metrics.overall_trend
        },
        "timestamp": time.time()
    }))

async def send_transcript(websocket: WebSocket, asr_result):
    """Send transcript update to client."""
    words = []
    for segment in asr_result.segments:
        for word_info in segment.words:
            words.append(word_info.get('word', ''))
    
    await websocket.send_text(json.dumps({
        "type": "transcript",
        "data": {
            "text": asr_result.full_text,
            "is_partial": asr_result.is_partial,
            "words": words
        },
        "timestamp": time.time()
    }))

async def send_tips(websocket: WebSocket, tips: list):
    """Send coaching tips to client."""
    tip_data = []
    for tip in tips:
        tip_data.append({
            "id": tip.id,
            "category": tip.category,
            "message": tip.message,
            "severity": tip.severity
        })
    
    await websocket.send_text(json.dumps({
        "type": "tip",
        "data": {"tips": tip_data},
        "timestamp": time.time()
    }))

async def generate_summary(session: SessionState) -> dict:
    """Generate end-of-session summary."""
    # Force process any remaining audio
    transcript = ""
    if session.asr:
        session.asr.force_process()
        transcript = session.asr.get_transcript()
    
    # Gather all stats
    audio_stats = session.audio.get_session_stats()
    vision_stats = session.vision.get_session_stats() if session.vision else {}
    fusion_stats = session.fusion.get_session_stats()
    coach_stats = session.coach.get_session_stats()
    
    # Save session
    await session.recorder.save_session(
        transcript=transcript,
        audio_stats=audio_stats,
        vision_stats=vision_stats,
        fusion_stats=fusion_stats,
        coach_stats=coach_stats
    )
    
    return {
        "session_id": session.recorder.session_id,
        "duration_seconds": session.get_duration(),
        "final_scores": fusion_stats.get('final_scores', {}),
        "audio_stats": audio_stats,
        "vision_stats": vision_stats,
        "coach_stats": coach_stats,
        "transcript": transcript
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host=config.server.HOST,
        port=config.server.PORT,
        reload=config.server.DEBUG,
        log_level="info"
    )

