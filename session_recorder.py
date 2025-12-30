"""
Session Recorder: Saves session data (transcript, metrics, features) to disk.
Optional extension for reviewing past sessions.
"""

import os
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import logging
import asyncio

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

from config import config

logger = logging.getLogger(__name__)

@dataclass
class SessionData:
    """Complete session data for saving."""
    session_id: str
    start_time: str
    end_time: str
    duration_seconds: float
    
    # Transcript
    full_transcript: str = ""
    
    # Audio stats
    audio_stats: Dict[str, Any] = None
    
    # Vision stats
    vision_stats: Dict[str, Any] = None
    
    # Fusion stats
    fusion_stats: Dict[str, Any] = None
    
    # Coach stats
    coach_stats: Dict[str, Any] = None
    
    # Metric timeline (sampled)
    metric_timeline: List[Dict[str, Any]] = None

class SessionRecorder:
    """
    Records session data for later review.
    Saves JSON files with transcript, metrics, and timeline.
    """
    
    def __init__(self):
        self.enabled = config.session.ENABLE_RECORDING
        self.recording_dir = config.session.RECORDING_DIR
        
        self.session_id: Optional[str] = None
        self.start_time: Optional[float] = None
        
        # Metric timeline (sampled every second)
        self.metric_timeline: List[Dict[str, Any]] = []
        self.last_sample_time: float = 0
        self.sample_interval: float = 1.0  # seconds
        
        # Audio buffer for optional saving
        self.audio_chunks: List[bytes] = []
        
        # Ensure recording directory exists
        if self.enabled:
            os.makedirs(self.recording_dir, exist_ok=True)
            logger.info(f"SessionRecorder initialized: saving to {self.recording_dir}")
    
    def start_session(self) -> str:
        """Start a new recording session."""
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.metric_timeline = []
        self.last_sample_time = 0
        self.audio_chunks = []
        
        logger.info(f"Session started: {self.session_id}")
        return self.session_id
    
    def sample_metrics(self, metrics: Dict[str, Any]):
        """
        Sample current metrics for timeline.
        Called on every metric update, but only saves periodically.
        """
        if not self.enabled:
            return
        
        now = time.time()
        if now - self.last_sample_time >= self.sample_interval:
            self.last_sample_time = now
            
            # Add timestamp
            sample = {
                "timestamp": now - self.start_time if self.start_time else now,
                **metrics
            }
            self.metric_timeline.append(sample)
    
    def add_audio_chunk(self, chunk: bytes):
        """Add audio chunk for optional saving."""
        if self.enabled and config.session.SAVE_AUDIO:
            self.audio_chunks.append(chunk)
    
    async def save_session(
        self,
        transcript: str,
        audio_stats: Dict[str, Any],
        vision_stats: Dict[str, Any],
        fusion_stats: Dict[str, Any],
        coach_stats: Dict[str, Any]
    ) -> Optional[str]:
        """
        Save session data to disk.
        Returns the path to the saved file.
        """
        if not self.enabled or not self.session_id:
            return None
        
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        session_data = SessionData(
            session_id=self.session_id,
            start_time=datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else "",
            end_time=datetime.fromtimestamp(end_time).isoformat(),
            duration_seconds=duration,
            full_transcript=transcript,
            audio_stats=audio_stats,
            vision_stats=vision_stats,
            fusion_stats=fusion_stats,
            coach_stats=coach_stats,
            metric_timeline=self.metric_timeline
        )
        
        # Create session directory
        session_dir = os.path.join(self.recording_dir, self.session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save main session data
        session_path = os.path.join(session_dir, "session.json")
        
        try:
            data_dict = asdict(session_data)
            
            if HAS_AIOFILES:
                async with aiofiles.open(session_path, 'w') as f:
                    await f.write(json.dumps(data_dict, indent=2))
            else:
                with open(session_path, 'w') as f:
                    json.dump(data_dict, f, indent=2)
            
            logger.info(f"Session saved: {session_path}")
            
            # Save transcript separately if enabled
            if config.session.SAVE_TRANSCRIPT and transcript:
                transcript_path = os.path.join(session_dir, "transcript.txt")
                if HAS_AIOFILES:
                    async with aiofiles.open(transcript_path, 'w') as f:
                        await f.write(transcript)
                else:
                    with open(transcript_path, 'w') as f:
                        f.write(transcript)
            
            # Save audio if enabled and we have data
            if config.session.SAVE_AUDIO and self.audio_chunks:
                audio_path = os.path.join(session_dir, "audio.raw")
                audio_data = b''.join(self.audio_chunks)
                if HAS_AIOFILES:
                    async with aiofiles.open(audio_path, 'wb') as f:
                        await f.write(audio_data)
                else:
                    with open(audio_path, 'wb') as f:
                        f.write(audio_data)
                logger.info(f"Audio saved: {len(audio_data)} bytes")
            
            return session_path
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return None
    
    def reset(self):
        """Reset for new session."""
        self.session_id = None
        self.start_time = None
        self.metric_timeline = []
        self.last_sample_time = 0
        self.audio_chunks = []
        logger.info("SessionRecorder reset")

def generate_summary_report(session_path: str) -> str:
    """
    Generate a human-readable summary report from saved session data.
    """
    try:
        with open(session_path) as f:
            data = json.load(f)
        
        duration = data.get('duration_seconds', 0)
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        
        audio_stats = data.get('audio_stats', {})
        fusion_stats = data.get('fusion_stats', {})
        coach_stats = data.get('coach_stats', {})
        
        final_scores = fusion_stats.get('final_scores', {})
        
        report = f"""
=== SESSION SUMMARY ===
Session ID: {data.get('session_id', 'Unknown')}
Duration: {minutes}m {seconds}s
Start: {data.get('start_time', 'Unknown')}

=== SPEECH METRICS ===
Total Words: {audio_stats.get('total_words', 0)}
Filler Words: {audio_stats.get('filler_words', 0)} ({audio_stats.get('filler_ratio', 0)*100:.1f}%)
Average WPM: {audio_stats.get('average_wpm', 0):.0f}
Speech Ratio: {audio_stats.get('speech_ratio', 0)*100:.0f}%

=== FINAL SCORES (0-100) ===
Pace: {final_scores.get('pace', 0.5)*100:.0f}
Energy: {final_scores.get('energy', 0.5)*100:.0f}
Vocal Variety: {final_scores.get('pitch_variety', 0.5)*100:.0f}
Filler Words: {final_scores.get('filler_words', 1)*100:.0f}
Eye Contact: {final_scores.get('eye_contact', 0.5)*100:.0f}
Presence: {final_scores.get('presence', 0.5)*100:.0f}
Overall: {final_scores.get('overall', 0.5)*100:.0f}

=== COACHING FEEDBACK ===
Total Tips: {coach_stats.get('total_tips_given', 0)}
Positive Tips: {coach_stats.get('positive_tips', 0)}
Corrective Tips: {coach_stats.get('corrective_tips', 0)}

=== TRANSCRIPT ===
{data.get('full_transcript', '(No transcript available)')[:500]}...

=== END OF REPORT ===
"""
        return report
        
    except Exception as e:
        return f"Failed to generate report: {e}"

