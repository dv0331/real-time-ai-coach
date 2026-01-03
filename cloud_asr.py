"""
Cloud ASR Pipeline - OpenAI Whisper API integration

This module provides speech-to-text using OpenAI's Whisper API,
enabling the app to work online without a local GPU.

Usage:
    # Set environment variable
    export OPENAI_API_KEY="sk-..."
    
    # Or pass directly
    asr = CloudASRPipeline(api_key="sk-...")
"""

import asyncio
import logging
import time
import os
import io
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import httpx

logger = logging.getLogger(__name__)

OPENAI_WHISPER_URL = "https://api.openai.com/v1/audio/transcriptions"


@dataclass
class WordInfo:
    """Information about a transcribed word."""
    word: str
    start: float = 0.0
    end: float = 0.0
    confidence: float = 1.0


@dataclass
class TranscriptSegment:
    """A segment of transcribed text."""
    text: str
    start: float
    end: float
    words: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ASRResult:
    """Result from ASR processing."""
    full_text: str
    segments: List[TranscriptSegment]
    is_partial: bool = False
    language: str = "en"
    duration: float = 0.0


class CloudASRPipeline:
    """
    Cloud-based ASR using OpenAI's Whisper API.
    
    Pros:
    - No GPU required
    - Works online/in cloud deployment
    - High accuracy
    
    Cons:
    - Costs money per minute
    - Higher latency than local
    - Requires buffering audio
    """
    
    def __init__(self,
                 api_key: str = None,
                 model: str = "whisper-1",
                 language: str = "en",
                 buffer_duration: float = 3.0,  # Buffer N seconds before sending
                 sample_rate: int = 16000):
        """
        Initialize cloud ASR pipeline.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Whisper model ("whisper-1")
            language: Language code
            buffer_duration: Seconds of audio to buffer before transcribing
            sample_rate: Audio sample rate (16000 recommended)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.language = language
        self.buffer_duration = buffer_duration
        self.sample_rate = sample_rate
        
        # Audio buffer (raw PCM samples)
        self.audio_buffer: List[bytes] = []
        self.buffer_samples = 0
        self.min_samples = int(buffer_duration * sample_rate)
        
        # Transcription state
        self.full_transcript = ""
        self.segments: List[TranscriptSegment] = []
        self.last_process_time = 0
        self.processing = False
        
        # Word callback for compatibility
        self.word_callback = None
        
        # Stats
        self.total_audio_sec = 0
        self.total_api_calls = 0
        self.total_latency_ms = 0
        
        if not self.api_key:
            logger.warning("OpenAI API key not set for cloud ASR!")
        else:
            logger.info(f"CloudASR initialized: model={model}, buffer={buffer_duration}s")
    
    def add_audio(self, audio_bytes: bytes):
        """
        Add audio chunk to buffer.
        
        Args:
            audio_bytes: Raw PCM audio (16-bit, 16kHz, mono)
        """
        self.audio_buffer.append(audio_bytes)
        self.buffer_samples += len(audio_bytes) // 2  # 16-bit = 2 bytes per sample
    
    def process(self) -> Optional[ASRResult]:
        """
        Process buffered audio if ready.
        Returns ASRResult if transcription available, None otherwise.
        
        Note: For real-time use, call process_async() instead.
        """
        # Check if we have enough audio
        if self.buffer_samples < self.min_samples:
            return None
        
        # Don't process if already processing
        if self.processing:
            return None
        
        # Run async transcription
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.process_async())
            return result
        finally:
            loop.close()
    
    async def process_async(self) -> Optional[ASRResult]:
        """
        Process buffered audio asynchronously.
        Returns ASRResult if transcription available.
        """
        if self.buffer_samples < self.min_samples:
            return None
        
        if self.processing:
            return None
        
        if not self.api_key:
            logger.warning("Cannot process: OpenAI API key not set")
            return None
        
        self.processing = True
        
        try:
            start_time = time.time()
            
            # Combine audio chunks
            combined_audio = b"".join(self.audio_buffer)
            audio_duration = len(combined_audio) / 2 / self.sample_rate
            
            # Convert to WAV format for API
            wav_buffer = self._create_wav(combined_audio)
            
            # Call Whisper API
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {
                    "file": ("audio.wav", wav_buffer, "audio/wav"),
                }
                data = {
                    "model": self.model,
                    "language": self.language,
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "word"
                }
                
                response = await client.post(
                    OPENAI_WHISPER_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                    data=data
                )
                
                if response.status_code != 200:
                    logger.error(f"Whisper API error: {response.status_code} - {response.text}")
                    return None
                
                result = response.json()
            
            # Parse response
            text = result.get("text", "").strip()
            words = result.get("words", [])
            
            if text:
                # Create segment
                segment = TranscriptSegment(
                    text=text,
                    start=self.total_audio_sec,
                    end=self.total_audio_sec + audio_duration,
                    words=[{"word": w.get("word", ""), "start": w.get("start", 0), "end": w.get("end", 0)} 
                           for w in words]
                )
                self.segments.append(segment)
                self.full_transcript += " " + text
                self.full_transcript = self.full_transcript.strip()
                
                # Call word callback for each word
                if self.word_callback and words:
                    for word_info in words:
                        self.word_callback(word_info.get("word", ""), word_info.get("start", 0))
            
            # Update stats
            latency = (time.time() - start_time) * 1000
            self.total_latency_ms += latency
            self.total_audio_sec += audio_duration
            self.total_api_calls += 1
            
            # Clear buffer
            self.audio_buffer = []
            self.buffer_samples = 0
            self.last_process_time = time.time()
            
            logger.info(f"Cloud ASR: '{text[:50]}...' ({latency:.0f}ms)")
            
            return ASRResult(
                full_text=text,
                segments=[segment] if text else [],
                is_partial=False,
                language=self.language,
                duration=audio_duration
            )
            
        except Exception as e:
            logger.error(f"Cloud ASR error: {e}")
            return None
        finally:
            self.processing = False
    
    def _create_wav(self, pcm_data: bytes) -> io.BytesIO:
        """Create WAV file from raw PCM data."""
        import struct
        
        buffer = io.BytesIO()
        
        # WAV header
        num_samples = len(pcm_data) // 2
        byte_rate = self.sample_rate * 2  # 16-bit mono
        block_align = 2
        
        # RIFF header
        buffer.write(b'RIFF')
        buffer.write(struct.pack('<I', 36 + len(pcm_data)))
        buffer.write(b'WAVE')
        
        # fmt chunk
        buffer.write(b'fmt ')
        buffer.write(struct.pack('<I', 16))  # Chunk size
        buffer.write(struct.pack('<H', 1))   # Audio format (PCM)
        buffer.write(struct.pack('<H', 1))   # Num channels (mono)
        buffer.write(struct.pack('<I', self.sample_rate))  # Sample rate
        buffer.write(struct.pack('<I', byte_rate))  # Byte rate
        buffer.write(struct.pack('<H', block_align))  # Block align
        buffer.write(struct.pack('<H', 16))  # Bits per sample
        
        # data chunk
        buffer.write(b'data')
        buffer.write(struct.pack('<I', len(pcm_data)))
        buffer.write(pcm_data)
        
        buffer.seek(0)
        return buffer
    
    def force_process(self):
        """Force process any remaining audio."""
        if self.audio_buffer and self.buffer_samples > 0:
            # Temporarily lower threshold
            old_min = self.min_samples
            self.min_samples = 1
            self.process()
            self.min_samples = old_min
    
    def get_transcript(self) -> str:
        """Get full transcript."""
        return self.full_transcript
    
    def reset(self):
        """Reset for new session."""
        self.audio_buffer = []
        self.buffer_samples = 0
        self.full_transcript = ""
        self.segments = []
        self.processing = False
        logger.info("CloudASR reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ASR statistics."""
        avg_latency = self.total_latency_ms / max(1, self.total_api_calls)
        return {
            "backend": "openai_whisper",
            "model": self.model,
            "total_audio_sec": round(self.total_audio_sec, 1),
            "total_api_calls": self.total_api_calls,
            "average_latency_ms": round(avg_latency, 1),
            "estimated_cost_usd": round(self.total_audio_sec / 60 * 0.006, 4)  # $0.006/min
        }


# ============================================================================
# Factory function for choosing ASR backend
# ============================================================================

def create_asr_pipeline(use_cloud: bool = False, api_key: str = None):
    """
    Create appropriate ASR pipeline based on deployment mode.
    
    Args:
        use_cloud: If True, use OpenAI Whisper API. If False, use local faster-whisper.
        api_key: OpenAI API key (required for cloud)
        
    Returns:
        ASR pipeline instance
    """
    if use_cloud:
        return CloudASRPipeline(api_key=api_key)
    else:
        # Import local ASR
        from asr_pipeline import ASRPipeline
        return ASRPipeline()


# ============================================================================
# Test
# ============================================================================

async def test_cloud_asr():
    """Test cloud ASR with sample audio."""
    import numpy as np
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable to test")
        return
    
    asr = CloudASRPipeline(api_key=api_key, buffer_duration=2.0)
    
    # Generate 3 seconds of silence (for testing structure)
    duration = 3.0
    samples = int(16000 * duration)
    silence = np.zeros(samples, dtype=np.int16)
    
    # Add to buffer
    asr.add_audio(silence.tobytes())
    
    print(f"Buffer: {asr.buffer_samples} samples")
    print(f"Stats: {asr.get_stats()}")
    
    # Note: Won't return meaningful transcription from silence
    result = await asr.process_async()
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(test_cloud_asr())
