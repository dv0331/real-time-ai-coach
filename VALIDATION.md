# Real-Time AI Coach - Validation & Performance Guide

## Performance Targets

### Latency Targets

| Component | Target | Acceptable | Notes |
|-----------|--------|------------|-------|
| Audio capture → backend | < 150ms | < 250ms | WebSocket + encoding |
| Audio → ASR result | < 1s | < 2s | Depends on model size |
| Video → vision features | < 100ms | < 200ms | @ 8 FPS |
| Feature → metrics | < 10ms | < 50ms | Pure computation |
| Metrics → UI update | < 20ms | < 50ms | WebSocket + render |
| **End-to-end** | **< 500ms** | **< 1s** | Capture to UI feedback |

### Throughput Targets

| Stream | Target Rate | Notes |
|--------|-------------|-------|
| Audio chunks | 10/sec | 100ms chunks, 16kHz |
| Video frames | 8/sec | 640x480 JPEG |
| Metrics updates | 10/sec | Matches audio rate |
| ASR updates | 1-2/sec | Batch processing |
| Coaching tips | 0.2-0.5/sec | Rate limited |

### Resource Usage (RTX 4060)

| Resource | Idle | Active Session | Max |
|----------|------|----------------|-----|
| GPU Memory | ~1GB | ~2-3GB | 4GB |
| GPU Utilization | 0% | 30-50% | 80% |
| CPU | 5% | 15-25% | 50% |
| RAM | 500MB | 1-2GB | 4GB |
| Network (local) | 0 | 500KB/s | 2MB/s |

---

## Validation Checklist

### Pre-Launch Checks

- [ ] Python 3.10+ installed
- [ ] CUDA toolkit installed (`nvcc --version` works)
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list | grep faster-whisper`)
- [ ] Server starts without errors
- [ ] No port conflicts (8000 available)

### Hardware Checks

- [ ] Webcam accessible
- [ ] Microphone accessible
- [ ] GPU detected: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Sufficient GPU memory free

### Functional Checks

#### Connection
- [ ] Browser loads http://localhost:8000
- [ ] WebSocket connects (no console errors)
- [ ] Status shows "Connected"

#### Media Capture
- [ ] Camera preview visible
- [ ] Video is mirrored (selfie view)
- [ ] No camera permission errors
- [ ] No microphone permission errors

#### Session Start
- [ ] "Start Session" button works
- [ ] Timer starts counting
- [ ] Status shows "Recording"
- [ ] Audio indicator activates when speaking

#### Audio Pipeline
- [ ] Transcript appears when speaking
- [ ] Filler words highlighted
- [ ] WPM updates realistically (100-200 range)
- [ ] Energy meter responds to voice

#### Vision Pipeline
- [ ] Face detected indicator works
- [ ] "No face" warning appears when looking away
- [ ] Eye contact indicator responds to gaze
- [ ] Presence score reflects framing

#### Metrics
- [ ] All 6 metric bars update
- [ ] Overall score ring animates
- [ ] Trend indicator updates
- [ ] No stuck/frozen metrics

#### Coaching
- [ ] Tips appear when metrics are low
- [ ] Tips don't flicker rapidly
- [ ] Different tip types appear
- [ ] Positive reinforcement tips appear

#### Session End
- [ ] "End Session" button works
- [ ] Summary modal appears
- [ ] Statistics are reasonable
- [ ] Session saved to `sessions/` folder

### UX "Wow Moments"

These should feel delightful:

- [ ] 🎯 **Instant feedback** - Tips appear within 1 second of issue
- [ ] 📊 **Smooth animations** - Metrics animate, don't jump
- [ ] 🎬 **Video quality** - Clear preview, good frame rate
- [ ] 💬 **Accurate transcript** - Most words correct
- [ ] 🔄 **Reliable detection** - Face/gaze tracking works consistently
- [ ] ⚡ **Responsive** - No lag or stuttering
- [ ] 📈 **Progress visibility** - Trend indicator shows improvement

---

## Performance Tuning

### If Latency is High (>1s)

1. **Reduce ASR model size**:
   ```python
   # config.py
   MODEL_SIZE: str = "tiny"  # fastest
   ```

2. **Increase audio chunk size**:
   ```python
   CHUNK_DURATION_MS: int = 200  # less overhead
   ```

3. **Reduce video FPS**:
   ```python
   TARGET_FPS: int = 5  # less GPU work
   ```

4. **Check GPU utilization**:
   ```bash
   nvidia-smi -l 1  # watch in real-time
   ```

### If GPU Memory is Full

1. **Use smaller model**:
   ```python
   MODEL_SIZE: str = "tiny"  # ~1GB instead of 2GB
   ```

2. **Switch to CPU for vision**:
   MediaPipe already uses CPU efficiently.

3. **Reduce batch sizes** (if implementing batching)

### If CPU is High

1. **Reduce video FPS**:
   ```python
   TARGET_FPS: int = 5
   ```

2. **Increase smoothing** (less frequent updates):
   ```python
   EMA_ALPHA: float = 0.2  # slower updates
   ```

### If Transcription is Inaccurate

1. **Use larger model**:
   ```python
   MODEL_SIZE: str = "small"  # or "medium"
   ```

2. **Ensure clear audio**:
   - Reduce background noise
   - Speak closer to microphone
   - Use external microphone if possible

3. **Check audio format**:
   - Must be 16kHz, 16-bit, mono

---

## Benchmarking

### Quick Latency Test

Add to `server.py` for debugging:

```python
import time

# In process_loop:
start = time.time()
# ... processing ...
print(f"Processing time: {(time.time() - start) * 1000:.1f}ms")
```

### ASR Speed Test

```python
from faster_whisper import WhisperModel
import numpy as np
import time

model = WhisperModel("base", device="cuda")
audio = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds

start = time.time()
segments, _ = model.transcribe(audio)
list(segments)  # Force evaluation
print(f"ASR time for 5s audio: {time.time() - start:.2f}s")
```

Expected results (RTX 4060):
- tiny: ~0.3s
- base: ~0.5s  
- small: ~1.0s
- medium: ~2.5s

### Vision Speed Test

```python
import mediapipe as mp
import numpy as np
import time
import cv2

face_mesh = mp.solutions.face_mesh.FaceMesh()
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

times = []
for _ in range(100):
    start = time.time()
    face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    times.append(time.time() - start)

print(f"Avg vision time: {np.mean(times) * 1000:.1f}ms")
print(f"Max FPS: {1 / np.mean(times):.0f}")
```

Expected: ~10-15ms per frame (60+ FPS possible)

---

## Common Issues & Solutions

### "Transcript is delayed"

**Cause**: ASR batching accumulates too much audio.

**Fix**:
```python
# config.py
MIN_AUDIO_LENGTH_SEC: float = 0.3  # Process sooner
```

### "Tips keep appearing and disappearing"

**Cause**: Score oscillation around threshold.

**Fix**:
```python
# config.py
SCORE_HYSTERESIS: float = 0.15  # Wider band
TIP_COOLDOWN_SEC: float = 5.0   # Longer cooldown
```

### "Face detection is unreliable"

**Cause**: Poor lighting or camera angle.

**Fix**:
- Improve lighting (face should be well-lit)
- Position camera at eye level
- Reduce detection confidence:
  ```python
  MIN_DETECTION_CONFIDENCE: float = 0.3
  ```

### "Eye contact score always low"

**Cause**: Gaze estimation is sensitive to camera position.

**Fix**:
```python
# config.py
GAZE_CENTER_TOLERANCE: float = 0.25  # More forgiving
```

### "No audio being captured"

**Cause**: Browser audio context suspended.

**Fix**: The frontend already handles this, but ensure:
- User clicks "Start Session" (user gesture required)
- Microphone permission granted
- No other app using microphone

---

## Session Recording Validation

After a session, check `sessions/<session_id>/`:

- [ ] `session.json` exists and is valid JSON
- [ ] `transcript.txt` contains readable text
- [ ] Metric timeline has data points
- [ ] Timestamps are reasonable

Sample validation script:

```python
import json
from pathlib import Path

sessions_dir = Path("sessions")
for session_dir in sessions_dir.iterdir():
    session_file = session_dir / "session.json"
    if session_file.exists():
        with open(session_file) as f:
            data = json.load(f)
        print(f"Session {data['session_id']}:")
        print(f"  Duration: {data['duration_seconds']:.1f}s")
        print(f"  Words: {data['audio_stats']['total_words']}")
        print(f"  Timeline points: {len(data['metric_timeline'])}")
```

---

## Success Criteria

Your PoC is working correctly if:

1. ✅ **< 500ms latency** from speech to transcript
2. ✅ **Accurate transcription** (>90% words correct)
3. ✅ **Reliable face tracking** (works when looking at camera)
4. ✅ **Relevant tips** (tips match actual issues)
5. ✅ **Stable metrics** (no rapid flickering)
6. ✅ **Useful summary** (actionable session report)
7. ✅ **< 4GB GPU memory** (runs on RTX 4060)
8. ✅ **No crashes** during 5+ minute sessions

---

## Next Level: Optional Extensions

Once the core works, consider adding:

### 1. Emotion Detection (Audio)

Use a prosody model for emotional analysis:
```python
# Add to requirements.txt
transformers
torch

# In audio_pipeline.py, add emotion inference
from transformers import pipeline
emotion_classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")
```

### 2. Emotion Detection (Video)

Add facial expression analysis:
```python
# Use MediaPipe's face mesh to detect:
# - Mouth corners (smile detection)
# - Eyebrow position (surprise/concern)
# - Eye openness (engagement)
```

### 3. Advanced LLM Coaching

Enable contextual feedback with conversation history:
```python
# Keep last N tips and adjust LLM prompt
# Include what the user is actually saying
# Provide content-aware feedback
```

### 4. Session Comparison

Build a simple dashboard to compare sessions over time:
```python
# Load multiple session.json files
# Plot score trends
# Show improvement areas
```

