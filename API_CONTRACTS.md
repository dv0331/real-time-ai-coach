# Real-Time AI Coach - API Contracts

Complete JSON schema documentation for WebSocket communication.

---

## WebSocket Endpoint

```
ws://localhost:8000/ws
```

---

## Client → Server Messages

### 1. Control Message

Start, stop, or reset a session.

```json
{
    "type": "control",
    "action": "start" | "stop" | "reset",
    "timestamp": 1735570000.123
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"control"` |
| `action` | string | `"start"`, `"stop"`, or `"reset"` |
| `timestamp` | float | Unix timestamp (seconds) |

---

### 2. Audio Chunk

Send audio data for processing.

```json
{
    "type": "audio",
    "data": "base64_encoded_pcm_audio...",
    "timestamp": 1735570000.123
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"audio"` |
| `data` | string | Base64-encoded PCM audio (16-bit, 16kHz, mono) |
| `timestamp` | float | Unix timestamp |

**Audio Format:**
- Sample rate: 16000 Hz
- Bit depth: 16-bit signed integer
- Channels: 1 (mono)
- Chunk duration: ~100ms (1600 samples)
- Encoding: Base64

---

### 3. Video Frame

Send video frame for processing.

```json
{
    "type": "video",
    "data": "base64_encoded_jpeg...",
    "timestamp": 1735570000.123
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"video"` |
| `data` | string | Base64-encoded JPEG image |
| `timestamp` | float | Unix timestamp |

**Video Format:**
- Resolution: 640x480 (default)
- Format: JPEG
- Quality: 70%
- Frame rate: 5-10 FPS

---

## Server → Client Messages

### 1. Status Message

Session state updates.

```json
{
    "type": "status",
    "data": {
        "status": "connected" | "started" | "stopped" | "reset",
        "message": "Human-readable status message"
    },
    "timestamp": 1735570000.123
}
```

---

### 2. Metrics Update

Real-time performance metrics (sent every ~100ms).

```json
{
    "type": "metrics",
    "data": {
        "scores": {
            "overall": 0.75,
            "pace": 0.80,
            "energy": 0.70,
            "pitch_variety": 0.65,
            "filler_words": 0.90,
            "eye_contact": 0.85,
            "presence": 0.75,
            "stability": 0.80
        },
        "raw": {
            "wpm": 142.5,
            "energy_db": -25.3,
            "filler_ratio": 0.02
        },
        "flags": {
            "is_speaking": true,
            "face_detected": true,
            "looking_at_camera": true
        },
        "trend": 1
    },
    "timestamp": 1735570000.123
}
```

#### Scores Object

All scores are normalized to 0.0 - 1.0 (higher is better).

| Field | Description | Good Range |
|-------|-------------|------------|
| `overall` | Weighted average of all scores | > 0.7 |
| `pace` | Speaking pace (WPM) score | > 0.7 |
| `energy` | Voice energy/projection | > 0.6 |
| `pitch_variety` | Vocal variety (1.0 = expressive, low = monotone) | > 0.5 |
| `filler_words` | Filler word avoidance (1.0 = no fillers) | > 0.8 |
| `eye_contact` | Gaze toward camera | > 0.7 |
| `presence` | Face framing and visibility | > 0.6 |
| `stability` | Head motion stability (1.0 = still) | > 0.5 |

#### Raw Object

Unprocessed metric values for display.

| Field | Type | Description |
|-------|------|-------------|
| `wpm` | float | Words per minute |
| `energy_db` | float | Voice energy in decibels |
| `filler_ratio` | float | Filler words / total words |

#### Flags Object

Boolean state indicators.

| Field | Type | Description |
|-------|------|-------------|
| `is_speaking` | bool | Voice activity detected |
| `face_detected` | bool | Face visible in frame |
| `looking_at_camera` | bool | Gaze toward camera |

#### Trend

| Value | Meaning |
|-------|---------|
| `-1` | Declining (performance dropping) |
| `0` | Stable |
| `1` | Improving |

---

### 3. Transcript Update

Incremental speech-to-text results.

```json
{
    "type": "transcript",
    "data": {
        "text": "Hello everyone, welcome to my presentation",
        "is_partial": false,
        "words": ["Hello", "everyone", "welcome", "to", "my", "presentation"]
    },
    "timestamp": 1735570000.123
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Transcribed text segment |
| `is_partial` | bool | True if still processing (Vosk only) |
| `words` | array | Individual words from this segment |

---

### 4. Coaching Tip

Real-time coaching feedback.

```json
{
    "type": "tip",
    "data": {
        "tips": [
            {
                "id": "eye_contact_low",
                "category": "eye_contact",
                "message": "Look at the camera more - imagine your audience there",
                "severity": "warning"
            },
            {
                "id": "pace_good",
                "category": "pace", 
                "message": "Great pacing! Keep it up",
                "severity": "info"
            }
        ]
    },
    "timestamp": 1735570000.123
}
```

#### Tip Object

| Field | Type | Values |
|-------|------|--------|
| `id` | string | Unique tip identifier |
| `category` | string | `pace`, `energy`, `pitch`, `fillers`, `eye_contact`, `presence`, `stability`, `overall` |
| `message` | string | Human-readable tip |
| `severity` | string | `info`, `warning`, `critical` |

#### Tip IDs

| ID | Category | Condition |
|----|----------|-----------|
| `pace_slow` | pace | WPM < 120 |
| `pace_fast` | pace | WPM > 160 |
| `pace_good` | pace | WPM 120-160 |
| `energy_low` | energy | energy_score < 0.4 |
| `energy_good` | energy | energy_score > 0.8 |
| `monotone` | pitch | pitch_variety_score < 0.4 |
| `expressive` | pitch | pitch_variety_score > 0.7 |
| `filler_words` | fillers | filler_ratio > 5% |
| `filler_good` | fillers | filler_word_score > 0.9 |
| `eye_contact_low` | eye_contact | eye_contact_score < 0.4 |
| `eye_contact_good` | eye_contact | eye_contact_score > 0.8 |
| `presence_low` | presence | presence_score < 0.5 |
| `no_face` | presence | face not detected |
| `too_fidgety` | stability | stability_score < 0.4 |
| `doing_great` | overall | overall_score > 0.8 |
| `improving` | overall | trend = 1 |

---

### 5. LLM Tip (Optional)

Enhanced tip from local LLM (when enabled).

```json
{
    "type": "llm_tip",
    "data": {
        "message": "Try emphasizing key points by pausing briefly before important words"
    },
    "timestamp": 1735570000.123
}
```

---

### 6. End-of-Session Summary

Complete session statistics (sent when session stops).

```json
{
    "type": "summary",
    "data": {
        "session_id": "20241230_143022",
        "duration_seconds": 180.5,
        "final_scores": {
            "overall": 0.72,
            "pace": 0.78,
            "energy": 0.68,
            "pitch_variety": 0.62,
            "filler_words": 0.88,
            "eye_contact": 0.75,
            "presence": 0.70,
            "stability": 0.80
        },
        "audio_stats": {
            "duration_seconds": 180.5,
            "total_words": 450,
            "filler_words": 12,
            "filler_ratio": 0.027,
            "average_wpm": 142.3,
            "speech_ratio": 0.75,
            "average_energy_db": -28.5,
            "pitch_variability": 35.2
        },
        "vision_stats": {
            "frames_processed": 1440,
            "average_gaze_x": 0.48,
            "average_gaze_y": 0.42
        },
        "coach_stats": {
            "total_tips_given": 24,
            "positive_tips": 12,
            "corrective_tips": 12,
            "tip_breakdown": {
                "pace": 4,
                "energy": 3,
                "eye_contact": 8,
                "fillers": 5,
                "overall": 4
            }
        },
        "transcript": "Hello everyone, welcome to my presentation about..."
    },
    "timestamp": 1735570000.123
}
```

---

## Message Flow Example

```
Client                          Server
  |                                |
  |---[connect]------------------>|
  |<--[status: connected]---------|
  |                                |
  |---[control: start]----------->|
  |<--[status: started]-----------|
  |                                |
  |---[audio chunk]--------------->|
  |---[video frame]--------------->|
  |<--[metrics]-------------------|
  |<--[transcript]----------------|
  |                                |
  |---[audio chunk]--------------->|
  |---[video frame]--------------->|
  |<--[metrics]-------------------|
  |<--[tip]------------------------|
  |                                |
  |  ... (continues) ...           |
  |                                |
  |---[control: stop]------------->|
  |<--[summary]-------------------|
  |<--[status: stopped]-----------|
  |                                |
```

---

## Error Handling

The server may send error status messages:

```json
{
    "type": "status",
    "data": {
        "status": "error",
        "message": "Failed to process audio: invalid format"
    },
    "timestamp": 1735570000.123
}
```

Clients should handle WebSocket close events and attempt reconnection.

---

## Rate Limits

| Stream | Rate | Notes |
|--------|------|-------|
| Audio | ~10 chunks/sec | 100ms chunks |
| Video | 5-10 frames/sec | Configurable |
| Metrics | ~10/sec | Server sends on each audio chunk |
| Tips | 2-5/sec max | Cooldown prevents spam |
| Transcript | As needed | Sent when ASR produces output |

