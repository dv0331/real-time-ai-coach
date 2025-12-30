# 🎭 Actor's AI Coach - Real-Time Rehearsal Assistant

> **"Practice makes perfect. AI makes practice smarter."**

A fully local, real-time AI coaching system designed specifically for **actors** to rehearse scenes, practice monologues, and refine their craft with instant feedback on delivery, emotion, presence, and timing.

## 🎬 The Problem We Solve

Actors need to rehearse constantly, but:
- **No immediate feedback** when practicing alone
- **Can't see themselves** objectively during performance
- **Coaches are expensive** and not always available
- **Self-recording** requires reviewing footage after the fact

**Actor's AI Coach** gives you a **live co-pilot** that watches your performance and provides real-time guidance - like having a director in your pocket.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ACTOR'S AI COACH                                  │
│                        Real-Time Rehearsal System                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐         WebSocket          ┌──────────────────────────────┐
│                  │  ───────────────────────►  │                              │
│   WEB BROWSER    │    Audio: 16kHz PCM        │      PYTHON BACKEND          │
│                  │    Video: JPEG @ 8 FPS     │      (FastAPI + GPU)         │
│  ┌────────────┐  │                            │                              │
│  │  Webcam    │  │  ◄───────────────────────  │  ┌────────────────────────┐  │
│  │  + Mic     │  │    Metrics + Tips JSON     │  │   AUDIO PIPELINE       │  │
│  └────────────┘  │                            │  │   • VAD (speech detect)│  │
│                  │                            │  │   • Energy/Volume      │  │
│  ┌────────────┐  │                            │  │   • Pitch Variability  │  │
│  │  Live      │  │                            │  │   • Pace (WPM)         │  │
│  │  Metrics   │  │                            │  │   • Pause Detection    │  │
│  │  + Tips    │  │                            │  └────────────────────────┘  │
│  └────────────┘  │                            │                              │
│                  │                            │  ┌────────────────────────┐  │
│  ┌────────────┐  │                            │  │   VISION PIPELINE      │  │
│  │  Script    │  │                            │  │   • Face Detection     │  │
│  │  Mode      │  │                            │  │   • Eye Contact        │  │
│  │  (Lines)   │  │                            │  │   • Head Pose          │  │
│  └────────────┘  │                            │  │   • Expression Proxy   │  │
│                  │                            │  │   • Presence Score     │  │
└──────────────────┘                            │  └────────────────────────┘  │
                                                │                              │
                                                │  ┌────────────────────────┐  │
                                                │  │   ASR (faster-whisper) │  │
                                                │  │   • Live Transcription │  │
                                                │  │   • Word Timestamps    │  │
                                                │  │   • Filler Detection   │  │
                                                │  └────────────────────────┘  │
                                                │                              │
                                                │  ┌────────────────────────┐  │
                                                │  │   ACTOR COACH ENGINE   │  │
                                                │  │   • Delivery Analysis  │  │
                                                │  │   • Emotional Range    │  │
                                                │  │   • Timing/Pacing      │  │
                                                │  │   • Presence/Command   │  │
                                                │  │   • Script Comparison  │  │
                                                │  └────────────────────────┘  │
                                                │                              │
                                                │  ┌────────────────────────┐  │
                                                │  │   OPTIONAL: Local LLM  │  │
                                                │  │   (Ollama/llama.cpp)   │  │
                                                │  │   For richer feedback  │  │
                                                │  └────────────────────────┘  │
                                                └──────────────────────────────┘
```

---

## 🎯 Actor-Specific Features

### 1. **Delivery Analysis**
- **Pace Control**: Too fast? Too slow? Get real-time WPM feedback
- **Energy Levels**: Are you projecting enough? Too quiet for the scene?
- **Vocal Variety**: Avoid monotone delivery - track pitch variation
- **Pauses**: Master dramatic beats and comedic timing

### 2. **Emotional Expression**
- **Emotional Range**: Track how varied your emotional delivery is
- **Intensity Matching**: Is your energy matching the scene's requirements?
- **Consistency**: Maintain character throughout the scene

### 3. **Physical Presence**
- **Eye Contact**: Critical for auditions - are you connecting with the camera/audience?
- **Head Movement**: Natural movement vs. being too static or too erratic
- **Framing**: Stay in frame, maintain good camera presence
- **Stillness vs. Motion**: Some scenes need stillness, others need energy

### 4. **Script Mode** (Coming Soon)
- Load your script/sides
- Practice with highlighted current line
- Track accuracy of line delivery
- Get feedback on interpretation

### 5. **Session Recording**
- Save your practice sessions
- Review transcript with timestamps
- Track improvement over time

---

## 🚀 Quick Start

### Prerequisites
- **Hardware**: i9 + RTX 4060 (or similar GPU)
- **Python**: 3.10+ with pip
- **Browser**: Chrome/Edge (for best WebRTC support)

### Installation (Windows)

```powershell
# 1. Navigate to project
cd "C:\Users\wwwdo\Desktop\real-time AI coach"

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
python server.py

# 5. Open browser
# Go to http://localhost:8000
```

### Installation (macOS/Linux)

```bash
# 1. Navigate to project
cd ~/real-time-ai-coach

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
python server.py

# 5. Open browser
# Go to http://localhost:8000
```

---

## 📊 Metrics Explained (For Actors)

| Metric | What It Measures | Why It Matters for Acting |
|--------|------------------|---------------------------|
| **Pace (WPM)** | Words per minute | Control your tempo - rushed delivery loses impact |
| **Energy** | Volume/projection level | Fill the space, be heard, match scene intensity |
| **Vocal Variety** | Pitch range variation | Avoid monotone - bring your lines to life |
| **Pause Ratio** | Time spent in silence | Master dramatic beats and comedic timing |
| **Eye Contact** | Gaze direction | Connect with camera/scene partner |
| **Presence** | Face framing + stability | Command attention, own the frame |
| **Filler Words** | "um", "uh", "like" | Stay in character - fillers break immersion |

---

## 🎭 Coaching Tips (What the AI Tells You)

The coach provides context-aware tips based on your performance:

**Delivery:**
- "Slow down - let the words land"
- "Pick up the pace - this scene needs energy"
- "Great variation in your delivery!"

**Energy:**
- "Project more - fill the space"
- "Bring the energy down - this is an intimate moment"
- "Your energy is perfect for this intensity"

**Presence:**
- "Find your light - face the camera"
- "You're drifting out of frame"
- "Strong presence - you're commanding the space"

**Technique:**
- "Eliminate filler words - stay in character"
- "Use the pause - don't rush through it"
- "Vary your pitch - avoid monotone"

---

## 📁 Project Structure

```
real-time AI coach/
├── server.py              # FastAPI WebSocket server
├── config.py              # Tuning parameters
├── requirements.txt       # Python dependencies
│
├── audio_pipeline.py      # Audio feature extraction
├── vision_pipeline.py     # Face/gaze/presence analysis
├── asr_pipeline.py        # Speech-to-text (faster-whisper)
├── fusion_engine.py       # Combine all metrics + smoothing
├── coach_engine.py        # Actor-specific coaching logic
├── session_recorder.py    # Save sessions to disk
├── actor_coach.py         # NEW: Actor-specific coaching rules
│
├── frontend/
│   ├── index.html         # Main UI
│   ├── styles.css         # Styling
│   └── client.js          # WebSocket + media capture
│
├── sessions/              # Saved practice sessions
│   └── 20241230_143022/
│       ├── session.json
│       ├── transcript.txt
│       └── metrics.json
│
└── scripts/               # Your scripts/sides for practice
    └── example_monologue.txt
```

---

## ⚙️ Configuration & Tuning

Edit `config.py` to tune for your setup:

```python
class Config:
    # Performance vs Quality
    VISION_FPS = 8          # Lower = less GPU, less responsive
    AUDIO_CHUNK_MS = 100    # Smaller = lower latency, more CPU
    
    # Actor-specific thresholds
    IDEAL_WPM_MIN = 120     # Below this = "slow down"
    IDEAL_WPM_MAX = 160     # Above this = "pick up pace"
    
    # Smoothing (prevent tip flicker)
    SMOOTHING_ALPHA = 0.3   # Higher = more responsive, more jittery
    TIP_COOLDOWN_SEC = 3.0  # Min time between tips of same type
    
    # Optional LLM
    USE_LLM = False         # Enable Ollama for richer feedback
    OLLAMA_MODEL = "llama3.2"
```

---

## 🎯 Performance Targets

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **End-to-end latency** | < 500ms | Time from speech to tip appearing |
| **Video processing** | 8-10 FPS | GPU utilization should be < 50% |
| **ASR latency** | < 2 sec | Time from speech to transcript |
| **Tip stability** | No flicker | Tips should change smoothly |
| **Memory usage** | < 4 GB | Monitor with Task Manager |

---

## ✅ Validation Checklist

### Before Your First Session
- [ ] Server starts without errors
- [ ] Browser shows "Connected" status
- [ ] Camera/mic permissions granted
- [ ] Can see yourself in preview

### During Session
- [ ] Transcript appears as you speak
- [ ] Metrics update in real-time
- [ ] Tips appear (not too frequently)
- [ ] No lag or stuttering

### Quality Check
- [ ] WPM roughly matches your actual speed
- [ ] Eye contact tracks correctly
- [ ] Tips are relevant and helpful
- [ ] Session can be saved

---

## 🔮 Coming Soon

- [ ] **Script Mode**: Load your sides, practice with prompts
- [ ] **Emotion Detection**: Audio-based emotion classification
- [ ] **Character Profiles**: Save settings per character
- [ ] **Self-Tape Mode**: Optimized for audition recordings
- [ ] **Scene Partner AI**: Virtual reader for scenes
- [ ] **Progress Tracking**: See improvement over time

---

## 🎬 Use Cases

### Audition Prep
1. Load your sides (coming soon)
2. Practice with real-time feedback
3. Review recordings
4. Nail the audition

### Monologue Practice
1. Select "Free Practice" mode
2. Perform your monologue
3. Get feedback on delivery, presence, emotion
4. Refine and repeat

### Self-Tape Sessions
1. Use as a practice tool before recording
2. Get feedback on framing and eye line
3. Perfect your takes before pressing record

### Accent/Dialect Work
1. Practice with ASR feedback
2. Track your pronunciation
3. Monitor pace and rhythm

---

## 🤝 Contributing

This is an open-source project. Contributions welcome!

---

## 📜 License

MIT License - Use freely for your acting career!

---

**Break a leg! 🎭**
