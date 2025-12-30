# Real-Time AI Coach - Setup Guide

Complete step-by-step instructions for Windows, macOS, and Linux.

## Prerequisites

### Required
- Python 3.10+ (3.11 recommended)
- NVIDIA GPU with CUDA support (RTX 4060 = perfect)
- CUDA Toolkit 11.8 or 12.x
- cuDNN 8.x
- Modern browser (Chrome/Edge/Firefox with WebRTC support)
- Webcam + Microphone

### Optional
- Ollama (for LLM-enhanced coaching tips)

---

## Step 1: CUDA Setup (GPU Acceleration)

### Windows

1. **Check GPU**: Open Command Prompt and run:
   ```cmd
   nvidia-smi
   ```
   You should see your RTX 4060 listed.

2. **Install CUDA Toolkit**:
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Select: Windows → x86_64 → 11 or 10 → exe (local)
   - Run installer, choose Express installation
   - Restart computer

3. **Install cuDNN**:
   - Download from: https://developer.nvidia.com/cudnn (requires NVIDIA account)
   - Extract ZIP file
   - Copy contents to CUDA installation:
     - Copy `bin\*.dll` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`
     - Copy `include\*.h` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include`
     - Copy `lib\x64\*.lib` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64`

4. **Verify**:
   ```cmd
   nvcc --version
   ```

### macOS (Apple Silicon)

No CUDA support, but faster-whisper works on CPU with good performance on M1/M2/M3.

### Linux (Ubuntu/Debian)

```bash
# Install CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
nvidia-smi
```

---

## Step 2: Python Environment Setup

### Windows

```powershell
# Navigate to project
cd "C:\Users\wwwdo\Desktop\real-time AI coach"

# Create virtual environment
python -m venv venv

# Activate (PowerShell)
.\venv\Scripts\Activate.ps1

# Or activate (Command Prompt)
venv\Scripts\activate.bat

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### macOS/Linux

```bash
# Navigate to project
cd ~/path/to/real-time-ai-coach

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

---

## Step 3: Install Additional Dependencies

### webrtcvad (Voice Activity Detection)

**Windows** may need Visual C++ Build Tools:
1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++"
3. Then: `pip install webrtcvad`

**macOS**:
```bash
pip install webrtcvad
```

**Linux**:
```bash
sudo apt install python3-dev
pip install webrtcvad
```

### faster-whisper with GPU

```bash
# Already in requirements.txt, but ensure CUDA version matches:
pip install faster-whisper

# Test GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Step 4: Optional - Install Vosk (Fallback ASR)

If faster-whisper fails, Vosk provides a lighter alternative:

```bash
pip install vosk

# Download model (run once)
python -c "
import os
import urllib.request
import zipfile

model_url = 'https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

print('Downloading Vosk model...')
zip_path = os.path.join(model_dir, 'vosk-model.zip')
urllib.request.urlretrieve(model_url, zip_path)

print('Extracting...')
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(model_dir)
os.remove(zip_path)
print('Done!')
"
```

---

## Step 5: Optional - Install Ollama (LLM Tips)

For enhanced AI coaching tips:

### Windows
1. Download from: https://ollama.ai/download
2. Install and run
3. Open terminal: `ollama pull mistral`

### macOS
```bash
brew install ollama
ollama serve &
ollama pull mistral
```

### Linux
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull mistral
```

To enable in the app, edit `config.py`:
```python
@dataclass
class CoachConfig:
    USE_LLM: bool = True  # Change to True
```

---

## Step 6: Run the Application

### Start the Server

```bash
# Make sure venv is activated
python server.py
```

You should see:
```
============================================================
Real-Time AI Coach Server Starting...
============================================================
Server: http://0.0.0.0:8000
Open http://localhost:8000 in your browser
============================================================
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Open the Frontend

1. Open your browser
2. Navigate to: http://localhost:8000
3. Allow camera and microphone access when prompted
4. Click "Start Session" to begin!

---

## Troubleshooting

### "CUDA out of memory"

Reduce model size in `config.py`:
```python
MODEL_SIZE: str = "tiny"  # Instead of "base"
```

### "webrtcvad installation failed"

Windows: Install Visual C++ Build Tools (see Step 3)
macOS: `brew install portaudio`
Linux: `sudo apt install python3-dev`

### "No module named 'cv2'"

```bash
pip install opencv-python-headless
```

### "faster-whisper not using GPU"

Check CUDA installation:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show RTX 4060
```

If False, reinstall PyTorch with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "WebSocket connection failed"

1. Check if server is running
2. Try a different port: edit `config.py` → `PORT: int = 8080`
3. Check firewall settings

### "Camera not detected"

1. Check if another app is using the camera
2. Try a different browser
3. Check browser permissions: Settings → Privacy → Camera

### "Audio crackling/distorted"

Increase chunk size in `config.py`:
```python
CHUNK_DURATION_MS: int = 200  # Instead of 100
```

---

## Performance Tuning

### For Lower Latency (if GPU is strong)

```python
# config.py
AUDIO_CHUNK_MS: int = 50      # Smaller chunks
TARGET_FPS: int = 10          # More video frames
MODEL_SIZE: str = "base"      # Larger, more accurate
```

### For Lower GPU Usage

```python
# config.py
AUDIO_CHUNK_MS: int = 200     # Larger chunks
TARGET_FPS: int = 5           # Fewer video frames
MODEL_SIZE: str = "tiny"      # Smaller model
DEVICE: str = "cpu"           # Use CPU instead
```

### Memory Usage Estimates

| Model Size | GPU Memory | Accuracy |
|------------|------------|----------|
| tiny       | ~1GB       | Good     |
| base       | ~1.5GB     | Better   |
| small      | ~2GB       | Great    |
| medium     | ~5GB       | Excellent|

Your RTX 4060 (8GB) can handle up to "small" comfortably.

---

## Quick Test Checklist

1. [ ] Server starts without errors
2. [ ] Browser can access http://localhost:8000
3. [ ] Camera preview shows your face
4. [ ] Microphone permission granted
5. [ ] Click "Start Session" - no errors in console
6. [ ] Speak - transcript appears
7. [ ] Face indicator shows "detected"
8. [ ] Metrics update in real-time
9. [ ] Coaching tips appear
10. [ ] Click "End Session" - summary appears

---

## Next Steps

1. **Practice with the coach** - do a few 2-minute sessions
2. **Review sessions** - check `sessions/` folder for saved data
3. **Tune parameters** - adjust thresholds in `config.py`
4. **Enable LLM tips** - install Ollama for richer feedback
5. **Extend** - see README.md for optional extensions

