# 🚀 Deployment Guide - Real-Time AI Coach

This guide explains how to deploy the Actor's AI Coach both **locally** and **online** using cloud services.

---

## 📊 Deployment Options Comparison

| Feature | 🖥️ LOCAL Mode | ☁️ CLOUD Mode |
|---------|--------------|---------------|
| **GPU Required** | ✅ Yes (RTX 3060+) | ❌ No |
| **Cost** | Free | ~$0.01-0.05/session |
| **Privacy** | 100% Local | Data sent to OpenAI |
| **Latency** | ~200ms | ~500-1000ms |
| **Works Online** | Only localhost | ✅ Yes, anywhere |
| **Setup Complexity** | High (CUDA, models) | Low (just API key) |

---

## 🔑 Using OpenAI API (Cloud Mode)

### Step 1: Get an OpenAI API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Go to **API Keys** → **Create new secret key**
4. Copy your key (starts with `sk-...`)

### Step 2: Set Environment Variable

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "sk-your-api-key-here"
```

**Windows (permanently):**
```powershell
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-your-api-key-here", "User")
```

**Linux/macOS:**
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

### Step 3: Enable Cloud Mode

Edit `config.py` and change the deployment mode:

```python
@dataclass
class DeploymentConfig:
    MODE: str = "cloud"  # Change from "local" to "cloud"
```

Or programmatically:

```python
from config import setup_for_cloud

# Option 1: Use environment variable
setup_for_cloud()

# Option 2: Pass API key directly
setup_for_cloud("sk-your-api-key-here")
```

### Step 4: Run the Server

```bash
python server.py
```

You'll see:
```
☁️  CLOUD MODE - Using OpenAI APIs
   LLM: OpenAI gpt-4o-mini
   ASR: OpenAI whisper-1
```

---

## 💰 Cloud Mode Costs (Estimated)

| Service | Model | Cost | Per Session (5 min) |
|---------|-------|------|---------------------|
| **ASR** | Whisper-1 | $0.006/min | ~$0.03 |
| **LLM Tips** | GPT-4o-mini | $0.15/1M tokens | ~$0.01 |
| **Vision** | GPT-4o | $5/1M tokens | ~$0.02 (optional) |

**Total: ~$0.04-0.06 per 5-minute session**

---

## 🌐 Deploying Online

### Option 1: Railway (Recommended - Easy)

1. Create account at [railway.app](https://railway.app)
2. Connect your GitHub repo
3. Add environment variable: `OPENAI_API_KEY`
4. Deploy!

```bash
# railway.json
{
  "build": { "builder": "NIXPACKS" },
  "deploy": { "startCommand": "python server.py" }
}
```

### Option 2: Render

1. Create account at [render.com](https://render.com)
2. New Web Service → Connect repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `python server.py`
5. Add `OPENAI_API_KEY` in Environment

### Option 3: Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Deploy
fly launch
fly secrets set OPENAI_API_KEY=sk-your-key
fly deploy
```

### Option 4: Docker

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OPENAI_API_KEY=""
EXPOSE 8000

CMD ["python", "server.py"]
```

```bash
docker build -t ai-coach .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... ai-coach
```

### Option 5: AWS/GCP/Azure (For Production)

For high-traffic production deployments:

1. **Container**: Use ECS, Cloud Run, or AKS
2. **Serverless**: AWS Lambda + API Gateway (with WebSocket support)
3. **GPU Instance**: If you want local models online (expensive)

---

## ⚙️ Configuration Options

### `config.py` - DeploymentConfig

```python
@dataclass
class DeploymentConfig:
    # "local" = GPU + Ollama | "cloud" = OpenAI APIs
    MODE: str = "cloud"
    
    # OpenAI API key (or set OPENAI_API_KEY env var)
    OPENAI_API_KEY: str = None
    
    # Cloud model settings
    OPENAI_LLM_MODEL: str = "gpt-4o-mini"      # Fast, cheap
    OPENAI_ASR_MODEL: str = "whisper-1"        # Only option
    OPENAI_VISION_MODEL: str = "gpt-4o"        # For face analysis
    
    # Use GPT-4 Vision instead of MediaPipe for emotion
    USE_VISION_FOR_EMOTION: bool = False
    
    # ASR buffer (longer = better accuracy, more latency)
    CLOUD_ASR_BUFFER_SEC: float = 3.0
```

### Model Options

| Setting | Options | Notes |
|---------|---------|-------|
| `OPENAI_LLM_MODEL` | `gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo` | mini is best for real-time |
| `OPENAI_ASR_MODEL` | `whisper-1` | Only option currently |
| `OPENAI_VISION_MODEL` | `gpt-4o`, `gpt-4-turbo` | For face emotion |

---

## 🔒 Security Best Practices

### API Key Security

❌ **Never do this:**
```python
OPENAI_API_KEY = "sk-actual-key-here"  # In code
```

✅ **Always do this:**
```python
# Environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
```

### Production Checklist

- [ ] API key stored in environment variables or secrets manager
- [ ] HTTPS enabled (most platforms do this automatically)
- [ ] Rate limiting implemented
- [ ] User authentication added (if multi-user)
- [ ] CORS configured properly
- [ ] Error logging without exposing sensitive data

---

## 🔄 Switching Between Modes

### Runtime Switch

```python
from config import config, setup_for_cloud, setup_for_local

# Switch to cloud
setup_for_cloud("sk-your-api-key")

# Switch back to local
setup_for_local()
```

### Environment-Based

```python
import os

# In config.py
MODE = os.environ.get("DEPLOYMENT_MODE", "local")
```

Then set in deployment:
```bash
DEPLOYMENT_MODE=cloud python server.py
```

---

## 🐛 Troubleshooting

### "OPENAI_API_KEY not set"

```bash
# Check if set
echo $OPENAI_API_KEY  # Linux/Mac
echo $env:OPENAI_API_KEY  # PowerShell
```

### "Rate limit exceeded"

OpenAI has rate limits. Solutions:
1. Increase `LLM_PROMPT_INTERVAL_SEC` in config
2. Increase `CLOUD_ASR_BUFFER_SEC` to reduce API calls
3. Upgrade OpenAI tier

### "Whisper API timeout"

Increase buffer time:
```python
CLOUD_ASR_BUFFER_SEC: float = 5.0  # Default is 3.0
```

### High latency in cloud mode

This is expected. Cloud mode trades speed for accessibility:
- Local: ~200ms latency
- Cloud: ~500-1500ms latency

---

## 📱 Mobile Access

Once deployed online, access from any device:

1. Deploy to cloud platform
2. Get your app URL (e.g., `https://my-coach.railway.app`)
3. Open URL on phone/tablet
4. Allow camera/microphone permissions
5. Start practicing!

---

## 🎯 Recommended Setup

### For Development/Personal Use
```
MODE = "local"
GPU + Ollama
Free, fast, private
```

### For Demo/Testing
```
MODE = "cloud"
OPENAI_LLM_MODEL = "gpt-4o-mini"
Low cost, works anywhere
```

### For Production
```
MODE = "cloud"
OPENAI_LLM_MODEL = "gpt-4o-mini"
USE_VISION_FOR_EMOTION = True  # Optional, more accurate
+ Authentication
+ Rate limiting
+ Monitoring
```

---

## 🆘 Need Help?

1. Check the logs: `python server.py` shows detailed startup info
2. Test API key: `python -c "from llm_coach import test_llm_coach; import asyncio; asyncio.run(test_llm_coach())"`
3. Check cloud ASR: `python cloud_asr.py`

---

**Happy coaching! 🎭**
