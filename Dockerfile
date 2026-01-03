# Dockerfile for Scivora - AI Acting Coach
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies (cloud mode only - no CUDA)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    python-multipart \
    openai \
    aiohttp \
    numpy \
    pillow \
    opencv-python-headless \
    mediapipe

# Copy application code
COPY . .

# Environment variables
ENV OPENAI_API_KEY=""
ENV DEPLOYMENT_MODE="cloud"
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the server
CMD ["python", "server.py"]
