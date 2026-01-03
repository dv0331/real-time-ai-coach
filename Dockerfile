# Dockerfile for Scivora - AI Acting Coach (Cloud Mode)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy cloud requirements (minimal dependencies)
COPY requirements-cloud.txt .

# Install Python dependencies for cloud mode only
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy application code
COPY . .

# Environment variables
ENV OPENAI_API_KEY=""
ENV DEPLOYMENT_MODE="cloud"
ENV PORT=10000

# Expose port
EXPOSE 10000

# Run the server
CMD ["python", "server.py"]
