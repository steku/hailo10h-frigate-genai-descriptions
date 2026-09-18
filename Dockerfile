# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FRIGATE_LOG_DIR=/app/logs \
    HEF_MODEL_PATH=/usr/local/hailo/resources/models/hailo10h/Qwen2-VL-2B-Instruct.hef \
    MODEL_ID=Qwen2-VL-2B-Instruct.hef

# Install only minimal runtime libraries required by HailoRT, Pillow, and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies directly using pre-installed pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Automatically install any local HailoRT .deb or .whl files placed in ./wheels if provided
COPY ./wheels* /tmp/wheels/
RUN if [ -d "/tmp/wheels" ]; then \
        if ls /tmp/wheels/*.deb 1> /dev/null 2>&1; then \
            dpkg -i /tmp/wheels/*.deb || apt-get install -f -y; \
        fi && \
        if ls /tmp/wheels/*.whl 1> /dev/null 2>&1; then \
            pip install --no-cache-dir /tmp/wheels/*.whl; \
        fi && \
        rm -rf /tmp/wheels; \
    fi

# Copy application source code
COPY hailo_frigate_server.py .

# Create logs directory
RUN mkdir -p /app/logs

EXPOSE 8888

# Launch the FastAPI gateway server
CMD ["python", "hailo_frigate_server.py"]
