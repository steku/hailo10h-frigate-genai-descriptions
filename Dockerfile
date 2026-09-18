# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FRIGATE_LOG_DIR=/app/logs \
    HEF_MODEL_PATH=/usr/local/hailo/resources/models/hailo10h/Qwen2-VL-2B-Instruct.hef \
    MODEL_ID=Qwen2-VL-2B-Instruct.hef \
    LD_LIBRARY_PATH=/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH}

# Install minimal runtime libraries required by HailoRT (including OpenMP libgomp1), Pillow, and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies directly using pre-installed pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install HailoRT wheel, deb, and shared libraries from the wheel directory
COPY wheel* /tmp/wheels/
RUN if [ -d "/tmp/wheels" ]; then \
        if ls /tmp/wheels/*.deb 1> /dev/null 2>&1; then \
            dpkg -i /tmp/wheels/*.deb || apt-get install -f -y; \
        fi && \
        if ls /tmp/wheels/*.whl 1> /dev/null 2>&1; then \
            pip install --no-cache-dir /tmp/wheels/*.whl; \
        fi && \
        if ls /tmp/wheels/*.so* 1> /dev/null 2>&1; then \
            cp -P /tmp/wheels/*.so* /usr/lib/; \
        fi && \
        find /usr/local/lib/python3.10 -name "libhailort.so*" -exec cp -P {} /usr/lib/ \; 2>/dev/null || true; \
        ldconfig; \
        rm -rf /tmp/wheels; \
    fi

# Copy application source code
COPY hailo_frigate_server.py .

# Create logs directory
RUN mkdir -p /app/logs

EXPOSE 8888

# Launch the FastAPI gateway server
CMD ["python", "hailo_frigate_server.py"]
