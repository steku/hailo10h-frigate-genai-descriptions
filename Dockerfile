# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FRIGATE_LOG_DIR=/app/logs \
    HEF_MODEL_PATH=/usr/local/hailo/resources/models/hailo10h/Qwen2-VL-2B-Instruct.hef \
    MODEL_ID=Qwen2-VL-2B-Instruct.hef \
    LD_LIBRARY_PATH=/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH}

# Install minimal runtime libraries required by HailoRT (including OpenMP libgomp1 and libusb), Pillow, and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libusb-1.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies directly using pre-installed pip
COPY requirements.txt .
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Copy and install HailoRT deb and wheel from wheels directory
COPY wheels/ /tmp/wheels/
RUN dpkg -x /tmp/wheels/*.deb / && \
    ldconfig && \
    pip install --no-cache-dir --root-user-action=ignore /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# Copy application source code
COPY hailo_frigate_server.py .

# Create logs directory
RUN mkdir -p /app/logs

EXPOSE 8888

# Launch the FastAPI gateway server
CMD ["python", "hailo_frigate_server.py"]
