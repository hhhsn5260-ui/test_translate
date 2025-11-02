FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        python3-setuptools \
        python3-wheel \
        ffmpeg \
        git && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY translate_agent ./translate_agent
COPY scripts ./scripts
COPY main.py .

RUN pip install --upgrade pip && \
    pip install .

# Whisper 将首次调用时下载模型，可提前拉取
RUN python -c "import whisper; whisper.load_model('base')" && \
    rm -rf ~/.cache/whisper

ENTRYPOINT ["translate-video"]
