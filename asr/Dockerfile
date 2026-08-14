FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GIGAAM_CACHE=/data/models/gigaam \
    DEVICE=cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        ffmpeg \
        libsndfile1 \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /work

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        torch torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

RUN python -m pip install --no-cache-dir \
    "gigaam @ git+https://github.com/salute-developers/GigaAM.git"

COPY app /work/app

ENTRYPOINT ["python", "-m", "app.transcribe"]
