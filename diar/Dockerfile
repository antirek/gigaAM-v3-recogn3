FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/models/hf \
    TORCH_HOME=/data/models/torch \
    NEMO_CACHE_DIR=/data/models/nemo

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        ffmpeg \
        libsndfile1 \
        git \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /work

# Blackwell (sm_120): PyTorch CUDA 12.8 wheels
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        torch torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# NeMo ASR (Sortformer). Install torch first so NeMo does not downgrade it.
RUN python -m pip install --no-cache-dir Cython packaging 'numpy<2.3' \
    && python -m pip install --no-cache-dir \
        'nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main'

COPY app /work/app

ENTRYPOINT ["python", "-m", "app.diarize"]
