# ─────────────────────────────────────────────────────────────────────
# Wan2.2-Animate-14B RunPod Serverless
#
# Replaces the never-finished Moore-AnimateAnyone stub (moore_animate_handler.py
# — its animate_character() was a TODO that just copied the reference image).
# Wan2.2-Animate is an officially released, actively maintained model
# (Wan-Video/Alibaba, Sept 2025, Apache 2.0) with a real two-stage pipeline:
#   1. preprocess_data.py       — pose/face extraction + retargeting from the
#                                  driving video (ViTPose + YOLO, both ONNX;
#                                  SAM2 only used in --replace_flag mode,
#                                  which we don't use — subprocessed from a
#                                  clone of the original research repo, since
#                                  Diffusers hasn't reimplemented this step)
#   2. WanAnimatePipeline (diffusers) — the actual diffusion generation, via
#                                  the official released `diffusers` Python
#                                  API (NOT the research repo's generate.py —
#                                  no flash-attn build needed as a result)
#
# Peak VRAM measured by Wan-Video at ~42.5GB (H100, single-GPU) for the
# animate-14B task — this needs an A100 80GB / H100, NOT a 24GB card.
#
# KNOWN RISK (see plan's milestone 1): ViTPose/YOLO here are trained on real
# human footage; our eventual driving input is a synthetic rendered mannequin.
# This Dockerfile/handler is deliberately validated FIRST against a stock real
# human video (no Gothos-specific code involved) before that risk is tested.
#
# Build with: docker buildx build --platform linux/amd64 -f Dockerfile.wan_animate -t ochidi1/wan-animate:latest --load .
# ─────────────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

LABEL maintainer="gothos"
LABEL description="Wan2.2-Animate-14B - pose-conditioned character animation, RunPod serverless handler"

# ── System dependencies ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/python3.10 /usr/bin/python3

# ── Clone Wan2.2 repo — only used for its preprocess/ scripts, not generate.py ──
RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/wan22
WORKDIR /opt/wan22

# ── Python deps ───────────────────────────────────────────────────────
# Deliberately NOT `pip install -r requirements.txt` from the cloned repo —
# that pulls in flash-attn (slow/fragile to build, and only needed by the
# repo's own generate.py, which we don't run) and pins transformers to an old
# range meant for that generate.py, which would fight the newer transformers
# WanAnimatePipeline (diffusers) actually needs. Installing an explicit,
# curated set instead, covering exactly what's used:
#   - torch/torchvision/torchaudio : pinned cu124 build for a stable ABI. The
#                                     floor is set by sam2 (needs torch>=2.5.1)
#                                     — an earlier 2.4.0 pin here got silently
#                                     upgraded to a cu130 PyPI wheel when sam2
#                                     installed, which left torchaudio 2.4.0
#                                     linked against the old ABI and made
#                                     `from diffusers import WanAnimatePipeline`
#                                     die on libtorchaudio.so. Keep these three
#                                     in lockstep and at/above sam2's floor.
#   - diffusers>=0.39.0            : first release with WanAnimatePipeline
#   - transformers/accelerate/safetensors/huggingface_hub : diffusers deps
#   - onnxruntime-gpu              : runs vitpose_h_wholebody.onnx + yolov10m.onnx
#   - sam2                         : imported UNCONDITIONALLY by
#                                     process_pipepline.py at module load time
#                                     even though we only use --retarget_flag,
#                                     not --replace_flag (which is the only
#                                     mode that actually exercises SAM2)
#   - moviepy<2                    : code uses the removed `moviepy.editor` API
#   - decord, loguru               : driving-video frame IO / logging
#   - matplotlib, tqdm             : human_visualization.py imports matplotlib at
#                                     module load (pose colormaps); tqdm is used
#                                     by the pipeline loops
#   - hydra-core, omegaconf        : sam_utils/video_predictor config loading
#   - ftfy, regex                  : WanAnimatePipeline's prompt_clean() calls
#                                     ftfy.fix_text() behind a lazy import — with
#                                     ftfy absent it dies on `name 'ftfy' is not
#                                     defined`. This path only runs when a prompt
#                                     is actually passed, which is why it stayed
#                                     hidden while the handler was dropping the
#                                     prompt entirely.
#
# The list above is now the FULL third-party import closure of
# wan/modules/animate/preprocess/*.py (audited across all 8 modules:
# PIL, cv2, decord, diffusers, hydra, loguru, matplotlib, moviepy, numpy,
# omegaconf, onnxruntime, sam2, torch, tqdm), not a guess — the earlier
# "KNOWN RISK" note about un-audited imports is what let a missing matplotlib
# take down preprocess_data.py at import time on the first real job.
# NOT installing the old Moore-AnimateAnyone Dockerfile's heavy mmcv/mmdet/mmpose
# stack — Wan's own preprocessing is ONNX-based and doesn't need it.
#
# KNOWN RISK: pose2d_utils.py/retarget_pose.py/utils.py/human_visualization.py/
# sam_utils.py/video_predictor.py (the rest of the preprocessing module) were
# not individually import-audited beyond process_pipepline.py + pose2d.py —
# if the first build fails on a missing import from one of those, add it here.
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
      --index-url https://download.pytorch.org/whl/cu124

# Pin the CUDA torch build for every later resolution too. Without this, any
# dependency with a `torch>=…` floor above the installed version drags in a
# fresh PyPI torch wheel (a different CUDA build) and silently breaks the ABI.
COPY constraints.txt /tmp/constraints.txt

RUN python -m pip install --no-cache-dir -c /tmp/constraints.txt \
      "diffusers>=0.39.0" \
      "transformers>=4.51.0" \
      accelerate safetensors huggingface_hub \
      onnxruntime-gpu \
      "moviepy<2" \
      decord \
      loguru \
      matplotlib \
      tqdm \
      hydra-core \
      omegaconf \
      ftfy \
      regex \
      runpod \
      opencv-python-headless \
      "imageio[ffmpeg]" \
      requests \
      && python -m pip install --no-cache-dir -c /tmp/constraints.txt "git+https://github.com/facebookresearch/sam2.git"

RUN python -c "import torch, diffusers; print(f'torch={torch.__version__} cuda={torch.version.cuda} diffusers={diffusers.__version__}')" && \
    python -c "from diffusers import WanAnimatePipeline; print('WanAnimatePipeline import OK')"

# Import stage 1 exactly the way the handler subprocesses it (same cwd, same
# same-directory imports). This turns a missing preprocessing dependency into a
# BUILD failure instead of a GPU-minutes failure on the first real job — which
# is how the missing matplotlib was found.
RUN cd /opt/wan22/wan/modules/animate/preprocess && \
    python -c "from process_pipepline import ProcessPipeline; print('preprocess import chain OK')"

# Exercise the pipeline's prompt-cleaning path (no weights needed). This is the
# code diffusers runs on every prompted generation, and it was failing at
# inference time with "name 'ftfy' is not defined" — a build-time check is far
# cheaper than an H100 job to catch it.
RUN python -c "from diffusers.pipelines.wan.pipeline_wan_animate import prompt_clean; \
assert prompt_clean('  a MAN in a black coat  ') == 'a MAN in a black coat'; \
print('prompt_clean OK')"

ENV PIP_BREAK_SYSTEM_PACKAGES=1

# ── Handler ───────────────────────────────────────────────────────────
WORKDIR /app
COPY wan_animate_handler.py /app/handler.py

ENV PYTHONUNBUFFERED=1
# Model weights live on a persistent RunPod network volume, downloaded on
# first cold start — same convention as every other worker in this repo.
# Two separate HF repos land under here (see handler docstring for why):
#   MODEL_DIR/preprocess_checkpoint/process_checkpoint/  <- ONNX pose/det models
#   MODEL_DIR/diffusers_checkpoint/                      <- WanAnimatePipeline weights
ENV MODEL_DIR=/runpod-volume/wan_animate
ENV WAN22_REPO=/opt/wan22
ENV HF_HOME=/runpod-volume/huggingface-cache

CMD ["python", "-u", "/app/handler.py"]
