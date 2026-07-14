# wan-animate-worker

GitHub Actions builder for the Wan2.2-Animate-14B RunPod serverless worker
(`ochidi1/wan-animate`), the pose-conditioned V2V engine behind Gothos'
BVH → character pose transfer (`/api/pose-transfer/video`, `model=pose-video`).

Cross-building this image locally on Apple Silicon is impractical (CUDA 12.4
devel base + torch cu124 + onnxruntime-gpu under QEMU), so it builds here, the
same way `ardy-worker` and `gaussiangpt-worker` do.

Source of truth for the handler and Dockerfile is the main repo
(`experimental/runpod/wan_animate_handler.py` + `Dockerfile.wan_animate`);
copies here are what CI actually builds.

## Pipeline

1. **Preprocess** (subprocess, cloned Wan2.2 research repo): ViTPose + YOLO
   extract pose/face conditioning from the driving video and retarget it onto
   the character reference → `src_ref.png`, `src_pose.mp4`, `src_face.mp4`.
2. **Generate** (in-process, official `diffusers` `WanAnimatePipeline`): the
   character performs the driving motion, steered by `prompt` /
   `negative_prompt`.

## Endpoint requirements

- **GPU:** A100 80GB or H100 — ~42.5GB peak VRAM. A 24GB card will OOM.
- **Volume:** 150GB at `/runpod-volume` (two HF repos + cache).
- **Container disk:** 30GB.
- **Env:** `HF_TOKEN` (optional), `MODEL_DIR=/runpod-volume/wan_animate`.

Weights download on first cold start, so budget several minutes for it.

## Deploy

Push to `main` (or run the workflow manually) → `ochidi1/wan-animate:latest`.
Point the RunPod template at that tag, then put the endpoint id in the main
repo's `app/config.ini` under `[keys] WAN_ANIMATE_RUNPOD_ENDPOINT_ID`.
