"""
Wan2.2-Animate-14B - RunPod Serverless Handler

Real pose-conditioned character animation, replacing the never-finished
Moore-AnimateAnyone stub (moore_animate_handler.py's animate_character() was a
TODO that just copied the reference image frame after frame).

Model: Wan-AI/Wan2.2-Animate-14B (Apache 2.0, Wan-Video/Alibaba, Sept 2025)
Requires: CUDA 12.4+, ~42.5GB peak VRAM measured by Wan-Video on H100 for the
animate-14B task (single GPU) — use an A100 80GB or H100, not a 24GB card.

Two-stage pipeline, deliberately split across two different codebases:
  1. Preprocessing (subprocess, original research repo — Diffusers has not
     reimplemented this yet; its own docs say so explicitly): runs
     wan/modules/animate/preprocess/preprocess_data.py from a full clone of
     github.com/Wan-Video/Wan2.2, which extracts pose/face conditioning from
     the driving video (ONNX ViTPose + YOLO detection) and retargets it to the
     character reference. In non-replace mode (what this handler uses) it
     writes exactly three files under --save_path: src_ref.png, src_pose.mp4,
     src_face.mp4 (confirmed by reading process_pipepline.py's __call__).
  2. Generation (in-process, official `diffusers` WanAnimatePipeline —
     released in diffusers, not a PR branch: see
     https://huggingface.co/docs/diffusers/en/api/pipelines/wan). Loads the
     three files stage 1 produced and calls the documented Python API
     directly, with real, confirmed tunables (num_inference_steps,
     guidance_scale, height/width, mode) instead of guessing at generate.py's
     undocumented CLI flags — this replaced an earlier draft of this handler
     that subprocessed the research repo's own generate.py; that approach is
     gone now in favor of this more maintainable, documented path.

Two separate HF repos are needed (different formats, not interchangeable):
  - Wan-AI/Wan2.2-Animate-14B            → only its process_checkpoint/
    subfolder is used, for stage 1's --ckpt_path (ONNX pose/det models).
  - Wan-AI/Wan2.2-Animate-14B-Diffusers  → the diffusers-format checkpoint
    WanAnimatePipeline.from_pretrained() loads for stage 2.

Operations:
  - animate  : character_url + pose_video_url (a driving video) + optional
               prompt / negative_prompt / fps
               -> a new video of the character performing that motion
  - health   : status, VRAM, whether model weights are present

KNOWN OPEN RISK (see plan doc): stage 1's ViTPose/YOLO are trained on real
human footage. Gothos will eventually drive this from a synthetic rendered
mannequin, not real photographic video — validate this handler FIRST against
a stock real human video (the `animate` op doesn't care where its inputs came
from) before wiring up the BVH-driven path.
"""

import os
import time
import base64
import logging
import tempfile
import subprocess
import shutil

import runpod
import requests
import torch
from PIL import Image
from diffusers import WanAnimatePipeline
from diffusers.utils import export_to_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("wan_animate")

# ── Constants ────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/wan_animate")
PREPROCESS_CKPT_DIR = os.path.join(MODEL_DIR, "preprocess_checkpoint")  # process_checkpoint/ subfolder only
DIFFUSERS_MODEL_DIR = os.path.join(MODEL_DIR, "diffusers_checkpoint")   # full -Diffusers repo
WAN22_REPO = os.environ.get("WAN22_REPO", "/opt/wan22")
PREPROCESS_SCRIPT_DIR = os.path.join(WAN22_REPO, "wan", "modules", "animate", "preprocess")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
OUTPUT_DIR = os.path.join(MODEL_DIR, "outputs")
# This worker has no file-serving mechanism of its own, so (unlike the LTX/
# Sulphur workers, which default to false and expect a separate upload step)
# it always returns base64 unless explicitly told not to.
RETURN_BASE64 = os.environ.get("RETURN_BASE64", "true").lower() in ("1", "true", "yes")

_models_ready = False
_pipe = None


def ensure_models_downloaded():
    global _models_ready
    if _models_ready:
        return
    from huggingface_hub import snapshot_download

    if HF_TOKEN:
        os.environ["HF_TOKEN"] = HF_TOKEN
        os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(PREPROCESS_CKPT_DIR) or not os.listdir(PREPROCESS_CKPT_DIR):
        logger.info("Downloading preprocessing checkpoints (process_checkpoint/ only, "
                    "not the full Wan-AI/Wan2.2-Animate-14B repo — we don't use its "
                    "generate.py) ...")
        t0 = time.time()
        snapshot_download(
            repo_id="Wan-AI/Wan2.2-Animate-14B",
            local_dir=PREPROCESS_CKPT_DIR,
            allow_patterns=["process_checkpoint/**"],
        )
        logger.info(f"Preprocessing checkpoints downloaded in {time.time() - t0:.1f}s")
    else:
        logger.info("Preprocessing checkpoints already present.")

    if not os.path.isdir(DIFFUSERS_MODEL_DIR) or not os.listdir(DIFFUSERS_MODEL_DIR):
        logger.info("Downloading Wan-AI/Wan2.2-Animate-14B-Diffusers (generation checkpoint) ...")
        t0 = time.time()
        snapshot_download(repo_id="Wan-AI/Wan2.2-Animate-14B-Diffusers", local_dir=DIFFUSERS_MODEL_DIR)
        logger.info(f"Diffusers checkpoint downloaded in {time.time() - t0:.1f}s")
    else:
        logger.info("Diffusers checkpoint already present.")

    _models_ready = True


def get_pipeline():
    global _pipe
    if _pipe is None:
        ensure_models_downloaded()
        logger.info("Loading WanAnimatePipeline ...")
        t0 = time.time()
        _pipe = WanAnimatePipeline.from_pretrained(DIFFUSERS_MODEL_DIR, torch_dtype=torch.bfloat16)
        _pipe.to("cuda")
        logger.info(f"Pipeline loaded in {time.time() - t0:.1f}s")
    return _pipe


# ═══════════════════════════════════════════════════════════════════════
#  INPUT / OUTPUT HELPERS (mirrors the pattern in sulphur2_handler.py)
# ═══════════════════════════════════════════════════════════════════════

_TEXTUAL_CONTENT_TYPES = ("text/html", "text/plain", "application/json", "application/xml", "text/xml")


def download_or_decode(data: str, suffix: str) -> str:
    """Fetch a URL or decode base64 into a temp file with the given suffix,
    failing fast if the bytes aren't real media (an HTML error page saved
    with a media extension would otherwise blow up deep inside the pipeline
    with a cryptic error)."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    content_type = ""
    source = "base64 input"
    if data.startswith("http"):
        source = data
        resp = requests.get(data, timeout=120)
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").lower()
        tmp.write(resp.content)
    else:
        if "," in data:
            data = data.split(",", 1)[1]
        tmp.write(base64.b64decode(data))
    tmp.close()

    size = os.path.getsize(tmp.name)
    with open(tmp.name, "rb") as f:
        head = f.read(512)
    problem = None
    if size == 0:
        problem = "got 0 bytes"
    elif head.lstrip()[:1] in (b"<", b"{") or any(t in content_type for t in _TEXTUAL_CONTENT_TYPES):
        problem = f"body is text/HTML, not media (content-type={content_type or 'n/a'})"
    if problem:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise ValueError(f"Invalid {suffix} input from {source[:200]}: {problem} ({size} bytes)")
    return tmp.name


def _run(cmd, cwd=None, timeout=600):
    logger.info(f"Running: {' '.join(cmd)} (cwd={cwd})")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    logger.info(f"Exit {proc.returncode} in {dt:.1f}s")
    if proc.returncode != 0:
        logger.error(f"STDOUT (tail):\n{proc.stdout[-4000:]}")
        logger.error(f"STDERR (tail):\n{proc.stderr[-4000:]}")
        raise RuntimeError(f"preprocess_data.py failed (exit {proc.returncode}): "
                           f"{proc.stderr[-1500:] or proc.stdout[-1500:]}")
    return proc


def _video_to_pil_frames(path):
    import decord
    vr = decord.VideoReader(path)
    frames = vr.get_batch(range(len(vr))).asnumpy()
    return [Image.fromarray(f) for f in frames]


# ═══════════════════════════════════════════════════════════════════════
#  OPERATIONS
# ═══════════════════════════════════════════════════════════════════════

def handle_health(_input):
    info = {
        "status": "healthy",
        "model_dir": MODEL_DIR,
        "preprocess_checkpoints_downloaded": os.path.isdir(PREPROCESS_CKPT_DIR) and bool(os.listdir(PREPROCESS_CKPT_DIR)),
        "diffusers_checkpoint_downloaded": os.path.isdir(DIFFUSERS_MODEL_DIR) and bool(os.listdir(DIFFUSERS_MODEL_DIR)),
        "pipeline_loaded": _pipe is not None,
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(props.total_memory / 1e9, 1)
    return info


def handle_animate(input_data):
    character_url = input_data.get("character_url") or input_data.get("reference_image")
    pose_video_url = input_data.get("pose_video_url") or input_data.get("driving_video_url")
    if not character_url or not pose_video_url:
        raise ValueError("animate requires both character_url (reference image) and "
                          "pose_video_url (driving video)")

    fps = int(input_data.get("fps", 24))
    height = int(input_data.get("height", 720))
    width = int(input_data.get("width", 1280))
    num_inference_steps = int(input_data.get("num_inference_steps", 20))
    seed = input_data.get("seed")

    # WanAnimatePipeline is text-conditioned as well as pose-conditioned, so the
    # caller's prompt describes the look while the driving video supplies the
    # motion. An earlier version of this handler dropped both prompt fields on
    # the floor, which made the panel's prompt box appear to do nothing.
    prompt = (input_data.get("prompt") or "").strip()
    negative_prompt = (input_data.get("negative_prompt") or "").strip()
    # Wan-Animate's documented default is 1.0, and it needs to stay there. An
    # earlier version of this handler raised it to 3.5 whenever a prompt was
    # present, on the theory that CFG is what makes a prompt "bite" — in practice
    # that overcooked the render and the tail of the clip blew out into a blur.
    # The prompt still conditions the model at 1.0 (it's fed to the text encoder
    # regardless; CFG only amplifies the positive-vs-negative contrast), so the
    # scale is left alone unless a caller deliberately overrides it.
    #
    # Corollary: at guidance_scale 1.0 there is no negative branch, so
    # negative_prompt is inert. It's still forwarded for callers who raise the
    # scale.
    guidance_scale = float(input_data.get("guidance_scale") or 1.0)

    workdir = tempfile.mkdtemp(prefix="wan_animate_")
    try:
        character_path = download_or_decode(character_url, ".png")
        driving_path = download_or_decode(pose_video_url, ".mp4")

        preprocess_dir = os.path.join(workdir, "process_results")
        os.makedirs(preprocess_dir, exist_ok=True)

        # Stage 1 (subprocess, original repo). cwd is the preprocess script's
        # own directory — it does `from process_pipepline import ProcessPipeline`
        # as a same-directory import with no package qualifier.
        _run([
            "python", os.path.join(PREPROCESS_SCRIPT_DIR, "preprocess_data.py"),
            "--ckpt_path", os.path.join(PREPROCESS_CKPT_DIR, "process_checkpoint"),
            "--video_path", driving_path,
            "--refer_path", character_path,
            "--save_path", preprocess_dir,
            "--resolution_area", str(width), str(height),
            "--fps", str(fps),
            "--retarget_flag",
            # Deliberately NOT --replace_flag (animating a clean single
            # character, not compositing into existing footage) or --use_flux
            # (skips the extra Flux relighting pass — not needed for a first
            # working version, and our eventual driving input may not have a
            # clearly recognizable face for it to work from).
        ], cwd=PREPROCESS_SCRIPT_DIR, timeout=600)

        ref_path = os.path.join(preprocess_dir, "src_ref.png")
        pose_path = os.path.join(preprocess_dir, "src_pose.mp4")
        face_path = os.path.join(preprocess_dir, "src_face.mp4")
        for p in (ref_path, pose_path, face_path):
            if not os.path.exists(p):
                raise RuntimeError(f"preprocess_data.py did not produce expected output {p}")

        # Stage 2 (in-process, official diffusers pipeline).
        pipe = get_pipeline()
        generator = torch.Generator(device="cuda").manual_seed(int(seed)) if seed is not None else None
        t0 = time.time()
        pipe_kwargs = dict(
            image=Image.open(ref_path).convert("RGB"),
            pose_video=_video_to_pil_frames(pose_path),
            face_video=_video_to_pil_frames(face_path),
            mode="animate",
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        if prompt:
            pipe_kwargs["prompt"] = prompt
        if negative_prompt:
            pipe_kwargs["negative_prompt"] = negative_prompt
        output = pipe(**pipe_kwargs).frames[0]
        logger.info(f"Generation took {time.time() - t0:.1f}s")

        out_path = os.path.join(workdir, "output.mp4")
        export_to_video(output, out_path, fps=fps)

        result = {"success": True, "size_bytes": os.path.getsize(out_path)}
        if RETURN_BASE64:
            with open(out_path, "rb") as f:
                result["video_base64"] = base64.b64encode(f.read()).decode("utf-8")
        else:
            # No file-serving mechanism on this worker yet — copy to the
            # persistent volume so it's at least retrievable via a follow-up
            # mechanism if RETURN_BASE64 is ever turned off.
            persisted = os.path.join(OUTPUT_DIR, f"{int(time.time())}.mp4")
            shutil.copy(out_path, persisted)
            result["output_path"] = persisted
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


OPERATION_MAP = {
    "animate": handle_animate,
    "health": handle_health,
}


def handler(event):
    try:
        input_data = event.get("input", {})
        operation = input_data.get("operation") or input_data.get("action", "animate")
        logger.info(f"Operation: {operation}")
        if operation not in OPERATION_MAP:
            return {"error": f"Unknown operation: {operation}", "available_operations": list(OPERATION_MAP.keys())}
        return OPERATION_MAP[operation](input_data)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Handler error: {e}\n{tb}")
        return {"error": str(e), "traceback": tb}


if __name__ == "__main__":
    logger.info("=" * 64)
    logger.info("  Wan2.2-Animate-14B RunPod Serverless Handler")
    logger.info(f"  Model dir  : {MODEL_DIR}")
    logger.info(f"  Wan22 repo : {WAN22_REPO}")
    logger.info(f"  Return b64 : {RETURN_BASE64}")
    logger.info("=" * 64)
    logger.info("Ensuring models are downloaded ...")
    ensure_models_downloaded()
    if os.environ.get("PRELOAD_PIPELINE", "1").lower() in ("1", "true", "yes"):
        try:
            get_pipeline()
        except Exception as e:
            logger.warning(f"Pipeline pre-load failed (will load on first request): {e}")
    runpod.serverless.start({"handler": handler})
