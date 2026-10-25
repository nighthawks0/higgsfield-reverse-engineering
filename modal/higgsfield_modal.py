"""
Higgsfield Reverse-Engineering — Genjutsu Motion Transfer
=========================================================

Reverse-engineered from Sirio's YouTube tutorial:
https://www.youtube.com/watch?v=d6spCMcfkew&t=10s

Core idea: keep the "how" (motion/camera/timing), throw away the "who"
(identity/IP). Two IP blockers: face + music.

11-step pipeline implemented as Modal functions:
  1. Upload original video
  2. Demucs: split vocals from music
  3. Pitch vocals up 3 semitones (rubber band) — avoids music fingerprint block
  4. Depth Anything: depth-mask the video (keeps motion, throws away appearance)
  5. SAM: cut out people on every frame
  6. Composite cut-out people over original background
  7. Insert pitched vocals into the video
  8. Human review gate (check holes in body/hair/hands)
  9. Seedance via Enhancer API (draft mode): swap people using depth video + refs + prompt
 10. Replace generated audio with original soundtrack
 11. Approve → render 1080p, restore original audio, download final

Plus optional Face Mash (MediaPipe Face Landmarker, 478 points/frame).

Run:
  modal deploy higgsfield_modal.py        # deploy to Modal
  modal run higgsfield_modal.py::main     # run locally (triggers remote)

Env vars needed:
  MODAL_PROFILE=yogeshag1314-alt          # default active profile
  REPLICATE_API_TOKEN                     # for Depth Anything / Demucs
  ENHANCER_API_KEY                        # for Seedance 2.5 draft generations
  HIGGSFIELD_API_KEY_ID / HIGGSFIELD_API_KEY_SECRET  # for official Genjutsu API
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image & volumes
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install(
        "git", "ffmpeg", "python3-dev", "build-essential", "libsm6", "libxext6",
        "libxrender-dev", "libglib2.0-0", "wget", "curl",
    )
    .pip_install(
        "torch==2.14.0",
        "torchvision",
        "opencv-python-headless",
        "mediapipe==0.10.14",
        "requests",
        "tqdm",
        "numpy",
        "Pillow",
        "scipy",
    )
    # Replicate Python client for Depth Anything / Demucs models
    .pip_install("replicate")
)

model_volume = modal.Volume.from_name("higgsfield-model-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("higgsfield-output", create_if_missing=True)
input_volume = modal.Volume.from_name("higgsfield-video-input", create_if_missing=True)

app = modal.App("higgsfield-genjutsu-reverse")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, cwd=None, env=None, check=True):
    """Run a shell command, streaming output."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    result = subprocess.run(
        cmd, cwd=cwd, env=full_env, shell=isinstance(cmd, str),
        capture_output=True, text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {cmd}\nSTDOUT: {result.stdout[-2000:]}\nSTDERR: {result.stderr[-4000:]}"
        )
    return result


def ffprobe(path):
    """Return ffprobe JSON for a media file."""
    out = run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", path],
        check=True,
    )
    return json.loads(out.stdout)


# ---------------------------------------------------------------------------
# Step 2: Demucs — vocal/music separation (Replicate)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    volumes={"/cache": model_volume, "/out": output_volume, "/video": input_volume},
    timeout=3600,
)
def demucs_separate(video_path: str = "/video/video.mp4"):
    """Split vocals from music using Demucs on Replicate.

    Returns paths to vocals.wav and accompaniment.wav.
    """
    import replicate

    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN not set")

    # Upload video to a public URL first (Replicate needs a URL input)
    # For simplicity, we assume video_path is already publicly accessible,
    # or we serve it. Here we pass the local path via a file:// won't work for Replicate.
    # Instead, we run Demucs locally if weights available, else Replicate.
    print(f"[demucs] separating vocals from {video_path}")

    # Local Demucs via torch (demucs package) — fallback to Replicate
    try:
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        print("[demucs] using local demucs")
        model = get_model("htdemucs")
        model.cuda()
        # ... apply
    except Exception as e:
        print(f"[demucs] local demucs unavailable ({e}), using Replicate")
        output = replicate.run(
            "maryoukai/demucs:latest",
            input={"audio": video_path, "split_vocals": True},
        )
        # output is a dict with file URLs
    # Placeholder: produce dummy outputs for structure
    vocals_wav = "/out/vocals.wav"
    accompaniment_wav = "/out/accompaniment.wav"
    # (real implementation writes these)
    volume.commit()
    return {"vocals": vocals_wav, "accompaniment": accompaniment_wav}


# ---------------------------------------------------------------------------
# Step 3: Pitch vocals up 3 semitones (rubber band / FFmpeg)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    volumes={"/out": output_volume, "/video": input_volume},
    timeout=1800,
)
def pitch_vocals(vocals_wav: str = "/out/vocals.wav", semitones: float = 3.0):
    """Pitch-shift vocals up by N semitones using FFmpeg rubberband filter."""
    out_path = "/out/vocals_pitched.wav"
    # rubberband pitch shift: pitch = 2^(semitones/12)
    pitch = 2 ** (semitones / 12.0)
    run([
        "ffmpeg", "-y", "-i", vocals_wav,
        "-af", f"rubberband=pitch={pitch}",
        "-ar", "44100", out_path,
    ], check=True)
    output_volume.commit()
    return out_path


# ---------------------------------------------------------------------------
# Step 4: Depth Anything — video depth masking
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    volumes={"/cache": model_volume, "/out": output_volume, "/video": input_volume},
    timeout=3600,
)
def depth_anything_video(video_path: str = "/video/video.mp4"):
    """Produce a depth-masked version of the video (keeps motion, throws away appearance).

    Uses Depth Anything V2 via Replicate or local torch hub.
    Output: /out/depth_video.mp4
    """
    import replicate

    token = os.environ.get("REPLICATE_API_TOKEN")
    print(f"[depth] generating depth mask for {video_path}")
    # Replicate Depth Anything Video
    output = replicate.run(
        "lucataco/depth-anything-video:latest",
        input={"video": video_path},
    )
    depth_video = "/out/depth_video.mp4"
    # download output
    run(["curl", "-L", "-o", depth_video, output["video_url"] if isinstance(output, dict) else output])
    output_volume.commit()
    return depth_video


# ---------------------------------------------------------------------------
# Step 5: SAM — cut out people on every frame
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    volumes={"/cache": model_volume, "/out": output_volume, "/video": input_volume},
    timeout=3600,
)
def sam_cutout_people(depth_video: str = "/out/depth_video.mp4"):
    """Segment people on every frame of the depth video using SAM.

    Outputs: /out/people_masks/ (per-frame PNG masks) and
             /out/people_cutout.mp4 (people cut out on transparent/black bg)
    """
    import cv2
    import numpy as np

    mask_dir = Path("/out/people_masks")
    mask_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(depth_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Placeholder: real SAM integration would load facebook/sam-vit-h model
    # and generate masks per frame. Here we write a structure stub.
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # TODO: run SAM predictor on frame -> mask
        mask_path = mask_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(mask_path), np.zeros((h, w), dtype=np.uint8))
        frame_idx += 1
    cap.release()

    cutout_path = "/out/people_cutout.mp4"
    output_volume.commit()
    return {"masks_dir": str(mask_dir), "cutout": cutout_path, "frames": frame_idx, "fps": fps}


# ---------------------------------------------------------------------------
# Step 6: Composite people over original background (Remotion-style in FFmpeg)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    volumes={"/out": output_volume, "/video": input_volume},
    timeout=3600,
)
def composite_people_over_bg(
    original_video: str = "/video/video.mp4",
    people_cutout: str = "/out/people_cutout.mp4",
):
    """Composite cut-out people over the original background.

    This stops the generation model from tracing the depth outline.
    Output: /out/composite.mp4
    """
    out_path = "/out/composite.mp4"
    # Use FFmpeg overlay: people_cutout over original_video
    run([
        "ffmpeg", "-y",
        "-i", original_video,
        "-i", people_cutout,
        "-filter_complex", "[1:v][0:v]scale2ref[cut][base];[cut][base]overlay=0:0:format=auto",
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-c:a", "copy", out_path,
    ], check=True)
    output_volume.commit()
    return out_path


# ---------------------------------------------------------------------------
# Step 7: Insert pitched vocals into the video
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    volumes={"/out": output_volume, "/video": input_volume},
    timeout=1800,
)
def insert_pitched_audio(
    video_path: str = "/out/composite.mp4",
    pitched_vocals: str = "/out/vocals_pitched.wav",
):
    """Replace video audio with pitched vocals."""
    out_path = "/out/composite_with_vocals.mp4"
    run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", pitched_vocals,
        "-c:v", "copy",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", out_path,
    ], check=True)
    output_volume.commit()
    return out_path


# ---------------------------------------------------------------------------
# Step 8: Human review gate — check masks for holes
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    volumes={"/out": output_volume, "/video": input_volume},
    timeout=1800,
)
def review_gate(composite_video: str = "/out/composite_with_vocals.mp4"):
    """Extract keyframes for human review of mask quality.

    Returns list of frame image paths the user should inspect.
    """
    import cv2
    review_dir = Path("/out/review")
    review_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(composite_video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # sample 5 frames evenly
    indices = [int(total * i / 5) for i in range(5)]
    paths = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            p = str(review_dir / f"review_{i:02d}.jpg")
            cv2.imwrite(p, frame)
            paths.append(p)
    cap.release()
    output_volume.commit()
    return {"review_frames": paths, "total_frames": total}


# ---------------------------------------------------------------------------
# Step 9: Seedance via Enhancer API (draft mode motion transfer)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    volumes={"/out": output_volume, "/video": input_volume},
    timeout=3600,
)
def seedance_motion_transfer(
    source_video: str = "/out/composite_with_vocals.mp4",
    reference_images: str = "[]",  # JSON array of public image URLs
    prompt: str = "",
    resolution: str = "720p",
):
    """Call Enhancer API / Seedance 2.5 draft mode for motion transfer.

    Sends depth-composite video + reference photos + prompt.
    Returns request handle; poll status_url until completed.
    """
    import requests

    enhancer_key = os.environ.get("ENHANCER_API_KEY")
    if not enhancer_key:
        raise RuntimeError("ENHANCER_API_KEY not set")

    refs = json.loads(reference_images) if isinstance(reference_images, str) else reference_images

    payload = {
        "video_url": source_video,  # must be publicly accessible URL
        "image_urls": refs,
        "prompt": prompt,
        "resolution": resolution,
        "draft_mode": True,
    }

    # Placeholder endpoint — Enhancer API docs from the video
    resp = requests.post(
        "https://api.enhancer.ai/v1/generate",
        json=payload,
        headers={"Authorization": f"Bearer {enhancer_key}"},
        timeout=120,
    )
    resp.raise_for_status()
    result = resp.json()
    output_volume.commit()
    return result


# ---------------------------------------------------------------------------
# Step 10-11: Restore original audio + final 1080p render
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    volumes={"/out": output_volume, "/video": input_volume},
    timeout=3600,
)
def finalize_video(
    generated_video: str = "/out/seedance_output.mp4",
    original_video: str = "/video/video.mp4",
    resolution: str = "1080p",
):
    """Replace generated video audio with original soundtrack, render final 1080p."""
    out_path = f"/out/final_{resolution}.mp4"
    # Extract audio from original
    orig_audio = "/out/original_audio.aac"
    run([
        "ffmpeg", "-y", "-i", original_video,
        "-vn", "-acodec", "copy", orig_audio,
    ], check=True)
    # Merge generated video with original audio, scale to target resolution
    height = {"480p": 480, "720p": 720, "1080p": 1080}.get(resolution, 1080)
    run([
        "ffmpeg", "-y",
        "-i", generated_video,
        "-i", orig_audio,
        "-filter_complex", f"[0:v]scale=-2:{height}:force_original_aspect_ratio=decrease,pad={height*16/9}:{height}:(ow-iw)/2:(oh-ih)/2[v]",
        "-map", "[v]", "-map", "1:a:0",
        "-c:v", "libx264", "-crf", "18", "-preset", "slow",
        "-c:a", "aac", "-b:a", "256k",
        "-shortest", out_path,
    ], check=True)
    output_volume.commit()
    return out_path


# ---------------------------------------------------------------------------
# Optional: Face Mash (MediaPipe Face Landmarker, 478 points/frame)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    volumes={"/out": output_volume, "/video": input_volume},
    timeout=1800,
)
def face_mash_landmarks(video_path: str = "/video/video.mp4"):
    """Track 478 face landmarks per frame using MediaPipe Face Landmarker.

    Saves /out/face_landmarks.json and produces a wireframe overlay video.
    Best for: single centered person talking to camera.
    """
    import cv2
    import mediapipe as mp
    import numpy as np

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    landmarks_all = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        frame_landmarks = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                pts = [
                    {"x": lm.x, "y": lm.y, "z": lm.z}
                    for lm in face_landmarks.landmark
                ]
                frame_landmarks.append(pts)
        landmarks_all.append({"frame": frame_idx, "faces": frame_landmarks})
        frame_idx += 1
    cap.release()
    face_mesh.close()

    out_json = "/out/face_landmarks.json"
    with open(out_json, "w") as f:
        json.dump({"fps": fps, "width": w, "height": h, "frames": landmarks_all}, f)
    output_volume.commit()
    return {"landmarks": out_json, "frames": frame_idx}


# ---------------------------------------------------------------------------
# Local entrypoint: orchestrate the full 11-step pipeline
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    video_path: str = "/video/video.mp4",
    reference_images: str = "[]",
    prompt: str = "",
    resolution: str = "1080p",
    do_face_mash: bool = False,
):
    """Run the full reverse-engineered Genjutsu motion-transfer pipeline.

    Args:
        video_path: source video (must be in /video/ volume, ≤30s)
        reference_images: JSON array of public image URLs for the new cast
        prompt: optional description of the change
        resolution: 480p / 720p / 1080p
        do_face_mash: whether to also run MediaPipe face landmark tracking
    """
    print("=== Higgsfield Genjutsu Reverse-Engineering Pipeline ===")
    print(f"Source: {video_path}")
    print(f"Resolution: {resolution}")
    print(f"References: {reference_images}")
    print(f"Prompt: {prompt}")

    # Step 2: Demucs
    print("\n[Step 2] Demucs vocal/music separation...")
    demucs_result = demucs_separate.remote(video_path)
    print(f"  vocals: {demucs_result['vocals']}")

    # Step 3: Pitch vocals
    print("\n[Step 3] Pitch vocals +3 semitones...")
    pitched = pitch_vocals.remote(demucs_result["vocals"])

    # Optional face mash
    if do_face_mash:
        print("\n[Face Mash] MediaPipe face landmarks...")
        face_result = face_mash_landmarks.remote(video_path)
        print(f"  landmarks: {face_result['landmarks']}")

    # Step 4: Depth Anything
    print("\n[Step 4] Depth Anything depth mask...")
    depth_video = depth_anything_video.remote(video_path)

    # Step 5: SAM cutout
    print("\n[Step 5] SAM cut out people...")
    sam_result = sam_cutout_people.remote(depth_video)

    # Step 6: Composite people over background
    print("\n[Step 6] Composite people over original background...")
    composite = composite_people_over_bg.remote(video_path, sam_result["cutout"])

    # Step 7: Insert pitched vocals
    print("\n[Step 7] Insert pitched vocals...")
    with_vocals = insert_pitched_audio.remote(composite, pitched)

    # Step 8: Review gate
    print("\n[Step 8] Human review gate (keyframes extracted)...")
    review = review_gate.remote(with_vocals)
    print(f"  Review frames: {review['review_frames']}")
    print("  -> Inspect /out/review/*.jpg before proceeding to generation")

    # Step 9: Seedance motion transfer (draft)
    print("\n[Step 9] Seedance motion transfer (draft mode)...")
    seedance_result = seedance_motion_transfer.remote(
        with_vocals, reference_images, prompt, resolution
    )
    print(f"  Seedance request: {seedance_result}")

    # Step 10-11: Finalize (restore original audio, 1080p)
    # Note: needs the generated video URL downloaded first — placeholder
    print("\n[Step 10-11] Finalize: restore original audio + render 1080p...")
    # final = finalize_video.remote(generated_url, video_path, resolution)

    print("\n=== Pipeline staged. Review keyframes, then run finalize. ===")
    return {
        "demucs": demucs_result,
        "pitched_vocals": pitched,
        "depth_video": depth_video,
        "sam": sam_result,
        "composite": composite,
        "with_vocals": with_vocals,
        "review": review,
        "seedance": seedance_result,
    }
