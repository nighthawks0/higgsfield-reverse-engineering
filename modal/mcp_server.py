"""
Higgsfield Base-1 Reverse-Engineering — MCP Server
==================================================

Exposes the open-source Higgsfield reverse-engineering pipeline as MCP tools.
Each tool triggers a Modal GPU function remotely. No external API keys required.

Run locally (stdio):
  python mcp_server.py

Run as HTTP (StreamableHTTP):
  python mcp_server.py --http --port 8000

Register in Hermes ~/.hermes/config.yaml:
  mcp_servers:
    higgsfield-base1:
      command: "python"
      args: ["/root/higgsfield-reverse/modal/mcp_server.py"]
"""

import os
import sys
import argparse
import asyncio
from typing import Any, Dict

import modal

MODAL_APP_NAME = "higgsfield-genjutsu-base1"

try:
    app = modal.App.lookup(MODAL_APP_NAME, create_if_missing=True)
except Exception as e:
    app = None
    _lookup_error = str(e)
else:
    _lookup_error = None


def _modal_function(name: str):
    if app is None:
        raise RuntimeError(f"Modal app '{MODAL_APP_NAME}' not found: {_lookup_error}")
    return getattr(app, name)


# ---------------------------------------------------------------------------
# MCP Server (MCP 2.x MCPServer)
# ---------------------------------------------------------------------------

from mcp.server.mcpserver import MCPServer

server = MCPServer(
    name="higgsfield-base1",
    instructions="Reverse-engineered Higgsfield pipeline (Genjutsu Motion Transfer, Lipsync Studio, Soul ID, Video Enhancement) running open-source models on Modal GPU.",
)


@server.tool()
async def higgsfield_depth_anything(video_path: str = "/video/video.mp4") -> str:
    """Generate depth masks from a source video using Depth Anything V2."""
    fn = _modal_function("depth_anything_v2")
    result = fn.remote(video_path)
    return str(result)


@server.tool()
async def higgsfield_sam_segment(video_path: str = "/video/video.mp4") -> str:
    """Segment people on every frame using SAM 2."""
    fn = _modal_function("sam2_segment")
    result = fn.remote(video_path)
    return str(result)


@server.tool()
async def higgsfield_demucs(video_path: str = "/video/video.mp4") -> str:
    """Split vocals from music using Demucs (open source, Modal GPU)."""
    fn = _modal_function("demucs_local")
    result = fn.remote(video_path)
    return str(result)


@server.tool()
async def higgsfield_pitch_vocals(
    vocals_wav: str = "/out/vocals.wav",
    semitones: float = 3.0,
) -> str:
    """Pitch-shift vocals up by N semitones using FFmpeg rubberband."""
    fn = _modal_function("pitch_vocals")
    result = fn.remote(vocals_wav, semitones)
    return str(result)


@server.tool()
async def higgsfield_controlnet_motion_transfer(
    video_path: str = "/video/video.mp4",
    reference_images: str = "[]",
    prompt: str = "",
    resolution: str = "720p",
) -> str:
    """Open-source motion transfer: ControlNet depth + IP-Adapter references."""
    fn = _modal_function("controlnet_motion_transfer")
    result = fn.remote(video_path, reference_images, prompt, resolution)
    return str(result)


@server.tool()
async def higgsfield_wav2lip_lipsync(
    video_path: str = "/video/video.mp4",
    audio_path: str = "/out/vocals_pitched.wav",
) -> str:
    """Lip-sync video to audio using Wav2Lip (open source)."""
    fn = _modal_function("wav2lip_lipsync")
    result = fn.remote(video_path, audio_path)
    return str(result)


@server.tool()
async def higgsfield_instantid_character(
    reference_image: str,
    prompt: str = "",
) -> str:
    """Zero-training character consistency using InstantID (replaces Soul ID)."""
    fn = _modal_function("instantid_character")
    result = fn.remote(reference_image, prompt)
    return str(result)


@server.tool()
async def higgsfield_enhance_video(
    input_video: str = "/out/motion_transfer.mp4",
    upscale: str = "1080p",
    fps_boost: bool = False,
) -> str:
    """Upscale + frame interpolation using Real-ESRGAN + RIFE."""
    fn = _modal_function("enhance_video")
    result = fn.remote(input_video, upscale, fps_boost)
    return str(result)


@server.tool()
async def higgsfield_face_mash_landmarks(video_path: str = "/video/video.mp4") -> str:
    """MediaPipe 478-point face landmarks per frame (face mash variant)."""
    fn = _modal_function("face_mash_landmarks")
    result = fn.remote(video_path)
    return str(result)


@server.tool()
async def higgsfield_finalize_video(
    generated_video: str = "/out/motion_transfer.mp4",
    original_video: str = "/video/video.mp4",
    resolution: str = "1080p",
) -> str:
    """Replace generated video audio with original soundtrack, render final."""
    fn = _modal_function("finalize_video")
    result = fn.remote(generated_video, original_video, resolution)
    return str(result)


@server.tool()
async def higgsfield_run_pipeline(
    video_path: str = "/video/video.mp4",
    reference_images: str = "[]",
    prompt: str = "",
    resolution: str = "720p",
    do_face_mash: bool = True,
    do_enhance: bool = True,
) -> str:
    """Run the full base-1 pipeline end-to-end."""
    fn = _modal_function("main")
    result = fn.remote(
        video_path, reference_images, prompt, resolution,
        do_face_mash, False, do_enhance,
    )
    return str(result)


# ---------------------------------------------------------------------------
# Main — stdio or HTTP transport
# ---------------------------------------------------------------------------

async def main_stdio():
    await server.run_stdio_async()


async def main_http(port: int):
    starlette_app = server.streamable_http_app()
    import uvicorn
    config = uvicorn.Config(starlette_app, host="0.0.0.0", port=port)
    uvicorn_server = uvicorn.Server(config)
    await uvicorn_server.serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Higgsfield Base-1 MCP Server")
    parser.add_argument("--http", action="store_true", help="Run HTTP transport")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    args = parser.parse_args()

    if args.http:
        asyncio.run(main_http(args.port))
    else:
        asyncio.run(main_stdio())
