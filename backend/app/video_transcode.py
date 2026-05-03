"""
Re-encode uploaded videos to H.264/AAC MP4 for HTML5 <video> playback.
Uses the FFmpeg binary from imageio-ffmpeg (no system FFmpeg required).
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import imageio_ffmpeg

from .config import config


def _video_temp_suffix(original_name: str = "") -> str:
    suf = Path(original_name or "").suffix.lower()
    if suf in (".mp4", ".avi", ".mov", ".webm", ".mkv", ".m4v"):
        return suf
    return ".mp4"


def transcode_video_bytes_to_playable_mp4(
    video_bytes: bytes,
    original_name: str = "",
) -> Optional[bytes]:
    """
    Decode with FFmpeg and mux to MP4 (yuv420p + AAC) for broad browser support.
    Output is capped to PREVIEW_TRANSCODE_MAX_SEC from the start (same window as analysis).
    """
    max_sec = int(config.PREVIEW_TRANSCODE_MAX_SEC)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    suffix = _video_temp_suffix(original_name)
    in_path: Optional[str] = None
    out_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f_in:
            f_in.write(video_bytes)
            in_path = f_in.name
        out_path = in_path + ".web_preview.mp4"

        def run(with_audio: bool) -> bool:
            if os.path.isfile(out_path):
                try:
                    os.unlink(out_path)
                except OSError:
                    pass
            cmd = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                in_path,
                "-t",
                str(max_sec),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ]
            if with_audio:
                cmd.extend(["-c:a", "aac", "-b:a", "128k"])
            else:
                cmd.append("-an")
            cmd.append(out_path)
            r = subprocess.run(cmd, capture_output=True, timeout=600)
            return (
                r.returncode == 0
                and os.path.isfile(out_path)
                and os.path.getsize(out_path) > 0
            )

        if not run(with_audio=True) and not run(with_audio=False):
            return None

        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            if p and os.path.isfile(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
