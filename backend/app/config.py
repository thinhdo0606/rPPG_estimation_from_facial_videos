"""
Configuration for Heart Rate Estimation Web Backend
"""
import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    MODEL_PATH = BASE_DIR / "models" / "MTTS_CSTM_fp16.pth"
    
    # Model parameters (must match training config)
    FACE_SIZE = 36
    CLIP_LENGTH = 128  # Number of frames per prediction
    FRAME_RATE = 30
    
    # Video upload / decode (memory-safe for long HD clips)
    MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "512"))
    MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
    # Only the first N seconds are decoded and analyzed (supports 30 / 45 / 60 s clips)
    MAX_VIDEO_DURATION_SEC = int(os.getenv("MAX_VIDEO_DURATION_SEC", "60"))
    # Web preview transcode length (FFmpeg); defaults to same cap as analysis
    PREVIEW_TRANSCODE_MAX_SEC = int(os.getenv("PREVIEW_TRANSCODE_MAX_SEC", str(MAX_VIDEO_DURATION_SEC)))
    # Downscale wide/tall frames before face detection (keeps quality, avoids OOM)
    VIDEO_DECODE_MAX_EDGE = int(os.getenv("VIDEO_DECODE_MAX_EDGE", "720"))
    
    # API settings
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    
    # CORS - Allow frontend
    CORS_ORIGINS = [
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
    
    # HR calculation
    HR_MIN = 40   # BPM
    HR_MAX = 180  # BPM
    
    # Real-time settings
    REALTIME_BUFFER_SIZE = 150  # Frames to buffer
    FACE_STABLE_THRESHOLD = 0.85  # 85% frames must have face
    
    # Device
    DEVICE = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"

config = Config()

