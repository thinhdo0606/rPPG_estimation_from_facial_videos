"""
Configuration for Heart Rate Estimation Web Backend
"""
import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    MODEL_PATH = BASE_DIR / "models" / "ts_cst_net_final.pth"
    
    # Model parameters (must match training config)
    FACE_SIZE = 36
    CLIP_LENGTH = 128  # Number of frames per prediction
    FRAME_RATE = 30
    
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

