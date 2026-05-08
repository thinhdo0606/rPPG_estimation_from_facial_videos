"""
FastAPI Backend for Heart Rate Estimation Web App
Supports video upload and real-time webcam
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import time
import json
import asyncio
import os
from collections import deque
from pathlib import Path

import torch

try:
    import psutil
except Exception:
    psutil = None

from .config import config
from .preprocessing import (
    FaceProcessor,
    decode_video_bytes,
    decode_base64_frames,
    decode_base64_frame,
    extract_first_frame_jpeg_base64,
)
from .model_service import get_estimator
from .video_transcode import transcode_video_bytes_to_playable_mp4

# ==================== APP SETUP ====================

app = FastAPI(
    title="Heart Rate Estimation Web API",
    description="Estimate heart rate from facial video using TS-CST Net",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face processor
face_processor = FaceProcessor()
BENCHMARK_LOGS = deque(maxlen=5000)


def _safe_percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float32), q))


def _resource_snapshot() -> Dict[str, float]:
    cpu_percent = 0.0
    ram_mb = 0.0
    if psutil is not None:
        process = psutil.Process()
        cpu_percent = float(psutil.cpu_percent(interval=None))
        ram_mb = float(process.memory_info().rss / (1024 ** 2))

    gpu_mem_alloc_mb = 0.0
    gpu_mem_reserved_mb = 0.0
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        gpu_mem_alloc_mb = float(torch.cuda.memory_allocated() / (1024 ** 2))
        gpu_mem_reserved_mb = float(torch.cuda.memory_reserved() / (1024 ** 2))

    return {
        "cpu_percent": cpu_percent,
        "ram_mb": ram_mb,
        "gpu_mem_alloc_mb": gpu_mem_alloc_mb,
        "gpu_mem_reserved_mb": gpu_mem_reserved_mb,
    }


def _record_benchmark(entry: Dict):
    BENCHMARK_LOGS.append(entry)


# ==================== DATA MODELS ====================

class FramesRequest(BaseModel):
    frames: List[str]
    fps: Optional[float] = 30.0


class HeartRateResponse(BaseModel):
    success: bool
    heart_rate: float
    confidence: float
    snr_db: float
    unit: str = "BPM"
    processing_time_ms: float
    ppg_signal: Optional[List[float]] = None
    benchmark: Optional[Dict] = None
    message: Optional[str] = None


class VideoAnalysisResponse(BaseModel):
    success: bool
    heart_rate: float
    confidence: float
    snr_db: float
    unit: str = "BPM"
    video_duration: float
    fps: float
    total_frames: int
    processing_time_ms: float
    ppg_signal: Optional[List[float]] = None
    benchmark: Optional[Dict] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # Fix Pydantic v2 warning
    
    status: str
    model_loaded: bool
    device: str
    version: str


# ==================== REST ENDPOINTS ====================

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Heart Rate Estimation Web API",
        "docs": "/docs",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "upload_video": "/api/predict/video",
            "realtime": "/api/predict/realtime",
            "websocket": "/ws/realtime"
        }
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health"""
    try:
        estimator = get_estimator()
        return HealthResponse(
            status="healthy",
            model_loaded=estimator.model is not None,
            device=config.DEVICE,
            version="1.0.0"
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            device=config.DEVICE,
            version="1.0.0"
        )


@app.get("/api/benchmark/summary", response_model=dict)
async def benchmark_summary():
    """Return benchmark summary for recent requests."""
    if len(BENCHMARK_LOGS) == 0:
        model_size_mb = 0.0
        model_path = Path(config.MODEL_PATH)
        if model_path.exists():
            model_size_mb = float(model_path.stat().st_size / (1024 ** 2))
        return {
            "count": 0,
            "model_size_mb": model_size_mb,
            "device": config.DEVICE,
            "message": "No benchmark records yet. Call /api/predict/video or /api/predict/realtime first."
        }

    keys = [
        "decode_ms",
        "preprocess_ms",
        "inference_ms",
        "forward_ms",
        "postprocess_ms",
        "end_to_end_ms",
        "inference_fps",
        "pipeline_fps",
        "cpu_percent",
        "ram_mb",
        "gpu_mem_alloc_mb",
        "gpu_mem_reserved_mb",
    ]
    summary = {}
    for key in keys:
        vals = [float(x.get(key, 0.0)) for x in BENCHMARK_LOGS]
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "p50": _safe_percentile(vals, 50),
            "p95": _safe_percentile(vals, 95),
        }

    model_size_mb = 0.0
    model_path = Path(config.MODEL_PATH)
    if model_path.exists():
        model_size_mb = float(model_path.stat().st_size / (1024 ** 2))

    return {
        "count": len(BENCHMARK_LOGS),
        "model_size_mb": model_size_mb,
        "clip_length": config.CLIP_LENGTH,
        "device": config.DEVICE,
        "summary": summary,
    }


@app.post("/api/benchmark/reset", response_model=dict)
async def benchmark_reset():
    """Clear benchmark records."""
    BENCHMARK_LOGS.clear()
    return {"success": True, "message": "Benchmark logs cleared."}


@app.post("/api/predict/video", response_model=VideoAnalysisResponse)
async def predict_from_video(video: UploadFile = File(...)):
    """
    Analyze uploaded video file for heart rate

    - Formats: MP4, AVI, MOV, WebM (MIME types may vary by browser)
    - Clips: about 5–60 seconds (e.g. 30 s, 45 s, 1 min). Only the first
      config.MAX_VIDEO_DURATION_SEC seconds are decoded.
    - File size: up to config.MAX_UPLOAD_MB (default 512 MB)
    """
    start_time = time.perf_counter()
    
    try:
        # Validate file type
        allowed_types = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo']
        if video.content_type and video.content_type not in allowed_types:
            # Still try to process, some browsers report wrong MIME type
            pass
        
        # Read video bytes
        video_bytes = await video.read()
        
        if len(video_bytes) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Video file too large. Maximum size is {config.MAX_UPLOAD_MB} MB.",
            )
        
        # Decode video to frames
        t_decode_0 = time.perf_counter()
        original_name = video.filename or ""
        frames, fps = decode_video_bytes(video_bytes, original_name=original_name)
        t_decode_1 = time.perf_counter()
        
        if len(frames) < config.CLIP_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Video too short. Need at least {config.CLIP_LENGTH} frames (~4 seconds at 30fps). Got {len(frames)} frames."
            )
        
        video_duration = len(frames) / fps if fps > 0 else 0
        
        # Preprocess
        t_pre_0 = time.perf_counter()
        faces_array = face_processor.process_frames(frames)
        t_pre_1 = time.perf_counter()
        
        # Predict
        estimator = get_estimator()
        t_inf_0 = time.perf_counter()
        result = estimator.predict(faces_array)
        t_inf_1 = time.perf_counter()
        
        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000.0
        decode_ms = (t_decode_1 - t_decode_0) * 1000.0
        preprocess_ms = (t_pre_1 - t_pre_0) * 1000.0
        stage_inference_ms = (t_inf_1 - t_inf_0) * 1000.0
        forward_ms = float(result.get("benchmark", {}).get("forward_ms", 0.0))
        postprocess_ms = float(result.get("benchmark", {}).get("postprocess_ms", 0.0))
        inference_ms = float(result.get("benchmark", {}).get("inference_ms", stage_inference_ms))
        inference_fps = float(config.CLIP_LENGTH / max(inference_ms / 1000.0, 1e-8))
        pipeline_fps = float(config.CLIP_LENGTH / max(processing_time / 1000.0, 1e-8))
        resources = _resource_snapshot()
        benchmark = {
            "decode_ms": decode_ms,
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "forward_ms": forward_ms,
            "postprocess_ms": postprocess_ms,
            "end_to_end_ms": processing_time,
            "inference_fps": inference_fps,
            "pipeline_fps": pipeline_fps,
            **resources,
        }
        _record_benchmark({"endpoint": "video", **benchmark})
        
        return VideoAnalysisResponse(
            success=True,
            heart_rate=result['heart_rate'],
            confidence=result['confidence'],
            snr_db=result['snr_db'],
            unit=result['unit'],
            video_duration=video_duration,
            fps=fps,
            total_frames=len(frames),
            processing_time_ms=processing_time,
            ppg_signal=result.get('ppg_signal'),
            benchmark=benchmark,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (time.perf_counter() - start_time) * 1000
        return VideoAnalysisResponse(
            success=False,
            heart_rate=0,
            confidence=0,
            snr_db=0,
            video_duration=0,
            fps=0,
            total_frames=0,
            processing_time_ms=processing_time,
            message=str(e)
        )


@app.post("/api/preview/transcode")
async def preview_transcode(video: UploadFile = File(...)):
    """
    Convert upload to H.264/AAC MP4 so the browser can play it (e.g. AVI / exotic codecs).
    Only the first PREVIEW_TRANSCODE_MAX_SEC seconds are included (matches analysis cap).
    """
    try:
        video_bytes = await video.read()
        if len(video_bytes) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Video file too large. Maximum size is {config.MAX_UPLOAD_MB} MB.",
            )
        original_name = video.filename or ""
        mp4 = transcode_video_bytes_to_playable_mp4(video_bytes, original_name=original_name)
        if not mp4:
            raise HTTPException(
                status_code=500,
                detail="Could not convert video for playback. Try an H.264 MP4 or install/update FFmpeg dependencies.",
            )
        return Response(
            content=mp4,
            media_type="video/mp4",
            headers={"Content-Disposition": 'inline; filename="preview.mp4"'},
        )
    except HTTPException:
        raise


@app.post("/api/preview/thumbnail", response_model=dict)
async def preview_thumbnail(video: UploadFile = File(...)):
    """
    Return a JPEG thumbnail (first frame) as base64 for formats the browser
    cannot play in <video> (e.g. many AVI codecs).
    """
    try:
        video_bytes = await video.read()
        if len(video_bytes) > config.MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"Video file too large. Maximum size is {config.MAX_UPLOAD_MB} MB.",
            )
        original_name = video.filename or ""
        b64 = extract_first_frame_jpeg_base64(video_bytes, original_name=original_name)
        if not b64:
            return {"success": False, "image_base64": None, "message": "Could not read a frame from this video."}
        return {"success": True, "image_base64": b64, "message": None}
    except HTTPException:
        raise
    except Exception as e:
        return {"success": False, "image_base64": None, "message": str(e)}


@app.post("/api/predict/realtime", response_model=HeartRateResponse)
async def predict_realtime(request: FramesRequest):
    """
    Predict heart rate from real-time captured frames
    
    Send 128+ frames captured from webcam at ~30 FPS
    """
    start_time = time.perf_counter()
    
    try:
        if len(request.frames) < config.CLIP_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {config.CLIP_LENGTH} frames, got {len(request.frames)}"
            )
        
        # Decode frames
        t_decode_0 = time.perf_counter()
        frames = decode_base64_frames(request.frames)
        t_decode_1 = time.perf_counter()
        
        if len(frames) < config.CLIP_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode enough frames. Got {len(frames)} valid frames"
            )
        
        # Preprocess
        t_pre_0 = time.perf_counter()
        faces_array = face_processor.process_frames(frames)
        t_pre_1 = time.perf_counter()
        
        # Predict
        estimator = get_estimator()
        t_inf_0 = time.perf_counter()
        result = estimator.predict(faces_array)
        t_inf_1 = time.perf_counter()
        
        end_time = time.perf_counter()
        processing_time = (end_time - start_time) * 1000.0
        decode_ms = (t_decode_1 - t_decode_0) * 1000.0
        preprocess_ms = (t_pre_1 - t_pre_0) * 1000.0
        stage_inference_ms = (t_inf_1 - t_inf_0) * 1000.0
        forward_ms = float(result.get("benchmark", {}).get("forward_ms", 0.0))
        postprocess_ms = float(result.get("benchmark", {}).get("postprocess_ms", 0.0))
        inference_ms = float(result.get("benchmark", {}).get("inference_ms", stage_inference_ms))
        inference_fps = float(config.CLIP_LENGTH / max(inference_ms / 1000.0, 1e-8))
        pipeline_fps = float(config.CLIP_LENGTH / max(processing_time / 1000.0, 1e-8))
        resources = _resource_snapshot()
        benchmark = {
            "decode_ms": decode_ms,
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "forward_ms": forward_ms,
            "postprocess_ms": postprocess_ms,
            "end_to_end_ms": processing_time,
            "inference_fps": inference_fps,
            "pipeline_fps": pipeline_fps,
            **resources,
        }
        _record_benchmark({"endpoint": "realtime", **benchmark})
        
        return HeartRateResponse(
            success=True,
            heart_rate=result['heart_rate'],
            confidence=result['confidence'],
            snr_db=result['snr_db'],
            unit=result['unit'],
            processing_time_ms=processing_time,
            ppg_signal=result.get('ppg_signal'),
            benchmark=benchmark,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (time.perf_counter() - start_time) * 1000
        return HeartRateResponse(
            success=False,
            heart_rate=0,
            confidence=0,
            snr_db=0,
            processing_time_ms=processing_time,
            message=str(e)
        )


# ==================== WEBSOCKET FOR REAL-TIME ====================

class ConnectionManager:
    """Manage WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.frame_buffers: Dict[str, List[str]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.frame_buffers[client_id] = []
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.frame_buffers:
            del self.frame_buffers[client_id]
    
    async def send_json(self, client_id: str, data: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(data)


manager = ConnectionManager()


@app.websocket("/ws/realtime/{client_id}")
async def websocket_realtime(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time heart rate estimation
    
    Client sends frames continuously, server responds when enough frames collected
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get('type') == 'frame':
                # Add frame to buffer
                frame_data = data.get('frame')
                if frame_data:
                    manager.frame_buffers[client_id].append(frame_data)
                    
                    # Send progress update
                    buffer_size = len(manager.frame_buffers[client_id])
                    await manager.send_json(client_id, {
                        'type': 'progress',
                        'frames_collected': buffer_size,
                        'frames_needed': config.CLIP_LENGTH,
                        'progress': min(buffer_size / config.CLIP_LENGTH, 1.0)
                    })
                    
                    # Process when enough frames
                    if buffer_size >= config.CLIP_LENGTH:
                        await manager.send_json(client_id, {
                            'type': 'processing',
                            'message': 'Analyzing heart rate...'
                        })
                        
                        # Get frames and clear buffer
                        frames_to_process = manager.frame_buffers[client_id][:config.CLIP_LENGTH]
                        manager.frame_buffers[client_id] = []
                        
                        try:
                            # Decode and process
                            frames = decode_base64_frames(frames_to_process)
                            
                            if len(frames) >= config.CLIP_LENGTH:
                                faces_array = face_processor.process_frames(frames)
                                
                                estimator = get_estimator()
                                result = estimator.predict(faces_array)
                                
                                await manager.send_json(client_id, {
                                    'type': 'result',
                                    'success': True,
                                    'heart_rate': result['heart_rate'],
                                    'confidence': result['confidence'],
                                    'unit': 'BPM',
                                    'ppg_signal': result.get('ppg_signal', [])[:50]  # Send partial PPG
                                })
                            else:
                                await manager.send_json(client_id, {
                                    'type': 'error',
                                    'message': 'Not enough valid frames'
                                })
                        except Exception as e:
                            await manager.send_json(client_id, {
                                'type': 'error',
                                'message': str(e)
                            })
            
            elif data.get('type') == 'reset':
                # Clear buffer
                manager.frame_buffers[client_id] = []
                await manager.send_json(client_id, {
                    'type': 'reset_confirmed'
                })
            
            elif data.get('type') == 'ping':
                await manager.send_json(client_id, {'type': 'pong'})
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        manager.disconnect(client_id)


# ==================== STARTUP ====================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("=" * 60)
    print("  HEART RATE ESTIMATION WEB API")
    print("=" * 60)
    
    try:
        get_estimator()
        print("  Model loaded successfully!")
    except Exception as e:
        print(f"  Warning: Could not load model: {e}")
    
    print(f"  Device: {config.DEVICE}")
    print(f"  CORS Origins: {config.CORS_ORIGINS}")
    print("=" * 60)

# ==================== SERVE FRONTEND ====================

# This must be at the end so it doesn't override API routes
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

    @app.exception_handler(404)
    async def not_found_exception_handler(request, exc):
        """Fallback to index.html for React SPA routing"""
        return FileResponse("static/index.html")

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG
    )

