"""
FastAPI Backend for Heart Rate Estimation Web App
Supports video upload and real-time webcam
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import time
import json
import asyncio

from .config import config
from .preprocessing import (
    FaceProcessor, 
    decode_video_bytes, 
    decode_base64_frames,
    decode_base64_frame
)
from .model_service import get_estimator

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


# ==================== DATA MODELS ====================

class FramesRequest(BaseModel):
    frames: List[str]
    fps: Optional[float] = 30.0


class HeartRateResponse(BaseModel):
    success: bool
    heart_rate: float
    confidence: float
    unit: str = "BPM"
    processing_time_ms: float
    ppg_signal: Optional[List[float]] = None
    message: Optional[str] = None


class VideoAnalysisResponse(BaseModel):
    success: bool
    heart_rate: float
    confidence: float
    unit: str = "BPM"
    video_duration: float
    fps: float
    total_frames: int
    processing_time_ms: float
    ppg_signal: Optional[List[float]] = None
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


@app.post("/api/predict/video", response_model=VideoAnalysisResponse)
async def predict_from_video(video: UploadFile = File(...)):
    """
    Analyze uploaded video file for heart rate
    
    - Upload a video file (mp4, avi, mov)
    - Video should be 5-30 seconds
    - Face should be visible throughout
    """
    start_time = time.time()
    
    try:
        # Validate file type
        allowed_types = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo']
        if video.content_type and video.content_type not in allowed_types:
            # Still try to process, some browsers report wrong MIME type
            pass
        
        # Read video bytes
        video_bytes = await video.read()
        
        if len(video_bytes) > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(
                status_code=400,
                detail="Video file too large. Maximum size is 100MB."
            )
        
        # Decode video to frames
        frames, fps = decode_video_bytes(video_bytes)
        
        if len(frames) < config.CLIP_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Video too short. Need at least {config.CLIP_LENGTH} frames (~4 seconds at 30fps). Got {len(frames)} frames."
            )
        
        video_duration = len(frames) / fps if fps > 0 else 0
        
        # Preprocess
        diff_frames, mean_frame = face_processor.process_frames(frames)
        
        # Predict
        estimator = get_estimator()
        result = estimator.predict(diff_frames, mean_frame)
        
        processing_time = (time.time() - start_time) * 1000
        
        return VideoAnalysisResponse(
            success=True,
            heart_rate=result['heart_rate'],
            confidence=result['confidence'],
            unit=result['unit'],
            video_duration=video_duration,
            fps=fps,
            total_frames=len(frames),
            processing_time_ms=processing_time,
            ppg_signal=result.get('ppg_signal')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        return VideoAnalysisResponse(
            success=False,
            heart_rate=0,
            confidence=0,
            video_duration=0,
            fps=0,
            total_frames=0,
            processing_time_ms=processing_time,
            message=str(e)
        )


@app.post("/api/predict/realtime", response_model=HeartRateResponse)
async def predict_realtime(request: FramesRequest):
    """
    Predict heart rate from real-time captured frames
    
    Send 128+ frames captured from webcam at ~30 FPS
    """
    start_time = time.time()
    
    try:
        if len(request.frames) < config.CLIP_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {config.CLIP_LENGTH} frames, got {len(request.frames)}"
            )
        
        # Decode frames
        frames = decode_base64_frames(request.frames)
        
        if len(frames) < config.CLIP_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode enough frames. Got {len(frames)} valid frames"
            )
        
        # Preprocess
        diff_frames, mean_frame = face_processor.process_frames(frames)
        
        # Predict
        estimator = get_estimator()
        result = estimator.predict(diff_frames, mean_frame)
        
        processing_time = (time.time() - start_time) * 1000
        
        return HeartRateResponse(
            success=True,
            heart_rate=result['heart_rate'],
            confidence=result['confidence'],
            unit=result['unit'],
            processing_time_ms=processing_time,
            ppg_signal=result.get('ppg_signal')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        return HeartRateResponse(
            success=False,
            heart_rate=0,
            confidence=0,
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
                                diff_frames, mean_frame = face_processor.process_frames(frames)
                                
                                estimator = get_estimator()
                                result = estimator.predict(diff_frames, mean_frame)
                                
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


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.DEBUG
    )

