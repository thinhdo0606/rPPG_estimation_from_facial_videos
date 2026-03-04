"""
Preprocessing module for video frames
Handles face detection and frame normalization
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional
import base64
from .config import config


class FaceProcessor:
    """Face detection and preprocessing using MediaPipe"""
    
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        self.face_size = config.FACE_SIZE
        
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in frame using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            h, w = frame.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            
            return (x, y, width, height)
        
        return None
    
    def check_face_in_oval(self, frame: np.ndarray, oval_bounds: dict) -> Tuple[bool, float]:
        """
        Check if face is within the oval guide area
        
        Args:
            frame: BGR image
            oval_bounds: {cx, cy, rx, ry} - center and radii of oval
            
        Returns:
            (is_inside, overlap_ratio)
        """
        bbox = self.detect_face(frame)
        if bbox is None:
            return False, 0.0
        
        x, y, w, h = bbox
        face_cx = x + w // 2
        face_cy = y + h // 2
        
        # Check if face center is inside oval
        cx = oval_bounds.get('cx', frame.shape[1] // 2)
        cy = oval_bounds.get('cy', frame.shape[0] // 2)
        rx = oval_bounds.get('rx', 100)
        ry = oval_bounds.get('ry', 130)
        
        # Normalized distance from center
        dx = (face_cx - cx) / rx
        dy = (face_cy - cy) / ry
        
        # Inside oval if (dx^2 + dy^2) <= 1
        distance = dx**2 + dy**2
        is_inside = distance <= 1.2  # Allow some tolerance
        
        overlap_ratio = max(0, 1 - distance)
        
        return is_inside, overlap_ratio
    
    def crop_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop and resize face region"""
        x, y, w, h = bbox
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (self.face_size, self.face_size))
        return face_resized
    
    def process_frames(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Process list of frames for model input"""
        if len(frames) < config.CLIP_LENGTH:
            raise ValueError(f"Need at least {config.CLIP_LENGTH} frames, got {len(frames)}")
        
        frames = frames[:config.CLIP_LENGTH]
        
        cropped_faces = []
        last_bbox = None
        
        for frame in frames:
            bbox = self.detect_face(frame)
            
            if bbox is None:
                if last_bbox is None:
                    h, w = frame.shape[:2]
                    size = min(h, w) // 2
                    x = (w - size) // 2
                    y = (h - size) // 2
                    bbox = (x, y, size, size)
                else:
                    bbox = last_bbox
            else:
                last_bbox = bbox
            
            face = self.crop_face(frame, bbox)
            cropped_faces.append(face)
        
        faces_array = np.array(cropped_faces, dtype=np.float32)
        faces_array = faces_array / 255.0
        
        diff_frames = np.diff(faces_array, axis=0)
        mean_frame = np.mean(faces_array, axis=0, keepdims=True)
        
        diff_frames = np.transpose(diff_frames, (0, 3, 1, 2))
        mean_frame = np.transpose(mean_frame, (0, 3, 1, 2))
        
        return diff_frames, mean_frame
    
    def __del__(self):
        if hasattr(self, 'face_detection'):
            self.face_detection.close()


def decode_video_bytes(video_bytes: bytes) -> List[np.ndarray]:
    """Decode video from bytes to list of frames"""
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name
    
    try:
        frames = []
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return frames, fps
        
    finally:
        os.unlink(temp_path)


def decode_base64_frame(b64_str: str) -> Optional[np.ndarray]:
    """Decode single base64 encoded frame"""
    try:
        if ',' in b64_str:
            b64_str = b64_str.split(',')[1]
        
        img_bytes = base64.b64decode(b64_str)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return frame
    except:
        return None


def decode_base64_frames(frames_data: List[str]) -> List[np.ndarray]:
    """Decode base64 encoded frames"""
    frames = []
    for b64_str in frames_data:
        frame = decode_base64_frame(b64_str)
        if frame is not None:
            frames.append(frame)
    return frames

