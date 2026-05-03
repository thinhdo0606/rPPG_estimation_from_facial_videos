"""
Preprocessing module for video frames
Handles face detection and frame normalization
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional
import base64
from pathlib import Path
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
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            cropped_faces.append(face)
        
        faces_array = np.array(cropped_faces, dtype=np.float32)
        # The images are already [0, 255] because they come from cv2.
        
        # (T, H, W, C) -> (T, C, H, W)
        faces_array = np.transpose(faces_array, (0, 3, 1, 2))
        
        return faces_array
    
    def __del__(self):
        if hasattr(self, 'face_detection'):
            self.face_detection.close()


def _resize_frame_max_edge(frame: np.ndarray, max_edge: int) -> np.ndarray:
    """Resize so longest side <= max_edge (keeps aspect ratio)."""
    h, w = frame.shape[:2]
    m = max(h, w)
    if m <= max_edge:
        return frame
    scale = max_edge / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _video_temp_suffix(original_name: str = "") -> str:
    suf = Path(original_name or "").suffix.lower()
    if suf in (".mp4", ".avi", ".mov", ".webm", ".mkv", ".m4v"):
        return suf
    return ".mp4"


def extract_first_frame_jpeg_base64(video_bytes: bytes, original_name: str = "") -> Optional[str]:
    """
    Read the first decodable frame via OpenCV and return JPEG as raw base64 (no data-URL prefix).
    Uses the file extension from original_name so codecs (e.g. AVI) are probed correctly.
    """
    import tempfile
    import os

    suffix = _video_temp_suffix(original_name)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        cap = cv2.VideoCapture(temp_path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        edge = int(config.VIDEO_DECODE_MAX_EDGE)
        if edge > 0:
            frame = _resize_frame_max_edge(frame, edge)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    finally:
        os.unlink(temp_path)


def decode_video_bytes(video_bytes: bytes, original_name: str = ""):
    """
    Decode video from bytes to a list of frames and FPS.

    - Reads at most config.MAX_VIDEO_DURATION_SEC worth of frames (plus one read
      after cap may report low FPS) to cap RAM for long files.
    - Downscales each frame if wider/taller than config.VIDEO_DECODE_MAX_EDGE
      before face detection / stacking.
    """
    import tempfile
    import os

    suffix = _video_temp_suffix(original_name)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(video_bytes)
        temp_path = f.name

    try:
        frames = []
        cap = cv2.VideoCapture(temp_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3 or fps > 240.0:
            fps = float(config.FRAME_RATE)

        max_sec = float(config.MAX_VIDEO_DURATION_SEC)
        max_frames = int(max_sec * fps) + 2
        max_frames = max(max_frames, config.CLIP_LENGTH + 1)
        max_frames = min(max_frames, int(max_sec * 120) + 2)

        edge = int(config.VIDEO_DECODE_MAX_EDGE)
        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if edge > 0:
                frame = _resize_frame_max_edge(frame, edge)
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

