import os
import cv2
import numpy as np
import sys
from pathlib import Path

# Thêm đường dẫn để có thể import các module của backend
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir))

from app.preprocessing import FaceProcessor
from app.model_service import get_estimator
from app.config import config

def calculate_ground_truth(gt_path):
    """Tính trung bình nhịp tim từ file ground truth"""
    try:
        with open(gt_path, 'r') as f:
            lines = f.readlines()
            
        hr_values = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    hr_values.append(float(line))
                except ValueError:
                    pass
                    
        if len(hr_values) > 0:
            return np.mean(hr_values)
        return None
    except Exception as e:
        print(f"Error reading ground truth {gt_path}: {e}")
        return None

def extract_frames(video_path, max_frames=config.CLIP_LENGTH):
    """Trích xuất frames từ video (giới hạn bằng CLIP_LENGTH để xử lý nhanh)"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
        
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
    cap.release()
    return frames, fps

def evaluate_mmse_dataset(dataset_dir):
    print(f"Evaluating dataset at: {dataset_dir}")
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print("MMSE_dataset directory not found!")
        return

    # Initialize model and processor
    face_processor = FaceProcessor()
    estimator = get_estimator()
    
    results = []
    
    # Iterate through subject folders
    for subject_dir in sorted(dataset_path.iterdir()):
        if not subject_dir.is_dir():
            continue
            
        video_path = subject_dir / "video.avi"
        gt_path = subject_dir / "Pulse Rate_BPM.txt"
        
        if not video_path.exists() or not gt_path.exists():
            continue
            
        print(f"\n--- Processing: {subject_dir.name} ---")
        
        # 1. Read Ground Truth
        gt_hr = calculate_ground_truth(gt_path)
        if gt_hr is None:
            print("Failed to read Ground Truth.")
            continue
            
        # 2. Process Video
        print(f"Reading video {video_path.name}...")
        frames, fps = extract_frames(video_path)
        print(f"Read {len(frames)} frames. FPS: {fps:.1f}")
        
        if len(frames) < config.CLIP_LENGTH:
            print(f"Video too short ({len(frames)} frames). Required: {config.CLIP_LENGTH}")
            continue
            
        # 3. Predict
        try:
            print("Processing faces...")
            faces_array = face_processor.process_frames(frames)
            
            print("Predicting heart rate...")
            result = estimator.predict(faces_array)
            pred_hr = result['heart_rate']
            confidence = result['confidence'] * 100
            
            # Error calculation
            error = abs(pred_hr - gt_hr)
            results.append({
                'subject': subject_dir.name,
                'gt_hr': gt_hr,
                'pred_hr': pred_hr,
                'error': error,
                'confidence': confidence
            })
            
            print(f"Ground Truth: {gt_hr:.2f} BPM")
            print(f"Prediction  : {pred_hr:.2f} BPM (Confidence: {confidence:.1f}%)")
            print(f"Absolute Err: {error:.2f} BPM")
            
        except Exception as e:
            print(f"Error processing {subject_dir.name}: {e}")
            
    # Summary
    if len(results) > 0:
        print("\n" + "="*50)
        print("EVALUATION SUMMARY ON MMSE_DATASET")
        print("="*50)
        print(f"{'Subject':<15} | {'Ground Truth':<15} | {'Prediction':<15} | {'Error (AE)':<10}")
        print("-" * 65)
        
        total_error = 0
        for r in results:
            print(f"{r['subject']:<15} | {r['gt_hr']:<15.2f} | {r['pred_hr']:<15.2f} | {r['error']:<10.2f}")
            total_error += r['error']
            
        mae = total_error / len(results)
        print("-" * 65)
        print(f"Mean Absolute Error (MAE): {mae:.2f} BPM")
        print("="*50)

if __name__ == "__main__":
    # dataset_dir path
    dataset_dir = os.path.join(str(backend_dir.parent), "MMSE_dataset")
    evaluate_mmse_dataset(dataset_dir)
