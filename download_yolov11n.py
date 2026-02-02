"""
Download YOLOv11n Pose Model

This script downloads the YOLOv11n pose estimation model from Ultralytics.
"""

from ultralytics import YOLO
import os

def download_yolov11n_pose():
    """Download YOLOv11n pose model"""
    model_path = "yolov11n-pose.pt"
    
    if os.path.exists(model_path):
        print(f"✓ Model already exists at {model_path}")
        return True
    
    try:
        print("Downloading YOLOv11n-pose model...")
        # This will automatically download the model
        model = YOLO("yolo11n-pose.pt")
        print(f"✓ Successfully downloaded YOLOv11n-pose model")
        return True
    except Exception as e:
        print(f"❌ Failed to download YOLOv11n-pose: {e}")
        print("The system will fall back to YOLOv8n-pose if available")
        return False

if __name__ == "__main__":
    download_yolov11n_pose()
