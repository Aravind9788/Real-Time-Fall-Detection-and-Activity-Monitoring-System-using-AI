"""
YOLO Detection & Activity Classification Module

This module handles person detection using YOLOv11n/YOLOv8n pose estimation and
classifies activities (walking, sitting, sleeping, lying, falling) based on
enhanced geometric rules and velocity calculations.
"""

import numpy as np
import torch
import os
from ultralytics import YOLO
from modules.fall_detection_config import (
    MIN_KEYPOINT_CONFIDENCE,
    MIN_CRITICAL_JOINTS_CONFIDENCE,
    FALL_VELOCITY_THRESHOLD,
    SLOW_MOVEMENT_THRESHOLD,
    STANDING_MOVEMENT_THRESHOLD,
    LYING_ANGLE_THRESHOLD,
    MINOR_FALL_ANGLE_THRESHOLD,
    YOLO_MODEL_PATH,
    YOLO_FALLBACK_PATH,
    YOLO_CONFIDENCE_THRESHOLD
)


# Initialize YOLO model with YOLOv11n or fallback to YOLOv8n
model = None

def load_model():
    """Lazy load YOLO model when needed"""
    global model
    if model is not None:
        return model
    
    if os.path.exists(YOLO_MODEL_PATH):
        try:
            model = YOLO(YOLO_MODEL_PATH)
            print(f"✓ Loaded {YOLO_MODEL_PATH}")
        except Exception as e:
            print(f"Warning: Failed to load {YOLO_MODEL_PATH}: {e}")
            model = None
    
    if model is None and os.path.exists(YOLO_FALLBACK_PATH):
        try:
            model = YOLO(YOLO_FALLBACK_PATH)
            print(f"✓ Loaded fallback model {YOLO_FALLBACK_PATH}")
        except Exception as e:
            print(f"Warning: Failed to load fallback model: {e}")
            model = None
    
    if model is None:
        print(f"⚠ Warning: No YOLO model found. Detection functions can still be tested.")
        return None
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model.to('cuda')
        print("✓ YOLO model moved to CUDA")
    else:
        print("⚠ YOLO running on CPU")
    
    return model

# Try to load model on import, but don't fail if it doesn't exist
try:
    model = load_model()
except Exception as e:
    print(f"⚠ Warning: Could not load YOLO model: {e}")
    print("Detection classification functions can still be used for testing.")
    model = None


def classify_activity(keypoints, conf, movement_speed=None, vertical_velocity=None, geometric_features=None, com_velocity=None):
    """
    Activity classification using geometric rules.
    
    NOTE: Fall detection is now handled exclusively by the custom YOLO model (best.pt).
    This function only classifies non-fall activities: STANDING, WALKING, SITTING, SLEEPING, LYING
    
    Args:
        keypoints: Array of body keypoints from YOLO pose estimation (17, 2)
        conf: Confidence scores for each keypoint (17,)
        movement_speed: Optional horizontal movement speed in pixels/second
        vertical_velocity: Optional vertical velocity of hip (positive=down, negative=up)
        geometric_features: Optional dict with pre-calculated geometric features
        com_velocity: Optional center-of-mass velocity dict
        
    Returns:
        str: Activity classification (STANDING, WALKING, SITTING, SLEEPING, LYING, UNKNOWN)
    """
    from modules.fall_detection_config import (
        COM_FALL_VELOCITY_THRESHOLD, LATERAL_FALL_ANGLE_THRESHOLD,
        LIMB_EXTENSION_THRESHOLD, ALIGNMENT_TWIST_THRESHOLD
    )
    
    try:
        # ==================== STEP 1: Confidence Validation ====================
        critical_joints = [5, 6, 11, 12]  # Left/Right Shoulders and Hips
        visible_critical = sum(1 for j in critical_joints if conf[j] >= MIN_CRITICAL_JOINTS_CONFIDENCE)
        
        if visible_critical < 2:
            # Fallback: Check if nose and at least one hip visible for lying detection
            if conf[0] >= MIN_KEYPOINT_CONFIDENCE and (conf[11] >= MIN_KEYPOINT_CONFIDENCE or conf[12] >= MIN_KEYPOINT_CONFIDENCE):
                nose_y = keypoints[0][1]
                hip_y = keypoints[11][1] if conf[11] >= MIN_KEYPOINT_CONFIDENCE else keypoints[12][1]
                if nose_y >= hip_y - 10:
                    return "LYING"
            return "UNKNOWN"
        
        # ==================== STEP 2: Extract/Calculate Geometric Features ====================
        if geometric_features is not None:
            body_angle = geometric_features.get('body_angle', 0)
            angle_valid = geometric_features.get('body_angle_valid', False)
            torso_length = geometric_features.get('torso_length', 0)
            shoulder_alignment = geometric_features.get('shoulder_alignment', 0)
            shoulder_valid = geometric_features.get('shoulder_alignment_valid', False)
            hip_alignment = geometric_features.get('hip_alignment', 0)
            hip_valid = geometric_features.get('hip_alignment_valid', False)
            limb_ext = geometric_features.get('limb_extension', {})
        else:
            # Calculate basic features manually if not provided
            sho_y = (keypoints[5][1] + keypoints[6][1]) / 2
            sho_x = (keypoints[5][0] + keypoints[6][0]) / 2
            hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
            hip_x = (keypoints[11][0] + keypoints[12][0]) / 2
            
            dy = hip_y - sho_y
            dx = hip_x - sho_x
            body_angle = abs(np.degrees(np.arctan2(dx, dy)))
            angle_valid = True
            torso_length = np.sqrt(dx**2 + dy**2)
            shoulder_alignment = 0
            shoulder_valid = False
            hip_alignment = 0
            hip_valid = False
            limb_ext = {'is_valid': False}
        
        if not angle_valid:
            return "UNKNOWN"
        
        # ==================== STEP 3: NON-FALL ACTIVITY CLASSIFICATION ====================
        # Fall detection is handled by the custom model (best.pt)
        # Geometric rules now ONLY classify: LYING, SITTING, STANDING, WALKING, SLEEPING
        
        # === HORIZONTAL / LYING DETECTION ===
        if body_angle > LYING_ANGLE_THRESHOLD:
            return "LYING"
        
        # ==================== STEP 5: TILTED BODY DETECTION ====================
        # Previously: Geometric fall detection was here (returned "MINOR FALL")
        # NOW: Fall detection is EXCLUSIVELY handled by custom model (best.pt) with velocity filtering
        # Tilted bodies are classified as LYING - state manager will handle if it's actually a fall
        if body_angle > MINOR_FALL_ANGLE_THRESHOLD:
            # Check if head is below feet (typical lying indicator)
            if conf[0] >= MIN_KEYPOINT_CONFIDENCE:
                nose_y = keypoints[0][1]
                # Find lowest foot position
                left_ankle_y = keypoints[15][1] if conf[15] >= MIN_KEYPOINT_CONFIDENCE else 0
                right_ankle_y = keypoints[16][1] if conf[16] >= MIN_KEYPOINT_CONFIDENCE else 0
                max_foot_y = max(left_ankle_y, right_ankle_y)
                
                if max_foot_y > 0 and nose_y > max_foot_y + 20:
                    # Head below feet = lying
                    return "LYING"
            
            # Significantly tilted body = classify as LYING
            # Model will determine if it's a fall (with velocity filtering)
            return "LYING"
        
        # ==================== STEP 6: SITTING VS STANDING/WALKING ====================
        # Sitting detection based on leg geometry
        if conf[13] >= MIN_KEYPOINT_CONFIDENCE and conf[15] >= MIN_KEYPOINT_CONFIDENCE:
            # Left leg check
            left_knee_y = keypoints[13][1]
            left_ankle_y = keypoints[15][1]
            left_hip_y = keypoints[11][1] if conf[11] >= MIN_KEYPOINT_CONFIDENCE else 1e9
            
            # Sitting: knees roughly at hip level, ankles below knees
            if left_hip_y < 1e9:
                knee_hip_diff = abs(left_knee_y - left_hip_y)
                if knee_hip_diff < torso_length * 0.6 and left_ankle_y > left_knee_y - 20:
                    return "SITTING"
        
        # Check torso-to-height ratio (sitting people have shorter apparent height)
        if conf[0] >= MIN_KEYPOINT_CONFIDENCE and conf[15] >= MIN_KEYPOINT_CONFIDENCE:
            nose_y = keypoints[0][1]
            ankle_y = keypoints[15][1]
            total_height = abs(ankle_y - nose_y)
            
            if total_height > 0 and torso_length / total_height > 0.65:
                # Torso takes up most of visible height = sitting
                return "SITTING"
        
        # Fallback sitting detection: check if hips are close to knees
        if conf[11] >= MIN_KEYPOINT_CONFIDENCE and conf[13] >= MIN_KEYPOINT_CONFIDENCE:
            hip_knee_dist = abs(keypoints[11][1] - keypoints[13][1])
            if hip_knee_dist < torso_length * 0.5:
                return "SITTING"
        
        # ==================== STEP 7: STANDING VS WALKING ====================
        if movement_speed is not None:
            if movement_speed < STANDING_MOVEMENT_THRESHOLD:
                return "STANDING"
            else:
                return "WALKING"
        
        # Default to standing
        return "STANDING"
        
    except Exception as e:
        print(f"Error in classify_activity: {e}")
        import traceback
        traceback.print_exc()
        return "UNKNOWN"


def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
