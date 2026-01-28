"""
YOLO Detection & Activity Classification Module

This module handles person detection using YOLOv8 pose estimation and
classifies activities (walking, sitting, sleeping, lying, falling) based
on skeleton keypoints.
"""

import numpy as np
from ultralytics import YOLO


# Initialize YOLO model
model = YOLO("yolov8n-pose.pt")  # small & fast
FALL_URL = "http://127.0.0.1:5000/trigger"


def classify_activity(keypoints, conf, movement_speed=None, vertical_velocity=None):
    """
    Advanced skeleton-based activity classification with velocity-based fall detection.
    
    Args:
        keypoints: Array of body keypoints from YOLO pose estimation
        conf: Confidence scores for each keypoint
        movement_speed: Optional horizontal movement speed in pixels/second
        vertical_velocity: Optional vertical velocity of head/shoulders (positive=down, negative=up)
        
    Returns:
        str: Activity classification (STANDING, WALKING, SITTING, SLEEPING, LYING, MINOR FALL, UNKNOWN)
    """
    try:
        # Check confidence of critical joints
        critical_joints = [5, 6, 11, 12] # Shoulders and Hips
        if any(conf[j] < 0.5 for j in critical_joints):
            if conf[0] > 0.5 and (conf[11] > 0.5 or conf[12] > 0.5):
                nose_y = keypoints[0][1]
                hip_y = (keypoints[11][1] + keypoints[12][1]) / 2 if (conf[11] > 0.5 and conf[12] > 0.5) else (keypoints[11][1] if conf[11] > 0.5 else keypoints[12][1])
                if nose_y > hip_y - 10:
                    return "LYING"
            return "UNKNOWN"

        # Calculate Midpoints and distances
        sho_y = (keypoints[5][1] + keypoints[6][1]) / 2
        sho_x = (keypoints[5][0] + keypoints[6][0]) / 2
        hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
        hip_x = (keypoints[11][0] + keypoints[12][0]) / 2
        
        dy = hip_y - sho_y
        dx = hip_x - sho_x
        angle = abs(np.degrees(np.arctan2(dx, dy)))
        
        # Height and width for ratio
        torso_len = np.sqrt(dx**2 + dy**2)
        sho_width = abs(keypoints[5][0] - keypoints[6][0])
        
        # ==================== VELOCITY-BASED FALL DETECTION ====================
        # CRITICAL: Check vertical velocity FIRST to detect actual falls
        # A real fall has HIGH downward velocity (gravity acceleration)
        # Thresholds (pixels/second):
        #   - Slow lying down: 0-80 px/s
        #   - Fast movement/fall: > 150 px/s (gravity = ~300-500 px/s at 640x480)
        
        if vertical_velocity is not None and vertical_velocity > 150:
            # HIGH downward velocity = ACTUAL FALL
            # Even if not fully horizontal yet, this is a fall in progress
            print(f"⚠️ FALL DETECTED: Vertical velocity = {vertical_velocity:.1f} px/s")
            return "MINOR FALL"
        
        # If very horizontal but LOW velocity, it's lying down, not falling
        if vertical_velocity is not None and angle > 60:
            if vertical_velocity < 80:
                # Slow descent = lying down intentionally
                return "LYING"
            else:
                # Fast descent + horizontal = fall that already landed
                return "MINOR FALL"
        
        # 1. LYING (Horizontal and low, but only if no high velocity was detected above)
        # Lowered threshold to 60 degrees to be more sensitive to lying down
        if angle > 60:
            return "LYING"
        
        # 2. MINOR FALL (Significant tilt but not fully flat, or head dropped)
        # Only trigger if we see abnormal posture without velocity data
        if angle > 40:
            # If ankles/knees are visible and head is very low
            if conf[15] > 0.5 and keypoints[0][1] > keypoints[11][1]:
                return "LYING"
            # Only return MINOR FALL here if we don't have velocity data
            # (if we had velocity data, we would have caught it above)
            if vertical_velocity is None:
                return "MINOR FALL"
            else:
                # We have velocity data and it's low, so this is controlled movement
                return "STANDING"

        # 3. SITTING vs WALKING/STANDING
        # Sitting: Vertical torso, but hips are lower (shorter total height)
        if conf[13] > 0.5 and conf[14] > 0.5 and conf[15] > 0.5:
            knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
            ank_y = (keypoints[15][1] + keypoints[16][1]) / 2
            
            upper_leg = abs(knee_y - hip_y)
            lower_leg = abs(ank_y - knee_y)
            
            # If knees are at similar horizontal level to hips, they are likely sitting
            if upper_leg < lower_leg * 0.5:
                return "SITTING"
                
        # Fallback for sitting: Torso length relative to total visible height
        if conf[15] > 0.5:
            total_h = abs(keypoints[15][1] - sho_y)
            if torso_len / total_h > 0.6: # Torso takes up too much of height
                return "SITTING"

        # 4. STANDING vs WALKING - Use movement speed if provided
        if movement_speed is not None:
            # Threshold: < 15 pixels/sec is STANDING, >= 15 is WALKING
            # At 640x480 resolution, walking is typically 30-100 pixels/sec
            if movement_speed < 15:
                return "STANDING"
        
        return "WALKING"

    except Exception:
        return "UNKNOWN"


def format_duration(seconds):
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration (e.g., "5m 30s")
    """
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"
