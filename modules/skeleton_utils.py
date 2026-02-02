"""
Skeleton Keypoint Utilities

Provides normalization and geometric calculations for YOLO pose keypoints.
"""

import numpy as np


def normalize_keypoints(keypoints, confidences, frame_width, frame_height):
    """
    Normalize keypoints relative to frame dimensions and hip center.
    
    Args:
        keypoints: (17, 2) array of (x, y) coordinates
        confidences: (17,) array of confidence scores
        frame_width: Width of the frame
        frame_height: Height of the frame
    
    Returns:
        tuple: (normalized_keypoints, is_valid) where normalized_keypoints is (17, 2)
               and is_valid is a boolean indicating if normalization succeeded
    """
    from modules.fall_detection_config import MIN_KEYPOINT_CONFIDENCE
    
    # Check if critical joints are present
    left_hip_idx, right_hip_idx = 11, 12
    
    if confidences[left_hip_idx] < MIN_KEYPOINT_CONFIDENCE and confidences[right_hip_idx] < MIN_KEYPOINT_CONFIDENCE:
        return keypoints, False
    
    # Calculate hip center
    hip_center, hip_valid = calculate_hip_center(keypoints, confidences)
    if not hip_valid:
        return keypoints, False
    
    # Normalize coordinates (0-1 range)
    normalized = keypoints.copy()
    normalized[:, 0] = keypoints[:, 0] / frame_width
    normalized[:, 1] = keypoints[:, 1] / frame_height
    
    return normalized, True


def calculate_hip_center(keypoints, confidences):
    """
    Calculate the center point between left and right hips.
    
    Args:
        keypoints: (17, 2) array of (x, y) coordinates
        confidences: (17,) array of confidence scores
    
    Returns:
        tuple: (hip_center, is_valid) where hip_center is (x, y) and is_valid is boolean
    """
    from modules.fall_detection_config import MIN_KEYPOINT_CONFIDENCE
    
    left_hip_idx, right_hip_idx = 11, 12
    
    left_hip_conf = confidences[left_hip_idx]
    right_hip_conf = confidences[right_hip_idx]
    
    # Both hips visible
    if left_hip_conf >= MIN_KEYPOINT_CONFIDENCE and right_hip_conf >= MIN_KEYPOINT_CONFIDENCE:
        hip_center = (keypoints[left_hip_idx] + keypoints[right_hip_idx]) / 2
        return hip_center, True
    
    # Only one hip visible
    if left_hip_conf >= MIN_KEYPOINT_CONFIDENCE:
        return keypoints[left_hip_idx], True
    if right_hip_conf >= MIN_KEYPOINT_CONFIDENCE:
        return keypoints[right_hip_idx], True
    
    # No hips visible
    return np.array([0, 0]), False


def calculate_body_angle(keypoints, confidences):
    """
    Calculate the angle of the body relative to vertical (0° = upright, 90° = horizontal).
    
    Args:
        keypoints: (17, 2) array of (x, y) coordinates
        confidences: (17,) array of confidence scores
    
    Returns:
        tuple: (angle, is_valid) where angle is in degrees and is_valid is boolean
    """
    from modules.fall_detection_config import MIN_KEYPOINT_CONFIDENCE
    
    # Shoulder indices: 5 (left), 6 (right)
    # Hip indices: 11 (left), 12 (right)
    
    # Calculate shoulder center
    left_shoulder_conf = confidences[5]
    right_shoulder_conf = confidences[6]
    
    if left_shoulder_conf >= MIN_KEYPOINT_CONFIDENCE and right_shoulder_conf >= MIN_KEYPOINT_CONFIDENCE:
        shoulder_center = (keypoints[5] + keypoints[6]) / 2
    elif left_shoulder_conf >= MIN_KEYPOINT_CONFIDENCE:
        shoulder_center = keypoints[5]
    elif right_shoulder_conf >= MIN_KEYPOINT_CONFIDENCE:
        shoulder_center = keypoints[6]
    else:
        return 0.0, False
    
    # Calculate hip center
    hip_center, hip_valid = calculate_hip_center(keypoints, confidences)
    if not hip_valid:
        return 0.0, False
    
    # Calculate angle
    dx = hip_center[0] - shoulder_center[0]
    dy = hip_center[1] - shoulder_center[1]
    
    # Angle relative to vertical (y-axis)
    # atan2(dx, dy) gives angle from vertical
    angle = abs(np.degrees(np.arctan2(dx, dy)))
    
    return angle, True


def calculate_torso_length(keypoints, confidences):
    """
    Calculate the length of the torso (shoulder to hip distance).
    
    Args:
        keypoints: (17, 2) array of (x, y) coordinates
        confidences: (17,) array of confidence scores
    
    Returns:
        tuple: (torso_length, is_valid) where torso_length is in pixels and is_valid is boolean
    """
    from modules.fall_detection_config import MIN_KEYPOINT_CONFIDENCE, MIN_TORSO_LENGTH
    
    # Calculate shoulder center
    left_shoulder_conf = confidences[5]
    right_shoulder_conf = confidences[6]
    
    if left_shoulder_conf >= MIN_KEYPOINT_CONFIDENCE and right_shoulder_conf >= MIN_KEYPOINT_CONFIDENCE:
        shoulder_center = (keypoints[5] + keypoints[6]) / 2
    elif left_shoulder_conf >= MIN_KEYPOINT_CONFIDENCE:
        shoulder_center = keypoints[5]
    elif right_shoulder_conf >= MIN_KEYPOINT_CONFIDENCE:
        shoulder_center = keypoints[6]
    else:
        return 0.0, False
    
    # Calculate hip center
    hip_center, hip_valid = calculate_hip_center(keypoints, confidences)
    if not hip_valid:
        return 0.0, False
    
    # Calculate distance
    dx = hip_center[0] - shoulder_center[0]
    dy = hip_center[1] - shoulder_center[1]
    torso_length = np.sqrt(dx**2 + dy**2)
    
    # Check if torso is reasonable size
    if torso_length < MIN_TORSO_LENGTH:
        return torso_length, False
    
    return torso_length, True


def calculate_head_hip_distance(keypoints, confidences):
    """
    Calculate the distance between head (nose) and hip center.
    
    Args:
        keypoints: (17, 2) array of (x, y) coordinates
        confidences: (17,) array of confidence scores
    
    Returns:
        tuple: (distance, is_valid) where distance is in pixels and is_valid is boolean
    """
    from modules.fall_detection_config import MIN_KEYPOINT_CONFIDENCE
    
    # Nose index: 0
    if confidences[0] < MIN_KEYPOINT_CONFIDENCE:
        return 0.0, False
    
    nose = keypoints[0]
    
    # Calculate hip center
    hip_center, hip_valid = calculate_hip_center(keypoints, confidences)
    if not hip_valid:
        return 0.0, False
    
    # Calculate distance
    dx = hip_center[0] - nose[0]
    dy = hip_center[1] - nose[1]
    distance = np.sqrt(dx**2 + dy**2)
    
    return distance, True


def get_geometric_features(keypoints, confidences, frame_width, frame_height):
    """
    Extract all geometric features for fall detection.
    
    Args:
        keypoints: (17, 2) array of (x, y) coordinates
        confidences: (17,) array of confidence scores
        frame_width: Width of the frame
        frame_height: Height of the frame
    
    Returns:
        dict: Dictionary containing all geometric features with validity flags
    """
    features = {}
    
    # Normalize keypoints
    normalized_kp, norm_valid = normalize_keypoints(keypoints, confidences, frame_width, frame_height)
    features['normalized_keypoints'] = normalized_kp
    features['normalization_valid'] = norm_valid
    
    # Hip center
    hip_center, hip_valid = calculate_hip_center(keypoints, confidences)
    features['hip_center'] = hip_center
    features['hip_center_valid'] = hip_valid
    
    # Body angle
    angle, angle_valid = calculate_body_angle(keypoints, confidences)
    features['body_angle'] = angle
    features['body_angle_valid'] = angle_valid
    
    # Torso length
    torso_length, torso_valid = calculate_torso_length(keypoints, confidences)
    features['torso_length'] = torso_length
    features['torso_length_valid'] = torso_valid
    
    # Head-hip distance
    head_hip_dist, dist_valid = calculate_head_hip_distance(keypoints, confidences)
    features['head_hip_distance'] = head_hip_dist
    features['head_hip_distance_valid'] = dist_valid
    
    return features


def calculate_shoulder_alignment(keypoints, confidences):
    """
    Calculate shoulder line angle relative to horizontal.
    Helps detect side falls where shoulders are tilted.
    
    Args:
        keypoints: (17, 2) array of (x, y) coordinates
        confidences: (17,) array of confidence scores
    
    Returns:
        tuple: (angle, is_valid) where:
            - angle: 0° = shoulders level (upright), 90° = vertical (side fall)
            - is_valid: boolean indicating if calculation succeeded
    """
    from modules.fall_detection_config import MIN_KEYPOINT_CONFIDENCE
    
    left_shoulder_idx, right_shoulder_idx = 5, 6
    left_conf = confidences[left_shoulder_idx]
    right_conf = confidences[right_shoulder_idx]
    
    # Need both shoulders visible
    if left_conf < MIN_KEYPOINT_CONFIDENCE or right_conf < MIN_KEYPOINT_CONFIDENCE:
        return 0.0, False
    
    left_shoulder = keypoints[left_shoulder_idx]
    right_shoulder = keypoints[right_shoulder_idx]
    
    # Calculate angle of shoulder line relative to horizontal
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    
    # Angle from horizontal (0° = level, 90° = vertical)
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    
    return angle, True


def calculate_hip_alignment(keypoints, confidences):
    """
    Calculate hip line angle relative to horizontal.
    Helps detect side falls and body twisting.
    
    Args:
        keypoints: (17, 2) array of (x, y) coordinates
        confidences: (17,) array of confidence scores
    
    Returns:
        tuple: (angle, is_valid) where:
            - angle: 0° = hips level, 90° = vertical (side fall)
            - is_valid: boolean indicating if calculation succeeded
    """
    from modules.fall_detection_config import MIN_KEYPOINT_CONFIDENCE
    
    left_hip_idx, right_hip_idx = 11, 12
    left_conf = confidences[left_hip_idx]
    right_conf = confidences[right_hip_idx]
    
    # Need both hips visible
    if left_conf < MIN_KEYPOINT_CONFIDENCE or right_conf < MIN_KEYPOINT_CONFIDENCE:
        return 0.0, False
    
    left_hip = keypoints[left_hip_idx]
    right_hip = keypoints[right_hip_idx]
    
    # Calculate angle of hip line relative to horizontal
    dx = right_hip[0] - left_hip[0]
    dy = right_hip[1] - left_hip[1]
    
    # Angle from horizontal (0° = level, 90° = vertical)
    angle = abs(np.degrees(np.arctan2(dy, dx)))
    
    return angle, True


def calculate_limb_extension(keypoints, confidences):
    """
    Analyze limb positions to detect fall-related protective extensions.
    During falls, people often extend arms/legs outward for protection.
    
    Args:
        keypoints: (17, 2) array of (x, y) coordinates
        confidences: (17,) array of confidence scores
    
    Returns:
        dict: {
            'arms_extended': bool - Are arms away from body?
            'legs_spread': bool - Are legs spread apart?
            'extension_score': float - Overall extension (0-1, higher = more extended)
            'is_valid': bool - Is analysis valid?
        }
    """
    from modules.fall_detection_config import MIN_KEYPOINT_CONFIDENCE
    
    result = {
        'arms_extended': False,
        'legs_spread': False,
        'extension_score': 0.0,
        'is_valid': False
    }
    
    # Keypoint indices
    left_shoulder, right_shoulder = 5, 6
    left_elbow, right_elbow = 7, 8
    left_wrist, right_wrist = 9, 10
    left_hip, right_hip = 11, 12
    left_knee, right_knee = 13, 14
    left_ankle, right_ankle = 15, 16
    
    # Check shoulder/hip visibility for reference
    hip_center, hip_valid = calculate_hip_center(keypoints, confidences)
    if not hip_valid:
        return result
    
    # Analyze arm extension
    arm_extension_scores = []
    
    # Left arm
    if (confidences[left_shoulder] >= MIN_KEYPOINT_CONFIDENCE and 
        confidences[left_wrist] >= MIN_KEYPOINT_CONFIDENCE):
        shoulder_wrist_dist = np.linalg.norm(keypoints[left_wrist] - keypoints[left_shoulder])
        shoulder_wrist_dx = abs(keypoints[left_wrist][0] - keypoints[left_shoulder][0])
        # Extension = horizontal distance / total distance
        if shoulder_wrist_dist > 0:
            arm_extension_scores.append(shoulder_wrist_dx / shoulder_wrist_dist)
    
    # Right arm
    if (confidences[right_shoulder] >= MIN_KEYPOINT_CONFIDENCE and 
        confidences[right_wrist] >= MIN_KEYPOINT_CONFIDENCE):
        shoulder_wrist_dist = np.linalg.norm(keypoints[right_wrist] - keypoints[right_shoulder])
        shoulder_wrist_dx = abs(keypoints[right_wrist][0] - keypoints[right_shoulder][0])
        if shoulder_wrist_dist > 0:
            arm_extension_scores.append(shoulder_wrist_dx / shoulder_wrist_dist)
    
    # Analyze leg spread
    leg_spread_score = 0.0
    if (confidences[left_hip] >= MIN_KEYPOINT_CONFIDENCE and 
        confidences[right_hip] >= MIN_KEYPOINT_CONFIDENCE and
        confidences[left_ankle] >= MIN_KEYPOINT_CONFIDENCE and
        confidences[right_ankle] >= MIN_KEYPOINT_CONFIDENCE):
        
        hip_width = np.linalg.norm(keypoints[right_hip] - keypoints[left_hip])
        ankle_width = np.linalg.norm(keypoints[right_ankle] - keypoints[left_ankle])
        
        # Legs are spread if ankles are wider than hips
        if hip_width > 0:
            leg_spread_score = min(ankle_width / hip_width, 2.0) / 2.0  # Cap at 2x hip width
            result['legs_spread'] = leg_spread_score > 0.7
    
    # Calculate overall extension score
    if arm_extension_scores:
        avg_arm_extension = np.mean(arm_extension_scores)
        result['arms_extended'] = avg_arm_extension > 0.5
        result['extension_score'] = (avg_arm_extension * 0.6 + leg_spread_score * 0.4)
        result['is_valid'] = True
    elif leg_spread_score > 0:
        result['extension_score'] = leg_spread_score
        result['is_valid'] = True
    
    return result


def calculate_center_of_mass(keypoints, confidences):
    """
    Calculate approximate center of mass from all visible keypoints.
    More stable than single-point tracking.
    
    Args:
        keypoints: (17, 2) array of (x, y) coordinates
        confidences: (17,) array of confidence scores
    
    Returns:
        tuple: (center_of_mass, is_valid) where center_of_mass is (x, y)
    """
    from modules.fall_detection_config import MIN_KEYPOINT_CONFIDENCE
    
    # Weight different body parts (core > limbs)
    weights = np.array([
        0.5,   # 0: Nose
        0.3, 0.3, 0.3, 0.3,  # 1-4: Eyes, Ears
        1.5, 1.5,  # 5-6: Shoulders (heavy weight)
        1.0, 1.0,  # 7-8: Elbows
        0.5, 0.5,  # 9-10: Wrists
        2.0, 2.0,  # 11-12: Hips (heaviest weight - core)
        1.0, 1.0,  # 13-14: Knees
        0.5, 0.5   # 15-16: Ankles
    ])
    
    # Filter by confidence
    valid_mask = confidences >= MIN_KEYPOINT_CONFIDENCE
    
    if np.sum(valid_mask) < 3:
        return np.array([0, 0]), False
    
    # Weighted average of valid keypoints
    valid_weights = weights[valid_mask]
    valid_keypoints = keypoints[valid_mask]
    
    center_of_mass = np.average(valid_keypoints, axis=0, weights=valid_weights)
    
    return center_of_mass, True


def get_enhanced_geometric_features(keypoints, confidences, frame_width, frame_height):
    """
    Extract ALL geometric features including new 360-degree fall detection features.
    
    Args:
        keypoints: (17, 2) array of (x, y) coordinates
        confidences: (17,) array of confidence scores
        frame_width: Width of the frame
        frame_height: Height of the frame
    
    Returns:
        dict: Dictionary containing all geometric features with validity flags
    """
    # Get base features
    features = get_geometric_features(keypoints, confidences, frame_width, frame_height)
    
    # Add new 360-degree features
    
    # Shoulder alignment (for side fall detection)
    shoulder_angle, shoulder_valid = calculate_shoulder_alignment(keypoints, confidences)
    features['shoulder_alignment'] = shoulder_angle
    features['shoulder_alignment_valid'] = shoulder_valid
    
    # Hip alignment (for side fall detection)
    hip_angle, hip_valid = calculate_hip_alignment(keypoints, confidences)
    features['hip_alignment'] = hip_angle
    features['hip_alignment_valid'] = hip_valid
    
    # Limb extension (protective gestures during fall)
    limb_ext = calculate_limb_extension(keypoints, confidences)
    features['limb_extension'] = limb_ext
    
    # Center of mass (more stable tracking point)
    com, com_valid = calculate_center_of_mass(keypoints, confidences)
    features['center_of_mass'] = com
    features['center_of_mass_valid'] = com_valid
    
    return features

