"""
Fall Detection Configuration

Centralized configuration for all fall detection thresholds and parameters.
Adjust these values to tune the system for your specific camera setup.
"""

# ==================== Confidence Thresholds ====================
# Minimum confidence for keypoint to be considered valid
MIN_KEYPOINT_CONFIDENCE = 0.5

# Minimum confidence for critical joints (shoulders, hips)
MIN_CRITICAL_JOINTS_CONFIDENCE = 0.5

# ==================== Velocity Thresholds (pixels/second) ====================
# Fast downward motion indicating a fall
FALL_VELOCITY_THRESHOLD = 150

# Controlled lying down (slow movement)
SLOW_MOVEMENT_THRESHOLD = 80

# Standing vs walking threshold (pixels/second)
# Increased to 30 to reduce false positives from tracking jitter
STANDING_MOVEMENT_THRESHOLD = 30

# ==================== Angle Thresholds (degrees) ====================
# Body horizontal - lying down
LYING_ANGLE_THRESHOLD = 60

# Significant tilt - potential fall
MINOR_FALL_ANGLE_THRESHOLD = 40

# ==================== Time Thresholds (seconds) ====================
# Time to wait before escalating minor fall to major fall
MAJOR_FALL_ESCALATION_TIME = 10

# Number of consecutive upright frames needed to confirm recovery
RECOVERY_CONFIRMATION_FRAMES = 10

# Cooldown time after recovery to suppress re-alerts
RECOVERY_COOLDOWN_TIME = 3.0

# Time spent lying before classifying as sleeping
LYING_TO_SLEEPING_TIME = 10

# ==================== Model Configuration ====================
# Primary YOLO model path (pose estimation)
YOLO_MODEL_PATH = "yolov11n-pose.pt"

# Fallback model if YOLOv11n not available
YOLO_FALLBACK_PATH = "yolov8n-pose.pt"

# Detection confidence threshold
YOLO_CONFIDENCE_THRESHOLD = 0.5

# Custom fall detection validator model (from LE2I dataset training)
FALL_VALIDATOR_MODEL_PATH = "best (1).pt"

# Confidence threshold for fall validator
FALL_VALIDATOR_CONFIDENCE = 0.6

# Use hybrid detection mode (model for falls, geometric for activities)
USE_HYBRID_DETECTION = True

# ==================== Geometric Feature Settings ====================
# Minimum torso length (pixels) to consider detection valid
MIN_TORSO_LENGTH = 30

# Maximum angle change per second (degrees/second) for normal movement
MAX_NORMAL_ANGLE_CHANGE_RATE = 30

# Minimum hip displacement for fall detection (pixels)
MIN_FALL_DISPLACEMENT = 50

# ==================== 360-Degree Fall Detection ====================
# Rapid pose change threshold (degrees/second)
RAPID_POSE_CHANGE_THRESHOLD = 100

# Minimum alignment difference for twist detection (degrees)
ALIGNMENT_TWIST_THRESHOLD = 30

# Limb extension score threshold (0-1 scale)
LIMB_EXTENSION_THRESHOLD = 0.7

# Center of mass velocity threshold (px/s)
COM_FALL_VELOCITY_THRESHOLD = 120

# Lateral (sideways) fall angle threshold (shoulder/hip tilt)
LATERAL_FALL_ANGLE_THRESHOLD = 65

# ==================== Sudden Fall Detection (Velocity Filtering) ====================
# Minimum velocity to classify a model "fall" prediction as a sudden fall (px/s)
# Falls with velocity above this threshold are considered sudden/rapid falls
SUDDEN_FALL_VELOCITY_THRESHOLD = 60  # Lowered to detect more falls (was 100)

# Maximum velocity to ignore as slow lying down (px/s)
# Movements below this threshold are considered controlled lying (e.g., going to sleep)
SLOW_LYING_VELOCITY_THRESHOLD = 30  # Lowered to be more strict about slow lying (was 50)

