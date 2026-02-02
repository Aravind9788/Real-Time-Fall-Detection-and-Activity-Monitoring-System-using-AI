"""
Hybrid Fall Detection Module

Combines custom YOLO model for fall detection with geometric rules for activity classification.
- Model-based: Binary fall/non-fall detection using best (1).pt
- Geometric-based: Walking, sitting, sleeping, standing classification
"""

from modules.fall_detector_validator import get_fall_validator
from modules.detection import classify_activity
from modules.fall_detection_config import (
    FALL_VALIDATOR_CONFIDENCE,
    USE_HYBRID_DETECTION,
    SUDDEN_FALL_VELOCITY_THRESHOLD,
    SLOW_LYING_VELOCITY_THRESHOLD
)


class HybridDetector:
    """
    Hybrid detection system that combines:
    1. Custom YOLO model for fall detection (binary: fall/non-fall)
    2. Geometric rules for activity classification (walking, sitting, sleeping, standing)
    """
    
    def __init__(self):
        """Initialize the hybrid detector"""
        self.fall_validator = None
        self.use_hybrid = USE_HYBRID_DETECTION
        
        if self.use_hybrid:
            try:
                self.fall_validator = get_fall_validator()
                if self.fall_validator and self.fall_validator.is_loaded():
                    print("✓ Hybrid detection mode enabled (Model + Geometric)")
                else:
                    print("⚠ Hybrid mode requested but model not loaded - using geometric only")
                    self.use_hybrid = False
            except Exception as e:
                print(f"⚠ Error initializing fall validator: {e}")
                print("  Falling back to geometric detection only")
                self.use_hybrid = False
        else:
            print("✓ Using geometric detection only (hybrid mode disabled)")
    
    def classify(self, frame, keypoints, conf, movement_speed=None, vertical_velocity=None, 
                 geometric_features=None, com_velocity=None):
        """
        Hybrid classification: Model ONLY for fall detection, geometric for other activities.
        
        Args:
            frame: Current video frame (numpy array)
            keypoints: Body keypoints from YOLO pose estimation (17, 2)
            conf: Confidence scores for each keypoint (17,)
            movement_speed: Optional horizontal movement speed in pixels/second
            vertical_velocity: Optional vertical velocity of hip
            geometric_features: Optional dict with pre-calculated geometric features
            com_velocity: Optional center-of-mass velocity dict
            
        Returns:
            str: Activity classification (STANDING, WALKING, SITTING, SLEEPING, LYING, MINOR FALL, UNKNOWN)
        """
        
        # STEP 1: Check for fall using ONLY the custom model
        if self.use_hybrid and self.fall_validator and self.fall_validator.is_loaded():
            try:
                # Run model-based fall detection
                validation = self.fall_validator.validate_fall(frame)
                
                if validation['is_fall'] and validation['confidence'] >= FALL_VALIDATOR_CONFIDENCE:
                    # Model detected fall - now check velocity to distinguish sudden vs slow
                    
                    # Get total velocity from com_velocity (center of mass)
                    total_velocity = None
                    if com_velocity and 'total_velocity' in com_velocity:
                        total_velocity = com_velocity['total_velocity']
                    
                    # VELOCITY FILTERING LOGIC
                    if total_velocity is not None:
                        # Case 1: High velocity = Sudden fall (accept model prediction)
                        if total_velocity >= SUDDEN_FALL_VELOCITY_THRESHOLD:
                            print(f"🚨 SUDDEN FALL DETECTED: velocity={total_velocity:.1f} px/s (threshold={SUDDEN_FALL_VELOCITY_THRESHOLD}), "
                                  f"model_conf={validation['confidence']:.2f}")
                            return "MINOR FALL"
                        
                        # Case 2: Low velocity = Slow lying (ignore fall prediction)
                        elif total_velocity <= SLOW_LYING_VELOCITY_THRESHOLD:
                            print(f"🛌 SLOW LYING DETECTED: velocity={total_velocity:.1f} px/s (threshold={SLOW_LYING_VELOCITY_THRESHOLD}), "
                                  f"ignoring model fall prediction")
                            # Continue to geometric classification (will likely classify as LYING)
                        
                        # Case 3: Medium velocity = Use model prediction with caution
                        else:
                            print(f"⚠️ UNCERTAIN VELOCITY: velocity={total_velocity:.1f} px/s "
                                  f"(between {SLOW_LYING_VELOCITY_THRESHOLD} and {SUDDEN_FALL_VELOCITY_THRESHOLD}), "
                                  f"accepting model prediction")
                            return "MINOR FALL"
                    
                    else:
                        # No velocity data available - trust model prediction
                        print(f"🤖 MODEL FALL DETECTED (no velocity data): class='{validation['class_name']}', "
                              f"conf={validation['confidence']:.2f}")
                        return "MINOR FALL"
                
                # Model says NO FALL - proceed to classify other activities
                # (Do NOT run geometric fall detection)
                
            except Exception as e:
                print(f"⚠ Error in model-based fall detection: {e}")
                # On error, fall back to geometric (only as emergency backup)
        
        # STEP 2: Use geometric rules ONLY for non-fall activities
        # Fall detection is exclusively handled by the model above
        # Geometric rules classify: Walking, Sitting, Sleeping, Standing, Lying
        activity = classify_activity(
            keypoints=keypoints,
            conf=conf,
            movement_speed=movement_speed,
            vertical_velocity=vertical_velocity,
            geometric_features=geometric_features,
            com_velocity=com_velocity
        )
        
        # Return the activity (will never be "MINOR FALL" from geometric rules)
        return activity
    
    def is_model_loaded(self):
        """Check if the custom fall detection model is loaded"""
        return self.use_hybrid and self.fall_validator and self.fall_validator.is_loaded()


# Global instance (singleton pattern)
_hybrid_detector_instance = None

def get_hybrid_detector():
    """Get or create the global hybrid detector instance"""
    global _hybrid_detector_instance
    if _hybrid_detector_instance is None:
        _hybrid_detector_instance = HybridDetector()
    return _hybrid_detector_instance
