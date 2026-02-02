"""
Fall Detection Validator using Custom Trained Model

This module uses the custom-trained YOLO model (best.pt) from the 
LE2I dataset repository as a validation layer for fall detection.
"""

import os
import torch
from ultralytics import YOLO


class FallDetectorValidator:
    """
    Secondary fall detection model for validation.
    Uses object detection (fall/non-fall classes) instead of pose estimation.
    """
    
    def __init__(self, model_path="best (1).pt", confidence_threshold=0.5):
        """
        Initialize the fall detector validator.
        
        Args:
            model_path: Path to the custom trained YOLO model
            confidence_threshold: Minimum confidence for fall detection
        """
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.model_loaded = False
        
        # Try to load the model
        if os.path.exists(model_path):
            try:
                self.model = YOLO(model_path)
                self.model_loaded = True
                print(f"✓ Loaded fall detection validator model: {model_path}")
                
                # Display model classes
                if hasattr(self.model, 'names'):
                    print(f"  Model classes: {list(self.model.names.values())}")
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    self.model.to('cuda')
                    print("  ✓ Fall validator model moved to CUDA")
                else:
                    print("  ⚠ Fall validator running on CPU")
                    
            except Exception as e:
                print(f"⚠ Warning: Could not load {model_path}: {e}")
                print("  System will use geometric fall detection only")
                self.model = None
                self.model_loaded = False
        else:
            print(f"⚠ Warning: Fall detector model not found at {model_path}")
            print("  System will use geometric fall detection only")
    
    def validate_fall(self, frame):
        """
        Validate if a fall is detected in the given frame.
        
        Args:
            frame: OpenCV image/frame (numpy array)
            
        Returns:
            dict: {
                'is_fall': bool,
                'confidence': float,
                'class_name': str,
                'bbox': list or None
            }
        """
        if self.model is None or not self.model_loaded:
            return {
                'is_fall': False,
                'confidence': 0.0,
                'class_name': 'model_not_loaded',
                'bbox': None,
                'error': 'Model not loaded'
            }
        
        try:
            # Run inference with standard YOLO input (just pass the frame)
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Parse results
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the detection with highest confidence
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                
                # Find highest confidence detection
                max_idx = confidences.argmax()
                max_conf = float(confidences[max_idx])
                class_id = int(classes[max_idx])
                class_name = results[0].names[class_id]
                
                # Get bounding box
                bbox = boxes.xyxy[max_idx].cpu().numpy().tolist()
                
                # Check if it's a "fall" class - handle various naming conventions
                # IMPORTANT: Exclude 'non-fall', 'no-fall', 'not-fall', etc.
                class_name_lower = class_name.lower().strip()
                
                # First check if it's explicitly a non-fall class
                if any(neg in class_name_lower for neg in ['non-fall', 'no-fall', 'not-fall', 'nofall']):
                    is_fall = False
                # Then check if it's a fall class
                elif class_name_lower in ['fall', 'falling', 'fallen']:
                    is_fall = True
                else:
                    # Fallback: check if 'fall' appears but not as part of 'non-fall'
                    is_fall = 'fall' in class_name_lower and 'non' not in class_name_lower
                
                return {
                    'is_fall': is_fall,
                    'confidence': max_conf,
                    'class_name': class_name,
                    'bbox': bbox
                }
            else:
                return {
                    'is_fall': False,
                    'confidence': 0.0,
                    'class_name': 'no_detection',
                    'bbox': None
                }
                
        except Exception as e:
            print(f"Error in fall validation: {e}")
            return {
                'is_fall': False,
                'confidence': 0.0,
                'class_name': 'error',
                'bbox': None,
                'error': str(e)
            }
    
    def is_loaded(self):
        """Check if the model is successfully loaded."""
        return self.model is not None


# Global instance (lazy loaded)
_validator_instance = None

def get_fall_validator():
    """Get or create the global fall validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = FallDetectorValidator()
    return _validator_instance
