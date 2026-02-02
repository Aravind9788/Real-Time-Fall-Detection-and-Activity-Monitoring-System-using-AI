"""
Person tracking logic module.

Handles person identity tracking using spatial context and ReID embeddings.
"""

import time
import numpy as np


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1, box2: Tuples of (x1, y1, x2, y2)
    
    Returns:
        float: IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


class PersonTracker:
    """
    Manages person identity tracking using spatial context and ReID.
    
    This class maintains spatial tracking information to prevent identity
    swapping when multiple people are close together.
    """
    
    def __init__(self, reid_manager):
        """
        Initialize PersonTracker.
        
        Args:
            reid_manager: ReID manager for person re-identification
        """
        self.reid_manager = reid_manager
        
        # Spatial tracking for identity stability
        self.person_last_bbox = {}  # persistent_id -> (x1, y1, x2, y2) - last known bbox
        self.person_velocity = {}    # persistent_id -> (vx, vy) - velocity for motion prediction
        self.person_center_history = {}  # persistent_id -> [(x, y, t), ...] - last N centers
        self.person_head_history = {}  # persistent_id -> [(y, t), ...] - last N head Y-positions
        self.person_com_history = {}  # persistent_id -> [(x, y, t), ...] - center of mass history
        self.tracker_to_persistent = {}  # yolo_id -> persistent_id mapping
    
    def match_with_spatial_context(self, current_bbox, current_embedding, frame_count):
        """Match person using both spatial proximity and ReID embedding.
        
        This hybrid approach prevents identity swapping when multiple people are close.
        
        Args:
            current_bbox: (x1, y1, x2, y2) current bounding box
            current_embedding: ReID embedding from current frame
            frame_count: Current frame number
        
        Returns:
            str or None: Matched persistent_id, or None if new person
        """
        if not self.person_last_bbox:
            # No existing people, use standard ReID
            return self.reid_manager.match_identity(current_embedding)
        
        # Calculate current center
        curr_cx = (current_bbox[0] + current_bbox[2]) / 2
        curr_cy = (current_bbox[1] + current_bbox[3]) / 2
        
        # Find spatial candidates (people within reasonable distance)
        spatial_candidates = []
        for pid, last_bbox in self.person_last_bbox.items():
            iou = calculate_iou(current_bbox, last_bbox)
            
            # Calculate center distance
            last_cx = (last_bbox[0] + last_bbox[2]) / 2
            last_cy = (last_bbox[1] + last_bbox[3]) / 2
            dist = np.sqrt((curr_cx - last_cx)**2 + (curr_cy - last_cy)**2)
            
            # Consider as candidate if IoU > 0.3 OR distance < 150 pixels
            if iou > 0.3 or dist < 150:
                spatial_candidates.append((pid, iou, dist))
        
        # If strong spatial match (high IoU), strongly prefer that identity
        for pid, iou, dist in spatial_candidates:
            if iou > 0.5:  # Strong overlap with previous position
                # Validate with ReID to ensure it's not a different person in same spot
                if pid in self.reid_manager.identity_bank:
                    reid_embedding = self.reid_manager.identity_bank[pid]['embedding']
                    similarity = np.dot(current_embedding, reid_embedding) if current_embedding is not None else 0
                    
                    # If ReID also agrees (similarity > 0.5) or IoU is very high, use this ID
                    if similarity > 0.5 or iou > 0.7:
                        # Update the embedding
                        self.reid_manager.identity_bank[pid]['embedding'] = (
                            self.reid_manager.identity_bank[pid]['embedding'] * 0.9 + current_embedding * 0.1
                        )
                        self.reid_manager.identity_bank[pid]['embedding'] /= np.linalg.norm(self.reid_manager.identity_bank[pid]['embedding'])
                        self.reid_manager.identity_bank[pid]['last_seen'] = time.time()
                        return pid
        
        # No strong spatial match, use ReID but only within spatial candidates
        if spatial_candidates and current_embedding is not None:
            candidate_ids = [pid for pid, _, _ in spatial_candidates]
            best_match_id = None
            best_similarity = -1
            
            for pid in candidate_ids:
                if pid in self.reid_manager.identity_bank and 'embedding' in self.reid_manager.identity_bank[pid]:
                    reid_embedding = self.reid_manager.identity_bank[pid]['embedding']
                    similarity = np.dot(current_embedding, reid_embedding)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = pid
            
            # Use ReID match if confidence is high enough
            if best_similarity > self.reid_manager.threshold:
                self.reid_manager.identity_bank[best_match_id]['embedding'] = (
                    self.reid_manager.identity_bank[best_match_id]['embedding'] * 0.9 + current_embedding * 0.1
                )
                self.reid_manager.identity_bank[best_match_id]['embedding'] /= np.linalg.norm(self.reid_manager.identity_bank[best_match_id]['embedding'])
                self.reid_manager.identity_bank[best_match_id]['last_seen'] = time.time()
                return best_match_id
        
        # No good match found, treat as new person
        new_id = f"Person_{self.reid_manager.next_persistent_id}"
        self.reid_manager.next_persistent_id += 1
        if current_embedding is not None:
            self.reid_manager.identity_bank[new_id] = {
                'embedding': current_embedding,
                'last_seen': time.time(),
                'first_seen': time.time()
            }
        return new_id
    
    def update_tracking(self, persistent_id, bbox, timestamp):
        """
        Update spatial tracking information for a person.
        
        Args:
            persistent_id: Person's persistent ID
            bbox: Bounding box (x1, y1, x2, y2)
            timestamp: Current timestamp
        """
        # Update last known bbox
        self.person_last_bbox[persistent_id] = bbox
        
        # Track center position history for movement detection
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        if persistent_id not in self.person_center_history:
            self.person_center_history[persistent_id] = []
        
        self.person_center_history[persistent_id].append((center_x, center_y, timestamp))
        
        # Keep only last 15 frames (about 0.5 seconds at 30fps)
        # Longer window provides more stable movement calculation
        if len(self.person_center_history[persistent_id]) > 15:
            self.person_center_history[persistent_id] = self.person_center_history[persistent_id][-15:]

    def update_head_position(self, persistent_id, head_y, timestamp):
        """
        Update head position history for vertical velocity calculation.
        
        Args:
            persistent_id: Person's persistent ID
            head_y: Y coordinate of the head/nose
            timestamp: Current timestamp
        """
        if persistent_id not in self.person_head_history:
            self.person_head_history[persistent_id] = []
        
        self.person_head_history[persistent_id].append((head_y, timestamp))
        
        # Keep only last 10 frames (about 0.5-1 second)
        if len(self.person_head_history[persistent_id]) > 10:
            self.person_head_history[persistent_id] = self.person_head_history[persistent_id][-10:]

    def get_vertical_velocity(self, persistent_id):
        """
        Calculate vertical velocity from head position history.
         Positive velocity = downward movement.
        
        Args:
            persistent_id: Person's persistent ID
            
        Returns:
            float or None: Vertical velocity in pixels per second, or None if insufficient data
        """
        if persistent_id not in self.person_head_history:
            return None
            
        history = self.person_head_history[persistent_id]
        if len(history) < 3:
            return None
            
        first_y, first_t = history[0]
        last_y, last_t = history[-1]
        time_diff = last_t - first_t
        
        if time_diff > 0:
            # Positive velocity = downward movement (Y increases downward in image coordinates)
            return (last_y - first_y) / time_diff
        
        return None
    
    def get_movement_speed(self, persistent_id):
        """
        Calculate movement speed from position history with smoothing.
        
        Args:
            persistent_id: Person's persistent ID
        
        Returns:
            float or None: Movement speed in pixels per second
        """
        if persistent_id not in self.person_center_history:
            return None
        
        history = self.person_center_history[persistent_id]
        if len(history) < 4:  # Need at least 4 frames for reliable measurement
            return None
        
        # Calculate displacement over entire history window
        # This reduces sensitivity to single-frame jitter
        first_x, first_y, first_t = history[0]
        last_x, last_y, last_t = history[-1]
        time_diff = last_t - first_t
        
        if time_diff <= 0:
            return None
        
        # Calculate total displacement
        total_displacement = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
        
        # Calculate speed
        speed = total_displacement / time_diff  # pixels per second
        
        # Apply smoothing: if speed is very low (< 5 px/s), likely just jitter
        if speed < 5:
            return 0.0
        
        # For medium speeds (5-30 px/s), check if movement is consistent
        # by looking at intermediate points
        if 5 <= speed <= 50:
            # Calculate variance in movement direction
            movements = []
            for i in range(1, len(history)):
                prev_x, prev_y, prev_t = history[i-1]
                curr_x, curr_y, curr_t = history[i]
                dx = curr_x - prev_x
                dy = curr_y - prev_y
                movements.append((dx, dy))
            
            # If movements are inconsistent (jitter), reduce reported speed
            if len(movements) > 2:
                # Check if movement vectors are generally in same direction
                total_dx = sum(abs(m[0]) for m in movements)
                total_dy = sum(abs(m[1]) for m in movements)
                net_dx = abs(sum(m[0] for m in movements))
                net_dy = abs(sum(m[1] for m in movements))
                
                # Consistency ratio: net movement / total movement
                # High ratio = consistent direction, Low ratio = random jitter
                consistency = 0
                if total_dx + total_dy > 0:
                    consistency = (net_dx + net_dy) / (total_dx + total_dy)
                
                # If consistency is low (< 0.3), it's likely jitter
                if consistency < 0.3:
                    return speed * consistency  # Reduce speed by consistency factor
        
        return speed
    
    def update_center_of_mass(self, persistent_id, com, timestamp):
        """
        Update center of mass position history for velocity calculation.
        
        Args:
            persistent_id: Person's persistent ID
            com: Center of mass (x, y) coordinates
            timestamp: Current timestamp
        """
        if persistent_id not in self.person_com_history:
            self.person_com_history[persistent_id] = []
        
        self.person_com_history[persistent_id].append((com[0], com[1], timestamp))
        
        # Keep only last 15 frames (about 0.5 seconds at 30fps)
        if len(self.person_com_history[persistent_id]) > 15:
            self.person_com_history[persistent_id] = self.person_com_history[persistent_id][-15:]
    
    def get_center_of_mass_velocity(self, persistent_id):
        """
        Calculate velocity of center of mass.
        More stable than single-point tracking for multi-directional fall detection.
        
        Args:
            persistent_id: Person's persistent ID
        
        Returns:
            dict or None: {
                'horizontal_velocity': float (px/s in X direction, positive = right),
                'vertical_velocity': float (px/s in Y direction, positive = down),
                'total_velocity': float (px/s magnitude),
                'direction_angle': float (degrees from horizontal, 0-360)
            } or None if insufficient data
        """
        if persistent_id not in self.person_com_history:
            return None
        
        history = self.person_com_history[persistent_id]
        if len(history) < 4:  # Need at least 4 frames
            return None
        
        first_x, first_y, first_t = history[0]
        last_x, last_y, last_t = history[-1]
        time_diff = last_t - first_t
        
        if time_diff <= 0:
            return None
        
        # Calculate velocity components
        vx = (last_x - first_x) / time_diff  # Horizontal velocity
        vy = (last_y - first_y) / time_diff  # Vertical velocity (positive = downward)
        
        # Total velocity magnitude
        total_vel = np.sqrt(vx**2 + vy**2)
        
        # Direction angle (0° = right, 90° = down, 180° = left, 270° = up)
        angle = np.degrees(np.arctan2(vy, vx))
        if angle < 0:
            angle += 360
        
        return {
            'horizontal_velocity': vx,
            'vertical_velocity': vy,
            'total_velocity': total_vel,
            'direction_angle': angle
        }
    
    def cleanup_person(self, persistent_id):
        """
        Remove all tracking data for a person.
        
        Args:
            persistent_id: Person's persistent ID
        """
        if persistent_id in self.person_last_bbox:
            del self.person_last_bbox[persistent_id]
        if persistent_id in self.person_velocity:
            del self.person_velocity[persistent_id]
        if persistent_id in self.person_center_history:
            del self.person_center_history[persistent_id]
        if persistent_id in self.person_head_history:
            del self.person_head_history[persistent_id]
        if persistent_id in self.person_com_history:
            del self.person_com_history[persistent_id]
        
        # Clean up tracker mapping
        yolo_keys = [k for k, v in self.tracker_to_persistent.items() if v == persistent_id]
        for k in yolo_keys:
            del self.tracker_to_persistent[k]
