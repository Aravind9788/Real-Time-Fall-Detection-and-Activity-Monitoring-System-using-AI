import threading
import queue
import time
import cv2
import numpy as np

# Import data_lock for thread safety
from modules.face_rec import data_lock
from modules.skeleton_utils import get_geometric_features
from modules.hybrid_detector import get_hybrid_detector

class FrameQueue:
    """
    Thread-safe bounded queue with drop-oldest policy.
    Optimized for real-time fall detection with low latency.
    """
    
    def __init__(self, maxsize=5):
        """
        Initialize frame queue.
        
        Args:
            maxsize: Maximum queue size (default: 5 frames for ~150-250ms latency at 30 FPS)
        """
        self.maxsize = maxsize
        self.queue = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()
        self.frames_queued = 0  # Total frames ever queued
        self.frames_dropped = 0  # Total frames dropped
        self.last_drop_warning = 0  # Timestamp of last drop warning
    
    def put(self, frame_data):
        """
        Add frame to queue. If full, drop oldest frame.
        
        Args:
            frame_data: Dictionary with frame, timestamp, and frame_number
        """
        with self.lock:
            if self.queue.full():
                try:
                    dropped_frame = self.queue.get_nowait()
                    self.frames_dropped += 1
                    
                    # Warn every 10 drops to avoid spam
                    if self.frames_dropped % 10 == 1:
                        current_time = time.time()
                        if current_time - self.last_drop_warning > 5.0:  # Max once per 5 seconds
                            print(f"⚠️ Frame queue full - dropping old frames (total dropped: {self.frames_dropped})")
                            self.last_drop_warning = current_time
                except queue.Empty:
                    pass
            
            try:
                self.queue.put_nowait(frame_data)
                self.frames_queued += 1
            except queue.Full:
                pass
    
    def get(self, timeout=1.0):
        """
        Get frame from queue with timeout.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Frame data dict or None if timeout
        """
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def size(self):
        """Get current number of frames in queue"""
        return self.queue.qsize()
    
    def get_stats(self):
        """
        Get queue statistics.
        
        Returns:
            dict: Statistics including queued, dropped, current size
        """
        return {
            'current_size': self.size(),
            'max_size': self.maxsize,
            'frames_queued': self.frames_queued,
            'frames_dropped': self.frames_dropped,
            'drop_rate': (self.frames_dropped / max(self.frames_queued, 1)) * 100
        }


class FrameProcessor:
    """
    Worker thread for processing video frames.
    
    Delegates tracking to PersonTracker and state management to ActivityStateManager.
    Includes ST-GCN based fall detection.
    """
    
    def __init__(self, frame_queue, alert_callback=None, result_callback=None):
        self.frame_queue = frame_queue
        self.alert_callback = alert_callback
        self.result_callback = result_callback
        
        self.running = False
        self.thread = None
        self.result_lock = threading.Lock()
        self.latest_result = None
        
        # Dependencies injected via inject_dependencies
        self.model = None
        self.person_tracker = None
        self.state_manager = None
        self.app_config = None
        
        self.detected_ids = None
        self.last_detection = None
        
        self.classify_activity = None
        self.rename_person = None
        self.reid_manager = None
        self.known_face_encodings = None
        self.known_face_names = None
        
        # Initialize hybrid detector (model + geometric)
        self.hybrid_detector = get_hybrid_detector()
        
        self.frame_count = 0
    
    def inject_dependencies(self, **kwargs):
        """Inject dependencies (modules and objects) from main.py"""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def start(self):
        """Start the worker thread"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        print("✓ Frame processor worker thread started")
    
    def stop(self):
        """Stop the worker thread gracefully"""
        if not self.running:
            return
        print("Stopping frame processor...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                print("⚠ Warning: Frame processor thread did not stop cleanly")
            else:
                print("✓ Frame processor stopped")
    
    def get_latest_result(self):
        """Get the latest processing result (thread-safe)"""
        with self.result_lock:
            return self.latest_result
    
    def _process_loop(self):
        """Main processing loop"""
        processed_count = 0
        last_stats_time = time.time()
        
        while self.running:
            frame_data = self.frame_queue.get(timeout=0.5)
            if frame_data is None:
                continue
            
            try:
                result = self._process_frame(frame_data)
                with self.result_lock:
                    self.latest_result = result
                
                if self.result_callback:
                    self.result_callback(result)
                
                processed_count += 1
                
                # Log stats every 100 frames
                if processed_count % 100 == 0:
                    stats = self.frame_queue.get_stats()
                    elapsed = time.time() - last_stats_time
                    fps = 100 / elapsed if elapsed > 0 else 0
                    
                    print(f"📊 Processed {processed_count} frames | "
                          f"Queue: {stats['current_size']}/{stats['max_size']} | "
                          f"Dropped: {stats['frames_dropped']} ({stats['drop_rate']:.1f}%) | "
                          f"Processing FPS: {fps:.1f}")
                    
                    last_stats_time = time.time()
                    
            except Exception as e:
                print(f"Error in frame processing loop: {e}")
                import traceback
                traceback.print_exc()
    
    def _process_frame(self, frame_data):
        """Process a single frame using PersonTracker and StateManager"""
        frame = frame_data['frame']
        self.frame_count += 1
        
        if frame is None or frame.size == 0:
            return {"error": "Invalid frame"}
        
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        now = time.time()
        
        # Run YOLO tracking
        results = self.model.track(frame, persist=True, conf=0.5, verbose=False)
        
        detected_ids_local = set()
        people_data = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0 and results[0].keypoints is not None:
            for i, kp in enumerate(results[0].keypoints.xy):
                # 1. Get Identifying Data
                yolo_id = int(results[0].boxes.id[i]) if results[0].boxes.id is not None and len(results[0].boxes.id) > i else f"unknown_{i}"
                
                box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                person_img = frame[y1:y2, x1:x2]
                current_bbox = (x1, y1, x2, y2)
                
                # 2. Tracking Update (PersonTracker)
                embedding = self.reid_manager.get_embedding(person_img)
                persistent_id = self.person_tracker.match_with_spatial_context(current_bbox, embedding, self.frame_count)
                
                self.person_tracker.tracker_to_persistent[str(yolo_id)] = persistent_id
                self.person_tracker.update_tracking(persistent_id, current_bbox, now)
                
                # Resolve Display ID
                pid = persistent_id
                
                from modules.database import manual_id_map
                if pid in manual_id_map:
                    pid = manual_id_map[pid]
                elif self.frame_count % 30 == 0:
                    # Face Recognition
                    from modules.face_rec import recognize_face
                    matched_name = recognize_face(person_img)
                    if matched_name:
                        if pid != matched_name:
                            self.rename_person(pid, matched_name)
                        pid = matched_name
                
                detected_ids_local.add(persistent_id)
                self.last_detection[persistent_id] = self.frame_count
                
                # 3. Activity Classification with Enhanced Geometric Features
                keypoints = kp.cpu().numpy()  # (17, 2)
                confidences = results[0].keypoints.conf[i].cpu().numpy()  # (17,)
                
                # Extract enhanced geometric features (includes 360-degree detection features)
                from modules.skeleton_utils import get_enhanced_geometric_features
                geometric_features = get_enhanced_geometric_features(keypoints, confidences, w, h)
                
                # Calculate velocity features
                movement_speed = self.person_tracker.get_movement_speed(persistent_id)
                
                # Track center of mass position for velocity calculation
                if geometric_features['center_of_mass_valid']:
                    com = geometric_features['center_of_mass']
                    self.person_tracker.update_center_of_mass(persistent_id, com, now)
                
                # Calculate center-of-mass velocity (multi-directional)
                com_velocity = self.person_tracker.get_center_of_mass_velocity(persistent_id)
                
                # Calculate vertical velocity of hip center (backward compatibility)
                if geometric_features['hip_center_valid']:
                    hip_y = geometric_features['hip_center'][1]
                    self.person_tracker.update_head_position(persistent_id, hip_y, now)
                
                vertical_velocity = self.person_tracker.get_vertical_velocity(persistent_id)
                
                # Classify activity using HYBRID detection (model for falls, geometric for activities)
                activity = self.hybrid_detector.classify(
                    frame=frame,  # Pass full frame for model-based fall detection
                    keypoints=keypoints, 
                    conf=confidences, 
                    movement_speed=movement_speed, 
                    vertical_velocity=vertical_velocity,
                    geometric_features=geometric_features,
                    com_velocity=com_velocity  # Multi-directional velocity
                )
                
                # 4. State Management
                center_coords = ((x1 + x2) / 2, (y1 + y2) / 2)
                with data_lock:
                    alert_data = self.state_manager.update_state(
                        persistent_id, pid, activity, center_coords, now
                    )
                    if persistent_id not in self.state_manager.person_state:
                         if hasattr(self, 'app_config'):
                             self.app_config.all_tracked_people.add(pid)
                
                if alert_data:
                    if self.alert_callback:
                        self.alert_callback(alert_data)
                    self.app_config.fall_events.append(alert_data)
                
                # 5. Visualization
                state = self.state_manager.get_state(persistent_id)
                self._draw_person_info(display_frame, x1, y1, x2, y2, pid, state)
                
                people_data.append({
                    "id": pid,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "activity": state
                })
        
        # 6. Cleanup Stale Data
        self._cleanup_stale_data(detected_ids_local)
        
        # 7. Update Global State
        if self.detected_ids is not None:
            self.detected_ids.clear()
            self.detected_ids.update(detected_ids_local)
        
        self._draw_ui_overlays(display_frame, len(detected_ids_local))
        
        return {
            "processed_frame": display_frame,
            "people": people_data,
            "frame_count": self.frame_count,
            "detected_count": len(detected_ids_local)
        }

    def _cleanup_stale_data(self, detected_ids_local):
        """Remove data for people not seen for a while"""
        ids_to_remove = []
        # Use state_manager's keys as the source of truth for active people
        for persistent_id in list(self.state_manager.person_state.keys()):
            if persistent_id not in detected_ids_local and (self.frame_count - self.last_detection.get(persistent_id, 0)) > 30:
                ids_to_remove.append(persistent_id)
        
        with data_lock:
            for persistent_id in ids_to_remove:
                self.state_manager.cleanup_person(persistent_id)
                self.person_tracker.cleanup_person(persistent_id)
                if persistent_id in self.last_detection:
                    del self.last_detection[persistent_id]
                
    def _draw_person_info(self, frame, x1, y1, x2, y2, pid, activity_state):
        """Draw bounding box and status info"""
        # Color coding
        if "FALL" in activity_state or activity_state == "LYING":
            color = (0, 0, 255)  # Red
            status_text = f"⚠️ {activity_state}"
        elif activity_state == "SLEEPING":
            color = (255, 0, 255)  # Purple
            status_text = f"😴 {activity_state}"
        elif activity_state == "WALKING":
            color = (0, 255, 0)  # Green
            status_text = f"🚶 {activity_state}"
        elif activity_state == "SITTING":
            color = (255, 165, 0)  # Orange
            status_text = f"🪑 {activity_state}"
        elif activity_state == "STANDING":
            color = (0, 255, 255)  # Yellow
            status_text = f"🧍 {activity_state}"
        else:
            color = (128, 128, 128)  # Gray
            status_text = activity_state
        
        # Draw box
        thickness = 4 if "FALL" in activity_state else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw text
        cv2.putText(frame, f"ID: {pid}", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, status_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Fall banner
        if "FALL" in activity_state:
            cv2.rectangle(frame, (x1, y2+5), (x2, y2+35), (0, 0, 255), -1)
            cv2.putText(frame, "FALL DETECTED!", (x1+5, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def _draw_ui_overlays(self, frame, count):
        """Draw system headers and overlays"""
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 50), (40, 40, 40), -1)
        cv2.putText(frame, "ELDERLY FALL DETECTION SYSTEM", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Count indicator
        cv2.rectangle(frame, (w-200, 0), (w, 50), (0, 100, 0) if count > 0 else (100, 100, 0), -1)
        text = f"Monitoring: {count}" if count > 0 else "Low Power"
        cv2.putText(frame, text, (w-190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
