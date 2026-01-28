"""
Frame Processing Worker Module

This module implements a producer-consumer pattern for real-time frame processing.
It uses a bounded queue to buffer incoming frames and a dedicated worker thread
to process them with YOLO pose detection, person tracking, and fall detection.
"""

import threading
import queue
import time
import cv2
import numpy as np
from collections import defaultdict


class FrameQueue:
    """Thread-safe bounded queue with drop-oldest policy"""
    
    def __init__(self, maxsize=2):
        self.maxsize = maxsize
        self.queue = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()
    
    def put(self, frame_data):
        """
        Add frame to queue. If full, drop oldest frame.
        
        Args:
            frame_data: Dictionary containing frame and metadata
        """
        with self.lock:
            if self.queue.full():
                try:
                    # Drop oldest frame
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.queue.put_nowait(frame_data)
            except queue.Full:
                pass  # Should not happen after dropping oldest
    
    def get(self, timeout=1.0):
        """Get frame from queue with timeout"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None


class FrameProcessor:
    """
    Worker thread for processing video frames.
    
    This class handles the continuous processing of frames from the queue,
    running YOLO detection, tracking, ReID, face recognition, and fall detection.
    """
    
    def __init__(self, frame_queue, alert_callback=None, result_callback=None):
        """
        Initialize frame processor.
        
        Args:
            frame_queue: FrameQueue instance for receiving frames
            alert_callback: Function to call when fall is detected
            result_callback: Function to call with processing results
        """
        self.frame_queue = frame_queue
        self.alert_callback = alert_callback
        self.result_callback = result_callback
        
        self.running = False
        self.thread = None
        
        # Latest processing result (thread-safe access)
        self.latest_result = None
        self.result_lock = threading.Lock()
        
        # Import references to global state (set by main.py)
        self.model = None
        self.person_state = None
        self.person_last_time = None
        self.person_last_bbox = None
        self.person_center_history = None
        self.person_head_history = None  # Track head Y-position for vertical velocity
        self.detected_ids = None
        self.last_detection = None
        self.tracker_to_persistent = None
        self.active_fall_event = None
        self.minor_fall_start_time = None
        self.lying_start_time = None
        self.recovery_mode = None
        self.recovery_confirm_count = None
        self.walking_time = None
        self.sitting_time = None
        self.sleeping_time = None
        self.standing_time = None
        self.all_tracked_people = None
        self.manual_id_map = None
        self.fall_events = None
        self.data_lock = None
        
        # Module functions (set by main.py)
        self.classify_activity = None
        self.match_with_spatial_context = None
        self.send_fall_alert = None
        self.rename_person = None
        self.reid_manager = None
        self.known_face_encodings = None
        self.known_face_names = None
        
        # Frame counter
        self.frame_count = 0
    
    def inject_dependencies(self, **kwargs):
        """
        Inject dependencies from main.py
        
        This allows the processor to access global state and functions
        without circular imports.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
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
    
    def _process_loop(self):
        """Main processing loop running in worker thread"""
        while self.running:
            frame_data = self.frame_queue.get(timeout=0.5)
            
            if frame_data is None:
                continue
            
            try:
                result = self._process_frame(frame_data)
                
                # Update latest result
                with self.result_lock:
                    self.latest_result = result
                
                # Call result callback if provided
                if self.result_callback:
                    self.result_callback(result)
                    
            except Exception as e:
                print(f"Error in frame processing loop: {e}")
                import traceback
                traceback.print_exc()
    
    def get_latest_result(self):
        """Get the latest processing result (thread-safe)"""
        with self.result_lock:
            return self.latest_result
    
    def _process_frame(self, frame_data):
        """
        Process a single frame with YOLO detection and tracking.
        
        This is where all the existing frame processing logic goes.
        """
        frame = frame_data['frame']
        self.frame_count += 1
        
        if frame is None or frame.size == 0:
            return {"error": "Invalid frame"}
        
        display_frame = frame.copy()
        now = time.time()
        
        # Run YOLO tracking
        results = self.model.track(frame, persist=True, conf=0.5, verbose=False)
        
        detected_ids_local = set()
        people_data = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0 and results[0].keypoints is not None:
            for i, kp in enumerate(results[0].keypoints.xy):
                # Get YOLO tracking ID
                yolo_id = int(results[0].boxes.id[i]) if results[0].boxes.id is not None and len(results[0].boxes.id) > i else f"unknown_{i}"
                
                # Get bounding box
                box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                
                yolo_id_str = str(yolo_id)
                pid = yolo_id_str
                
                if x2 > x1 and y2 > y1:
                    person_img = frame[y1:y2, x1:x2]
                    current_bbox = (x1, y1, x2, y2)
                    
                    # Person Re-Identification with Spatial Continuity
                    embedding = self.reid_manager.get_embedding(person_img)
                    persistent_id = self.match_with_spatial_context(current_bbox, embedding, self.frame_count)
                    self.tracker_to_persistent[yolo_id_str] = persistent_id
                    
                    # Update spatial tracking
                    self.person_last_bbox[persistent_id] = current_bbox
                    
                    # Track center position history
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    if persistent_id not in self.person_center_history:
                        self.person_center_history[persistent_id] = []
                    self.person_center_history[persistent_id].append((center_x, center_y, now))
                    if len(self.person_center_history[persistent_id]) > 10:
                        self.person_center_history[persistent_id] = self.person_center_history[persistent_id][-10:]
                    
                    pid = persistent_id
                    
                    # Check for Manual Name/Face Link
                    if pid in self.manual_id_map:
                        pid = self.manual_id_map[pid]
                    
                    # Face Recognition (every 30 frames)
                    elif self.frame_count % 30 == 0:
                        import face_recognition
                        with self.data_lock:
                            target_encodings = self.known_face_encodings[:]
                            target_names = self.known_face_names[:]
                        
                        if target_encodings:
                            rgb_person = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                            face_locations = face_recognition.face_locations(rgb_person, number_of_times_to_upsample=1)
                            if face_locations:
                                face_encodings = face_recognition.face_encodings(rgb_person, face_locations)
                                for fe in face_encodings:
                                    matches = face_recognition.compare_faces(target_encodings, fe, tolerance=0.6)
                                    if True in matches:
                                        first_match_index = matches.index(True)
                                        real_name = str(target_names[first_match_index])
                                        if pid != real_name:
                                            self.rename_person(pid, real_name)
                                        pid = real_name
                                        break
                    
                    center_coords = ((x1 + x2) / 2, (y1 + y2) / 2)
                    detected_ids_local.add(persistent_id)
                    self.last_detection[persistent_id] = self.frame_count
                    
                    keypoints = kp.cpu().numpy()
                    confidences = results[0].keypoints.conf[i].cpu().numpy()
                    
                    # Calculate horizontal movement speed (existing code)
                    movement_speed = None
                    if persistent_id in self.person_center_history and len(self.person_center_history[persistent_id]) >= 2:
                        history = self.person_center_history[persistent_id]
                        first_x, first_y, first_t = history[0]
                        last_x, last_y, last_t = history[-1]
                        time_diff = last_t - first_t
                        if time_diff > 0:
                            distance = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
                            movement_speed = distance / time_diff
                    
                    # ==================== VERTICAL VELOCITY CALCULATION ====================
                    # Track head/shoulder Y-position for fall detection
                    # Calculate vertical velocity (positive = downward movement)
                    vertical_velocity = None
                    
                    # Get head Y-position (use nose or average of shoulders)
                    if confidences[0] > 0.5:  # Nose detected
                        head_y = keypoints[0][1]
                    elif confidences[5] > 0.5 and confidences[6] > 0.5:  # Both shoulders
                        head_y = (keypoints[5][1] + keypoints[6][1]) / 2
                    elif confidences[5] > 0.5:  # Left shoulder only
                        head_y = keypoints[5][1]
                    elif confidences[6] > 0.5:  # Right shoulder only
                        head_y = keypoints[6][1]
                    else:
                        head_y = None
                    
                    if head_y is not None:
                        # Track head position history
                        if persistent_id not in self.person_head_history:
                            self.person_head_history[persistent_id] = []
                        
                        self.person_head_history[persistent_id].append((head_y, now))
                        
                        # Keep only last 10 frames (about 0.5-1 second at 15 FPS)
                        if len(self.person_head_history[persistent_id]) > 10:
                            self.person_head_history[persistent_id] = self.person_head_history[persistent_id][-10:]
                        
                        # Calculate vertical velocity if we have enough history
                        if len(self.person_head_history[persistent_id]) >= 3:
                            history = self.person_head_history[persistent_id]
                            first_y, first_t = history[0]
                            last_y, last_t = history[-1]
                            time_diff = last_t - first_t
                            
                            if time_diff > 0:
                                # Positive velocity = downward movement (Y increases downward in image coordinates)
                                vertical_velocity = (last_y - first_y) / time_diff
                                
                                # Debug output for falls
                                if vertical_velocity > 100:
                                    print(f"📉 Person {pid}: Vertical velocity = {vertical_velocity:.1f} px/s (downward)")
                    
                    # Call classify_activity with both horizontal and vertical velocity
                    activity = self.classify_activity(keypoints, confidences, movement_speed, vertical_velocity)
                    
                    with self.data_lock:
                        if persistent_id not in self.person_state:
                            self.person_state[persistent_id] = "UNKNOWN"
                            self.person_last_time[persistent_id] = now
                            self.all_tracked_people.add(pid)
                            print(f"✓ New person detected: ID {pid} (Internal: {persistent_id})")
                    
                    # State machine logic
                    new_state = activity
                    if activity == "UNKNOWN":
                        new_state = self.person_state.get(persistent_id, "UNKNOWN")
                    
                    is_currently_down = (new_state in ["LYING", "MINOR FALL", "SLEEPING"])
                    
                    if is_currently_down:
                        self.recovery_confirm_count[persistent_id] = 0
                        if persistent_id not in self.minor_fall_start_time:
                            self.minor_fall_start_time[persistent_id] = now
                        
                        if self.active_fall_event.get(persistent_id) == "MINOR" and (now - self.minor_fall_start_time[persistent_id] > 10):
                            alert_data = {
                                "person": pid,
                                "type": "MAJOR FALL",
                                "message": f"MAJOR FALL (ID {pid}) - No recovery after 10s",
                                "coords": center_coords,
                                "timestamp": time.strftime("%H:%M:%S", time.localtime(now))
                            }
                            self.send_fall_alert(alert_data["message"], pid, "MAJOR FALL", coords=center_coords)
                            
                            # Call alert callback
                            if self.alert_callback:
                                self.alert_callback(alert_data)
                            
                            self.active_fall_event[persistent_id] = "MAJOR"
                            self.fall_events.append({
                                "person": pid, "type": "MAJOR FALL", "time": now,
                                "timestamp": time.strftime("%H:%M:%S", time.localtime(now))
                            })
                    
                    elif new_state in ["WALKING", "SITTING", "STANDING"]:
                        if persistent_id in self.active_fall_event:
                            self.recovery_confirm_count[persistent_id] = self.recovery_confirm_count.get(persistent_id, 0) + 1
                            
                            if self.recovery_confirm_count[persistent_id] > 10:
                                alert_data = {
                                    "person": pid,
                                    "type": "RECOVERED",
                                    "message": f"RECOVERED (ID {pid})",
                                    "coords": center_coords,
                                    "timestamp": time.strftime("%H:%M:%S", time.localtime(now))
                                }
                                self.recovery_mode[persistent_id] = now + 3.0
                                self.send_fall_alert(alert_data["message"], pid, "RECOVERED", coords=center_coords)
                                
                                # Call alert callback
                                if self.alert_callback:
                                    self.alert_callback(alert_data)
                                
                                del self.active_fall_event[persistent_id]
                                if persistent_id in self.minor_fall_start_time: del self.minor_fall_start_time[persistent_id]
                                if persistent_id in self.lying_start_time: del self.lying_start_time[persistent_id]
                                self.recovery_confirm_count[persistent_id] = 0
                        else:
                            if persistent_id in self.minor_fall_start_time: del self.minor_fall_start_time[persistent_id]
                            if persistent_id in self.lying_start_time: del self.lying_start_time[persistent_id]
                            self.recovery_confirm_count[persistent_id] = 0
                    
                    if new_state == "LYING":
                        if persistent_id not in self.lying_start_time:
                            self.lying_start_time[persistent_id] = now
                        elif now - self.lying_start_time[persistent_id] > 10:
                            new_state = "SLEEPING"
                    
                    if persistent_id in self.recovery_mode:
                        if now > self.recovery_mode[persistent_id]:
                            del self.recovery_mode[persistent_id]
                        elif new_state == "MINOR FALL":
                            new_state = "STANDING"
                    
                    duration = now - self.person_last_time.get(persistent_id, now)
                    if duration > 0:
                        with self.data_lock:
                            prev_state = self.person_state.get(persistent_id, "UNKNOWN")
                            if prev_state == "WALKING": self.walking_time[pid] += duration
                            elif prev_state == "STANDING": self.standing_time[pid] += duration
                            elif prev_state == "SITTING": self.sitting_time[pid] += duration
                            elif prev_state == "SLEEPING": self.sleeping_time[pid] += duration
                    
                    self.person_last_time[persistent_id] = now
                    
                    is_initial_fall = False
                    prev_s = self.person_state.get(persistent_id, "UNKNOWN")
                    
                    if is_currently_down and persistent_id not in self.active_fall_event and prev_s not in ["LYING", "SLEEPING", "MINOR FALL"]:
                        is_initial_fall = True
                        alert_type = "MINOR FALL"
                    
                    if new_state != "UNKNOWN" and new_state != prev_s:
                        with self.data_lock:
                            self.person_state[persistent_id] = new_state
                    
                    if is_initial_fall:
                        alert_data = {
                            "person": pid,
                            "type": "MINOR FALL",
                            "message": f"MINOR FALL (ID {pid})",
                            "coords": center_coords,
                            "timestamp": time.strftime("%H:%M:%S", time.localtime(now))
                        }
                        self.active_fall_event[persistent_id] = "MINOR"
                        self.send_fall_alert(alert_data["message"], pid, "MINOR FALL", coords=center_coords)
                        
                        # Call alert callback
                        if self.alert_callback:
                            self.alert_callback(alert_data)
                        
                        self.fall_events.append({
                            "person": pid, "type": "MINOR FALL", "time": now,
                            "timestamp": time.strftime("%H:%M:%S", time.localtime(now))
                        })
                    
                    # Draw on frame with enhanced fall detection display
                    activity_state = self.person_state[persistent_id]
                    
                    # Color coding for fall detection
                    if "FALL" in activity_state or activity_state == "LYING":
                        color = (0, 0, 255)  # Red for falls/lying
                        status_text = f"⚠️ {activity_state}"
                    elif activity_state == "SLEEPING":
                        color = (255, 0, 255)  # Purple for sleeping
                        status_text = f"😴 {activity_state}"
                    elif activity_state == "WALKING":
                        color = (0, 255, 0)  # Green for walking
                        status_text = f"🚶 {activity_state}"
                    elif activity_state == "SITTING":
                        color = (255, 165, 0)  # Orange for sitting
                        status_text = f"🪑 {activity_state}"
                    elif activity_state == "STANDING":
                        color = (0, 255, 255)  # Yellow for standing
                        status_text = f"🧍 {activity_state}"
                    else:
                        color = (128, 128, 128)  # Gray for unknown
                        status_text = activity_state
                    
                    # Draw bounding box with thicker line for falls
                    thickness = 4 if "FALL" in activity_state else 2
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw ID label at top
                    cv2.putText(display_frame, f"ID: {pid}", (x1, y1-30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Draw activity status below ID
                    cv2.putText(display_frame, status_text, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Add warning banner for fall detection
                    if "FALL" in activity_state:
                        cv2.rectangle(display_frame, (x1, y2+5), (x2, y2+35), (0, 0, 255), -1)
                        cv2.putText(display_frame, "FALL DETECTED!", (x1+5, y2+25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    people_data.append({
                        "id": pid,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "activity": self.person_state[persistent_id]
                    })
        
        # Clean up people not detected
        ids_to_remove = []
        for persistent_id in list(self.person_state.keys()):
            if persistent_id not in detected_ids_local and (self.frame_count - self.last_detection.get(persistent_id, 0)) > 30:
                ids_to_remove.append(persistent_id)
        
        with self.data_lock:
            for persistent_id in ids_to_remove:
                if persistent_id in self.person_state: del self.person_state[persistent_id]
                if persistent_id in self.person_last_time: del self.person_last_time[persistent_id]
                if persistent_id in self.last_detection: del self.last_detection[persistent_id]
                if persistent_id in self.lying_start_time: del self.lying_start_time[persistent_id]
                if persistent_id in self.minor_fall_start_time: del self.minor_fall_start_time[persistent_id]
                if persistent_id in self.recovery_mode: del self.recovery_mode[persistent_id]
                if persistent_id in self.active_fall_event: del self.active_fall_event[persistent_id]
                if persistent_id in self.recovery_confirm_count: del self.recovery_confirm_count[persistent_id]
                if persistent_id in self.person_last_bbox: del self.person_last_bbox[persistent_id]
                if persistent_id in self.person_center_history: del self.person_center_history[persistent_id]
                if persistent_id in self.person_head_history: del self.person_head_history[persistent_id]
                yolo_keys = [k for k, v in self.tracker_to_persistent.items() if v == persistent_id]
                for k in yolo_keys: del self.tracker_to_persistent[k]
        
        # Add header banner showing system status
        h, w = display_frame.shape[:2]
        cv2.rectangle(display_frame, (0, 0), (w, 50), (40, 40, 40), -1)
        cv2.putText(display_frame, "ELDERLY FALL DETECTION SYSTEM", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        feature_text = "Minor Fall | Major Fall | Low Power"
        cv2.putText(display_frame, feature_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # People count indicator
        people_count = len(detected_ids_local)
        if people_count > 0:
            cv2.rectangle(display_frame, (w-200, 0), (w, 50), (0, 100, 0), -1)
            cv2.putText(display_frame, f"Monitoring: {people_count}", (w-190, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            cv2.rectangle(display_frame, (w-200, 0), (w, 50), (100, 100, 0), -1)
            cv2.putText(display_frame, "Low Power", (w-190, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Update detected_ids in global state
        self.detected_ids.clear()
        self.detected_ids.update(detected_ids_local)
        
        return {
            "processed_frame": display_frame,
            "people": people_data,
            "frame_count": self.frame_count,
            "detected_count": people_count
        }
