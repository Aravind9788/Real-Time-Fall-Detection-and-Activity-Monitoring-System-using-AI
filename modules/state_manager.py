"""
Activity state management module.

Handles state machine logic for activity detection, fall alerts, and time tracking.
"""

import time
from collections import defaultdict


class ActivityStateManager:
    """
    Manages activity state for tracked persons.
    
    Handles state transitions, fall detection escalation, recovery tracking,
    and time accumulation for different activities.
    """
    
    def __init__(self, send_fall_alert_func):
        """
        Initialize ActivityStateManager.
        
        Args:
            send_fall_alert_func: Function to send fall alerts
        """
        self.send_fall_alert = send_fall_alert_func
        
        # State tracking per person
        self.person_state = {}  # persistent_id -> current state
        self.person_last_time = {}  # persistent_id -> last update time
        self.active_fall_event = {}  # persistent_id -> "MINOR" or "MAJOR"
        self.minor_fall_start_time = {}  # persistent_id -> timestamp when minor fall started
        self.lying_start_time = {}  # persistent_id -> timestamp when lying started
        self.recovery_mode = {}  # persistent_id -> expiry_time (suppress minor fall alerts)
        self.recovery_confirm_count = {}  # persistent_id -> count of recovery frames
        
        # State smoothing to prevent flickering (sitting/walking changes)
        self.pending_state_change = {}  # persistent_id -> (new_state, count)
        self.state_change_threshold = 3  # Require 3 consecutive frames to change state
        
        # Time tracking dictionaries
        self.walking_time = defaultdict(float)
        self.sleeping_time = defaultdict(float)
        self.sitting_time = defaultdict(float)
        self.standing_time = defaultdict(float)
    
    def update_state(self, persistent_id, display_id, activity, center_coords, timestamp):
        """
        Update person's activity state and return any alerts.
        
        Args:
            persistent_id: Internal persistent ID
            display_id: Display ID (can be name or ID)
            activity: Detected activity string
            center_coords: (x, y) center coordinates
            timestamp: Current timestamp
        
        Returns:
            dict or None: Alert data if alert should be sent, None otherwise
        """
        now = timestamp
        
        # Initialize person if not seen before
        if persistent_id not in self.person_state:
            self.person_state[persistent_id] = "UNKNOWN"
            self.person_last_time[persistent_id] = now
        
        # Determine new state
        new_state = activity
        if activity == "UNKNOWN":
            new_state = self.person_state.get(persistent_id, "UNKNOWN")
        
        # Check if currently down
        is_currently_down = (new_state in ["LYING", "MINOR FALL", "SLEEPING"])
        
        alert_data = None
        
        # 1. Handle Risk States (Lying or Minor Fall)
        if is_currently_down:
            self.recovery_confirm_count[persistent_id] = 0  # Reset recovery counter
            
            if persistent_id not in self.minor_fall_start_time:
                self.minor_fall_start_time[persistent_id] = now
            
            # Escalation check: If they've been down for 10s and only have a Minor alert
            if self.active_fall_event.get(persistent_id) == "MINOR":
                if (now - self.minor_fall_start_time[persistent_id] > 10):
                    message = f"MAJOR FALL (ID {display_id}) - No recovery after 10s"
                    self.send_fall_alert(
                        message,
                        display_id,
                        "MAJOR FALL",
                        coords=center_coords
                    )
                    self.active_fall_event[persistent_id] = "MAJOR"
                    alert_data = {
                        "person": display_id,
                        "type": "MAJOR FALL",
                        "message": message,
                        "coords": center_coords,
                        "time": now,
                        "timestamp": time.strftime("%H:%M:%S", time.localtime(now))
                    }
        
        # 2. Handle Potential Recovery (Upright: WALKING, SITTING)
        elif new_state in ["WALKING", "SITTING"]:
            if persistent_id in self.active_fall_event:
                # Increment recovery counter
                self.recovery_confirm_count[persistent_id] = self.recovery_confirm_count.get(persistent_id, 0) + 1
                
                # Confirm recovery after 10 consecutive frames of upright activity
                if self.recovery_confirm_count[persistent_id] > 10:
                    self.recovery_mode[persistent_id] = now + 3.0
                    message = f"RECOVERED (ID {display_id})"
                    self.send_fall_alert(
                        message,
                        display_id,
                        "RECOVERED",
                        coords=center_coords
                    )
                    alert_data = {
                        "person": display_id,
                        "type": "RECOVERED",
                        "message": message,
                        "coords": center_coords,
                        "time": now,
                        "timestamp": time.strftime("%H:%M:%S", time.localtime(now))
                    }
                    del self.active_fall_event[persistent_id]
                    if persistent_id in self.minor_fall_start_time:
                        del self.minor_fall_start_time[persistent_id]
                    if persistent_id in self.lying_start_time:
                        del self.lying_start_time[persistent_id]
                    self.recovery_confirm_count[persistent_id] = 0
            else:
                # Even if no active fall, clear "down" timers if they are clearly upright
                if persistent_id in self.minor_fall_start_time:
                    del self.minor_fall_start_time[persistent_id]
                if persistent_id in self.lying_start_time:
                    del self.lying_start_time[persistent_id]
                self.recovery_confirm_count[persistent_id] = 0
        
        # 3. SLEEPING logic (sustained lying)
        if new_state == "LYING":
            if persistent_id not in self.lying_start_time:
                self.lying_start_time[persistent_id] = now
            elif now - self.lying_start_time[persistent_id] > 10:
                new_state = "SLEEPING"
        
        # 4. Special case: If in recovery mode, don't show "MINOR FALL"
        if persistent_id in self.recovery_mode:
            if now > self.recovery_mode[persistent_id]:
                del self.recovery_mode[persistent_id]
            elif new_state == "MINOR FALL":
                new_state = "STANDING"
        
        # 5. CRITICAL FIX: If there's an active fall event, keep showing  "MINOR FALL" or "MAJOR FALL"
        # Don't let "LYING" overwrite the fall state
        if persistent_id in self.active_fall_event:
            if new_state == "LYING":
                # Person is still on the ground - preserve the fall state
                if self.active_fall_event[persistent_id] == "MAJOR":
                    new_state = "MAJOR FALL"
                else:
                    new_state = "MINOR FALL"
        
        # Accumulate time for PREVIOUS activity (before state change)
        duration = now - self.person_last_time[persistent_id]
        if duration > 0:
            prev_state = self.person_state.get(persistent_id, "UNKNOWN")
            if prev_state == "WALKING":
                self.walking_time[display_id] += duration
            elif prev_state == "STANDING":
                self.standing_time[display_id] += duration
            elif prev_state == "SITTING":
                self.sitting_time[display_id] += duration
            elif prev_state == "SLEEPING":
                self.sleeping_time[display_id] += duration
        
        self.person_last_time[persistent_id] = now
        
        # Detect INITIAL FALL Alert
        is_initial_fall = False
        prev_s = self.person_state.get(persistent_id, "UNKNOWN")
        
        # Trigger MINOR FALL immediately if they go down and no active event
        if is_currently_down and persistent_id not in self.active_fall_event:
            if prev_s not in ["LYING", "SLEEPING", "MINOR FALL"]:
                is_initial_fall = True
                alert_type = "MINOR FALL"
        
        # ==================== STATE SMOOTHING: Prevent Flickering ====================
        # For non-critical states (SITTING, WALKING, STANDING), require multiple consecutive frames
        # Only falls and LYING bypass smoothing for immediate response
        # Special case: LYING → SLEEPING is also immediate (timer-based, not detection-based)
        critical_states = ["MINOR FALL", "MAJOR FALL", "LYING", "UNKNOWN"]
        
        should_update_state = False
        
        # Check if this is a timer-based LYING → SLEEPING transition (bypass smoothing)
        is_sleeping_upgrade = (prev_s == "LYING" and new_state == "SLEEPING")
        
        if new_state in critical_states or prev_s in critical_states or is_sleeping_upgrade:
            # Critical states OR timer-based sleeping upgrade: update immediately
            should_update_state = True
        elif new_state != prev_s:
            # Non-critical state change (e.g., SITTING <-> WALKING)
            # Check if we have a pending change
            if persistent_id in self.pending_state_change:
                pending_state, count = self.pending_state_change[persistent_id]
                
                if pending_state == new_state:
                    # Same pending state - increment counter
                    count += 1
                    if count >= self.state_change_threshold:
                        # Threshold reached - accept the change
                        should_update_state = True
                        del self.pending_state_change[persistent_id]
                    else:
                        # Keep pending
                        self.pending_state_change[persistent_id] = (new_state, count)
                else:
                    # Different state detected - reset pending
                    self.pending_state_change[persistent_id] = (new_state, 1)
            else:
                # Start new pending change
                self.pending_state_change[persistent_id] = (new_state, 1)
        else:
            # Same state as before - clear any pending changes
            if persistent_id in self.pending_state_change:
                del self.pending_state_change[persistent_id]
            should_update_state = False  # No change needed
        
        # Update state if changed and threshold met
        if should_update_state and new_state != "UNKNOWN" and new_state != prev_s:
            self.person_state[persistent_id] = new_state
        
        # Trigger initial fall alert
        if is_initial_fall:
            self.active_fall_event[persistent_id] = "MINOR"
            message = f"{alert_type} (ID {display_id})"
            self.send_fall_alert(
                message,
                display_id,
                alert_type,
                coords=center_coords
            )
            alert_data = {
                "person": display_id,
                "type": alert_type,
                "message": message,
                "coords": center_coords,
                "time": now,
                "timestamp": time.strftime("%H:%M:%S", time.localtime(now))
            }
        
        return alert_data
    
    def get_state(self, persistent_id):
        """
        Get current state for a person.
        
        Args:
            persistent_id: Person's persistent ID
        
        Returns:
            str: Current state or "UNKNOWN"
        """
        return self.person_state.get(persistent_id, "UNKNOWN")
    
    def cleanup_person(self, persistent_id):
        """
        Remove all state data for a person.
        
        Args:
            persistent_id: Person's persistent ID
        """
        if persistent_id in self.person_state:
            del self.person_state[persistent_id]
        if persistent_id in self.person_last_time:
            del self.person_last_time[persistent_id]
        if persistent_id in self.lying_start_time:
            del self.lying_start_time[persistent_id]
        if persistent_id in self.minor_fall_start_time:
            del self.minor_fall_start_time[persistent_id]
        if persistent_id in self.recovery_mode:
            del self.recovery_mode[persistent_id]
        if persistent_id in self.active_fall_event:
            del self.active_fall_event[persistent_id]
        if persistent_id in self.recovery_confirm_count:
            del self.recovery_confirm_count[persistent_id]
        if persistent_id in self.pending_state_change:
            del self.pending_state_change[persistent_id]
    
    def get_time_stats(self, display_id):
        """
        Get time statistics for a person.
        
        Args:
            display_id: Display ID (name or ID)
        
        Returns:
            dict: Dictionary with walking, sitting, sleeping, standing times
        """
        return {
            'walking': self.walking_time.get(display_id, 0),
            'sitting': self.sitting_time.get(display_id, 0),
            'sleeping': self.sleeping_time.get(display_id, 0),
            'standing': self.standing_time.get(display_id, 0)
        }
