"""
Camera module for smart elderly monitoring system.

Handles camera initialization, motion detection, and low-power mode.
"""

import cv2
import time
import sys
import numpy as np


def open_camera():
    """Robust camera opener for Windows"""
    for _ in range(3):  # Try up to 3 times
        # Using DSHOW (DirectShow) on Windows is much more stable for resolution changes
        c = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
        if c.isOpened():
            # Set buffer size to 1 for lowest latency
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return c
        time.sleep(0.5)
    return None


class SmartCamera:
    """
    Smart camera with motion detection and low-power mode.
    
    Automatically turns off the webcam when no motion is detected for a period,
    and periodically checks for motion to wake up.
    """
    
    def __init__(self, motion_timeout=10, sleep_check_interval=2.0, motion_threshold=10000):
        """
        Initialize SmartCamera.
        
        Args:
            motion_timeout: Seconds of no motion before entering sleep mode
            sleep_check_interval: Seconds between motion checks in sleep mode
            motion_threshold: Pixel sum threshold for motion detection
        """
        self.motion_timeout = motion_timeout
        self.sleep_check_interval = sleep_check_interval
        self.motion_threshold = motion_threshold
        
        # Camera state
        self.cap = None
        self.prev_gray = None
        self.last_motion_time = time.time()
        self.system_sleeping = False
        
        # Initialize camera
        self._init_camera()
    
    def _init_camera(self):
        """Initialize the camera"""
        self.cap = open_camera()
        if self.cap is None:
            print("Warning: Could not open camera")
    
    def _detect_motion(self, frame):
        """
        Detect motion in the current frame.
        
        Args:
            frame: Current frame (BGR)
        
        Returns:
            bool: True if motion detected, False otherwise
        """
        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        motion_detected = False
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            frame_delta = cv2.absdiff(self.prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            if np.sum(thresh) > self.motion_threshold:
                motion_detected = True
                self.last_motion_time = now
        
        self.prev_gray = gray
        return motion_detected
    
    def _enter_sleep_mode(self):
        """Enter low-power sleep mode"""
        if not self.system_sleeping:
            print("💤 Low Power Mode: No motion detected. Turning off webcam...")
            self.system_sleeping = True
            cv2.destroyAllWindows()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
    
    def _check_motion_while_sleeping(self):
        """Check for motion while in sleep mode"""
        # Wait before checking
        time.sleep(self.sleep_check_interval)
        
        # Re-open camera
        self.cap = open_camera()
        if self.cap is None:
            return False
        
        # Optimize sleep check by lowering resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        # Take a quick peek for motion
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            return False
        
        # Check for motion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            frame_delta = cv2.absdiff(self.prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            if np.sum(thresh) > self.motion_threshold:
                self.last_motion_time = time.time()  # Motion found!
                return True
        
        self.prev_gray = gray
        return False
    
    def _wake_from_sleep(self):
        """Wake up from sleep mode and restore full resolution"""
        if self.system_sleeping:
            print("☀️ Motion detected! Webcam and AI Reopened.")
            # Restore full resolution for AI analysis
            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.system_sleeping = False
    
    def get_frame(self):
        """
        Get a frame from the camera with motion detection.
        
        Returns:
            tuple: (frame, motion_detected) or (None, False) if no frame available
        """
        now = time.time()
        
        # System sleeps if no motion for the timeout period
        if now - self.last_motion_time > self.motion_timeout:
            self._enter_sleep_mode()
            
            # Check for motion while sleeping
            motion_found = self._check_motion_while_sleeping()
            if not motion_found:
                return None, False
            # If motion found, fall through to wake up and get frame
        
        # Wake from sleep if motion was detected
        if self.system_sleeping:
            self._wake_from_sleep()
        
        # Ensure camera is open
        if self.cap is None or not self.cap.isOpened():
            print("Camera lost. Attempting to reopen...")
            self.cap = open_camera()
            if self.cap is None:
                time.sleep(1)
                return None, False
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Empty frame or read failed. Retrying...")
            time.sleep(0.1)
            return None, False
        
        # Detect motion in the frame
        motion_detected = self._detect_motion(frame)
        
        return frame, motion_detected
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
