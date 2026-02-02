"""
Application configuration module.

Handles FastAPI app creation, CORS setup, and global state management.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from collections import defaultdict


def create_app():
    """Create and configure FastAPI application"""
    app = FastAPI(title="Elderly Activity Monitor")
    
    # Add CORS middleware to allow ngrok access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins (ngrok, localhost, etc.)
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
        allow_headers=["*"],  # Allows all headers
    )
    
    return app


# ==================== Global State Variables ====================
# These are shared across the application

# Alert state
fall = False
active_alerts = []  # List of unacknowledged falls

# Person tracking
all_tracked_people = set()  # Persistent list of all detected IDs
person_signatures = {}  # Store color histograms: Name -> Histogram

# Fall event history
fall_events = []

# Frame state
last_frame = None
frame_count = 0
last_detection = {}  # Track last frame when person was detected (persistent_id -> frame_number)
detected_ids = set()  # Track which people are detected in current frame

# WebSocket and Frame Processing
alert_queue = asyncio.Queue()  # Queue for broadcasting alerts to WebSocket clients
connected_alert_clients = set()  # Set of connected WebSocket clients for alerts
frame_queue = None  # FrameQueue instance (initialized in main)
frame_processor = None  # FrameProcessor instance (initialized in main)

# Status message display
status_message = ""
status_expiry = 0

# Module instances (initialized in main)
person_tracker = None
state_manager = None
