"""
Elderly Activity Monitor - Main Application

Entry point for the elderly monitoring system with fall detection,
activity tracking, and real-time web dashboard.
"""

import cv2
import time
import numpy as np
import sys
import threading
import face_recognition
import asyncio
import base64
from fastapi import FastAPI, Request, Form, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from collections import defaultdict
from datetime import datetime
import atexit
import queue

# Import custom modules
from modules.reid import reid_manager, tracker_to_persistent
from modules.face_rec import (
    known_face_encodings, known_face_names, register_face, 
    load_encodings, data_lock
)
from modules.detection import model, classify_activity, format_duration, FALL_URL
from modules.database import (
    init_db, log_activity_to_db, load_manual_id_map, 
    save_manual_id_map, load_stats_from_db, send_fall_alert, 
    DB_PATH, manual_id_map, get_fall_history
)
from modules.frame_processor import FrameQueue, FrameProcessor



# ==================== Global State Management ====================
# Flask app
# FastAPI app
app = FastAPI(title="Elderly Activity Monitor")
templates = Jinja2Templates(directory="templates")

# Add CORS middleware to allow ngrok access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (ngrok, localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Pydantic models for request validation
class FallAlert(BaseModel):
    person: str = "Unknown"

class RegisterRequest(BaseModel):
    name: str
    yolo_id: str = None
fall = False
active_alerts = []  # List of unacknowledged falls

# Track walking/sleeping/sitting/standing durations
walking_time = defaultdict(float)
sleeping_time = defaultdict(float)
sitting_time = defaultdict(float)
standing_time = defaultdict(float)

# Track current state per person
person_state = {}
person_last_time = {}
lying_start_time = {}  # Track when a person started lying down
minor_fall_start_time = {}  # Track duration of minor fall for escalation
recovery_mode = {}  # pid -> expiry_time (suppress minor fall alerts while getting up)
recovery_confirm_count = {}  # persistent_id -> count (frames of sustained upright activity)
active_fall_event = {}  # pid -> True (prevent multiple alerts for the same fall)

# All tracked people
all_tracked_people = set()  # Persistent list of all detected IDs
person_signatures = {}  # Store color histograms: Name -> Histogram

# Spatial tracking for identity stability
person_last_bbox = {}  # persistent_id -> (x1, y1, x2, y2) - last known bbox
person_velocity = {}    # persistent_id -> (vx, vy) - velocity for motion prediction
person_center_history = {}  # persistent_id -> [(x, y, t), ...] - last N centers for movement detection
person_head_history = {}  # persistent_id -> [(y, t), ...] - last N head Y-positions for vertical velocity

# Track fall events with timestamps (history)
fall_events = []
last_frame = None

# Frame processing variables
frame_count = 0  # Current frame number
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


# ==================== Helper Functions ====================
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


def match_with_spatial_context(current_bbox, current_embedding, frame_count):
    """Match person using both spatial proximity and ReID embedding.
    
    This hybrid approach prevents identity swapping when multiple people are close.
    
    Args:
        current_bbox: (x1, y1, x2, y2) current bounding box
        current_embedding: ReID embedding from current frame
        frame_count: Current frame number
    
    Returns:
        str or None: Matched persistent_id, or None if new person
    """
    if not person_last_bbox:
        # No existing people, use standard ReID
        return reid_manager.match_identity(current_embedding)
    
    # Calculate current center
    curr_cx = (current_bbox[0] + current_bbox[2]) / 2
    curr_cy = (current_bbox[1] + current_bbox[3]) / 2
    
    # Find spatial candidates (people within reasonable distance)
    spatial_candidates = []
    for pid, last_bbox in person_last_bbox.items():
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
            if pid in reid_manager.identity_bank:
                reid_embedding = reid_manager.identity_bank[pid]['embedding']
                similarity = np.dot(current_embedding, reid_embedding) if current_embedding is not None else 0
                
                # If ReID also agrees (similarity > 0.5) or IoU is very high, use this ID
                if similarity > 0.5 or iou > 0.7:
                    # Update the embedding
                    reid_manager.identity_bank[pid]['embedding'] = (
                        reid_manager.identity_bank[pid]['embedding'] * 0.9 + current_embedding * 0.1
                    )
                    reid_manager.identity_bank[pid]['embedding'] /= np.linalg.norm(reid_manager.identity_bank[pid]['embedding'])
                    reid_manager.identity_bank[pid]['last_seen'] = time.time()
                    return pid
    
    # No strong spatial match, use ReID but only within spatial candidates
    if spatial_candidates and current_embedding is not None:
        candidate_ids = [pid for pid, _, _ in spatial_candidates]
        best_match_id = None
        best_similarity = -1
        
        for pid in candidate_ids:
            if pid in reid_manager.identity_bank and 'embedding' in reid_manager.identity_bank[pid]:
                reid_embedding = reid_manager.identity_bank[pid]['embedding']
                similarity = np.dot(current_embedding, reid_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = pid
        
        # Use ReID match if confidence is high enough
        if best_similarity > reid_manager.threshold:
            reid_manager.identity_bank[best_match_id]['embedding'] = (
                reid_manager.identity_bank[best_match_id]['embedding'] * 0.9 + current_embedding * 0.1
            )
            reid_manager.identity_bank[best_match_id]['embedding'] /= np.linalg.norm(reid_manager.identity_bank[best_match_id]['embedding'])
            reid_manager.identity_bank[best_match_id]['last_seen'] = time.time()
            return best_match_id
    
    # No good match found, treat as new person
    new_id = f"Person_{reid_manager.next_persistent_id}"
    reid_manager.next_persistent_id += 1
    if current_embedding is not None:
        reid_manager.identity_bank[new_id] = {
            'embedding': current_embedding,
            'last_seen': time.time(),
            'first_seen': time.time()
        }
    return new_id


def rename_person(old_id, new_name):
    """Safely rename a person and transfer all their stats and mappings."""
    with data_lock:
        manual_id_map[str(old_id)] = new_name
        
        # Transfer current stats
        walking_time[new_name] = walking_time.get(new_name, 0) + walking_time.pop(old_id, 0)
        sitting_time[new_name] = sitting_time.get(new_name, 0) + sitting_time.pop(old_id, 0)
        sleeping_time[new_name] = sleeping_time.get(new_name, 0) + sleeping_time.pop(old_id, 0)
        standing_time[new_name] = standing_time.get(new_name, 0) + standing_time.pop(old_id, 0)
        
        # Update set of all people (remove old ID if it's not the same as new name)
        if str(old_id) in all_tracked_people and str(old_id) != new_name:
            all_tracked_people.remove(old_id)
        all_tracked_people.add(new_name)
        
        # Save mappings immediately
        save_manual_id_map()
        print(f"👤 Renamed {old_id} to {new_name} and transferred stats.")


# ==================== FastAPI Routes ====================
@app.post("/trigger")
async def trigger(alert: FallAlert = None):
    global fall
    if alert is None:
        alert = FallAlert()
    pid = alert.person
    # Extract type from pid if sent as string "MAJOR FALL (ID 1)"
    fall_type = "FALL"
    if "MAJOR" in str(pid): fall_type = "MAJOR FALL"
    elif "MINOR" in str(pid): fall_type = "MINOR FALL"
    elif "RECOVERED" in str(pid): fall_type = "RECOVERED"
    
    now = time.time()
    
    # Add to active alerts if not already there recently
    if not any(a['person'] == pid for a in active_alerts):
        active_alerts.append({
            "person": pid,
            "type": fall_type,
            "time": time.strftime("%H:%M:%S", time.localtime(now)),
            "timestamp": now
        })
    
    fall = True
    return {"status": "OK"}

@app.post("/api/acknowledge/{pid}")
async def acknowledge(pid: str):
    global active_alerts
    active_alerts = [a for a in active_alerts if str(a['person']) != str(pid)]
    return {"status": "success"}

@app.get("/fall")
async def check():
    global fall
    if fall:
        fall = False
        return {"fall": True}
    return {"fall": False}


@app.post("/register")
async def register(
    name: str = Form(None),
    yolo_id: str = Form(None),
    front: UploadFile = File(None),
    back: UploadFile = File(None)
):
    global last_frame, status_message, status_expiry
    
    if not name:
        return {"status": "error", "message": "Missing name"}
    
    # CASE 1: Manual ID naming (No face needed)
    if yolo_id and (str(yolo_id) in person_state or str(yolo_id) in all_tracked_people):
        rename_person(str(yolo_id), name)
        status_message = f"ID {yolo_id} is now {name}"
        status_expiry = time.time() + 5
        return {"status": "success", "message": f"Successfully named body {yolo_id} as {name}"}

    # CASE 2: Uploaded files or Live frame
    to_process = []
    if front:
        # Read uploaded file content
        front_bytes = await front.read()
        to_process.append(front_bytes)
    if back:
        back_bytes = await back.read()
        to_process.append(back_bytes)
    
    if not to_process:
        if last_frame is not None:
            to_process = [last_frame]
        else:
            return {"status": "error", "message": "No photos uploaded or live frame available"}
    
    if register_face(to_process, name, yolo_model=model if not (front or back) else None):
        status_message = f"Registered: {name}"
        status_expiry = time.time() + 5
        return {"status": "success", "message": f"Successfully registered {name}"}
    return {"status": "error", "message": "No face detected in provided images"}


@app.get("/api/history")
async def activity_history():
    """Get history for charts"""
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT date, person_id, walking, sitting, sleeping FROM activity ORDER BY date DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return [{"date": r[0], "pid": r[1], "walk": r[2], "sit": r[3], "sleep": r[4]} for r in rows]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Enhanced modern dashboard"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/report")
async def api_report():
    """API endpoint for JSON report with sorting and limiting"""
    with data_lock:
        # Snapshot current state for reporting
        current_state_snapshot = person_state.copy()
        # Unnamed IDs are those in current state that don't have a manual mapping
        unnamed_ids = [pid for pid in current_state_snapshot if pid not in manual_id_map]
        
        # Sort people: currently active first, then by total monitored time
        all_pids = list(all_tracked_people)
        
        def get_activity_score(display_id):
            # Check if this person (by name or ID) is currently active
            is_active = display_id in current_state_snapshot  # If it's a persistent_id
            if not is_active:
                # Check if any persistent_id mapped to this name is active
                is_active = any(k in current_state_snapshot for k, v in manual_id_map.items() if v == display_id)
            
            total_time = (walking_time.get(display_id, 0) + standing_time.get(display_id, 0) + 
                         sitting_time.get(display_id, 0) + sleeping_time.get(display_id, 0))
            return (is_active, total_time)

        sorted_pids = sorted(all_pids, key=get_activity_score, reverse=True)
        
        report_data = []
        for display_id in sorted_pids[:10]:
            # Try to find the persistent ID associated with this name to get the current activity
            internal_id = display_id
            for k, v in manual_id_map.items():
                if v == display_id:
                    internal_id = k
                    break

            report_data.append({
                "person": display_id,
                "walking_dur": format_duration(walking_time.get(display_id, 0)),
                "standing_dur": format_duration(standing_time.get(display_id, 0)),
                "sleeping_dur": format_duration(sleeping_time.get(display_id, 0)),
                "sitting_dur": format_duration(sitting_time.get(display_id, 0)),
                "current_activity": current_state_snapshot.get(internal_id, "AWAY"),
                "is_active": internal_id in current_state_snapshot
            })
            
        alerts_copy = active_alerts.copy()
        # Get fall history from database instead of memory
        falls_copy = get_fall_history(limit=10)

    return {
        "people": report_data,
        "falls": falls_copy, 
        "active_alerts": alerts_copy,
        "unnamed_ids": unnamed_ids
    }


@app.websocket("/ws/camera")
async def websocket_camera(websocket: WebSocket):
    """
    WebSocket endpoint for receiving camera frames from frontend.
    
    This endpoint runs in the WebSocket receive loop and only handles frame ingestion.
    All processing is delegated to the worker thread via the frame queue.
    """
    await websocket.accept()
    print("✓ Camera WebSocket client connected")
    
    try:
        frame_number = 0
        while True:
            # Receive frame data from client
            data = await websocket.receive_json()
            frame_number += 1
            
            try:
                # Decode base64 image
                img_data_base64 = data.get('frame', '')
                if ',' in img_data_base64:
                    img_data_base64 = img_data_base64.split(',')[1]
                
                img_data = base64.b64decode(img_data_base64)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None and frame.size > 0:
                    # Push frame to queue (non-blocking, drops old frames if full)
                    frame_queue.put({
                        'frame': frame,
                        'timestamp': time.time(),
                        'frame_number': frame_number
                    })
                    
                    # Optionally send acknowledgment (keep it minimal to avoid blocking)
                    # await websocket.send_json({"status": "received"})
                    
            except Exception as e:
                print(f"Error decoding frame: {e}")
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        print("✗ Camera WebSocket client disconnected")
    except Exception as e:
        print(f"Camera WebSocket error: {e}")
        import traceback
        traceback.print_exc()


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for broadcasting alerts and processed frames to frontend.
    
    This endpoint sends:
    - Fall detection alerts (MINOR/MAJOR/RECOVERED)
    - Active person tracking data
    - Optional processed frames with bounding boxes
    """
    await websocket.accept()
    connected_alert_clients.add(websocket)
    print(f"✓ Alert WebSocket client connected (total: {len(connected_alert_clients)})")
    
    try:
        while True:
            # Get latest processing result
            result = frame_processor.get_latest_result() if frame_processor else None
            
            if result and 'error' not in result:
                # Prepare data to send
                send_data = {
                    "type": "update",
                    "people": result.get('people', []),
                    "frame_count": result.get('frame_count', 0),
                    "detected_count": result.get('detected_count', 0)
                }
                
                # Optionally include processed frame (comment out if you want alerts-only)
                if 'processed_frame' in result:
                    _, buffer = cv2.imencode('.jpg', result['processed_frame'], [cv2.IMWRITE_JPEG_QUALITY, 70])
                    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                    send_data['processed_frame'] = jpg_as_text
                
                # Send update
                await websocket.send_json(send_data)
            else:
                # Send blank/waiting frame if no result yet
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Waiting for camera...", (150, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(blank_frame, "Starting frame processor...", (120, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                _, buffer = cv2.imencode('.jpg', blank_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_json({
                    "type": "update",
                    "processed_frame": jpg_as_text,
                    "detected_count": 0,
                    "frame_count": 0,
                    "people": []
                })
            
            # Check for alerts in the queue
            try:
                # Non-blocking check for alerts
                while not alert_queue.empty():
                    alert_data = await asyncio.wait_for(alert_queue.get(), timeout=0.01)
                    await websocket.send_json({
                        "type": "alert",
                        **alert_data
                    })
            except asyncio.TimeoutError:
                pass
            
            # Small delay to avoid overwhelming the client (~20 FPS)
            await asyncio.sleep(0.05)
            
    except WebSocketDisconnect:
        connected_alert_clients.discard(websocket)
        print(f"✗ Alert WebSocket client disconnected (remaining: {len(connected_alert_clients)})")
    except Exception as e:
        connected_alert_clients.discard(websocket)
        print(f"Alert WebSocket error: {e}")
        import traceback
        traceback.print_exc()




@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    """WebSocket endpoint for real-time video streaming"""
    await websocket.accept()
    print("✓ WebSocket client connected")
    try:
        while True:
            # Get current frame from global last_frame
            if last_frame is not None:
                # Encode frame as JPEG for efficient transmission
                _, buffer = cv2.imencode('.jpg', last_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                # Convert to base64 for WebSocket transmission
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                # Send to client
                await websocket.send_json({
                    "type": "frame",
                    "data": jpg_as_text
                })
            await asyncio.sleep(0.033)  # ~30 FPS
    except WebSocketDisconnect:
        print("✗ WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


def run_server():
    """Runs FastAPI server in background thread with uvicorn"""
    try:
        print("FastAPI: Starting server...")
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    except Exception as e:
        print(f"FastAPI Error: {e}")
        import traceback
        traceback.print_exc()


# ==================== Camera Functions ====================
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


# ==================== Main Execution ====================
if __name__ == "__main__":
    # Load stats from database
    load_stats_from_db(walking_time, sitting_time, sleeping_time, all_tracked_people)
    
    # Initialize frame queue and processor
    print("Initializing frame processing system...")
    frame_queue = FrameQueue(maxsize=2)
    
    # Alert callback function to push alerts to WebSocket queue
    def handle_alert(alert_data):
        """Push alert to queue for WebSocket broadcasting"""
        try:
            asyncio.run_coroutine_threadsafe(alert_queue.put(alert_data), asyncio.get_event_loop())
        except Exception as e:
            print(f"Error queuing alert: {e}")
    
    # Create frame processor
    frame_processor = FrameProcessor(
        frame_queue=frame_queue,
        alert_callback=handle_alert
    )
    
    # Inject dependencies into frame processor
    frame_processor.inject_dependencies(
        model=model,
        person_state=person_state,
        person_last_time=person_last_time,
        person_last_bbox=person_last_bbox,
        person_center_history=person_center_history,
        person_head_history=person_head_history,
        detected_ids=detected_ids,
        last_detection=last_detection,
        tracker_to_persistent=tracker_to_persistent,
        active_fall_event=active_fall_event,
        minor_fall_start_time=minor_fall_start_time,
        lying_start_time=lying_start_time,
        recovery_mode=recovery_mode,
        recovery_confirm_count=recovery_confirm_count,
        walking_time=walking_time,
        sitting_time=sitting_time,
        sleeping_time=sleeping_time,
        standing_time=standing_time,
        all_tracked_people=all_tracked_people,
        manual_id_map=manual_id_map,
        fall_events=fall_events,
        data_lock=data_lock,
        classify_activity=classify_activity,
        match_with_spatial_context=match_with_spatial_context,
        send_fall_alert=send_fall_alert,
        rename_person=rename_person,
        reid_manager=reid_manager,
        known_face_encodings=known_face_encodings,
        known_face_names=known_face_names
    )
    
    # Start frame processor worker thread
    frame_processor.start()
    
    # Register shutdown handler
    def cleanup():
        """Graceful shutdown of frame processor"""
        print("\nShutting down frame processor...")
        if frame_processor:
            frame_processor.stop()
    
    atexit.register(cleanup)
    
    # Start DB sync thread
    db_sync_thread = threading.Thread(
        target=log_activity_to_db, 
        args=(all_tracked_people, walking_time, sitting_time, sleeping_time, 
              data_lock, reid_manager, save_manual_id_map),
        daemon=True
    )
    db_sync_thread.start()
    
    # Start FastAPI server thread
    print("Starting FastAPI server thread...")
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    time.sleep(1)  # Give FastAPI time to start
    print("✓ FastAPI server running at http://127.0.0.1:8000")
    print("  Access the dashboard at: http://127.0.0.1:8000/")
    print("  WebSocket camera endpoint: ws://127.0.0.1:8000/ws/camera")
    print("  WebSocket alerts endpoint: ws://127.0.0.1:8000/ws/alerts")
    print("\n✓ Real-time frame processing system initialized")
    print("  Frame queue size: 2 (drops old frames when full)")
    print("  Worker thread: Running")
    
    # Keep the server running
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nProgram exited.")
        cleanup()
        sys.exit(0)
    
    # OLD BACKEND CAMERA CODE - DISABLED
    # cap = open_camera()
    # if cap is None:
        print("Error: Cannot open camera")
        sys.exit(1)

    frame_count = 0
    start_time = time.time()
    last_detection = {}  # Track last frame when person was detected
    last_motion_time = time.time()
    prev_gray = None
    system_sleeping = False

    while True:
        frame_count += 1
        if cap is None or not cap.isOpened():
            print("Camera lost. Attempting to reopen...")
            cap = open_camera()
            if cap is None:
                time.sleep(1)
                continue

        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Empty frame or read failed. Retrying...")
            time.sleep(0.1)
            continue
        
        # --- Motion Detection (Low Power Mode) ---
        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        motion_detected = False
        if prev_gray is not None and prev_gray.shape == gray.shape:
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            if np.sum(thresh) > 10000:  # Adjust sensitivity
                motion_detected = True
                last_motion_time = now
        else:
            # Reset if shape mismatched or first frame
            prev_gray = gray
            continue  # Skip one frame to establish baseline
        prev_gray = gray

        # System sleeps if no motion for 10 seconds
        if now - last_motion_time > 10:
            if not system_sleeping:
                print("💤 Low Power Mode: No motion detected. Turning off webcam...")
                system_sleeping = True
                cv2.destroyAllWindows()
                cap.release()  # Turn off the webcam hardware
            
            # In sleep mode, wait 2 seconds, then re-open camera to check for motion
            time.sleep(2.0)
            cap = open_camera()
            if cap is None:
                continue
            
            # Optimize sleep check by lowering resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            
            # Take a quick peek for motion
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0: 
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if prev_gray is not None and prev_gray.shape == gray.shape:
                frame_delta = cv2.absdiff(prev_gray, gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                if np.sum(thresh) > 10000:
                    last_motion_time = time.time()  # Motion found!
            prev_gray = gray
            continue
        
        if system_sleeping:
            print("☀️ Motion detected! Webcam and AI Reopened.")
            # Restore full resolution for AI analysis
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            system_sleeping = False

        last_frame = frame.copy()  # Store for registration
        try:
            # Make a copy to display
            display_frame = frame.copy()

            # Run YOLO tracking
            results = model.track(frame, persist=True, conf=0.5, verbose=False)

            detected_ids = set()  # Track which people are in this frame
            
            if results[0].keypoints is not None:
                for i, kp in enumerate(results[0].keypoints.xy):
                    # Get YOLO tracking ID as fallback
                    yolo_id = int(results[0].boxes.id[i]) if results[0].boxes.id is not None else f"unknown_{i}"
                    
                    # Try Face Recognition for ID consistency
                    box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                    # Ensure box is within frame boundaries
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
                    
                    yolo_id_str = str(yolo_id)
                    pid = yolo_id_str  # Fallback
                    
                    if x2 > x1 and y2 > y1:
                        person_img = frame[y1:y2, x1:x2]
                        current_bbox = (x1, y1, x2, y2)
                        
                        # 1. Person Re-Identification with Spatial Continuity
                        # This hybrid approach prevents identity swapping
                        embedding = reid_manager.get_embedding(person_img)
                        persistent_id = match_with_spatial_context(current_bbox, embedding, frame_count)
                        tracker_to_persistent[yolo_id_str] = persistent_id
                        
                        # Update spatial tracking information
                        person_last_bbox[persistent_id] = current_bbox
                        
                        # Track center position history for movement detection
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        if persistent_id not in person_center_history:
                            person_center_history[persistent_id] = []
                        person_center_history[persistent_id].append((center_x, center_y, now))
                        # Keep only last 10 frames (about 0.33 seconds at 30fps)
                        if len(person_center_history[persistent_id]) > 10:
                            person_center_history[persistent_id] = person_center_history[persistent_id][-10:]
                        
                        pid = persistent_id  # Current resolved ID (fallback)
                        
                        # 2. Check for Manual Name/Face Link
                        if pid in manual_id_map:
                            pid = manual_id_map[pid]
                        
                        # 3. Optional: Face Recognition to "Name" the Identity
                        elif frame_count % 30 == 0:
                            with data_lock:
                                target_encodings = known_face_encodings[:]
                                target_names = known_face_names[:]
                            
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
                                            # Link this Persistent ID to a Real Name
                                            if pid != real_name:
                                                rename_person(pid, real_name)
                                            pid = real_name
                                            break
                    
                    # Resolved coords (center of box)
                    center_coords = ((x1 + x2) / 2, (y1 + y2) / 2)

                    # Use the final resolved identity (Persistent ID or Name)
                    detected_ids.add(persistent_id)
                    last_detection[persistent_id] = frame_count
                    
                    keypoints = kp.cpu().numpy()
                    confidences = results[0].keypoints.conf[i].cpu().numpy()

                    now = time.time()
                    
                    # Calculate movement speed from position history
                    movement_speed = None
                    if persistent_id in person_center_history and len(person_center_history[persistent_id]) >= 2:
                        history = person_center_history[persistent_id]
                        # Calculate average speed over last few frames
                        first_x, first_y, first_t = history[0]
                        last_x, last_y, last_t = history[-1]
                        time_diff = last_t - first_t
                        if time_diff > 0:
                            distance = np.sqrt((last_x - first_x)**2 + (last_y - first_y)**2)
                            movement_speed = distance / time_diff  # pixels per second
                    
                    activity = classify_activity(keypoints, confidences, movement_speed)

                    with data_lock:
                        if persistent_id not in person_state:
                            person_state[persistent_id] = "UNKNOWN"
                            person_last_time[persistent_id] = now
                            all_tracked_people.add(pid)  # Store resolved name for DB/Reporting
                            print(f"✓ New person detected: ID {pid} (Internal: {persistent_id})")

                    # State Machine logic for ESCALATION and RECOVERY
                    new_state = activity
                    if activity == "UNKNOWN":
                        new_state = person_state.get(persistent_id, "UNKNOWN")
                    
                    # 1. Handle Risk States (Lying or Minor Fall)
                    is_currently_down = (new_state in ["LYING", "MINOR FALL", "SLEEPING"])
                    
                    if is_currently_down:
                        recovery_confirm_count[persistent_id] = 0  # Reset recovery counter
                        if persistent_id not in minor_fall_start_time:
                            minor_fall_start_time[persistent_id] = now
                        
                        # Escalation check: If they've been down for 10s and only have a Minor alert
                        if active_fall_event.get(persistent_id) == "MINOR" and (now - minor_fall_start_time[persistent_id] > 10):
                            send_fall_alert(f"MAJOR FALL (ID {pid}) - No recovery after 10s", pid, "MAJOR FALL", coords=center_coords)
                            active_fall_event[persistent_id] = "MAJOR"
                            fall_events.append({
                                "person": pid, "type": "MAJOR FALL", "time": now,
                                "timestamp": time.strftime("%H:%M:%S", time.localtime(now))
                            })
                    
                    # 2. Handle Potential Recovery (Upright: WALKING, SITTING)
                    elif new_state in ["WALKING", "SITTING"]:
                        if persistent_id in active_fall_event:
                            # Increment recovery counter
                            recovery_confirm_count[persistent_id] = recovery_confirm_count.get(persistent_id, 0) + 1
                            
                            # Confirm recovery after 10 consecutive frames of upright activity
                            if recovery_confirm_count[persistent_id] > 10:
                                recovery_mode[persistent_id] = now + 3.0
                                send_fall_alert(f"RECOVERED (ID {pid})", pid, "RECOVERED", coords=center_coords)
                                del active_fall_event[persistent_id]
                                if persistent_id in minor_fall_start_time: del minor_fall_start_time[persistent_id]
                                if persistent_id in lying_start_time: del lying_start_time[persistent_id]
                                recovery_confirm_count[persistent_id] = 0
                        else:
                            # Even if no active fall, clear "down" timers if they are clearly upright
                            if persistent_id in minor_fall_start_time: del minor_fall_start_time[persistent_id]
                            if persistent_id in lying_start_time: del lying_start_time[persistent_id]
                            recovery_confirm_count[persistent_id] = 0

                    # 3. SLEEPING logic (sustained lying)
                    if new_state == "LYING":
                        if persistent_id not in lying_start_time: 
                            lying_start_time[persistent_id] = now
                        elif now - lying_start_time[persistent_id] > 10: 
                            new_state = "SLEEPING"

                    # 4. Special case: If in recovery mode, don't show "MINOR FALL"
                    if persistent_id in recovery_mode:
                        if now > recovery_mode[persistent_id]: 
                            del recovery_mode[persistent_id]
                        elif new_state == "MINOR FALL": 
                            new_state = "STANDING"
                    
                    # Accumulate time for CURRENT activity (every frame)
                    duration = now - person_last_time[persistent_id]
                    if duration > 0:
                        with data_lock:
                            prev_state = person_state.get(persistent_id, "UNKNOWN")
                            if prev_state == "WALKING": walking_time[pid] += duration
                            elif prev_state == "STANDING": standing_time[pid] += duration
                            elif prev_state == "SITTING": sitting_time[pid] += duration
                            elif prev_state == "SLEEPING": sleeping_time[pid] += duration
                        
                    person_last_time[persistent_id] = now

                    # Detect INITIAL FALL Alert
                    is_initial_fall = False
                    prev_s = person_state.get(persistent_id, "UNKNOWN")
                    
                    # Trigger MINOR FALL immediately if they go down and no active event
                    if is_currently_down and persistent_id not in active_fall_event and prev_s not in ["LYING", "SLEEPING", "MINOR FALL"]:
                        is_initial_fall = True
                        alert_type = "MINOR FALL"

                    # Update state if changed
                    if new_state != "UNKNOWN" and new_state != prev_s:
                        with data_lock:
                            person_state[persistent_id] = new_state

                    # Only trigger initial fall alert
                    if is_initial_fall:
                        active_fall_event[persistent_id] = "MINOR"
                        send_fall_alert(f"{alert_type} (ID {pid})", pid, alert_type, coords=center_coords)
                        fall_events.append({
                            "person": pid,
                            "type": alert_type,
                            "time": now,
                            "timestamp": time.strftime("%H:%M:%S", time.localtime(now))
                        })

                    # Overlay text
                    walk_str = format_duration(walking_time[pid])
                    sit_str = format_duration(sitting_time[pid])
                    sleep_str = format_duration(sleeping_time[pid])
                    
                    # Color code based on state
                    color = (0, 0, 255) if "FALL" in person_state[persistent_id] or person_state[persistent_id] == "LYING" else (0, 255, 255) if person_state[persistent_id] == "WALKING" else (255, 0, 0)
                    
                    cv2.putText(display_frame, f"ID {pid}: {person_state[persistent_id]}", (20, 60 + (i % 5) * 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(display_frame, f"W:{walk_str} S:{sit_str} Sl:{sleep_str}", (20, 90 + (i % 5) * 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # No keypoints detected
                cv2.putText(display_frame, "No pose detected", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Clean up people not detected for 30 frames (about 1 second at 30fps)
            ids_to_remove = []
            for persistent_id in list(person_state.keys()):
                if persistent_id not in detected_ids and (frame_count - last_detection.get(persistent_id, 0)) > 30:
                    ids_to_remove.append(persistent_id)
            
            with data_lock:
                for persistent_id in ids_to_remove:
                    if persistent_id in person_state: del person_state[persistent_id]
                    if persistent_id in person_last_time: del person_last_time[persistent_id]
                    if persistent_id in last_detection: del last_detection[persistent_id]
                    if persistent_id in lying_start_time: del lying_start_time[persistent_id]
                    if persistent_id in minor_fall_start_time: del minor_fall_start_time[persistent_id]
                    if persistent_id in recovery_mode: del recovery_mode[persistent_id]
                    if persistent_id in active_fall_event: del active_fall_event[persistent_id]
                    if persistent_id in recovery_confirm_count: del recovery_confirm_count[persistent_id]
                    
                    # Clean up spatial tracking
                    if persistent_id in person_last_bbox: del person_last_bbox[persistent_id]
                    if persistent_id in person_velocity: del person_velocity[persistent_id]
                    if persistent_id in person_center_history: del person_center_history[persistent_id]
                    
                    # Clean up tracker mapping to prevent stale entries
                    yolo_keys = [k for k, v in tracker_to_persistent.items() if v == persistent_id]
                    for k in yolo_keys: del tracker_to_persistent[k]
                    
                    print(f"Removed internal ID {persistent_id} from active tracking")

            # Log progress every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"Frame {frame_count} | FPS: {fps:.1f} | People tracked: {len(person_state)}")

            # Show status message if active
            if time.time() < status_expiry:
                cv2.rectangle(display_frame, (0, 0), (w, 40), (0, 255, 0), -1)
                cv2.putText(display_frame, status_message, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # Always show camera feed
            cv2.imshow("Elderly Monitor", display_frame)

            # Press ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        except Exception as e:
            print(f"Error at frame {frame_count}: {e}")
            import traceback
            traceback.print_exc()
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    print("Program exited.")
    sys.exit(0)
