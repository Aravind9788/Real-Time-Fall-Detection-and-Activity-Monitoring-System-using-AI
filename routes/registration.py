"""
Registration routes for face and person registration
"""

import time
from fastapi import APIRouter, Form, File, UploadFile
from typing import Optional, Dict, Set, Any
from collections import defaultdict

router = APIRouter(tags=["registration"])

# Global state (will be injected from main.py)
last_frame = None
status_message = ""
status_expiry = 0
person_state: Dict[str, str] = {}
all_tracked_people: Set[str] = set()
model = None
rename_person_func = None
register_face_func = None


def init_registration_state(
    frame,
    state: Dict[str, str],
    tracked_people: Set[str],
    yolo_model,
    rename_func,
    register_func
):
    """Initialize global state references from main.py"""
    global last_frame, person_state, all_tracked_people, model, rename_person_func, register_face_func
    last_frame = frame
    person_state = state
    all_tracked_people = tracked_people
    model = yolo_model
    rename_person_func = rename_func
    register_face_func = register_func


@router.post("/register")
async def register(
    name: str = Form(None),
    yolo_id: str = Form(None),
    front: UploadFile = File(None),
    back: UploadFile = File(None)
):
    """Register a new person with face recognition or manual ID naming"""
    global last_frame, status_message, status_expiry
    
    if not name:
        return {"status": "error", "message": "Missing name"}
    
    # CASE 1: Manual ID naming (No face needed)
    if yolo_id and (str(yolo_id) in person_state or str(yolo_id) in all_tracked_people):
        rename_person_func(str(yolo_id), name)
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
    
    if register_face_func(to_process, name, yolo_model=model if not (front or back) else None):
        status_message = f"Registered: {name}"
        status_expiry = time.time() + 5
        return {"status": "success", "message": f"Successfully registered {name}"}
    return {"status": "error", "message": "No face detected in provided images"}
