"""
Alert-related routes for fall detection system
"""

import time
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter(tags=["alerts"])

# Global state (will be injected from main.py)
fall = False
active_alerts: List[Dict[str, Any]] = []


class FallAlert(BaseModel):
    person: str = "Unknown"


def init_alerts_state(fall_state: bool, alerts_list: List[Dict[str, Any]]):
    """Initialize global state references from main.py"""
    global fall, active_alerts
    fall = fall_state
    active_alerts = alerts_list


@router.post("/trigger")
async def trigger(alert: FallAlert = None):
    """Trigger a fall alert"""
    global fall
    if alert is None:
        alert = FallAlert()
    pid = alert.person
    # Extract type from pid if sent as string "MAJOR FALL (ID 1)"
    fall_type = "FALL"
    if "MAJOR" in str(pid): 
        fall_type = "MAJOR FALL"
    elif "MINOR" in str(pid): 
        fall_type = "MINOR FALL"
    elif "RECOVERED" in str(pid): 
        fall_type = "RECOVERED"
    
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


@router.post("/api/acknowledge/{pid}")
async def acknowledge(pid: str):
    """Acknowledge a fall alert for a specific person"""
    global active_alerts
    active_alerts = [a for a in active_alerts if str(a['person']) != str(pid)]
    return {"status": "success"}


@router.get("/fall")
async def check():
    """Check if there's a fall alert"""
    global fall
    if fall:
        fall = False
        return {"fall": True}
    return {"fall": False}
