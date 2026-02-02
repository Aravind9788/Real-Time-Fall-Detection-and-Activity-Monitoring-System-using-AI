"""
API data endpoints for activity history and reports
"""

import sqlite3
from fastapi import APIRouter
from typing import Dict, Set, List, Any
from collections import defaultdict

router = APIRouter(tags=["api"])

# Global state (will be injected from main.py)
person_state: Dict[str, str] = {}
all_tracked_people: Set[str] = set()
manual_id_map: Dict[str, str] = {}
walking_time: Dict[str, float] = defaultdict(float)
sitting_time: Dict[str, float] = defaultdict(float)
sleeping_time: Dict[str, float] = defaultdict(float)
standing_time: Dict[str, float] = defaultdict(float)
active_alerts: List[Dict[str, Any]] = []
data_lock = None
DB_PATH = ""
format_duration_func = None
get_fall_history_func = None


def init_api_state(
    state: Dict[str, str],
    tracked_people: Set[str],
    id_map: Dict[str, str],
    walk_time: Dict[str, float],
    sit_time: Dict[str, float],
    sleep_time: Dict[str, float],
    stand_time: Dict[str, float],
    alerts_list: List[Dict[str, Any]],
    lock,
    db_path: str,
    format_func,
    fall_history_func
):
    """Initialize global state references from main.py"""
    global person_state, all_tracked_people, manual_id_map
    global walking_time, sitting_time, sleeping_time, standing_time
    global active_alerts, data_lock, DB_PATH
    global format_duration_func, get_fall_history_func
    
    person_state = state
    all_tracked_people = tracked_people
    manual_id_map = id_map
    walking_time = walk_time
    sitting_time = sit_time
    sleeping_time = sleep_time
    standing_time = stand_time
    active_alerts = alerts_list
    data_lock = lock
    DB_PATH = db_path
    format_duration_func = format_func
    get_fall_history_func = fall_history_func


@router.get("/api/history")
async def activity_history():
    """Get activity history for charts"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT date, person_id, walking, sitting, sleeping FROM activity ORDER BY date DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return [{"date": r[0], "pid": r[1], "walk": r[2], "sit": r[3], "sleep": r[4]} for r in rows]


@router.get("/api/report")
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
                "walking_dur": format_duration_func(walking_time.get(display_id, 0)),
                "standing_dur": format_duration_func(standing_time.get(display_id, 0)),
                "sleeping_dur": format_duration_func(sleeping_time.get(display_id, 0)),
                "sitting_dur": format_duration_func(sitting_time.get(display_id, 0)),
                "current_activity": current_state_snapshot.get(internal_id, "AWAY"),
                "is_active": internal_id in current_state_snapshot
            })
            
        alerts_copy = active_alerts.copy()
        # Get fall history from database instead of memory
        falls_copy = get_fall_history_func(limit=10)

    return {
        "people": report_data,
        "falls": falls_copy, 
        "active_alerts": alerts_copy,
        "unnamed_ids": unnamed_ids
    }
