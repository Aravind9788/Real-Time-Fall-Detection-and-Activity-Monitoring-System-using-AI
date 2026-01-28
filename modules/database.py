"""
Database & Logging Module

This module handles SQLite database operations for activity tracking,
fall event logging, and persistent ID mapping storage.
"""

import sqlite3
import pickle
import time
import os
import requests
import threading
from datetime import datetime, date


# Database configuration
DB_PATH = "monitor_data.db"
ID_MAP_FILE = "manual_id_map.pickle"

# Global variables (will be imported by main.py)
manual_id_map = {}
last_global_alert_time = 0
last_alert_coords = {}
last_alert_pid = {}


def init_db():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Table for daily activity summaries
    c.execute('''CREATE TABLE IF NOT EXISTS activity
                 (date TEXT, person_id TEXT, walking REAL, sitting REAL, sleeping REAL, PRIMARY KEY(date, person_id))''')
    # Table for fall events
    c.execute('''CREATE TABLE IF NOT EXISTS falls
                 (timestamp DATETIME, person_id TEXT, type TEXT)''')
    conn.commit()
    conn.close()


def log_activity_to_db(all_tracked_people, walking_time, sitting_time, sleeping_time, data_lock, reid_manager_instance, save_manual_id_map_func):
    """
    Sync current in-memory stats to DB and save ReID banks every minute
    
    This function runs in a background thread and periodically saves activity data.
    
    Args:
        all_tracked_people: Set of all tracked person IDs
        walking_time: Dict mapping person ID to walking duration
        sitting_time: Dict mapping person ID to sitting duration
        sleeping_time: Dict mapping person ID to sleeping duration
        data_lock: Threading lock for thread-safe access
        reid_manager_instance: ReID manager instance to save
        save_manual_id_map_func: Function to save manual ID mappings
    """
    while True:
        time.sleep(60)
        # Snapshot the data while holding lock to minimize contention
        with data_lock:
            stats_snapshot = []
            for pid in list(all_tracked_people):
                stats_snapshot.append((str(pid), walking_time.get(pid, 0), 
                                     sitting_time.get(pid, 0), sleeping_time.get(pid, 0)))
        
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            today = str(date.today())
            for pid, w, s, sl in stats_snapshot:
                c.execute('''INSERT OR REPLACE INTO activity (date, person_id, walking, sitting, sleeping)
                             VALUES (?, ?, ?, ?, ?)''', (today, pid, w, s, sl))
            conn.commit()
            conn.close()
            
            # Save ReID and Identity mapping
            reid_manager_instance.prune_bank()
            reid_manager_instance.save_bank()
            save_manual_id_map_func()
            print("✓ Database and ReID banks synchronized.")
        except Exception as e:
            print(f"Sync Error: {e}")


def load_manual_id_map():
    """Load manual ID mappings from disk"""
    global manual_id_map
    if os.path.exists(ID_MAP_FILE):
        try:
            with open(ID_MAP_FILE, "rb") as f:
                manual_id_map = pickle.load(f)
            print(f"Loaded {len(manual_id_map)} manual ID mappings.")
        except Exception as e:
            print(f"Error loading manual ID map: {e}")


def save_manual_id_map():
    """Save manual ID mappings to disk"""
    try:
        with open(ID_MAP_FILE, "wb") as f:
            pickle.dump(manual_id_map, f)
    except Exception as e:
        print(f"Error saving manual ID map: {e}")


def load_stats_from_db(walking_time, sitting_time, sleeping_time, all_tracked_people):
    """
    Load today's activity statistics from database
    
    Args:
        walking_time: Dict to populate with walking durations
        sitting_time: Dict to populate with sitting durations
        sleeping_time: Dict to populate with sleeping durations
        all_tracked_people: Set to populate with tracked person IDs
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        today = str(date.today())
        c.execute("SELECT person_id, walking, sitting, sleeping FROM activity WHERE date=?", (today,))
        rows = c.fetchall()
        for r in rows:
            pid, w, s, sl = r
            # Use the person_id as stored in DB
            walking_time[pid] = w
            sitting_time[pid] = s
            sleeping_time[pid] = sl
            all_tracked_people.add(pid)
        conn.close()
        print(f"Loaded stats for {len(rows)} people from database.")
    except Exception as e:
        print(f"Error loading stats: {e}")


def send_fall_alert(alert_msg, pid, fall_type, coords=None):
    """
    Send fall alert and log to database with spatial-temporal squelching
    
    Args:
        alert_msg: Alert message to send
        pid: Person ID
        fall_type: Type of fall (MAJOR FALL, MINOR FALL, RECOVERED)
        coords: Optional (x, y) coordinates for spatial deduplication
    """
    global last_global_alert_time, last_alert_coords, last_alert_pid
    try:
        now = time.time()
        
        # Spatial-Temporal Squelch:
        # If we sent an alert of this type recently (< 5s) and it was in the same area (< 150px)
        # then it's likely a phantom ID/tracker drift for the same person.
        if coords and fall_type in last_alert_coords:
            prev_coords = last_alert_coords[fall_type]
            import numpy as np
            dist = np.sqrt((coords[0]-prev_coords[0])**2 + (coords[1]-prev_coords[1])**2)
            if dist < 150 and (now - last_global_alert_time) < 5:
                # If it's the SAME person (resolved name), definitely skip
                # If it's a DIFFERENT person but very close/recent, it's likely a ghost ID
                print(f"🤫 Squelching redundant {fall_type} for {pid} (likely ghost ID)")
                return
        
        # Update last alert state
        last_global_alert_time = now
        if coords: 
            last_alert_coords[fall_type] = coords
        last_alert_pid[fall_type] = pid

        print(f"⚠️  {alert_msg}! Sending alert...")
        # Log to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO falls (timestamp, person_id, type) VALUES (?, ?, ?)",
                  (datetime.now(), str(pid), fall_type))
        conn.commit()
        conn.close()
        
        from modules.detection import FALL_URL
        requests.post(FALL_URL, json={"person": alert_msg}, timeout=1)
        print("✓ Fall alert sent and logged successfully")
    except Exception as e:
        print(f"✗ Failed to send/log fall alert: {e}")


def get_fall_history(limit=50):
    """
    Retrieve fall event history from database.
    
    Args:
        limit: Maximum number of events to return (default 50)
        
    Returns:
        List of fall events with timestamp, person, and type
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            SELECT timestamp, person_id, type
            FROM falls
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = c.fetchall()
        conn.close()
        
        events = []
        for row in rows:
            # Parse timestamp and format for display
            try:
                dt = datetime.fromisoformat(str(row[0]))
                time_str = dt.strftime("%H:%M:%S")
                unix_time = dt.timestamp()
            except:
                time_str = str(row[0])[:8]  # Fallback to first 8 chars
                unix_time = 0
            
            events.append({
                "timestamp": time_str,
                "person": row[1],
                "type": row[2],
                "time": unix_time
            })
        
        return events
        
    except Exception as e:
        print(f"Error fetching fall history: {e}")
        return []


# Initialize database on module import
init_db()
load_manual_id_map()
