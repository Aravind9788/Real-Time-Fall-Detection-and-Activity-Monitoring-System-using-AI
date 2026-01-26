import cv2
import time
import numpy as np
import requests
import sys
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO
from collections import defaultdict
import sqlite3
import pickle
import face_recognition
from datetime import datetime, date
from flask import Flask, jsonify, request
import threading

data_lock = threading.Lock()

# ==================== Person Re-Identification (ReID) ====================
class ReIDManager:
    def __init__(self, threshold=0.75):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Using ResNet50 as a robust feature extractor (2048D embeddings)
        # Optimized for clothing, pose and lighting invariance
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Identity() # Remove classification head to get embeddings
        self.model.to(self.device)
        self.model.eval()
        
        self.threshold = threshold
        # Identity Store: persistent_id -> {'embedding': tensor, 'last_seen': timestamp}
        self.identity_bank = {}
        self.next_persistent_id = 1
        self.bank_file = "reid_bank.pickle"
        self.load_bank()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_bank(self):
        if os.path.exists(self.bank_file):
            try:
                with open(self.bank_file, "rb") as f:
                    data = pickle.load(f)
                    
                if isinstance(data, dict):
                    if 'bank' in data:
                        self.identity_bank = data['bank']
                        self.next_persistent_id = data.get('next_id', 1)
                    else:
                        # Legacy format: the whole pickle was the bank
                        self.identity_bank = data
                        # Estimate next_id from keys like "Person_N"
                        pids = [int(k.split('_')[1]) for k in data.keys() if isinstance(k, str) and k.startswith("Person_")]
                        self.next_persistent_id = max(pids) + 1 if pids else 1
                
                # Validation: Remove invalid entries that would cause KeyError
                valid_bank = {}
                for pid, entry in self.identity_bank.items():
                    if isinstance(entry, dict) and 'embedding' in entry:
                        valid_bank[pid] = entry
                    elif isinstance(entry, np.ndarray):
                        # Very old format where entry WAS the embedding
                        valid_bank[pid] = {'embedding': entry, 'last_seen': time.time()}
                
                self.identity_bank = valid_bank
                print(f"Loaded {len(self.identity_bank)} valid identities from ReID bank.")
            except Exception as e:
                print(f"Error loading ReID bank: {e}")

    def save_bank(self):
        try:
            with open(self.bank_file, "wb") as f:
                pickle.dump({'bank': self.identity_bank, 'next_id': self.next_persistent_id}, f)
        except Exception as e:
            print(f"Error saving ReID bank: {e}")

    @torch.no_grad()
    def get_embedding(self, person_crop):
        if person_crop.size == 0: return None
        img_t = self.transform(person_crop).unsqueeze(0).to(self.device)
        embedding = self.model(img_t)
        # L2 Normalize for cosine similarity
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy().flatten()

    def match_identity(self, current_embedding):
        if current_embedding is None: return None
        
        best_id = None
        max_sim = -1
        
        for pid, data in self.identity_bank.items():
            if 'embedding' not in data: continue
            # Cosine similarity (since embeddings are L2 normalized, it's just a dot product)
            similarity = np.dot(current_embedding, data['embedding'])
            if similarity > max_sim:
                max_sim = similarity
                best_id = pid
        
        if max_sim > self.threshold:
            # Update the representative embedding (moving average for robustness)
            self.identity_bank[best_id]['embedding'] = (
                self.identity_bank[best_id]['embedding'] * 0.9 + current_embedding * 0.1
            )
            # Re-normalize after update
            self.identity_bank[best_id]['embedding'] /= np.linalg.norm(self.identity_bank[best_id]['embedding'])
            self.identity_bank[best_id]['last_seen'] = time.time()
            return best_id
        
        # New identity
        new_id = f"Person_{self.next_persistent_id}"
        self.next_persistent_id += 1
        self.identity_bank[new_id] = {
            'embedding': current_embedding,
            'last_seen': time.time(),
            'first_seen': time.time()
        }
        return new_id

    def prune_bank(self, max_idle=3600, min_duration=5):
        """Remove short-lived 'ghost' IDs to prevent bank bloat"""
        now = time.time()
        to_delete = []
        for pid, data in self.identity_bank.items():
            idle_time = now - data['last_seen']
            duration = data['last_seen'] - data.get('first_seen', data['last_seen'])
            
            # If seen once and never again for an hour, or tracked for < 5s then gone
            if idle_time > max_idle or (idle_time > 300 and duration < min_duration):
                to_delete.append(pid)
        
        for pid in to_delete:
            del self.identity_bank[pid]
        if to_delete:
            print(f"🧹 Pruned {len(to_delete)} ghost identities from ReID bank.")

reid_manager = ReIDManager(threshold=0.85)
tracker_to_persistent = {} # Maps YOLO tracker_id -> persistent_id

# ==================== Face Recognition Setup ====================
FACES_DIR = "registered_faces"
ENCODINGS_FILE = "encodings.pickle"
if not os.path.exists(FACES_DIR): os.makedirs(FACES_DIR)

known_face_encodings = []
known_face_names = []

def load_encodings():
    global known_face_encodings, known_face_names
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
    print(f"Loaded {len(known_face_names)} registered faces.")

load_encodings()

def register_face(image_or_list, name, yolo_model=None):
    global known_face_encodings, known_face_names
    if image_or_list is None: return False
    
    images = image_or_list if isinstance(image_or_list, list) else [image_or_list]
    success_count = 0
    
    for image in images:
        try:
            if isinstance(image, np.ndarray):
                # If we have a YOLO model, try to crop people out first for better accuracy
                if yolo_model:
                    results = yolo_model(image, verbose=False)
                    if results[0].boxes:
                        for box in results[0].boxes.xyxy:
                            x1, y1, x2, y2 = map(int, box.cpu().numpy())
                            crop = image[y1:y2, x1:x2]
                            if process_single_image(crop, name):
                                success_count += 1
                        continue # Already processed crops for this frame
                
                if process_single_image(image, name):
                    success_count += 1
            else: # Flask FileStorage
                file_bytes = np.frombuffer(image.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None: continue
                if process_single_image(img, name):
                    success_count += 1
        except Exception as e:
            print(f"Face Registration Error: {e}")
            
    if success_count > 0:
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
        return True
    return False

def process_single_image(img, name):
    global known_face_encodings, known_face_names
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)
        encodings = face_recognition.face_encodings(rgb, boxes)
        if len(encodings) > 0:
            with data_lock:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
            return True
    except: pass
    return False

# ==================== Database Setup ====================
DB_PATH = "monitor_data.db"

def init_db():
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

init_db()

def log_activity_to_db():
    """Sync current in-memory stats to DB and save ReID banks every minute"""
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
            reid_manager.prune_bank()
            reid_manager.save_bank()
            save_manual_id_map()
            print("✓ Database and ReID banks synchronized.")
        except Exception as e:
            print(f"Sync Error: {e}")

# Start DB sync thread
threading.Thread(target=log_activity_to_db, daemon=True).start()

# ==================== Flask Server ====================
app = Flask(__name__)
fall = False
active_alerts = []  # List of unacknowledged falls

# Track walking/sleeping/sitting durations
walking_time = defaultdict(float)
sleeping_time = defaultdict(float)
sitting_time = defaultdict(float)

# Track current state per person
person_state = {}
person_last_time = {}
lying_start_time = {} # Track when a person started lying down
minor_fall_start_time = {} # Track duration of minor fall for escalation
recovery_mode = {} # pid -> expiry_time (suppress minor fall alerts while getting up)
recovery_confirm_count = {} # persistent_id -> count (frames of sustained upright activity)
active_fall_event = {} # pid -> True (prevent multiple alerts for the same fall)

# Squelch logic for ghost IDs/phantom bodies
last_global_alert_time = 0
last_alert_coords = {} # type -> (x, y)
last_alert_pid = {}   # type -> pid
all_tracked_people = set()  # Persistent list of all detected IDs
manual_id_map = {} # Manual link: YOLO_ID -> Registered Name
person_signatures = {} # Store color histograms: Name -> Histogram

ID_MAP_FILE = "manual_id_map.pickle"

def load_manual_id_map():
    global manual_id_map
    if os.path.exists(ID_MAP_FILE):
        try:
            with open(ID_MAP_FILE, "rb") as f:
                manual_id_map = pickle.load(f)
            print(f"Loaded {len(manual_id_map)} manual ID mappings.")
        except Exception as e:
            print(f"Error loading manual ID map: {e}")

def save_manual_id_map():
    try:
        with open(ID_MAP_FILE, "wb") as f:
            pickle.dump(manual_id_map, f)
    except Exception as e:
        print(f"Error saving manual ID map: {e}")

load_manual_id_map()

def get_color_signature(image):
    """Calculate color histogram for Re-Identification"""
    try:
        if image is None or image.size == 0: return None
        # Focus on the torso (center of the crop)
        h, w = image.shape[:2]
        torso = image[int(h*0.2):int(h*0.7), int(w*0.2):int(w*0.8)]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    except: return None

def compare_signatures(sig1, sig2):
    """Compare two color histograms (0.0 to 1.0)"""
    if sig1 is None or sig2 is None: return 0
    return cv2.compareHist(sig1, sig2, cv2.HISTCMP_CORREL)

status_message = ""
status_expiry = 0

def load_stats_from_db():
    global walking_time, sleeping_time, sitting_time, all_tracked_people
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

load_stats_from_db()

@app.route("/trigger", methods=["POST"])
def trigger():
    global fall
    data = request.get_json(silent=True) or {}
    pid = data.get("person", "Unknown")
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
    return "OK"

@app.route("/api/acknowledge/<pid>", methods=["POST"])
def acknowledge(pid):
    global active_alerts
    active_alerts = [a for a in active_alerts if str(a['person']) != str(pid)]
    return jsonify({"status": "success"})

@app.route("/fall")
def check():
    global fall
    if fall:
        fall = False
        return jsonify({"fall": True})
    return jsonify({"fall": False})

# Track fall events with timestamps (history)
fall_events = []

last_frame = None

def rename_person(old_id, new_name):
    """Safely rename a person and transfer all their stats and mappings."""
    with data_lock:
        manual_id_map[str(old_id)] = new_name
        
        # Transfer current stats
        walking_time[new_name] = walking_time.get(new_name, 0) + walking_time.pop(old_id, 0)
        sitting_time[new_name] = sitting_time.get(new_name, 0) + sitting_time.pop(old_id, 0)
        sleeping_time[new_name] = sleeping_time.get(new_name, 0) + sleeping_time.pop(old_id, 0)
        
        # Update set of all people (remove old ID if it's not the same as new name)
        if str(old_id) in all_tracked_people and str(old_id) != new_name:
            all_tracked_people.remove(old_id)
        all_tracked_people.add(new_name)
        
        # Save mappings immediately
        save_manual_id_map()
        print(f"👤 Renamed {old_id} to {new_name} and transferred stats.")

@app.route("/register", methods=["POST"])
def register():
    global last_frame, status_message, status_expiry
    name = request.form.get("name")
    target_id = request.form.get("yolo_id") # Register by active ID (can be persistent_id)
    
    if not name:
        return jsonify({"status": "error", "message": "Missing name"})
    
    # CASE 1: Manual ID naming (No face needed)
    if target_id and (str(target_id) in person_state or str(target_id) in all_tracked_people):
        rename_person(str(target_id), name)
        status_message = f"ID {target_id} is now {name}"
        status_expiry = time.time() + 5
        return jsonify({"status": "success", "message": f"Successfully named body {target_id} as {name}"})

    # CASE 2: Uploaded files or Live frame
    front_img = request.files.get("front")
    back_img = request.files.get("back")
    
    to_process = []
    if front_img: to_process.append(front_img)
    if back_img: to_process.append(back_img)
    
    if not to_process:
        if last_frame is not None:
            to_process = [last_frame]
        else:
            return jsonify({"status": "error", "message": "No photos uploaded or live frame available"})
    
    if register_face(to_process, name, yolo_model=model if not (front_img or back_img) else None):
        status_message = f"Registered: {name}"
        status_expiry = time.time() + 5
        return jsonify({"status": "success", "message": f"Successfully registered {name}"})
    return jsonify({"status": "error", "message": "No face detected in provided images"})

@app.route("/api/history")
def activity_history():
    """Get history for charts"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT date, person_id, walking, sitting, sleeping FROM activity ORDER BY date DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return jsonify([{"date": r[0], "pid": r[1], "walk": r[2], "sit": r[3], "sleep": r[4]} for r in rows])

@app.route("/")
def home():
    """Enhanced modern dashboard"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Elderly Monitor Pro</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .navbar { background-color: #212529; color: white; margin-bottom: 30px; }
            .card { border: none; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; transition: transform 0.2s; }
            .card:hover { transform: translateY(-5px); }
            .status-badge { padding: 8px 16px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
            .bg-walking { background-color: #d1e7dd; color: #0f5132; }
            .bg-sleeping { background-color: #cfe2ff; color: #084298; }
            .bg-sitting { background-color: #fff3cd; color: #856404; }
            .bg-major-fall { background-color: #dc3545; color: white; }
            .bg-minor-fall { background-color: #198754; color: white; }
            .bg-recovered { background-color: #d1e7dd; color: #0f5132; border: 1px solid #198754; }
            .bg-away { background-color: #e9ecef; color: #6c757d; }
            .alert-item { border-left: 5px solid #dc3545; background-color: #fff; padding: 15px; margin-bottom: 10px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; }
            .time-text { font-size: 0.8rem; color: #6c757d; }
            #live-time { font-weight: 500; }
        </style>
    </head>
    <body>
        <nav class="navbar px-4 py-3">
            <h4 class="mb-0">👴 Elderly Activity Monitor Pro</h4>
            <div class="d-flex align-items-center bg-dark p-2 rounded">
                <input type="text" id="reg-name" class="form-control form-control-sm me-2" placeholder="Person Name" style="width: 150px;">
                <div class="me-2 text-white small">
                    <label class="mb-0">Front:</label>
                    <input type="file" id="reg-front" class="form-control form-control-sm" accept="image/*" style="width: 100px; display: inline-block;">
                </div>
                <div class="me-2 text-white small">
                    <label class="mb-0">Side/Back:</label>
                    <input type="file" id="reg-back" class="form-control form-control-sm" accept="image/*" style="width: 100px; display: inline-block;">
                </div>
                <button class="btn btn-primary btn-sm" onclick="registerPerson()">Register Person</button>
                <div id="live-time" class="ms-3"></div>
            </div>
        </nav>

        <div class="container">
            <div id="alert-container"></div>

            <div id="unnamed-container" class="mb-4 d-none">
                <div class="card bg-dark text-white p-3">
                    <h6>👤 Active Unidentified Bodies (Manual Naming)</h6>
                    <div id="unnamed-list" class="d-flex flex-wrap"></div>
                    <small class="text-muted mt-2">If face detection fails, look at the camera window for the ID and name it here.</small>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-8">
                    <h5 class="mb-4">Live Activity Tracking</h5>
                    <div id="people-grid" class="row"></div>
                    
                    <h5 class="mt-4 mb-4">Activity Trends (Daily)</h5>
                    <div class="card p-3">
                        <canvas id="activityChart"></canvas>
                    </div>
                </div>
                <div class="col-md-4">
                    <h5 class="mb-4">Recent Fall History</h5>
                    <div id="fall-history" class="list-group"></div>
                </div>
            </div>
        </div>

        <script>
            let activityChart = null;

            function registerPerson(yoloId = null) {
                const name = document.getElementById('reg-name').value;
                if (!name) { alert('Enter a name first, then click name body'); return; }
                
                const formData = new FormData();
                formData.append('name', name);
                if (yoloId) formData.append('yolo_id', yoloId);
                
                const front = document.getElementById('reg-front').files[0];
                const back = document.getElementById('reg-back').files[0];
                
                if (front) formData.append('front', front);
                if (back) formData.append('back', back);
                
                if (!front && !back && !yoloId) {
                    if(!confirm("No photos selected. Register using current live webcam frame?")) return;
                }

                fetch('/register', { method: 'POST', body: formData })
                    .then(r => r.json())
                    .then(data => {
                        alert(data.message);
                        if(data.status === 'success') {
                            document.getElementById('reg-name').value = '';
                            if(document.getElementById('reg-front')) document.getElementById('reg-front').value = '';
                            if(document.getElementById('reg-back')) document.getElementById('reg-back').value = '';
                        }
                    });
            }

            function initChart(data) {
                const ctx = document.getElementById('activityChart').getContext('2d');
                if (activityChart) activityChart.destroy();
                
                const labels = [...new Set(data.map(d => d.date))].reverse();
                const datasets = [];
                const colors = { walk: '#198754', sit: '#ffc107', sleep: '#0d6efd' };

                ['walk', 'sit', 'sleep'].forEach(type => {
                    datasets.push({
                        label: type.charAt(0).toUpperCase() + type.slice(1),
                        data: labels.map(l => {
                            const entries = data.filter(d => d.date === l);
                            return entries.reduce((acc, curr) => acc + curr[type], 0);
                        }),
                        backgroundColor: colors[type]
                    });
                });

                activityChart = new Chart(ctx, {
                    type: 'bar',
                    data: { labels, datasets },
                    options: { scales: { y: { beginAtZero: true, title: { display: true, text: 'Minutes' } } } }
                });
            }

            function ackFall(pid) {
                fetch(`/api/acknowledge/${pid}`, { method: 'POST' })
                    .then(() => update());
            }

            function update() {
                fetch('/api/report')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('live-time').textContent = new Date().toLocaleTimeString();
                        
                        // Active Alerts
                        let alertHtml = '';
                        for (let l of data.active_alerts) {
                            let alertColor = l.type === 'MAJOR FALL' ? 'text-danger' : 'text-success';
                            let icon = l.type === 'RECOVERED' ? '✅' : '⚠️';
                            alertHtml += `
                                <div class="alert-item shadow-sm animated shake" style="border-left-color: ${l.type === 'MAJOR FALL' ? '#dc3545' : '#198754'}">
                                    <div>
                                        <h5 class="mb-0 ${alertColor}">${icon} ${l.type} DETECTED</h5>
                                        <p class="mb-0">Person ID: <strong>${l.person}</strong> at ${l.time}</p>
                                    </div>
                                    <button class="btn btn-outline-dark btn-sm" onclick="ackFall('${l.person}')">Acknowledge</button>
                                </div>`;
                        }
                        document.getElementById('alert-container').innerHTML = alertHtml;
                        
                        // Unnamed IDs
                        const unnamedList = document.getElementById('unnamed-list');
                        const unnamedContainer = document.getElementById('unnamed-container');
                        if (data.unnamed_ids && data.unnamed_ids.length > 0) {
                            unnamedContainer.classList.remove('d-none');
                            let unnamedHtml = '';
                            data.unnamed_ids.forEach(id => {
                                unnamedHtml += `<button class="btn btn-outline-warning btn-sm me-2 mb-2" onclick="registerPerson('${id}')">Name Body ${id}</button>`;
                            });
                            unnamedList.innerHTML = unnamedHtml;
                        } else {
                            unnamedContainer.classList.add('d-none');
                        }

                        // People Grid
                        let peopleHtml = '';
                        for (let p of data.people) {
                            let activity = p.current_activity;
                            let badgeCls = 'bg-' + activity.toLowerCase().replace(' ', '-');
                            let opacity = p.is_active ? '1.0' : '0.6';
                            
                            peopleHtml += `
                                <div class="col-md-6" style="opacity: ${opacity}">
                                    <div class="card p-3">
                                        <div class="d-flex justify-content-between align-items-center mb-3">
                                            <h6 class="mb-0">Person ID: ${p.person} ${p.is_active ? '' : '(Away)'}</h6>
                                            <span class="status-badge ${badgeCls}">${activity}</span>
                                        </div>
                                        <div class="row text-center">
                                            <div class="col-4">
                                                <div class="small text-muted">Walking</div>
                                                <div class="h6 mb-0">${p.walking_dur}</div>
                                            </div>
                                            <div class="col-4 border-start">
                                                <div class="small text-muted">Sitting</div>
                                                <div class="h6 mb-0">${p.sitting_dur}</div>
                                            </div>
                                            <div class="col-4 border-start">
                                                <div class="small text-muted">Sleeping</div>
                                                <div class="h6 mb-0">${p.sleeping_dur}</div>
                                            </div>
                                        </div>
                                    </div>
                                </div>`;
                        }
                        document.getElementById('people-grid').innerHTML = peopleHtml || '<p class="text-muted">No people currently detected...</p>';
                        
                        // Fall History
                        let historyHtml = '';
                        for (let f of data.falls.reverse()) {
                            let historyColor = f.type === 'MAJOR FALL' ? 'text-danger' : 'text-success';
                            let histIcon = f.type === 'RECOVERED' ? '✅' : '⚠️';
                            historyHtml += `
                                <div class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                                    <div>
                                        <div class="fw-bold">Person ${f.person}</div>
                                        <div class="small ${historyColor}">${histIcon} ${f.type}</div>
                                    </div>
                                    <span class="time-text">${f.timestamp}</span>
                                </div>`;
                        }
                        document.getElementById('fall-history').innerHTML = historyHtml || '<p class="text-muted small">No fall history recorded.</p>';
                    });

                // Update charts every 30 seconds
                if (!window.lastChartUpdate || (Date.now() - window.lastChartUpdate > 30000)) {
                    fetch('/api/history')
                        .then(r => r.json())
                        .then(data => {
                            initChart(data);
                            window.lastChartUpdate = Date.now();
                        });
                }
            }
            setInterval(update, 2000);
            update();
        </script>
    </body>
    </html>
    """
    return html

def format_duration(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"

@app.route("/api/report")
def api_report():
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
            is_active = display_id in current_state_snapshot # If it's a persistent_id
            if not is_active:
                # Check if any persistent_id mapped to this name is active
                is_active = any(k in current_state_snapshot for k, v in manual_id_map.items() if v == display_id)
            
            total_time = walking_time.get(display_id, 0) + sitting_time.get(display_id, 0) + sleeping_time.get(display_id, 0)
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
                "sleeping_dur": format_duration(sleeping_time.get(display_id, 0)),
                "sitting_dur": format_duration(sitting_time.get(display_id, 0)),
                "current_activity": current_state_snapshot.get(internal_id, "AWAY"),
                "is_active": internal_id in current_state_snapshot
            })
            
        alerts_copy = active_alerts.copy()
        falls_copy = fall_events[-10:].copy()

    return jsonify({
        "people": report_data,
        "falls": falls_copy, 
        "active_alerts": alerts_copy,
        "unnamed_ids": unnamed_ids
    })

def run_server():
    # Runs Flask server in background thread
    global server_ready
    try:
        print("Flask: Starting server...")
        app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"Flask Error: {e}")
        import traceback
        traceback.print_exc()

print("Starting Flask server thread...")
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

import time as time_module
time_module.sleep(1)  # Give Flask time to start
print("✓ Flask server should be running at http://127.0.0.1:5000")
print("  Access the report at: http://127.0.0.1:5000/")

# ==================== YOLOv8 Pose Model ====================
model = YOLO("yolov8n-pose.pt")  # small & fast
FALL_URL = "http://127.0.0.1:5000/trigger"
fall_cooldown = {}  # Prevent spam alerts for same person

def classify_activity(keypoints, conf):
    """
    Advanced skeleton-based activity classification with Sitting and Major/Minor Fall.
    """
    try:
        # Check confidence of critical joints
        critical_joints = [5, 6, 11, 12] # Shoulders and Hips
        if any(conf[j] < 0.5 for j in critical_joints):
            if conf[0] > 0.5 and (conf[11] > 0.5 or conf[12] > 0.5):
                nose_y = keypoints[0][1]
                hip_y = (keypoints[11][1] + keypoints[12][1]) / 2 if (conf[11] > 0.5 and conf[12] > 0.5) else (keypoints[11][1] if conf[11] > 0.5 else keypoints[12][1])
                if nose_y > hip_y - 10:
                    return "LYING"
            return "UNKNOWN"

        # Calculate Midpoints and distances
        sho_y = (keypoints[5][1] + keypoints[6][1]) / 2
        sho_x = (keypoints[5][0] + keypoints[6][0]) / 2
        hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
        hip_x = (keypoints[11][0] + keypoints[12][0]) / 2
        
        dy = hip_y - sho_y
        dx = hip_x - sho_x
        angle = abs(np.degrees(np.arctan2(dx, dy)))
        
        # Height and width for ratio
        torso_len = np.sqrt(dx**2 + dy**2)
        sho_width = abs(keypoints[5][0] - keypoints[6][0])
        
        # 1. LYING (Horizontal and low)
        # Lowered threshold to 60 degrees to be more sensitive to lying down
        if angle > 60:
            return "LYING"
        
        # 2. MINOR FALL (Significant tilt but not fully flat, or head dropped)
        if angle > 40:
            # If ankles/knees are visible and head is very low
            if conf[15] > 0.5 and keypoints[0][1] > keypoints[11][1]:
                return "LYING"
            return "MINOR FALL"

        # 3. SITTING vs WALKING
        # Sitting: Vertical torso, but hips are lower (shorter total height)
        if conf[13] > 0.5 and conf[14] > 0.5 and conf[15] > 0.5:
            knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
            ank_y = (keypoints[15][1] + keypoints[16][1]) / 2
            
            upper_leg = abs(knee_y - hip_y)
            lower_leg = abs(ank_y - knee_y)
            
            # If knees are at similar horizontal level to hips, they are likely sitting
            if upper_leg < lower_leg * 0.5:
                return "SITTING"
                
        # Fallback for sitting: Torso length relative to total visible height
        if conf[15] > 0.5:
            total_h = abs(keypoints[15][1] - sho_y)
            if torso_len / total_h > 0.6: # Torso takes up too much of height
                return "SITTING"

        return "WALKING"

    except Exception:
        return "UNKNOWN"

def send_fall_alert(alert_msg, pid, fall_type, coords=None):
    global last_global_alert_time, last_alert_coords, last_alert_pid
    try:
        now = time.time()
        
        # Spatial-Temporal Squelch:
        # If we sent an alert of this type recently (< 5s) and it was in the same area (< 150px)
        # then it's likely a phantom ID/tracker drift for the same person.
        if coords and fall_type in last_alert_coords:
            prev_coords = last_alert_coords[fall_type]
            dist = np.sqrt((coords[0]-prev_coords[0])**2 + (coords[1]-prev_coords[1])**2)
            if dist < 150 and (now - last_global_alert_time) < 5:
                # If it's the SAME person (resolved name), definitely skip
                # If it's a DIFFERENT person but very close/recent, it's likely a ghost ID
                print(f"🤫 Squelching redundant {fall_type} for {pid} (likely ghost ID)")
                return
        
        # Update last alert state
        last_global_alert_time = now
        if coords: last_alert_coords[fall_type] = coords
        last_alert_pid[fall_type] = pid

        print(f"⚠️  {alert_msg}! Sending alert...")
        # Log to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO falls (timestamp, person_id, type) VALUES (?, ?, ?)",
                  (datetime.now(), str(pid), fall_type))
        conn.commit()
        conn.close()
        
        requests.post(FALL_URL, json={"person": alert_msg}, timeout=1)
        print("✓ Fall alert sent and logged successfully")
    except Exception as e:
        print(f"✗ Failed to send/log fall alert: {e}")

# ==================== Camera Loop ====================
def open_camera():
    """Robust camera opener for Windows"""
    for _ in range(3): # Try up to 3 times
        # Using DSHOW (DirectShow) on Windows is much more stable for resolution changes
        c = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(0)
        if c.isOpened():
            # Set buffer size to 1 for lowest latency
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return c
        time.sleep(0.5)
    return None

cap = open_camera()
if cap is None:
    print("Error: Cannot open camera")
    exit()

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
        if np.sum(thresh) > 10000: # Adjust sensitivity
            motion_detected = True
            last_motion_time = now
    else:
        # Reset if shape mismatched or first frame
        prev_gray = gray
        continue # Skip one frame to establish baseline
    prev_gray = gray

    # System sleeps if no motion for 10 seconds
    if now - last_motion_time > 10:
        if not system_sleeping:
            print("💤 Low Power Mode: No motion detected. Turning off webcam...")
            system_sleeping = True
            cv2.destroyAllWindows()
            cap.release() # Turn off the webcam hardware
        
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
        if not ret or frame is None or frame.size == 0: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if prev_gray is not None and prev_gray.shape == gray.shape:
            frame_delta = cv2.absdiff(prev_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            if np.sum(thresh) > 10000:
                last_motion_time = time.time() # Motion found!
        prev_gray = gray
        continue
    
    if system_sleeping:
        print("☀️ Motion detected! Webcam and AI Reopened.")
        # Restore full resolution for AI analysis
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        system_sleeping = False

    last_frame = frame.copy() # Store for registration
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
                pid = yolo_id_str # Fallback
                
                if x2 > x1 and y2 > y1:
                    person_img = frame[y1:y2, x1:x2]
                    
                    # 1. Person Re-Identification (Deep Embeddings)
                    # Optimization: Use cached ID for active trackers, update embedding every 30 frames
                    if yolo_id_str in tracker_to_persistent and frame_count % 30 != 0:
                        persistent_id = tracker_to_persistent[yolo_id_str]
                    else:
                        embedding = reid_manager.get_embedding(person_img)
                        persistent_id = reid_manager.match_identity(embedding)
                        tracker_to_persistent[yolo_id_str] = persistent_id
                    
                    pid = persistent_id # Current resolved ID (fallback)
                    
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
                activity = classify_activity(keypoints, confidences)

                with data_lock:
                    if persistent_id not in person_state:
                        person_state[persistent_id] = "UNKNOWN"
                        person_last_time[persistent_id] = now
                        all_tracked_people.add(pid) # Store resolved name for DB/Reporting
                        print(f"✓ New person detected: ID {pid} (Internal: {persistent_id})")

                # State Machine logic for ESCALATION and RECOVERY
                new_state = activity
                if activity == "UNKNOWN":
                    new_state = person_state.get(persistent_id, "UNKNOWN")
                
                # 1. Handle Risk States (Lying or Minor Fall)
                is_currently_down = (new_state in ["LYING", "MINOR FALL", "SLEEPING"])
                
                if is_currently_down:
                    recovery_confirm_count[persistent_id] = 0 # Reset recovery counter
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
                    if persistent_id not in lying_start_time: lying_start_time[persistent_id] = now
                    elif now - lying_start_time[persistent_id] > 10: new_state = "SLEEPING"

                # 4. Special case: If in recovery mode, don't show "MINOR FALL"
                if persistent_id in recovery_mode:
                    if now > recovery_mode[persistent_id]: del recovery_mode[persistent_id]
                    elif new_state == "MINOR FALL": new_state = "WALKING"
                
                # Accumulate time for CURRENT activity (every frame)
                duration = now - person_last_time[persistent_id]
                if duration > 0:
                    with data_lock:
                        prev_state = person_state.get(persistent_id, "UNKNOWN")
                        if prev_state == "WALKING": walking_time[pid] += duration
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
