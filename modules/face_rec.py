"""
Face Recognition & Registration Module

This module handles face recognition using the face_recognition library,
including face registration, encoding storage, and color signature matching
for person identification.
"""

import os
import pickle
import cv2
import numpy as np
import face_recognition
import threading


# Setup
FACES_DIR = "registered_faces"
ENCODINGS_FILE = "encodings.pickle"
if not os.path.exists(FACES_DIR): 
    os.makedirs(FACES_DIR)

# Global storage
known_face_encodings = []
known_face_names = []
data_lock = threading.Lock()


def load_encodings():
    """Load face encodings from disk"""
    global known_face_encodings, known_face_names
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
    print(f"Loaded {len(known_face_names)} registered faces.")


def register_face(image_or_list, name, yolo_model=None):
    """
    Register a face with the given name
    
    Args:
        image_or_list: Single image (numpy array or FileStorage) or list of images
        name: Name to associate with the face
        yolo_model: Optional YOLO model to crop people first for better accuracy
        
    Returns:
        bool: True if registration successful, False otherwise
    """
    global known_face_encodings, known_face_names
    if image_or_list is None: 
        return False
    
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
                if img is None: 
                    continue
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
    """
    Process a single image for face registration
    
    Args:
        img: Image as numpy array
        name: Name to associate with detected face
        
    Returns:
        bool: True if face detected and encoded, False otherwise
    """
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
    except: 
        pass
    return False


def get_color_signature(image):
    """
    Calculate color histogram for Re-Identification
    
    Args:
        image: Person crop image
        
    Returns:
        Normalized histogram or None if failed
    """
    try:
        if image is None or image.size == 0: 
            return None
        # Focus on the torso (center of the crop)
        h, w = image.shape[:2]
        torso = image[int(h*0.2):int(h*0.7), int(w*0.2):int(w*0.8)]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    except: 
        return None


def compare_signatures(sig1, sig2):
    """
    Compare two color histograms
    
    Args:
        sig1, sig2: Color histograms to compare
        
    Returns:
        float: Correlation score (0.0 to 1.0)
    """
    if sig1 is None or sig2 is None: 
        return 0
    return cv2.compareHist(sig1, sig2, cv2.HISTCMP_CORREL)


# Load encodings on module import
load_encodings()


def recognize_face(person_img, tolerance=0.6):
    """
    Recognize a face in the given image crop.
    
    Args:
        person_img: BGR image crop of the person
        tolerance: Distance tolerance for face matching
        
    Returns:
        str: Name of the matched person, or None if no match/no face
    """
    try:
        # Use thread-safe copy of encodings/names
        with data_lock:
            target_encodings = known_face_encodings[:]
            target_names = known_face_names[:]
        
        if not target_encodings:
            return None
            
        rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        # Using upsample=1 for better accuracy on small crops
        face_locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)
        
        if not face_locations:
            return None
            
        face_encodings = face_recognition.face_encodings(rgb, face_locations)
        
        for fe in face_encodings:
            matches = face_recognition.compare_faces(target_encodings, fe, tolerance=tolerance)
            if True in matches:
                first_match_index = matches.index(True)
                return str(target_names[first_match_index])
                
        return None
    except Exception as e:
        print(f"Face Recognition Error: {e}")
        return None
