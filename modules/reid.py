"""
Person Re-Identification (ReID) Module

This module handles person re-identification using deep learning embeddings
from ResNet50. It maintains a persistent identity bank for tracking individuals
across frames and camera sessions.
"""

import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights


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


# Global instances
reid_manager = ReIDManager(threshold=0.85)
tracker_to_persistent = {}  # Maps YOLO tracker_id -> persistent_id
