"""
SafeFusion Demo v11 - Live threat detection with YOLOv8-Nano and YAMNet
Visual + Audio fusion with rule-based decision logic and stacked UI layout to prevent text overlap.

Fix v6: Text panel and camera feed are separate stacked regions, preventing overlap with YOLO labels.
Fix v7: Fetch YAMNet class map from public URL instead of relying on model's class_map_path() method.
Fix v8: Add camera-covered detection based on brightness and texture variance heuristics.
Fix v10: Decision CSV logging for offline evaluation (throttled, append-only).
Fix v11: Accurate timing with perf_counter, vision-only toggle, evidence snapshots, event history panel.
"""

import argparse
import cv2
import numpy as np
import os
import threading
import time
import collections
import urllib.request
import csv
import io
from datetime import datetime
from collections import deque
from typing import Optional, Dict, Tuple, List, Any

# Try to import audio dependencies
HAS_AUDIO = False
try:
    import sounddevice as sd
    import tensorflow as tf
    import tensorflow_hub as hub
    HAS_AUDIO = True
except ImportError as e:
    print(f"Audio dependencies not available: {e}")
    print("Running in video-only mode.")

# Try to import winsound for alerts
HAS_WINSOUND = False
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    print("winsound not available (alerts disabled)")

# ============================================================================
# CONFIGURATION
# ============================================================================

CAMERA_INDEX = 0
AUDIO_ENABLED = True
YOLO_MODEL = "yolov8n.pt"

# Fusion thresholds
TAU_W = 0.4      # Weapon confidence threshold for CRITICAL
TAU_P = 0.5      # Person confidence threshold
TAU_A = 0.6      # Audio distress threshold for HIGH
TAU_A_STAR = 0.8 # Audio distress threshold for WARNING (visual-only fallback)

# Temporal smoothing
CONFIDENCE_HOLD_SECONDS = 1.5

# Camera-covered detection thresholds
BRIGHTNESS_THRESHOLD = 25   # Mean pixel value below which frame is considered dark (0-255 scale)
TEXTURE_THRESHOLD = 15     # Standard deviation below which frame is considered featureless

# COCO classes to track
TRACKED_CLASSES = {"person", "knife", "scissors", "baseball bat"}
WEAPON_CLASSES = {"knife", "scissors", "baseball bat"}

# Audio distress keywords
DISTRESS_KEYWORDS = {
    "scream", "shout", "yell", "crying", "sobbing",
    "glass", "shatter", "alarm", "siren", "gunshot", "explosion", "battle cry"
}

# ============================================================================
# SHARED STATE (thread-safe)
# ============================================================================

class SharedState:
    """Thread-safe shared state for visual and audio detection results."""
    
    def __init__(self):
        self._lock = threading.Lock()
        
        # Visual detection results
        self.visual_detections = []  # List of (bbox, class_name, confidence)
        self.C_weapon = 0.0
        self.C_weapon_raw = 0.0  # unsmoothed max weapon conf from the latest frame
        self.C_person = 0.0
        self.latest_frame = None  # Latest camera frame for display
        self.camera_covered = False  # Whether camera lens is physically covered
        
        # Temporal smoothing histories
        self.weapon_history = deque()  # (timestamp, confidence)
        self.person_history = deque()  # (timestamp, confidence)
        
        # Audio detection results
        self.P_distress = 0.0
        self.audio_label = ""
        self.audio_top_label = ""  # Top predicted class label
        self.audio_top_score = 0.0  # Top predicted class score
        self.audio_best_distress_label = ""  # Best matching distress label
        
        # Manual overrides
        self.override_weapon = None   # None or float
        self.override_audio = None     # None or float
        
        # Vision-only mode
        self.vision_only = False  # When True, audio is ignored in fusion
        
        # Performance timing
        self.visual_inference_ms = 0.0
        self.audio_inference_ms = 0.0
        self.fusion_ms = 0.0
        self.visual_timing_history = deque(maxlen=30)
        self.audio_timing_history = deque(maxlen=30)
        
        # Running flag
        self.running = True
    
    def update_visual(self, detections: List[Tuple], inference_ms: float, frame, camera_covered: bool):
        """Update visual detection results with temporal smoothing. Frame is the raw camera frame for display."""
        with self._lock:
            self.visual_detections = detections
            self.latest_frame = frame
            self.camera_covered = camera_covered
            self.visual_timing_history.append(inference_ms)
            self.visual_inference_ms = sum(self.visual_timing_history) / len(self.visual_timing_history)
            
            # Extract max confidence per class
            weapon_conf = 0.0
            person_conf = 0.0
            for bbox, class_name, conf in detections:
                if class_name in WEAPON_CLASSES:
                    weapon_conf = max(weapon_conf, conf)
                elif class_name == "person":
                    person_conf = max(person_conf, conf)
            
            # Update temporal smoothing
            now = time.time()
            self.C_weapon_raw = weapon_conf
            self._update_history(self.weapon_history, now, weapon_conf)
            self._update_history(self.person_history, now, person_conf)
            
            self.C_weapon = self._get_max_in_window(self.weapon_history)
            self.C_person = self._get_max_in_window(self.person_history)
    
    def update_audio(self, distress_prob: float, label: str, inference_ms: float, 
                     top_label: str = "", top_score: float = 0.0, 
                     best_distress_label: str = ""):
        """Update audio detection results."""
        with self._lock:
            self.P_distress = distress_prob
            self.audio_label = label
            self.audio_top_label = top_label
            self.audio_top_score = top_score
            self.audio_best_distress_label = best_distress_label
            self.audio_timing_history.append(inference_ms)
            self.audio_inference_ms = sum(self.audio_timing_history) / len(self.audio_timing_history)
    
    def _update_history(self, history: deque, timestamp: float, confidence: float):
        """Add new entry to history and prune old entries."""
        history.append((timestamp, confidence))
        # Remove entries older than CONFIDENCE_HOLD_SECONDS
        cutoff = timestamp - CONFIDENCE_HOLD_SECONDS
        while history and history[0][0] < cutoff:
            history.popleft()
    
    def _get_max_in_window(self, history: deque) -> float:
        """Get maximum confidence within the current time window."""
        if not history:
            return 0.0
        return max(conf for _, conf in history)
    
    def set_override_weapon(self, value: Optional[float]):
        """Set manual weapon override."""
        with self._lock:
            self.override_weapon = value
    
    def set_override_audio(self, value: Optional[float]):
        """Set manual audio override."""
        with self._lock:
            self.override_audio = value
    
    def clear_overrides(self):
        """Clear all manual overrides."""
        with self._lock:
            self.override_weapon = None
            self.override_audio = None
    
    def toggle_vision_only(self):
        """Toggle vision-only mode."""
        with self._lock:
            self.vision_only = not self.vision_only
    
    def snapshot(self) -> Dict:
        """Get a snapshot of current state for main thread consumption."""
        with self._lock:
            # Apply overrides if set
            weapon_input = self.override_weapon if self.override_weapon is not None else self.C_weapon
            audio_input = self.override_audio if self.override_audio is not None else self.P_distress
            
            # In vision-only mode, force audio to 0.0 (except weapon override still works)
            if self.vision_only:
                audio_input = 0.0
            
            return {
                "visual_detections": list(self.visual_detections),
                "C_weapon": weapon_input,
                "C_weapon_raw": self.C_weapon_raw,
                "C_weapon_smoothed": self.C_weapon,
                "C_person": self.C_person,
                "P_distress": audio_input,
                "audio_label": self.audio_label,
                "audio_top_label": self.audio_top_label,
                "audio_top_score": self.audio_top_score,
                "audio_best_distress_label": self.audio_best_distress_label,
                "override_weapon": self.override_weapon,
                "override_audio": self.override_audio,
                "vision_only": self.vision_only,
                "visual_inference_ms": self.visual_inference_ms,
                "audio_inference_ms": self.audio_inference_ms if self.audio_timing_history else 0.0,
                "has_audio": len(self.audio_timing_history) > 0,
                "latest_frame": self.latest_frame,
                "camera_covered": self.camera_covered,
                "running": self.running
            }
    
    def stop(self):
        """Signal threads to stop."""
        with self._lock:
            self.running = False

# ============================================================================
# VISUAL PROCESSING UNIT (VPU) - Background Thread
# ============================================================================

def vpu_thread(shared_state: SharedState):
    """Background thread for YOLOv8-Nano visual object detection."""
    print("VPU: Starting visual processing thread...")
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"VPU: Failed to open camera at index {CAMERA_INDEX}")
        shared_state.stop()
        return
    
    # Initialize YOLO model
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_MODEL)
        print(f"VPU: YOLO model loaded: {YOLO_MODEL}")
    except Exception as e:
        print(f"VPU: Failed to load YOLO model: {e}")
        cap.release()
        shared_state.stop()
        return
    
    while shared_state.snapshot()["running"]:
        try:
            ret, frame = cap.read()
            if not ret:
                print("VPU: Failed to read frame from camera")
                break
            
            # Check if camera is covered (dark, low-texture frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean()
            texture = gray.std()
            covered = (brightness < BRIGHTNESS_THRESHOLD) and (texture < TEXTURE_THRESHOLD)
            
            # Run YOLO inference
            start_time = time.perf_counter()
            results = model(frame, verbose=False)
            inference_ms = (time.perf_counter() - start_time) * 1000
            
            # Process detections
            detections = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    if class_name in TRACKED_CLASSES and confidence > 0.15:
                        bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                        detections.append((bbox, class_name, confidence))
            
            # Update shared state
            shared_state.update_visual(detections, inference_ms, frame, covered)
            
        except Exception as e:
            print(f"VPU: Error in inference loop: {e}")
            continue
    
    cap.release()
    print("VPU: Visual processing thread stopped")

# ============================================================================
# AUDIO PROCESSING UNIT (APU) - Background Thread
# ============================================================================

def apu_thread(shared_state: SharedState):
    """Background thread for YAMNet audio distress detection."""
    if not HAS_AUDIO or not AUDIO_ENABLED:
        print("APU: Audio disabled or dependencies unavailable")
        return
    
    print("APU: Starting audio processing thread...")
    
    # Load YAMNet model
    try:
        yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
        print("APU: YAMNet model loaded")
    except Exception as e:
        print(f"APU: Failed to load YAMNet model: {e}")
        return
    
    # Get AudioSet class labels from public URL
    class_names = []
    try:
        url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
        with urllib.request.urlopen(url, timeout=10) as response:
            csv_text = response.read().decode("utf-8")
        reader = csv.reader(io.StringIO(csv_text))
        next(reader)  # skip header row
        class_names = [row[2] for row in reader]
        print(f"APU: AudioSet labels loaded ({len(class_names)} classes)")
    except Exception as e:
        print(f"APU: Failed to fetch class map from URL: {e}")
        print("APU: Falling back to video-only mode")
        return
    
    # Audio capture parameters
    SAMPLE_RATE = 16000
    CHANNELS = 1
    WINDOW_DURATION = 1.0  # seconds
    CHUNK_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
    
    # Initialize audio stream
    try:
        stream = sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE
        )
        stream.start()
        print("APU: Audio stream started")
    except Exception as e:
        print(f"APU: Failed to open audio stream: {e}")
        return
    
    # Processing interval
    last_process_time = time.time()
    PROCESS_INTERVAL = 0.5  # seconds
    
    audio_buffer = collections.deque(maxlen=CHUNK_SIZE * 2)
    
    while shared_state.snapshot()["running"]:
        try:
            # Read audio chunk
            audio_chunk, overflowed = stream.read(CHUNK_SIZE)
            if overflowed:
                print("APU: Audio buffer overflow")
            
            audio_buffer.extend(audio_chunk.flatten())
            
            # Process at interval
            current_time = time.time()
            if current_time - last_process_time >= PROCESS_INTERVAL:
                if len(audio_buffer) >= CHUNK_SIZE:
                    # Get last 1 second of audio (same as audio_check.py)
                    waveform = np.array(list(audio_buffer)[-CHUNK_SIZE:], dtype=np.float32)
                    
                    # Run YAMNet inference
                    start_time = time.perf_counter()
                    scores, embeddings, spectrogram = yamnet_model(waveform)
                    inference_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Get top predictions (for logging)
                    top_scores = tf.math.top_k(scores, k=5)
                    top_indices = top_scores.indices.numpy()[0]
                    top_values = top_scores.values.numpy()[0]
                    
                    # Get top label and score
                    top_label = ""
                    top_score = 0.0
                    if top_indices[0] < len(class_names):
                        top_label = class_names[top_indices[0]]
                        top_score = float(top_values[0])
                    
                    # Match against distress keywords across ALL classes (same as audio_check.py)
                    max_distress_prob = 0.0
                    matched_label = ""
                    
                    for idx, score in enumerate(scores.numpy()[0]):
                        if idx < len(class_names):
                            label = class_names[idx]
                            label_lower = label.lower()
                            for keyword in DISTRESS_KEYWORDS:
                                if keyword in label_lower:
                                    if score > max_distress_prob:
                                        max_distress_prob = float(score)
                                        matched_label = label
                    
                    # Update shared state with all audio info
                    shared_state.update_audio(max_distress_prob, matched_label, inference_ms,
                                             top_label, top_score, matched_label)
                
                last_process_time = current_time
            
        except Exception as e:
            print(f"APU: Error in audio processing: {e}")
            continue
    
    stream.stop()
    stream.close()
    print("APU: Audio processing thread stopped")

# ============================================================================
# FUSION LOGIC
# ============================================================================

def fusion_logic(C_weapon: float, C_person: float, P_distress: float) -> str:
    """
    Apply fusion logic to determine threat state.
    
    Returns: 'CRITICAL', 'HIGH', 'WARNING', or 'SAFE'
    """
    if C_weapon > TAU_W:
        return "CRITICAL"
    elif C_person > TAU_P and P_distress > TAU_A:
        return "HIGH"
    elif C_person == 0.0 and P_distress > TAU_A_STAR:
        return "WARNING"
    else:
        return "SAFE"

# ============================================================================
# VERIFICATION NOTE
# ============================================================================

def get_verification_note(threat_state: str, P_distress: float) -> str:
    """Get verification note based on threat state and audio corroboration."""
    if threat_state == "CRITICAL":
        if P_distress > 0.3:
            return "Weapon + audio corroboration -> high-confidence alert"
        else:
            return "Object only, no audio corroboration -> HUMAN VERIFICATION RECOMMENDED"
    elif threat_state == "HIGH":
        return "Person + distress audio jointly detected"
    elif threat_state == "WARNING":
        return "No visual input -- audio-only fallback triggered"
    else:
        return ""

# ============================================================================
# ALERT BEEP
# ============================================================================

def play_alert_beep(threat_state: str):
    """Play alert beep on thread for CRITICAL/HIGH transitions."""
    if not HAS_WINSOUND:
        return
    
    def beep_worker():
        try:
            if threat_state == "CRITICAL":
                winsound.Beep(1200, 180)  # 1200Hz for 180ms
            elif threat_state == "HIGH":
                winsound.Beep(900, 150)   # 900Hz for 150ms
        except Exception:
            pass  # Silently fail if beep doesn't work
    
    thread = threading.Thread(target=beep_worker, daemon=True)
    thread.start()

# ============================================================================
# EVIDENCE SNAPSHOT THREAD
# ============================================================================

class EvidenceSnapshotter:
    """Daemon thread to save evidence snapshots on threat state transitions."""
    
    def __init__(self, clip_name: str):
        self.clip_name = clip_name
        self.queue = collections.deque()
        self._lock = threading.Lock()
        self._running = True
        self._last_snapshot_time = 0.0
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
    
    def request_snapshot(self, frame: np.ndarray, state: str):
        """Request a snapshot to be saved (non-blocking)."""
        with self._lock:
            self.queue.append((frame.copy(), state, time.time()))
    
    def _worker(self):
        """Background worker to save snapshots."""
        while self._running:
            time.sleep(0.1)  # Check periodically
            with self._lock:
                if not self.queue:
                    continue
                frame, state, request_time = self.queue.popleft()
                # Throttle: at most one snapshot every 3 seconds
                if request_time - self._last_snapshot_time < 3.0:
                    continue
                self._last_snapshot_time = request_time
            
            # Save snapshot outside lock
            try:
                os.makedirs("evidence", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"evidence/{timestamp}_{state}_{self.clip_name}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Evidence: Saved snapshot to {filename}")
            except Exception as e:
                print(f"Evidence: Failed to save snapshot: {e}")
    
    def stop(self):
        """Stop the snapshot thread."""
        with self._lock:
            self._running = False

# ============================================================================
# EVENT HISTORY
# ============================================================================

class EventHistory:
    """Track recent state transitions for display."""
    
    def __init__(self, max_entries: int = 5):
        self.max_entries = max_entries
        self.history = collections.deque(maxlen=max_entries)
        self._lock = threading.Lock()
    
    def add_event(self, state: str, reason: str):
        """Add a state transition event."""
        with self._lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.history.appendleft((timestamp, state, reason))
    
    def get_history(self) -> List[Tuple[str, str, str]]:
        """Get the current history (newest first)."""
        with self._lock:
            return list(self.history)

# ============================================================================
# LAYOUT/RENDERING HELPER
# ============================================================================

def create_info_panel(
    threat_state: str,
    verification_note: str,
    C_weapon: float,
    C_person: float,
    P_distress: float,
    audio_label: str,
    override_weapon: Optional[float],
    override_audio: Optional[float],
    vision_only: bool,
    visual_ms: float,
    audio_ms: float,
    fusion_ms: float,
    has_audio: bool,
    camera_covered: bool,
    frame_width: int,
    event_history: List[Tuple[str, str, str]],
    log_status: str = ""
) -> np.ndarray:
    """
    Create the top info panel as a separate image region.
    This panel contains: banner, verification note, confidence values, performance stats.
    Returns the panel as a numpy array (BGR image).
    """
    # Define colors
    BANNER_COLORS = {
        "CRITICAL": (0, 0, 180),      # Dark red
        "HIGH": (0, 100, 180),        # Orange
        "WARNING": (0, 165, 180),     # Dark yellow
        "SAFE": (0, 150, 0)           # Dark green
    }
    
    banner_color = BANNER_COLORS.get(threat_state, (128, 128, 128))
    
    # Panel dimensions
    banner_height = 80
    note_height = 26 if verification_note else 0
    camera_height = 26  # Camera covered status line
    mode_height = 26    # Vision-only mode line
    conf_height = 60
    perf_height = 26
    log_height = 26 if log_status else 0
    keys_height = 26
    events_heading_height = 20
    events_height = min(len(event_history) * 18, 90)  # At most 5 events, max 90px
    
    total_height = (banner_height + note_height + camera_height + mode_height + 
                    conf_height + perf_height + log_height + keys_height + 
                    events_heading_height + events_height)
    
    # Create panel with dark background
    panel = np.full((total_height, frame_width, 3), (30, 30, 30), dtype=np.uint8)
    
    y_offset = 0
    
    # Draw banner
    panel[y_offset:y_offset + banner_height] = banner_color
    cv2.putText(
        panel, f"THREAT STATE: {threat_state}",
        (20, y_offset + 50),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2
    )
    y_offset += banner_height
    
    # Draw verification note
    if verification_note:
        cv2.putText(
            panel, verification_note,
            (20, y_offset + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        y_offset += note_height
    
    # Draw camera covered status
    camera_text = "Camera: COVERED" if camera_covered else "Camera: clear"
    camera_color = (0, 165, 255) if camera_covered else (100, 100, 100)  # Orange/yellow if covered, gray if clear
    cv2.putText(
        panel, camera_text,
        (20, y_offset + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, camera_color, 1
    )
    y_offset += camera_height
    
    # Draw vision-only mode status
    if vision_only:
        mode_text = "MODE: VISION-ONLY (audio ignored)"
        mode_color = (0, 200, 255)  # Bright warning color
    else:
        mode_text = "MODE: SAFEFUSION (audio + video)"
        mode_color = (100, 100, 100)  # Gray
    cv2.putText(
        panel, mode_text,
        (20, y_offset + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1
    )
    y_offset += mode_height
    
    # Draw confidence values
    conf_text = f"C_weapon: {C_weapon:.2f}  C_person: {C_person:.2f}  P_distress: {P_distress:.2f}"
    if audio_label:
        conf_text += f"  (audio: {audio_label})"
    
    # Add override indicators
    override_indicators = []
    if override_weapon is not None:
        override_indicators.append(f"W={override_weapon:.2f}")
    if override_audio is not None:
        override_audio_val = override_audio if override_audio is not None else P_distress
        override_indicators.append(f"A={override_audio_val:.2f}")
    if override_indicators:
        conf_text += f"  [OVERRIDE: {' '.join(override_indicators)}]"
    
    cv2.putText(
        panel, conf_text,
        (20, y_offset + 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    y_offset += conf_height
    
    # Draw thresholds line
    thresh_text = f"Thresholds: weapon>{TAU_W} person>{TAU_P} audio>{TAU_A} audio-only>{TAU_A_STAR}"
    cv2.putText(
        panel, thresh_text,
        (20, y_offset + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1
    )
    y_offset += perf_height
    
    # Draw logging status
    if log_status:
        cv2.putText(
            panel, log_status,
            (20, y_offset + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1
        )
        y_offset += log_height
    
    # Draw key legend
    keys_text = "Keys: [w] weapon override [s] audio override [n] clear audio [c] clear all [v] vision-only [q] quit"
    cv2.putText(
        panel, keys_text,
        (20, y_offset + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1
    )
    y_offset += keys_height
    
    # Draw event history heading
    cv2.putText(
        panel, "Recent events:",
        (20, y_offset + 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1
    )
    y_offset += events_heading_height
    
    # Draw event history entries
    for i, (timestamp, state, reason) in enumerate(event_history):
        event_text = f"{timestamp} {state}: {reason}"
        cv2.putText(
            panel, event_text,
            (30, y_offset + 12 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1
        )
    
    return panel

def draw_detections_on_frame(frame: np.ndarray, detections: List[Tuple]) -> np.ndarray:
    """
    Draw YOLO bounding boxes and labels on the camera frame.
    Weapons are labeled as "weapon" on screen.
    """
    annotated = frame.copy()
    
    for bbox, class_name, confidence in detections:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Determine label and color
        if class_name in WEAPON_CLASSES:
            label = f"weapon {confidence:.2f}"
            color = (0, 0, 255)  # Red for weapons
        else:
            label = f"{class_name} {confidence:.2f}"
            color = (0, 255, 0)  # Green for persons
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated, 
            (x1, y1 - label_h - 10), 
            (x1 + label_w, y1), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated, label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
    
    return annotated

# ============================================================================
# DECISION LOGGING (offline evaluation)
# ============================================================================

LOG_COLUMNS = [
    "timestamp",
    "clip_name",
    "c_weapon_smoothed",
    "c_weapon_raw",
    "c_person_smoothed",
    "p_distress",
    "visual_empty",
    "camera_covered",
    "manual_override",
    "threat_state",
    "visual_ms",
    "audio_ms",
    "fusion_ms",
    "mode",
    "audio_top_label",
    "audio_top_score",
    "audio_best_distress_label",
]

LOG_MIN_INTERVAL = 0.2  # at most 5 rows per second


def parse_args():
    parser = argparse.ArgumentParser(description="SafeFusion demo v11 with decision logging")
    parser.add_argument("--clip-name", default="unnamed", help="Label for this recording session")
    parser.add_argument("--log-file", default="logs/decisions.csv", help="CSV path for decision rows")
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Quit after N seconds; 0 means run until 'q' is pressed",
    )
    return parser.parse_args()


def manual_override_label(override_weapon, override_audio) -> str:
    if override_weapon is not None:
        return "weapon"
    if override_audio is not None:
        return "scream" if override_audio >= 0.9 else "noise"
    return "none"


def open_decision_log(log_path: str):
    """Create the log directory and open the CSV for append. Write header only on new files."""
    directory = os.path.dirname(log_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    write_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    
    # Check if existing file has the new columns
    has_mode_column = False
    has_audio_columns = False
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        try:
            with open(log_path, "r", newline="", encoding="utf-8") as f:
                first_line = f.readline()
                has_mode_column = "mode" in first_line.lower()
                has_audio_columns = ("audio_top_label" in first_line.lower() and
                                     "audio_top_score" in first_line.lower() and
                                     "audio_best_distress_label" in first_line.lower())
        except Exception:
            pass
    
    handle = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
    if write_header:
        writer.writeheader()
        handle.flush()
    
    return handle, writer, has_mode_column, has_audio_columns


def append_decision_row(writer, handle, row: dict, has_mode_column: bool, has_audio_columns: bool) -> None:
    """Append a decision row, handling missing columns in old CSV files."""
    if not has_mode_column:
        # Remove 'mode' from row if the CSV doesn't have that column
        row = {k: v for k, v in row.items() if k != "mode"}
    if not has_audio_columns:
        # Remove new audio columns if the CSV doesn't have them
        row = {k: v for k, v in row.items() if k not in ["audio_top_label", "audio_top_score", "audio_best_distress_label"]}
    writer.writerow(row)
    handle.flush()


# ============================================================================
# MAIN DISPLAY LOOP
# ============================================================================

def main():
    """Main entry point for SafeFusion demo."""
    args = parse_args()
    print("SafeFusion Demo v11 - Starting...")
    print(f"Main: Logging clip_name={args.clip_name} -> {args.log_file}")
    print("Keys: [w] weapon override [s] audio override [n] clear audio [c] clear all [v] vision-only [q] quit")
    
    log_handle = None
    log_writer = None
    has_mode_column = False
    has_audio_columns = False
    last_log_time = 0.0
    try:
        log_handle, log_writer, has_mode_column, has_audio_columns = open_decision_log(args.log_file)
        if not has_mode_column:
            print("Warning: Existing CSV does not have 'mode' column. Rows will be appended without it.")
        if not has_audio_columns:
            print("Warning: Existing CSV does not have audio detail columns. Rows will be appended without them.")
    except Exception as e:
        print(f"Main: Decision log could not be opened ({e}). Continuing without logging.")
    
    # Initialize shared state
    shared_state = SharedState()
    
    # Initialize evidence snapshotter
    snapshotter = EvidenceSnapshotter(args.clip_name)
    
    # Initialize event history
    event_history = EventHistory(max_entries=5)
    
    # Start background threads
    vpu = threading.Thread(target=vpu_thread, args=(shared_state,), daemon=True)
    apu = threading.Thread(target=apu_thread, args=(shared_state,), daemon=True)
    
    vpu.start()
    apu.start()
    
    # Wait for first frame to get dimensions
    print("Main: Waiting for first frame...")
    while True:
        snapshot = shared_state.snapshot()
        if snapshot["latest_frame"] is not None:
            break
        if not snapshot["running"]:
            print("Main: VPU stopped before providing first frame")
            if log_handle is not None:
                try:
                    log_handle.close()
                except Exception:
                    pass
            snapshotter.stop()
            return
        time.sleep(0.1)
    
    # Get camera dimensions from first frame
    first_frame = shared_state.snapshot()["latest_frame"]
    frame_height, frame_width = first_frame.shape[:2]
    print(f"Main: Camera resolution: {frame_width}x{frame_height}")
    
    # Main display loop
    last_threat_state = "SAFE"
    fps_start_time = time.time()
    fps_frame_count = 0
    session_start = time.time()
    
    while True:
        loop_start = time.time()
        if args.duration > 0 and (loop_start - session_start) >= args.duration:
            print(f"Main: Duration {args.duration}s reached, quitting")
            break
        
        # Get snapshot of shared state
        snapshot = shared_state.snapshot()
        
        if not snapshot["running"]:
            break
        
        # Get frame from VPU thread
        frame = snapshot["latest_frame"]
        if frame is None:
            # Create placeholder frame if no frame available yet
            frame = np.full((frame_height, frame_width, 3), (50, 50, 50), dtype=np.uint8)
            cv2.putText(
                frame, "WAITING FOR CAMERA...",
                (frame_width // 2 - 120, frame_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
        
        # Draw detections on frame
        annotated_frame = draw_detections_on_frame(frame, snapshot["visual_detections"])
        
        # Apply fusion logic
        fusion_start = time.perf_counter()
        threat_state = fusion_logic(
            snapshot["C_weapon"],
            snapshot["C_person"],
            snapshot["P_distress"]
        )
        fusion_ms = (time.perf_counter() - fusion_start) * 1000
        
        # Update fusion timing in shared state
        with shared_state._lock:
            shared_state.fusion_ms = fusion_ms
        
        # Handle state transitions for evidence snapshots and event history
        if threat_state != last_threat_state:
            # Generate reason string
            if threat_state == "CRITICAL":
                reason = f"weapon {snapshot['C_weapon']:.2f}"
            elif threat_state == "HIGH":
                reason = f"person + distress {snapshot['P_distress']:.2f}"
            elif threat_state == "WARNING":
                if snapshot["camera_covered"]:
                    reason = "camera covered"
                else:
                    reason = f"no visual + distress {snapshot['P_distress']:.2f}"
            elif threat_state == "SAFE":
                reason = "back to safe"
            else:
                reason = ""
            
            # Add to event history
            event_history.add_event(threat_state, reason)
            
            # Request evidence snapshot for threat states
            if threat_state in ["CRITICAL", "HIGH", "WARNING"]:
                snapshotter.request_snapshot(annotated_frame, threat_state)
        
        # Decision log (throttled; never crashes the demo)
        now_log = time.time()
        if log_writer is not None and (now_log - last_log_time) >= LOG_MIN_INTERVAL:
            try:
                audio_ms_value = (
                    f"{snapshot['audio_inference_ms']:.3f}" if snapshot["has_audio"] else ""
                )
                mode_value = "vision_only" if snapshot["vision_only"] else "safefusion"
                append_decision_row(log_writer, log_handle, {
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "clip_name": args.clip_name,
                    "c_weapon_smoothed": f"{snapshot['C_weapon_smoothed']:.6f}",
                    "c_weapon_raw": f"{snapshot['C_weapon_raw']:.6f}",
                    "c_person_smoothed": f"{snapshot['C_person']:.6f}",
                    "p_distress": f"{snapshot['P_distress']:.6f}",
                    "visual_empty": 1 if not snapshot["visual_detections"] else 0,
                    "camera_covered": 1 if snapshot["camera_covered"] else 0,
                    "manual_override": manual_override_label(
                        snapshot["override_weapon"], snapshot["override_audio"]
                    ),
                    "threat_state": threat_state,
                    "visual_ms": f"{snapshot['visual_inference_ms']:.3f}",
                    "audio_ms": audio_ms_value,
                    "fusion_ms": f"{fusion_ms:.3f}",
                    "mode": mode_value,
                    "audio_top_label": snapshot.get("audio_top_label", ""),
                    "audio_top_score": f"{snapshot.get('audio_top_score', 0.0):.6f}",
                    "audio_best_distress_label": snapshot.get("audio_best_distress_label", ""),
                }, has_mode_column, has_audio_columns)
                last_log_time = now_log
            except Exception:
                pass
        
        # Get verification note
        verification_note = get_verification_note(threat_state, snapshot["P_distress"])
        
        # Play alert on state transition
        if threat_state in ["CRITICAL", "HIGH"] and threat_state != last_threat_state:
            play_alert_beep(threat_state)
        last_threat_state = threat_state
        
        # Create info panel (separate from camera frame)
        audio_display_ms = snapshot["audio_inference_ms"] if snapshot["has_audio"] else 0.0
        audio_display_str = f"{audio_display_ms:.1f}ms" if snapshot["has_audio"] else "n/a"
        
        info_panel = create_info_panel(
            threat_state=threat_state,
            verification_note=verification_note,
            C_weapon=snapshot["C_weapon"],
            C_person=snapshot["C_person"],
            P_distress=snapshot["P_distress"],
            audio_label=snapshot["audio_label"],
            override_weapon=snapshot["override_weapon"],
            override_audio=snapshot["override_audio"],
            vision_only=snapshot["vision_only"],
            visual_ms=snapshot["visual_inference_ms"],
            audio_ms=audio_display_ms,
            fusion_ms=fusion_ms,
            has_audio=snapshot["has_audio"],
            camera_covered=snapshot["camera_covered"],
            frame_width=frame_width,
            event_history=event_history.get_history(),
            log_status=f"Logging: {args.clip_name} -> {args.log_file}"
        )
        
        # Combine info panel and camera frame using vertical stacking
        # This ensures text never overlaps with YOLO labels
        combined_frame = cv2.vconcat([info_panel, annotated_frame])
        
        # Calculate FPS
        fps_frame_count += 1
        fps_elapsed = time.time() - fps_start_time
        if fps_elapsed >= 1.0:
            fps = fps_frame_count / fps_elapsed
            fps_frame_count = 0
            fps_start_time = time.time()
        else:
            fps = 0.0
        
        # Draw FPS on the combined frame (in the info panel area)
        if fps > 0:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                combined_frame, fps_text,
                (frame_width - 120, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        # Display
        cv2.imshow("SafeFusion v11", combined_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            shared_state.set_override_weapon(0.95)
        elif key == ord('s'):
            shared_state.set_override_audio(0.95)
        elif key == ord('n'):
            shared_state.set_override_audio(0.0)
        elif key == ord('c'):
            shared_state.clear_overrides()
        elif key == ord('v'):
            shared_state.toggle_vision_only()
        
        # Calculate end-to-end timing
        e2e_ms = (time.time() - loop_start) * 1000
    
    # Cleanup
    shared_state.stop()
    snapshotter.stop()
    cv2.destroyAllWindows()
    
    # Wait for threads to finish
    vpu.join(timeout=2.0)
    apu.join(timeout=2.0)
    
    if log_handle is not None:
        try:
            log_handle.close()
        except Exception:
            pass
    
    print("SafeFusion Demo v11 - Stopped cleanly")

if __name__ == "__main__":
    main()
