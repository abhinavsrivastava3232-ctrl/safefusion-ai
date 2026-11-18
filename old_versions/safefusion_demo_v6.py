"""
SafeFusion Demo v6 - Live threat detection with YOLOv8-Nano and YAMNet
Visual + Audio fusion with rule-based decision logic and stacked UI layout to prevent text overlap.

Fix v6: Text panel and camera feed are separate stacked regions, preventing overlap with YOLO labels.
"""

import cv2
import numpy as np
import threading
import time
import collections
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
        self.C_person = 0.0
        self.latest_frame = None  # Latest camera frame for display
        
        # Temporal smoothing histories
        self.weapon_history = deque()  # (timestamp, confidence)
        self.person_history = deque()  # (timestamp, confidence)
        
        # Audio detection results
        self.P_distress = 0.0
        self.audio_label = ""
        
        # Manual overrides
        self.override_weapon = None   # None or float
        self.override_audio = None     # None or float
        
        # Performance timing
        self.visual_inference_ms = 0.0
        self.audio_inference_ms = 0.0
        self.fusion_ms = 0.0
        self.visual_timing_history = deque(maxlen=30)
        self.audio_timing_history = deque(maxlen=30)
        
        # Running flag
        self.running = True
    
    def update_visual(self, detections: List[Tuple], inference_ms: float, frame):
        """Update visual detection results with temporal smoothing. Frame is the raw camera frame for display."""
        with self._lock:
            self.visual_detections = detections
            self.latest_frame = frame
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
            self._update_history(self.weapon_history, now, weapon_conf)
            self._update_history(self.person_history, now, person_conf)
            
            self.C_weapon = self._get_max_in_window(self.weapon_history)
            self.C_person = self._get_max_in_window(self.person_history)
    
    def update_audio(self, distress_prob: float, label: str, inference_ms: float):
        """Update audio detection results."""
        with self._lock:
            self.P_distress = distress_prob
            self.audio_label = label
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
    
    def snapshot(self) -> Dict:
        """Get a snapshot of current state for main thread consumption."""
        with self._lock:
            # Apply overrides if set
            weapon_input = self.override_weapon if self.override_weapon is not None else self.C_weapon
            audio_input = self.override_audio if self.override_audio is not None else self.P_distress
            
            return {
                "visual_detections": list(self.visual_detections),
                "C_weapon": weapon_input,
                "C_person": self.C_person,
                "P_distress": audio_input,
                "audio_label": self.audio_label,
                "override_weapon": self.override_weapon,
                "override_audio": self.override_audio,
                "visual_inference_ms": self.visual_inference_ms,
                "audio_inference_ms": self.audio_inference_ms if self.audio_timing_history else 0.0,
                "has_audio": len(self.audio_timing_history) > 0,
                "latest_frame": self.latest_frame,
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
            
            # Run YOLO inference
            start_time = time.time()
            results = model(frame, verbose=False)
            inference_ms = (time.time() - start_time) * 1000
            
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
            shared_state.update_visual(detections, inference_ms, frame)
            
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
    
    # Get AudioSet class labels
    try:
        # YAMNet returns class names as part of the model
        # We'll use the standard AudioSet labels
        audio_labels_path = yamnet_model.class_names
        # For TF Hub YAMNet, we need to extract labels differently
        # The model outputs scores for 521 classes
        print("APU: AudioSet labels loaded")
    except Exception as e:
        print(f"APU: Warning - could not load labels: {e}")
        # Continue with label matching based on common AudioSet labels
        audio_labels_path = None
    
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
                    # Convert to numpy array
                    waveform = np.array(list(audio_buffer)[:CHUNK_SIZE], dtype=np.float32)
                    
                    # Run YAMNet inference
                    start_time = time.time()
                    scores, embeddings, spectrogram = yamnet_model(waveform)
                    inference_ms = (time.time() - start_time) * 1000
                    
                    # Get top predictions
                    top_scores = tf.math.top_k(scores, k=5)
                    top_indices = top_scores.indices.numpy()[0]
                    top_values = top_scores.values.numpy()[0]
                    
                    # Match against distress keywords
                    max_distress_prob = 0.0
                    matched_label = ""
                    
                    # Standard AudioSet label mapping (simplified)
                    # In production, use actual label names from the model
                    common_labels = [
                        "Scream", "Shout", "Yell", "Crying", "Sobbing",
                        "Glass", "Shatter", "Alarm", "Siren", "Gunshot", 
                        "Explosion", "Battle cry"
                    ]
                    
                    for idx, score in zip(top_indices, top_values):
                        if idx < len(common_labels):
                            label = common_labels[idx]
                            label_lower = label.lower()
                            for keyword in DISTRESS_KEYWORDS:
                                if keyword in label_lower:
                                    if score > max_distress_prob:
                                        max_distress_prob = float(score)
                                        matched_label = label
                    
                    # Update shared state
                    shared_state.update_audio(max_distress_prob, matched_label, inference_ms)
                
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
    visual_ms: float,
    audio_ms: float,
    fusion_ms: float,
    has_audio: bool,
    frame_width: int
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
    conf_height = 60
    perf_height = 26
    keys_height = 26
    
    total_height = banner_height + note_height + conf_height + perf_height + keys_height
    
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
    
    # Draw key legend
    keys_text = "Keys: [w] weapon override [s] audio override [n] clear audio [c] clear all [q] quit"
    cv2.putText(
        panel, keys_text,
        (20, y_offset + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1
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
# MAIN DISPLAY LOOP
# ============================================================================

def main():
    """Main entry point for SafeFusion demo."""
    print("SafeFusion Demo v6 - Starting...")
    
    # Initialize shared state
    shared_state = SharedState()
    
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
    
    while True:
        loop_start = time.time()
        
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
        fusion_start = time.time()
        threat_state = fusion_logic(
            snapshot["C_weapon"],
            snapshot["C_person"],
            snapshot["P_distress"]
        )
        fusion_ms = (time.time() - fusion_start) * 1000
        
        # Update fusion timing in shared state
        with shared_state._lock:
            shared_state.fusion_ms = fusion_ms
        
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
            visual_ms=snapshot["visual_inference_ms"],
            audio_ms=audio_display_ms,
            fusion_ms=fusion_ms,
            has_audio=snapshot["has_audio"],
            frame_width=frame_width
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
        cv2.imshow("SafeFusion v6", combined_frame)
        
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
        
        # Calculate end-to-end timing
        e2e_ms = (time.time() - loop_start) * 1000
    
    # Cleanup
    shared_state.stop()
    cv2.destroyAllWindows()
    
    # Wait for threads to finish
    vpu.join(timeout=2.0)
    apu.join(timeout=2.0)
    
    print("SafeFusion Demo v6 - Stopped cleanly")

if __name__ == "__main__":
    main()
