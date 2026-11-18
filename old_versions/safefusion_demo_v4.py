"""
SafeFusion v4 — multi-modal (webcam + microphone) threat-detection demo.

Architecture
  VPU thread  — OpenCV capture + YOLOv8-Nano
  APU thread  — sounddevice capture + YAMNet
  Main thread — snapshot shared state, fuse, draw HUD, handle keys

CPU-only. Prefer staying alive over failing loudly during a live demo.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

try:
    import winsound

    HAS_WINSOUND = True
except Exception:
    HAS_WINSOUND = False


# =============================================================================
# Config
# =============================================================================

CAMERA_INDEX = 0
AUDIO_ENABLED = True  # False = skip the APU thread entirely (video-only)
YOLO_MODEL = "yolov8n.pt"

WINDOW_NAME = "SafeFusion"
SAMPLE_RATE = 16_000
AUDIO_WINDOW_SECONDS = 1.0
AUDIO_INFER_PERIOD = 0.5
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"
YOLO_IMGSZ = 320
DISPLAY_WAIT_MS = 30

# Overlay / tracking floor (not a fusion trigger). Weak boxes still appear.
DISPLAY_CONF_MIN = 0.15

# Peak visual confidence is held this long so YOLO flicker cannot flap the banner.
CONFIDENCE_HOLD_SECONDS = 1.5

# Fusion thresholds.
TAU_W = 0.4  # weapon; YOLOv8n knife scores often sit in 0.35–0.55
TAU_P = 0.5
TAU_A = 0.6
TAU_A_STAR = 0.8

PERSON_CLASS = "person"
WEAPON_CLASSES = frozenset({"knife", "scissors", "baseball bat"})
TRACKED_CLASSES = frozenset({PERSON_CLASS}) | WEAPON_CLASSES

DISTRESS_KEYWORDS = (
    "scream",
    "shout",
    "yell",
    "crying",
    "sobbing",
    "glass",
    "shatter",
    "alarm",
    "gunshot",
    "explosion",
    "battle cry",
)

BANNER_HEIGHT = 80
NOTE_STRIP_HEIGHT = 26
PLACEHOLDER_H, PLACEHOLDER_W = 480, 640

BANNER_BGR = {
    "CRITICAL": (0, 0, 210),
    "HIGH": (0, 140, 255),
    "WARNING": (0, 220, 255),
    "SAFE": (0, 170, 0),
}
PERSON_BOX_BGR = (255, 200, 0)
WEAPON_BOX_BGR = (0, 0, 255)


# =============================================================================
# Shared state (lock-protected) + temporal smoothing
# =============================================================================

def placeholder_frame(message: str) -> np.ndarray:
    frame = np.full((PLACEHOLDER_H, PLACEHOLDER_W, 3), 32, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    (tw, _th), _ = cv2.getTextSize(message, font, scale, 2)
    x = max(12, (PLACEHOLDER_W - tw) // 2)
    cv2.putText(frame, message, (x, PLACEHOLDER_H // 2), font, scale, (220, 220, 220), 2, cv2.LINE_AA)
    return frame


class SharedState:
    """Latest sensor readings, held visual scores, and demo overrides."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.stop = threading.Event()

        self._weapon_history: Deque[tuple[float, float]] = deque()
        self._person_history: Deque[tuple[float, float]] = deque()

        self.C_weapon = 0.0
        self.C_person = 0.0
        self.P_distress = 0.0
        self.audio_label = "—"
        self.detections: list[dict] = []
        self.has_visual = False
        self.frame = placeholder_frame("Waiting for camera...")
        self.camera_ok = False
        self.audio_ok = False
        self.audio_status = "disabled" if not AUDIO_ENABLED else "starting"

        self.override_C_weapon: Optional[float] = None
        self.override_P_distress: Optional[float] = None

    def _held_max(self, history: Deque[tuple[float, float]], confidence: float, now: float) -> float:
        """Append a reading, drop samples older than CONFIDENCE_HOLD_SECONDS, return the window max."""
        history.append((now, float(confidence)))
        cutoff = now - CONFIDENCE_HOLD_SECONDS
        while history and history[0][0] < cutoff:
            history.popleft()
        if not history:
            return 0.0
        return max(score for _t, score in history)

    def snapshot(self) -> dict:
        with self._lock:
            frame = None if self.frame is None else self.frame.copy()
            return {
                "C_weapon": self.C_weapon,
                "C_person": self.C_person,
                "P_distress": self.P_distress,
                "audio_label": self.audio_label,
                "detections": list(self.detections),
                "has_visual": self.has_visual,
                "frame": frame,
                "camera_ok": self.camera_ok,
                "audio_ok": self.audio_ok,
                "audio_status": self.audio_status,
                "override_C_weapon": self.override_C_weapon,
                "override_P_distress": self.override_P_distress,
            }

    def publish_visual(
        self,
        frame: np.ndarray,
        detections: list[dict],
        weapon_conf: float,
        person_conf: float,
        camera_ok: bool,
    ) -> None:
        now = time.time()
        with self._lock:
            self.C_weapon = self._held_max(self._weapon_history, weapon_conf, now)
            self.C_person = self._held_max(self._person_history, person_conf, now)
            self.frame = frame
            self.detections = detections
            self.has_visual = bool(detections)
            self.camera_ok = camera_ok

    def publish_audio(self, P_distress: float, label: str) -> None:
        with self._lock:
            self.P_distress = float(P_distress)
            self.audio_label = label
            self.audio_ok = True
            self.audio_status = "live"

    def set_audio_status(self, status: str, ok: bool = False) -> None:
        with self._lock:
            self.audio_status = status
            self.audio_ok = ok

    def set_overrides(
        self,
        C_weapon: Optional[float] = None,
        P_distress: Optional[float] = None,
        clear: bool = False,
    ) -> None:
        with self._lock:
            if clear:
                self.override_C_weapon = None
                self.override_P_distress = None
                return
            if C_weapon is not None:
                self.override_C_weapon = float(C_weapon)
            if P_distress is not None:
                self.override_P_distress = float(P_distress)


# =============================================================================
# Fusion + verification note
# =============================================================================

def fuse_threat(
    C_weapon: float,
    C_person: float,
    P_distress: float,
    has_visual: bool,
) -> str:
    """Rule-based fusion. Live sensors and key overrides both call this."""
    if C_weapon > TAU_W:
        return "CRITICAL"
    if C_person > TAU_P and P_distress > TAU_A:
        return "HIGH"
    if (not has_visual) and P_distress > TAU_A_STAR:
        return "WARNING"
    return "SAFE"


def fusion_inputs_from_snapshot(snap: dict) -> tuple[float, float, float, bool, str]:
    """Replace fusion *inputs* when an override is set; do not bypass fuse_threat."""
    C_weapon = snap["C_weapon"]
    C_person = snap["C_person"]
    P_distress = snap["P_distress"]
    has_visual = snap["has_visual"]

    if not snap["audio_ok"] and snap["override_P_distress"] is None:
        P_distress = 0.0

    if snap["override_C_weapon"] is not None:
        C_weapon = snap["override_C_weapon"]
    if snap["override_P_distress"] is not None:
        P_distress = snap["override_P_distress"]

    if snap["override_C_weapon"] is not None and snap["override_P_distress"] is not None:
        override_text = "weapon+distress"
    elif snap["override_C_weapon"] is not None:
        override_text = "weapon (w)"
    elif snap["override_P_distress"] is not None:
        override_text = "scream (s)" if snap["override_P_distress"] >= 0.9 else "noise (n)"
    else:
        override_text = "none (live)"

    return C_weapon, C_person, P_distress, has_visual, override_text


def get_verification_note(threat_state: str, P_distress: float) -> Optional[str]:
    if threat_state == "CRITICAL":
        if P_distress > 0.3:
            return "Weapon + audio corroboration -> high-confidence alert"
        return "Object only, no audio corroboration -> HUMAN VERIFICATION RECOMMENDED"
    if threat_state == "HIGH":
        return "Person + distress audio jointly detected"
    if threat_state == "WARNING":
        return "No visual input -- audio-only fallback triggered"
    return None


# =============================================================================
# Alert beep (once on enter CRITICAL / HIGH)
# =============================================================================

def _beep_worker(threat_state: str) -> None:
    if not HAS_WINSOUND:
        return
    try:
        if threat_state == "CRITICAL":
            winsound.Beep(1200, 180)
        elif threat_state == "HIGH":
            winsound.Beep(900, 150)
    except Exception:
        return


def beep_on_enter(threat_state: str, previous_state: str) -> None:
    if threat_state not in ("CRITICAL", "HIGH"):
        return
    if threat_state == previous_state:
        return
    threading.Thread(
        target=_beep_worker,
        args=(threat_state,),
        daemon=True,
        name="SafeFusionBeep",
    ).start()


# =============================================================================
# Visual Processing Unit
# =============================================================================

def vpu_thread(state: SharedState) -> None:
    print("[VPU] Loading YOLOv8-Nano (first run downloads weights)...")
    model = None
    try:
        from ultralytics import YOLO

        model = YOLO(YOLO_MODEL)
        print("[VPU] YOLO ready.")
    except Exception as exc:
        print(f"[VPU] YOLO failed to load ({exc}). Camera will still run without detections.")

    cap = None
    last_open_attempt = 0.0

    while not state.stop.is_set():
        if cap is None or not cap.isOpened():
            now = time.time()
            if now - last_open_attempt < 1.5:
                state.publish_visual(
                    placeholder_frame("Waiting for camera..."),
                    [],
                    0.0,
                    0.0,
                    False,
                )
                time.sleep(0.2)
                continue
            last_open_attempt = now
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                print(f"[VPU] Could not open camera index {CAMERA_INDEX}. Retrying...")
                cap = None
                state.publish_visual(
                    placeholder_frame(f"Waiting for camera (index {CAMERA_INDEX})..."),
                    [],
                    0.0,
                    0.0,
                    False,
                )
                continue
            print(f"[VPU] Camera {CAMERA_INDEX} opened.")

        ok, frame = cap.read()
        if not ok or frame is None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
            state.publish_visual(
                placeholder_frame("Waiting for camera..."),
                [],
                0.0,
                0.0,
                False,
            )
            continue

        detections: list[dict] = []
        annotated = frame
        weapon_conf = 0.0
        person_conf = 0.0
        if model is not None:
            try:
                annotated, detections, weapon_conf, person_conf = run_yolo(model, frame)
            except Exception as exc:
                print(f"[VPU] Inference error (continuing): {exc}")
                annotated = frame

        state.publish_visual(annotated, detections, weapon_conf, person_conf, True)

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    print("[VPU] Stopped.")


def run_yolo(model, frame: np.ndarray):
    results = model.predict(
        frame,
        verbose=False,
        device="cpu",
        imgsz=YOLO_IMGSZ,
        conf=DISPLAY_CONF_MIN,
    )
    annotated = frame.copy()
    detections: list[dict] = []
    weapon_conf = 0.0
    person_conf = 0.0

    if not results:
        return annotated, detections, weapon_conf, person_conf

    result = results[0]
    names = result.names
    if result.boxes is None:
        return annotated, detections, weapon_conf, person_conf

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if isinstance(names, dict):
            class_name = str(names.get(cls_id, "")).lower()
        else:
            class_name = str(names[cls_id]).lower()
        if class_name not in TRACKED_CLASSES:
            continue
        if not (conf > DISPLAY_CONF_MIN):
            continue

        detections.append({"class": class_name, "confidence": conf})
        if class_name in WEAPON_CLASSES:
            weapon_conf = max(weapon_conf, conf)
            color = WEAPON_BOX_BGR
            label = f"weapon {conf:.2f}"
        else:
            person_conf = max(person_conf, conf)
            color = PERSON_BOX_BGR
            label = f"person {conf:.2f}"

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated, detections, weapon_conf, person_conf


# =============================================================================
# Audio Processing Unit
# =============================================================================

def apu_thread(state: SharedState) -> None:
    try:
        import sounddevice as sd
        import tensorflow as tf
        import tensorflow_hub as hub
    except Exception as exc:
        print(f"[APU] Audio stack unavailable ({exc}). Continuing video-only.")
        state.set_audio_status("unavailable")
        return

    print("[APU] Loading YAMNet (first run downloads the TF-Hub module)...")
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        yamnet = hub.load(YAMNET_HANDLE)
        class_names = load_yamnet_class_names(yamnet)
        distress_ids = [
            i
            for i, name in enumerate(class_names)
            if any(key in name.lower() for key in DISTRESS_KEYWORDS)
        ]
        print(f"[APU] YAMNet ready ({len(distress_ids)} distress-related labels).")
    except Exception as exc:
        print(f"[APU] YAMNet failed to load ({exc}). Continuing video-only.")
        state.set_audio_status("unavailable")
        return

    window_n = int(SAMPLE_RATE * AUDIO_WINDOW_SECONDS)
    ring: Deque[float] = deque(maxlen=window_n)
    ring_lock = threading.Lock()

    def on_audio(indata, frames, time_info, status):  # noqa: ARG001
        with ring_lock:
            ring.extend(indata[:, 0].tolist())

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=on_audio,
            blocksize=int(SAMPLE_RATE * 0.1),
        )
        stream.start()
    except Exception as exc:
        print(f"[APU] Microphone failed to open ({exc}). Continuing video-only.")
        state.set_audio_status("no microphone")
        return

    state.set_audio_status("live", ok=True)
    print("[APU] Microphone stream started.")

    try:
        while not state.stop.is_set():
            time.sleep(AUDIO_INFER_PERIOD)
            with ring_lock:
                if len(ring) < window_n // 4:
                    continue
                waveform = np.asarray(ring, dtype=np.float32)
            try:
                P_distress, label = score_distress(yamnet, waveform, class_names, distress_ids)
                state.publish_audio(P_distress, label)
            except Exception as exc:
                print(f"[APU] Inference error (continuing): {exc}")
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        print("[APU] Stopped.")


def load_yamnet_class_names(yamnet) -> list[str]:
    path = yamnet.class_map_path().numpy()
    if isinstance(path, bytes):
        path = path.decode("utf-8")
    try:
        import tensorflow as tf

        handle = tf.io.gfile.GFile(path, "r")
    except Exception:
        handle = open(path, "r", encoding="utf-8")
    names: list[str] = []
    with handle:
        for row in csv.DictReader(handle):
            names.append(row.get("display_name") or row.get("displayName") or "")
    return names


def score_distress(yamnet, waveform: np.ndarray, class_names: list[str], distress_ids: list[int]):
    import tensorflow as tf

    wave = np.clip(np.asarray(waveform, dtype=np.float32).reshape(-1), -1.0, 1.0)
    scores, _embeddings, _spectrogram = yamnet(wave)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    best_p = 0.0
    best_label = "none"
    for idx in distress_ids:
        p = float(mean_scores[idx])
        if p > best_p:
            best_p = p
            best_label = class_names[idx] if idx < len(class_names) else str(idx)
    return best_p, best_label


# =============================================================================
# HUD
# =============================================================================

def draw_hud(
    frame: np.ndarray,
    threat_state: str,
    C_weapon: float,
    C_person: float,
    P_distress: float,
    audio_label: str,
    audio_status: str,
    audio_ok: bool,
    override_text: str,
) -> np.ndarray:
    _h, w = frame.shape[:2]
    color = BANNER_BGR.get(threat_state, (80, 80, 80))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, BANNER_HEIGHT), color, -1)
    frame = cv2.addWeighted(overlay, 0.74, frame, 0.26, 0)

    cv2.putText(
        frame,
        f"THREAT STATE: {threat_state}",
        (16, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.45,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )

    note = get_verification_note(threat_state, P_distress)
    note_h = 0
    if note:
        note_h = NOTE_STRIP_HEIGHT
        cv2.rectangle(frame, (0, BANNER_HEIGHT), (w, BANNER_HEIGHT + note_h), (22, 22, 22), -1)
        note_color = (0, 220, 255) if "HUMAN VERIFICATION" in note else (210, 255, 210)
        cv2.putText(
            frame,
            note,
            (16, BANNER_HEIGHT + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            note_color,
            1,
            cv2.LINE_AA,
        )

    if audio_ok:
        audio_line = f"P_distress={P_distress:.2f}   label={audio_label}"
        audio_color = (235, 235, 235)
    else:
        audio_line = f"P_distress=n/a   audio={audio_status}"
        audio_color = (150, 150, 150)

    lines = [
        (f"C_weapon={C_weapon:.2f}   C_person={C_person:.2f}", (240, 240, 240)),
        (audio_line, audio_color),
        (
            f"override={override_text}    keys: [w]eapon  [s]cream  [n]oise  [c]lear  [q]uit",
            (240, 240, 240),
        ),
    ]
    y0 = BANNER_HEIGHT + note_h + 24
    for i, (text, col) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (16, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            col,
            1,
            cv2.LINE_AA,
        )
    return frame


def handle_key(raw_key: int, state: SharedState) -> bool:
    """Return False to quit. OpenCV waitKey is -1 when idle."""
    if raw_key is None or raw_key < 0:
        return True
    ch = chr(raw_key & 0xFF).lower()
    if ch == "q":
        return False
    if ch == "w":
        state.set_overrides(C_weapon=0.95)
        print("[DEMO] Override C_weapon=0.95 (expect CRITICAL)")
    elif ch == "s":
        state.set_overrides(P_distress=0.95)
        print("[DEMO] Override P_distress=0.95 (HIGH if person, WARNING if no visual)")
    elif ch == "n":
        state.set_overrides(P_distress=0.0)
        print("[DEMO] Override P_distress=0.0 (expect SAFE unless weapon)")
    elif ch == "c":
        state.set_overrides(clear=True)
        print("[DEMO] Overrides cleared")
    return True


# =============================================================================
# Main display loop
# =============================================================================

def main() -> None:
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "1")

    state = SharedState()
    print("SafeFusion v4")
    print(f"  CAMERA_INDEX={CAMERA_INDEX}  AUDIO_ENABLED={AUDIO_ENABLED}  YOLO_MODEL={YOLO_MODEL}")
    print("  Click the video window, then use w / s / n / c / q.")

    video = threading.Thread(target=vpu_thread, args=(state,), name="VPU", daemon=True)
    video.start()

    audio = None
    if AUDIO_ENABLED:
        audio = threading.Thread(target=apu_thread, args=(state,), name="APU", daemon=True)
        audio.start()
    else:
        state.set_audio_status("disabled")
        print("[APU] Skipped (AUDIO_ENABLED=False).")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    previous_state = "SAFE"

    try:
        while not state.stop.is_set():
            snap = state.snapshot()
            C_weapon, C_person, P_distress, has_visual, override_text = fusion_inputs_from_snapshot(snap)
            threat_state = fuse_threat(C_weapon, C_person, P_distress, has_visual)
            beep_on_enter(threat_state, previous_state)
            previous_state = threat_state

            frame = snap["frame"] if snap["frame"] is not None else placeholder_frame("Waiting for camera...")
            show_audio = snap["audio_ok"] or snap["override_P_distress"] is not None
            hud = draw_hud(
                frame,
                threat_state,
                C_weapon,
                C_person,
                P_distress,
                snap["audio_label"],
                snap["audio_status"],
                show_audio,
                override_text,
            )
            cv2.imshow(WINDOW_NAME, hud)
            if not handle_key(cv2.waitKey(DISPLAY_WAIT_MS), state):
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Shutting down...")
        state.stop.set()
        video.join(timeout=3.0)
        if audio is not None:
            audio.join(timeout=3.0)
        cv2.destroyAllWindows()
        print("Goodbye.")


if __name__ == "__main__":
    main()
