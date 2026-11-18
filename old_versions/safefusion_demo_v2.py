"""
SafeFusion v2 — same demo as safefusion_demo.py, plus:
  - lower overlay/track confidence (0.15) so weak boxes still draw
  - larger/bolder threat banner for room-scale readability
  - one-shot winsound beep on transition into CRITICAL or HIGH

Do not use this as a replacement for safefusion_demo.py (that file is the fallback).

Architecture (two sensor threads + a main-thread fusion/display loop):
  - Visual Processing Unit (VPU): webcam + YOLOv8-Nano
  - Audio Processing Unit (APU): microphone + YAMNet
  - Decision-Level Fusion Engine: rule-based threat state from shared scores

Designed for a live faculty demo: prefer staying up over failing loudly.
Press keys in the OpenCV window (window must be focused):
  w  simulate weapon   (C_weapon = 0.95  -> CRITICAL)
  s  simulate scream   (P_distress = 0.95 -> HIGH if person, WARNING if no visual)
  n  simulate noise    (P_distress = 0.0  -> typically SAFE)
  c  clear overrides   (back to live sensors)
  q  quit cleanly
"""

from __future__ import annotations

import csv
import os
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

try:
    import winsound

    HAS_WINSOUND = True
except Exception:
    HAS_WINSOUND = False

# =============================================================================
# Config — change these before a demo if the wrong camera/mic is selected
# =============================================================================

CAMERA_INDEX = 0
AUDIO_ENABLED = True  # set False to skip the APU thread entirely (video-only)

WINDOW_NAME = "SafeFusion"
YOLO_WEIGHTS = "yolov8n.pt"
YOLO_IMGSZ = 320  # smaller = faster on CPU; 640 is more accurate
DISPLAY_WAIT_MS = 30  # OpenCV waitKey period (~33 FPS cap; YOLO is usually slower)

SAMPLE_RATE = 16000
AUDIO_WINDOW_SEC = 1.0
AUDIO_INFER_INTERVAL_SEC = 0.5
YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

# COCO names we care about. Knife / scissors / baseball bat are treated as weapons.
PERSON_CLASS = "person"
WEAPON_CLASSES = frozenset({"knife", "scissors", "baseball bat"})
VISUAL_CLASSES = frozenset({PERSON_CLASS}) | WEAPON_CLASSES

# Substring match against YAMNet / AudioSet class names (case-insensitive).
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

# Overlay/track confidence only (does not change fusion trigger thresholds).
# Weak YOLO hits (e.g. knife at 0.20–0.30) still appear on-screen and in the console.
DISPLAY_CONF_THRESHOLD = 0.15

# Fusion thresholds (must match the spec exactly). Unchanged from v1.
TH_WEAPON = 0.5
TH_PERSON = 0.5
TH_DISTRESS_HIGH = 0.6
TH_DISTRESS_WARNING = 0.8

BANNER_COLORS = {
    "CRITICAL": (0, 0, 210),  # BGR red
    "HIGH": (0, 140, 255),  # orange
    "WARNING": (0, 220, 255),  # yellow
    "SAFE": (0, 170, 0),  # green
}
BOX_COLOR_PERSON = (255, 200, 0)
BOX_COLOR_WEAPON = (0, 0, 255)

PLACEHOLDER_H, PLACEHOLDER_W = 480, 640


# =============================================================================
# Shared state — lock-protected; sensor threads write, main thread reads
# =============================================================================

class SharedState:
    """Thread-safe snapshot of the latest VPU/APU outputs and demo overrides."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.stop = threading.Event()

        # Live sensor readings (written by VPU / APU).
        self.c_weapon = 0.0
        self.c_person = 0.0
        self.has_visual_detection = False
        self.p_distress = 0.0
        self.audio_label = "—"
        self.frame = _placeholder_frame("Starting SafeFusion...")

        # Capabilities / health (written once or when devices recover).
        self.camera_ok = False
        self.audio_ok = False
        self.audio_status = "disabled" if not AUDIO_ENABLED else "starting"

        # Manual overrides: None means "use live sensor value".
        self.override_c_weapon: Optional[float] = None
        self.override_p_distress: Optional[float] = None

    def snapshot(self) -> dict:
        with self._lock:
            frame = None if self.frame is None else self.frame.copy()
            return {
                "c_weapon": self.c_weapon,
                "c_person": self.c_person,
                "has_visual_detection": self.has_visual_detection,
                "p_distress": self.p_distress,
                "audio_label": self.audio_label,
                "frame": frame,
                "camera_ok": self.camera_ok,
                "audio_ok": self.audio_ok,
                "audio_status": self.audio_status,
                "override_c_weapon": self.override_c_weapon,
                "override_p_distress": self.override_p_distress,
            }

    def update_visual(
        self,
        frame: np.ndarray,
        c_weapon: float,
        c_person: float,
        has_visual_detection: bool,
        camera_ok: bool,
    ) -> None:
        with self._lock:
            self.frame = frame
            self.c_weapon = float(c_weapon)
            self.c_person = float(c_person)
            self.has_visual_detection = bool(has_visual_detection)
            self.camera_ok = bool(camera_ok)

    def update_audio(self, p_distress: float, label: str) -> None:
        with self._lock:
            self.p_distress = float(p_distress)
            self.audio_label = label
            self.audio_ok = True
            self.audio_status = "live"

    def set_audio_status(self, status: str, ok: bool = False) -> None:
        with self._lock:
            self.audio_status = status
            self.audio_ok = ok

    def set_overrides(
        self,
        c_weapon: Optional[float] = None,
        p_distress: Optional[float] = None,
        clear: bool = False,
    ) -> None:
        with self._lock:
            if clear:
                self.override_c_weapon = None
                self.override_p_distress = None
                return
            if c_weapon is not None:
                self.override_c_weapon = float(c_weapon)
            if p_distress is not None:
                self.override_p_distress = float(p_distress)


def _placeholder_frame(message: str) -> np.ndarray:
    frame = np.zeros((PLACEHOLDER_H, PLACEHOLDER_W, 3), dtype=np.uint8)
    frame[:] = (36, 36, 36)
    _put_centered(frame, message, y=PLACEHOLDER_H // 2, scale=0.8, color=(220, 220, 220))
    return frame


def _put_centered(frame: np.ndarray, text: str, y: int, scale: float, color) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
    x = max(10, (frame.shape[1] - tw) // 2)
    cv2.putText(frame, text, (x, y), font, scale, color, 2, cv2.LINE_AA)


# =============================================================================
# Fusion engine — called only from the main/display thread
# =============================================================================

def fuse_threat(
    c_weapon: float,
    c_person: float,
    p_distress: float,
    has_visual_detection: bool,
) -> str:
    """Rule-based decision-level fusion (exact spec)."""
    if c_weapon > TH_WEAPON:
        return "CRITICAL"
    if c_person > TH_PERSON and p_distress > TH_DISTRESS_HIGH:
        return "HIGH"
    if (not has_visual_detection) and p_distress > TH_DISTRESS_WARNING:
        return "WARNING"
    return "SAFE"


def apply_overrides(snap: dict) -> tuple[float, float, float, bool, str]:
    """Return fused inputs after applying any live-demo overrides."""
    c_weapon = snap["c_weapon"]
    c_person = snap["c_person"]
    p_distress = snap["p_distress"]
    has_visual = snap["has_visual_detection"]

    if snap["override_c_weapon"] is not None:
        c_weapon = snap["override_c_weapon"]
        # A weapon override is a visual event for fusion purposes.
        has_visual = True
    if snap["override_p_distress"] is not None:
        p_distress = snap["override_p_distress"]

    if snap["override_c_weapon"] is not None and snap["override_p_distress"] is not None:
        override_label = "weapon+distress"
    elif snap["override_c_weapon"] is not None:
        override_label = "weapon (w)"
    elif snap["override_p_distress"] is not None:
        if snap["override_p_distress"] >= 0.9:
            override_label = "scream (s)"
        else:
            override_label = "noise (n)"
    else:
        override_label = "none (live)"

    return c_weapon, c_person, p_distress, has_visual, override_label


# =============================================================================
# Visual Processing Unit (background thread)
# =============================================================================

def vpu_thread(state: SharedState) -> None:
    print("[VPU] Loading YOLOv8-Nano (first run downloads yolov8n.pt)...")
    model = None
    try:
        from ultralytics import YOLO

        model = YOLO(YOLO_WEIGHTS)
        print("[VPU] YOLO ready.")
    except Exception as exc:
        print(f"[VPU] YOLO failed to load ({exc}). Video will run without detections.")

    cap = None
    last_retry = 0.0

    while not state.stop.is_set():
        if cap is None or not cap.isOpened():
            now = time.time()
            if now - last_retry < 2.0 and cap is not None:
                state.update_visual(
                    _placeholder_frame("Waiting for camera..."),
                    0.0,
                    0.0,
                    False,
                    False,
                )
                time.sleep(0.2)
                continue
            last_retry = now
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                print(f"[VPU] Could not open camera index {CAMERA_INDEX}. Retrying...")
                state.update_visual(
                    _placeholder_frame(
                        f"Waiting for camera (index {CAMERA_INDEX})..."
                    ),
                    0.0,
                    0.0,
                    False,
                    False,
                )
                time.sleep(1.0)
                cap = None
                continue
            print(f"[VPU] Camera {CAMERA_INDEX} opened.")

        ok, frame = cap.read()
        if not ok or frame is None:
            state.update_visual(
                _placeholder_frame("Waiting for camera..."),
                0.0,
                0.0,
                False,
                False,
            )
            try:
                cap.release()
            except Exception:
                pass
            cap = None
            continue

        c_weapon, c_person, has_visual = 0.0, 0.0, False
        annotated = frame
        if model is not None:
            try:
                annotated, c_weapon, c_person, has_visual = _run_yolo(model, frame)
            except Exception as exc:
                print(f"[VPU] Inference error (continuing): {exc}")
                annotated = frame

        state.update_visual(annotated, c_weapon, c_person, has_visual, True)

    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    print("[VPU] Stopped.")


def _run_yolo(model, frame: np.ndarray):
    results = model.predict(
        frame,
        verbose=False,
        device="cpu",
        imgsz=YOLO_IMGSZ,
        classes=None,
        conf=DISPLAY_CONF_THRESHOLD,
    )
    annotated = frame.copy()
    c_weapon = 0.0
    c_person = 0.0
    has_visual = False
    detections = []

    if not results:
        return annotated, c_weapon, c_person, has_visual

    result = results[0]
    names = result.names
    if result.boxes is None or len(result.boxes) == 0:
        return annotated, c_weapon, c_person, has_visual

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if isinstance(names, dict):
            name = str(names.get(cls_id, "")).lower()
        else:
            name = str(names[cls_id]).lower()
        if name not in VISUAL_CLASSES:
            continue
        if conf < DISPLAY_CONF_THRESHOLD:
            continue
        detections.append((name, conf, box))

    if detections:
        print(
            "[VPU] detections: "
            + ", ".join(f"{n}={c:.2f}" for n, c, _ in detections)
        )

    for name, conf, box in detections:
        has_visual = True
        draw_name = name
        if name == PERSON_CLASS:
            c_person = max(c_person, conf)
            color = BOX_COLOR_PERSON
        else:
            c_weapon = max(c_weapon, conf)
            color = BOX_COLOR_WEAPON
            draw_name = f"weapon:{name}"

        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy.tolist()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{draw_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated, c_weapon, c_person, has_visual


# =============================================================================
# Audio Processing Unit (background thread)
# =============================================================================

def apu_thread(state: SharedState) -> None:
    """Capture ~1s of mono 16 kHz audio and score distress classes with YAMNet."""
    try:
        import sounddevice as sd
        import tensorflow as tf
        import tensorflow_hub as hub
    except Exception as exc:
        msg = f"audio imports failed ({exc})"
        print(f"[APU] {msg}. Continuing in video-only mode.")
        state.set_audio_status("unavailable")
        return

    print("[APU] Loading YAMNet from TensorFlow Hub (first run downloads the model)...")
    try:
        # Keep TF from grabbing every CPU thread and starving YOLO / the UI.
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        yamnet = hub.load(YAMNET_HANDLE)
        class_names = _load_yamnet_class_names(yamnet)
        distress_ids = _distress_class_indices(class_names)
        print(f"[APU] YAMNet ready. Tracking {len(distress_ids)} distress-related classes.")
        if not distress_ids:
            print("[APU] Warning: no AudioSet classes matched distress keywords.")
    except Exception as exc:
        print(f"[APU] YAMNet failed to load ({exc}). Continuing in video-only mode.")
        state.set_audio_status("unavailable")
        return

    window_samples = int(SAMPLE_RATE * AUDIO_WINDOW_SEC)
    audio_buf: deque[float] = deque(maxlen=window_samples)
    buf_lock = threading.Lock()

    def _callback(indata, frames, time_info, status):  # noqa: ARG001
        if status:
            # Non-fatal (underruns, etc.) — keep going for the demo.
            pass
        with buf_lock:
            audio_buf.extend(indata[:, 0].tolist())

    try:
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=_callback,
            blocksize=int(SAMPLE_RATE * 0.1),
        )
        stream.start()
    except Exception as exc:
        print(f"[APU] Microphone failed to open ({exc}). Continuing in video-only mode.")
        state.set_audio_status("no microphone")
        return

    state.set_audio_status("live", ok=True)
    print("[APU] Microphone stream started.")

    try:
        while not state.stop.is_set():
            time.sleep(AUDIO_INFER_INTERVAL_SEC)
            with buf_lock:
                if len(audio_buf) < window_samples // 4:
                    continue
                waveform = np.array(audio_buf, dtype=np.float32)
            try:
                p_distress, label = _yamnet_distress(yamnet, waveform, class_names, distress_ids)
                state.update_audio(p_distress, label)
            except Exception as exc:
                print(f"[APU] Inference error (continuing): {exc}")
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        print("[APU] Stopped.")


def _load_yamnet_class_names(yamnet) -> list[str]:
    """Read the 521 AudioSet class names shipped with the TF-Hub YAMNet module."""
    class_map_path = yamnet.class_map_path().numpy()
    if isinstance(class_map_path, bytes):
        class_map_path = class_map_path.decode("utf-8")
    names: list[str] = []
    with tf_gfile_open(class_map_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.append(row.get("display_name") or row.get("displayName") or "")
    return names


def tf_gfile_open(path: str):
    """Open a class-map CSV; TF Hub may store it in a virtual GFile path."""
    try:
        import tensorflow as tf

        return tf.io.gfile.GFile(path, "r")
    except Exception:
        return open(path, "r", encoding="utf-8")


def _distress_class_indices(class_names: list[str]) -> list[int]:
    ids = []
    for i, name in enumerate(class_names):
        lowered = name.lower()
        if any(keyword in lowered for keyword in DISTRESS_KEYWORDS):
            ids.append(i)
    return ids


def _yamnet_distress(yamnet, waveform: np.ndarray, class_names: list[str], distress_ids: list[int]):
    import tensorflow as tf

    if waveform.ndim > 1:
        waveform = waveform.reshape(-1)
    waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)
    scores, embeddings, spectrogram = yamnet(waveform)  # noqa: F841
    # Mean over the ~0.48s analysis frames in the 1s window.
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    best_p = 0.0
    best_label = "none"
    for i in distress_ids:
        p = float(mean_scores[i])
        if p > best_p:
            best_p = p
            best_label = class_names[i] if i < len(class_names) else str(i)
    return best_p, best_label


# =============================================================================
# Display / overlay helpers
# =============================================================================

def draw_hud(
    frame: np.ndarray,
    threat: str,
    c_weapon: float,
    c_person: float,
    p_distress: float,
    audio_label: str,
    audio_status: str,
    audio_ok: bool,
    override_label: str,
) -> np.ndarray:
    h, w = frame.shape[:2]
    banner_h = 80
    color = BANNER_COLORS.get(threat, (80, 80, 80))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)

    title = f"THREAT STATE: {threat}"
    cv2.putText(
        frame,
        title,
        (16, int(banner_h * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.45,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )

    if audio_ok:
        audio_txt = f"P_distress={p_distress:.2f}  label={audio_label}"
        audio_color = (230, 230, 230)
    else:
        audio_txt = f"P_distress=n/a  audio={audio_status}"
        audio_color = (150, 150, 150)

    lines = [
        f"C_weapon={c_weapon:.2f}   C_person={c_person:.2f}",
        audio_txt,
        f"override={override_label}    keys: [w]eapon  [s]cream  [n]oise  [c]lear  [q]uit",
    ]
    y0 = banner_h + 24
    for i, line in enumerate(lines):
        col = audio_color if i == 1 and not audio_ok else (240, 240, 240)
        cv2.putText(
            frame,
            line,
            (16, y0 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            col,
            1,
            cv2.LINE_AA,
        )
    return frame


def _play_threat_beep(threat: str) -> None:
    """Brief Windows beep; never raises (non-Windows / audio failures are ignored)."""
    if not HAS_WINSOUND:
        return
    try:
        if threat == "CRITICAL":
            winsound.Beep(1200, 180)
        elif threat == "HIGH":
            winsound.Beep(900, 150)
    except Exception:
        pass


def maybe_beep_on_transition(threat: str, last_threat: str) -> str:
    """Beep once when entering CRITICAL or HIGH; return threat for the next frame."""
    if threat in ("CRITICAL", "HIGH") and threat != last_threat:
        threading.Thread(
            target=_play_threat_beep,
            args=(threat,),
            daemon=True,
            name="AlertBeep",
        ).start()
    return threat


def handle_key(raw_key: int, state: SharedState) -> bool:
    """Return False if the app should quit. OpenCV waitKey is -1 when idle."""
    if raw_key is None or raw_key < 0:
        return True
    ch = chr(raw_key & 0xFF).lower()
    if ch == "q":
        return False
    if ch == "w":
        state.set_overrides(c_weapon=0.95)
        print("[DEMO] Override: C_weapon=0.95 (expect CRITICAL)")
    elif ch == "s":
        state.set_overrides(p_distress=0.95)
        print("[DEMO] Override: P_distress=0.95 (HIGH if person, WARNING if no visual)")
    elif ch == "n":
        state.set_overrides(p_distress=0.0)
        print("[DEMO] Override: P_distress=0.0 (noise rejection; expect SAFE)")
    elif ch == "c":
        state.set_overrides(clear=True)
        print("[DEMO] Overrides cleared — live sensors")
    return True


# =============================================================================
# Main — start threads, fusion + display loop, clean shutdown
# =============================================================================

def main() -> None:
    # Avoid OpenCV/Qt plugin surprises on some Windows Python installs.
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "1")

    state = SharedState()
    print("SafeFusion demo")
    print(f"  CAMERA_INDEX={CAMERA_INDEX}  AUDIO_ENABLED={AUDIO_ENABLED}")
    print("  Focus the video window, then use w/s/n/c/q.")

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
    last_threat = "SAFE"

    try:
        while not state.stop.is_set():
            snap = state.snapshot()
            c_weapon, c_person, p_distress, has_visual, override_label = apply_overrides(snap)

            # No live audio (disabled or failed): P_distress stays 0 unless overridden.
            audio_live = bool(snap["audio_ok"])
            if not audio_live and snap["override_p_distress"] is None:
                p_distress = 0.0

            threat = fuse_threat(c_weapon, c_person, p_distress, has_visual)
            last_threat = maybe_beep_on_transition(threat, last_threat)
            frame = snap["frame"] if snap["frame"] is not None else _placeholder_frame("No frame")
            show_audio = audio_live or snap["override_p_distress"] is not None
            hud = draw_hud(
                frame,
                threat,
                c_weapon,
                c_person,
                p_distress,
                snap["audio_label"],
                snap["audio_status"],
                show_audio,
                override_label,
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
