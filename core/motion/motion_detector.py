import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import os
from datetime import datetime
import math

# ================= CONFIG =================
MODEL_PATH = "yolov8n.pt"          # keep small for CPU
FRAME_SIZE = (640, 480)
FPS = 30

# Motion & proximity (looser to verify pipeline)
MOTION_THRESHOLD = 0.8             # mean optical-flow magnitude per ROI
MOTION_VARIANCE_THRESHOLD = 3.0    # chaotic motion variance threshold
PROXIMITY_THRESHOLD = 220          # absolute distance threshold (pixels)
PROXIMITY_DELTA = 15               # sudden collapse vs previous min distance

# Recording
BUFFER_SIZE = 90                   # ~3s pre-event
RECORD_AFTER = 150                 # ~5s post-event
CLIP_FOLDER = "saved_clips"

# Detection
CONFIDENCE = 0.25
MAX_PEOPLE = 6                     # cap for pairwise checks
MIN_BOX_AREA = 600                 # allow small persons so we don't miss detections
# =========================================

os.makedirs(CLIP_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

prev_gray = None
frame_buffer = deque(maxlen=BUFFER_SIZE)

recording = False
record_frames = 0
video_writer = None

prev_min_dist = None


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def roi_motion_from_flow(mag, box):
    x1, y1, x2, y2 = box
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(mag.shape[1], x2); y2 = min(mag.shape[0], y2)
    if x2 > x1 and y2 > y1:
        return float(np.mean(mag[y1:y2, x1:x2]))
    return 0.0


def motion_variance(mag, box):
    x1, y1, x2, y2 = box
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(mag.shape[1], x2); y2 = min(mag.shape[0], y2)
    roi = mag[y1:y2, x1:x2]
    if roi.size:
        return float(np.var(roi))
    return 0.0


def alert_user(event_time, people, reason):
    msg = f"{event_time} | People: {people} | Reason: {reason}\n"
    with open("alerts.log", "a") as log:
        log.write(msg)
    print("[ALERT]", msg.strip())


def main():
    global prev_gray, recording, record_frames, video_writer, prev_min_dist

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error: cannot open index 0")
        return

    print("[INFO] SMART SYSTEM RUNNING | CPU MODE")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] No frame from camera")
            break

        frame = cv2.resize(frame, FRAME_SIZE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------- OPTICAL FLOW ----------
        mag = None
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # ---------- PERSON DETECTION ----------
        # Robust handling of names and classes
        results = model.predict(frame, conf=CONFIDENCE, device='cpu', verbose=False)[0]
        names = results.names if hasattr(results, "names") else model.names

        boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            for b in results.boxes:
                cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls[0])
                name = names.get(cls_id, "") if isinstance(names, dict) else names[cls_id]
                if name == "person":
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    if (x2 - x1) * (y2 - y1) > MIN_BOX_AREA:
                        boxes.append((x1, y1, x2, y2))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        boxes = boxes[:MAX_PEOPLE]
        people = len(boxes)

        # ---------- MOTION ----------
        motions, variances = [], []
        if mag is not None and people > 0:
            for bx in boxes:
                motions.append(roi_motion_from_flow(mag, bx))
                variances.append(motion_variance(mag, bx))

        prev_gray = gray.copy()

        # ---------- PROXIMITY ----------
        prox_collapse = False
        min_dist = None
        if len(boxes) >= 2:
            centers = [center(b) for b in boxes]
            dists = [distance(centers[i], centers[j]) for i in range(len(centers)) for j in range(i + 1, len(centers))]
            if dists:
                min_dist = min(dists)
                if min_dist < PROXIMITY_THRESHOLD:
                    prox_collapse = True
                if prev_min_dist is not None and (prev_min_dist - min_dist) > PROXIMITY_DELTA:
                    prox_collapse = True
                prev_min_dist = min_dist
        else:
            prev_min_dist = None

        # ---------- PER-FRAME SUSPICION ----------
        high_motion_people = sum(1 for m in motions if m > MOTION_THRESHOLD)
        chaotic_motion = any(v > MOTION_VARIANCE_THRESHOLD for v in variances)

        # Rule: either chaotic multi-person motion OR proximity collapse
        abnormal = (high_motion_people >= 2 and chaotic_motion) or prox_collapse
        suspicious = abnormal

        # ---------- DEBUG PRINTS ----------
        print(f"[DEBUG] People={people} | Motions={np.round(motions,2)} | "
              f"Vars={np.round(variances,2)} | HighMotion={high_motion_people} | "
              f"Chaotic={chaotic_motion} | MinDist={None if min_dist is None else int(min_dist)} | "
              f"ProxCollapse={prox_collapse} | Suspicious={suspicious}")

        # ---------- BUFFER ----------
        frame_buffer.append(frame.copy())

        # ---------- CLIP SAVING ----------
        if suspicious and not recording:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = os.path.join(CLIP_FOLDER, f"event_{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(path, fourcc, FPS, FRAME_SIZE)
            for f in frame_buffer:
                video_writer.write(f)
            recording = True
            record_frames = RECORD_AFTER
            print(f"[INFO] Saving clip: {path}")
            reason = f"chaotic={chaotic_motion}, highMotion={high_motion_people}, proxCollapse={prox_collapse}"
            alert_user(ts, people, reason)

        if recording:
            video_writer.write(frame)
            record_frames -= 1
            if record_frames <= 0:
                video_writer.release()
                recording = False
                print("[INFO] Clip saved")

        # ---------- UI ----------
        status = "SUSPICIOUS" if suspicious else "NORMAL"
        color = (0, 0, 255) if suspicious else (0, 255, 0)

        cv2.rectangle(frame, (0, 0), (640, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"People: {people}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if min_dist is not None:
            cv2.putText(frame, f"MinDist: {int(min_dist)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        # Show motion summary on overlay
        if motions:
            m_txt = f"M:{np.round(np.mean(motions),2)} V:{np.round(np.mean(variances),2)} HM:{high_motion_people}"
            cv2.putText(frame, m_txt, (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Women Safety System – Smart (Per-frame)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
