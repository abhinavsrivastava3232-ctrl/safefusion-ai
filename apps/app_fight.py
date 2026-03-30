import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import os
from datetime import datetime
import math

# ================= CONFIG =================
MODEL_PATH = "yolov8n.pt"
FRAME_SIZE = (640, 480)
FPS = 30

MOTION_THRESHOLD = 2.5
PROXIMITY_THRESHOLD = 150
BURST_TRIGGER = 2
CONTEXT_SEC = 20

BUFFER_SIZE = 45
RECORD_AFTER = 45
# =========================================

model = YOLO(MODEL_PATH)

prev_gray = None
frame_buffer = deque(maxlen=BUFFER_SIZE)

frame_count = 0
motion_acc = []
burst_window = deque(maxlen=CONTEXT_SEC)

recording = False
record_frames = 0
video_writer = None

CLIP_FOLDER = "saved_clips"
os.makedirs(CLIP_FOLDER, exist_ok=True)


def center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)//2, (y1+y2)//2)


def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def roi_motion(prev, curr, box):
    x1,y1,x2,y2 = box
    roi_prev = prev[y1:y2, x1:x2]
    roi_curr = curr[y1:y2, x1:x2]
    if roi_prev.size and roi_curr.size:
        return np.mean(cv2.absdiff(roi_prev, roi_curr))
    return 0


def main():
    global prev_gray, recording, record_frames, video_writer, frame_count

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return

    print("[INFO] FINAL SMART SYSTEM RUNNING")

    prev_dist = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, FRAME_SIZE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ---------- PERSON DETECTION ----------
        results = model(frame, conf=0.25, verbose=False)[0]
        boxes = []

        if results.boxes:
            for box in results.boxes:
                if results.names[int(box.cls[0])] == "person":
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    if (x2-x1)*(y2-y1) > 2200:
                        boxes.append((x1,y1,x2,y2))
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)

        people = len(boxes)

        # ---------- MOTION PER PERSON ----------
        motions = []
        if prev_gray is not None:
            for b in boxes[:2]:  # max 2 persons
                motions.append(roi_motion(prev_gray, gray, b))

        prev_gray = gray.copy()

        # ---------- PROXIMITY CHANGE ----------
        prox_collapse = False
        if len(boxes) >= 2:
            d = distance(center(boxes[0]), center(boxes[1]))
            if prev_dist is not None and (prev_dist - d) > 40:
                prox_collapse = True
            prev_dist = d

        # ---------- 1 SECOND BURST ----------
        frame_count += 1
        motion_acc.append(motions)

        if frame_count >= FPS:
            frame_count = 0

            high_motion_people = sum(1 for m in motions if m > MOTION_THRESHOLD)

            abnormal = (
                (high_motion_people >= 2) or
                prox_collapse
            )

            burst_window.append(1 if abnormal else 0)
            motion_acc.clear()

        burst_count = sum(burst_window)
        suspicious = burst_count >= BURST_TRIGGER

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

        if recording:
            video_writer.write(frame)
            record_frames -= 1
            if record_frames <= 0:
                video_writer.release()
                recording = False
                print("[INFO] Clip saved")

        # ---------- UI ----------
        status = "SUSPICIOUS" if suspicious else "NORMAL"
        color = (0,0,255) if suspicious else (0,255,0)

        cv2.rectangle(frame,(0,0),(640,70),(0,0,0),-1)
        cv2.putText(frame,f"STATUS: {status}",
                    (10,25),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
        cv2.putText(frame,f"People:{people}  Bursts:{burst_count}",
                    (10,55),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.imshow("Women Safety System – Smart", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
