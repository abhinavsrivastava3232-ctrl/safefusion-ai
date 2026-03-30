import cv2
from ultralytics import YOLO
from collections import deque
import os
from datetime import datetime

# ================= CONFIG =================
MODEL_PATH = "best.pt"             # fight detection YOLOv8 weights
FRAME_SIZE = (640, 480)
FPS = 30

BUFFER_SIZE = 90                   # ~3s pre-event
RECORD_AFTER = 150                 # ~5s post-event
CLIP_FOLDER = "saved_clips"

CONFIDENCE = 0.25
# =========================================

os.makedirs(CLIP_FOLDER, exist_ok=True)

# Load fight detection model
model = YOLO(MODEL_PATH)

# Globals
frame_buffer = deque(maxlen=BUFFER_SIZE)
recording = False
record_frames = 0
video_writer = None

def alert_user(event_time, people):
    msg = f"{event_time} | People: {people} | FIGHT detected\n"
    with open("alerts.log", "a") as log:
        log.write(msg)
    print("[ALERT]", msg.strip())

def main():
    global recording, record_frames, video_writer

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return

    print("[INFO] Fight Detection System Running | CPU Mode")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, FRAME_SIZE)

        # ---------- DETECTION ----------
        results = model(frame, conf=CONFIDENCE)[0]

        fight_detected = False
        people = 0

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 1:  # Violence/Fight class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "FIGHT", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    fight_detected = True
                    people += 1

        # ---------- BUFFER ----------
        frame_buffer.append(frame.copy())

        # ---------- CLIP SAVING ----------
        if fight_detected and not recording:
            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = os.path.join(CLIP_FOLDER, f"fight_{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(path, fourcc, FPS, FRAME_SIZE)
            for f in frame_buffer:
                video_writer.write(f)
            recording = True
            record_frames = RECORD_AFTER
            print(f"[INFO] Saving clip: {path}")
            alert_user(ts, people)

        if recording:
            video_writer.write(frame)
            record_frames -= 1
            if record_frames <= 0:
                video_writer.release()
                recording = False
                print("[INFO] Clip saved")

        # ---------- UI ----------
        status = "FIGHT DETECTED" if fight_detected else "NORMAL"
        color = (0, 0, 255) if fight_detected else (0, 255, 0)

        cv2.rectangle(frame, (0, 0), (640, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"People: {people}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Women Safety Surveillance – Fight Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
