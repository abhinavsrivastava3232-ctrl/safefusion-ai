import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# ===== BALANCED DEMO CONFIG =====
MODEL_PATH = "yolov8n.pt"
FRAME_SIZE = (640, 480)

MOTION_THRESHOLD = 6.0      # balanced
TRIGGER_FRAMES = 5          # smoothing
# =================================

model = YOLO(MODEL_PATH)
prev_gray = None
motion_window = deque(maxlen=TRIGGER_FRAMES)


def main():
    global prev_gray

    cap = cv2.VideoCapture(0)
    print("[DEMO MODE] Balanced Live Surveillance")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, FRAME_SIZE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = model(frame, conf=0.2, verbose=False)[0]

        person_boxes = []

        if results.boxes:
            for box in results.boxes:
                cls = int(box.cls[0])
                label = results.names[cls]

                if label == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)

                    if area > 2500:   # ignore tiny junk
                        person_boxes.append((x1, y1, x2, y2))
                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      (255, 0, 0), 2)

        people = len(person_boxes)

        # -------- ROI MOTION (STABLE) --------
        motion = 0
        if prev_gray is not None and person_boxes:
            x1, y1, x2, y2 = person_boxes[0]
            roi_prev = prev_gray[y1:y2, x1:x2]
            roi_curr = gray[y1:y2, x1:x2]

            if roi_prev.size and roi_curr.size:
                diff = cv2.absdiff(roi_prev, roi_curr)
                motion = np.mean(diff)

        prev_gray = gray.copy()

        motion_window.append(motion)
        avg_motion = sum(motion_window) / len(motion_window)

        suspicious = avg_motion > MOTION_THRESHOLD

        status = "SUSPICIOUS" if suspicious else "NORMAL"
        color = (0, 0, 255) if suspicious else (0, 255, 0)

        # -------- UI --------
        cv2.rectangle(frame, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {status}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame,
                    f"People: {people}  Motion(avg): {avg_motion:.1f}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        cv2.imshow("Women Safety Demo (Balanced)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
