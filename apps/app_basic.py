import cv2
from ultralytics import YOLO
import math

# ================= CONFIG ================= #
YOLO_MODEL_PATH = "yolov8n.pt"

WEAPON_CLASSES = ["knife", "scissors"]
WEAPON_CONF_THRESHOLD = 0.45
WEAPON_CONFIRM_FRAMES = 5

HIGH_THREAT_THRESHOLD = 70
MODERATE_THREAT_THRESHOLD = 40
# ========================================== #

model = YOLO(YOLO_MODEL_PATH)

def estimate_distress(person_boxes):
    if len(person_boxes) < 2:
        return False

    centers = []
    for x1, y1, x2, y2 in person_boxes:
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))

    min_dist = float("inf")
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            min_dist = min(min_dist, d)

    return min_dist < 120


def compute_threat_score(weapon, distress, people):
    score = 0
    if weapon:
        score += 60
    if distress:
        score += 30
    if people >= 2:
        score += 10
    return min(score, 100)


def draw_overlay(frame, threat, weapon, distress, people):
    if threat >= HIGH_THREAT_THRESHOLD:
        status, color = "HIGH RISK", (0, 0, 255)
    elif threat >= MODERATE_THREAT_THRESHOLD:
        status, color = "MODERATE RISK", (0, 165, 255)
    else:
        status, color = "SAFE", (0, 255, 0)

    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)

    cv2.putText(frame, f"Status: {status} | Threat: {threat}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(frame,
                f"Weapon:{weapon} Distress:{distress} People:{people}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)

    if threat >= HIGH_THREAT_THRESHOLD:
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 255), -1)
        cv2.putText(frame, "POTENTIAL DANGER DETECTED",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return

    weapon_frame_count = 0
    prev_threat = 0

    print("[INFO] Women Safety AI running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        results = model(frame, verbose=False)[0]

        person_boxes = []
        weapon_this_frame = False

        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                label = results.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label == "person":
                    person_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (255, 0, 0), 2)
                    cv2.putText(frame, "Person",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 0, 0), 2)

                elif label in WEAPON_CLASSES and conf > WEAPON_CONF_THRESHOLD:
                    weapon_this_frame = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, f"{label.capitalize()} ({conf:.2f})",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)

        weapon_frame_count = weapon_frame_count + 1 if weapon_this_frame else max(weapon_frame_count - 1, 0)
        weapon_detected = weapon_frame_count >= WEAPON_CONFIRM_FRAMES

        distress = estimate_distress(person_boxes)

        raw_threat = compute_threat_score(
            weapon_detected,
            distress,
            len(person_boxes)
        )

        threat = int(0.7 * prev_threat + 0.3 * raw_threat)
        prev_threat = threat

        draw_overlay(frame, threat, weapon_detected, distress, len(person_boxes))

        cv2.imshow("Women Safety AI - FINAL DEMO", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
