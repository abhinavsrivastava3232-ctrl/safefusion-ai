import cv2
from ultralytics import YOLO

# ---------------- CONFIG ---------------- #

YOLO_MODEL_PATH = "yolov8n.pt"

WEAPON_CLASSES = [
    "knife",
    "scissors",
    "bottle",
    "baseball bat",
    "hammer",
    "wrench",
    "sports ball",
]

CONF_THRESHOLD = 0.25
WEAPON_CONFIRM_FRAMES = 5

# ---------------------------------------- #


def main():
    print("[INFO] Loading YOLOv8 model...")
    model = YOLO(YOLO_MODEL_PATH)

    weapon_frame_count = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not accessible.")
        return

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        results = model(frame, verbose=False, conf=CONF_THRESHOLD)[0]

        annotated = frame.copy()
        seen_labels = set()
        weapon_this_frame = False

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = results.names[cls_id]

                seen_labels.add(f"{label} ({conf:.2f})")

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label in WEAPON_CLASSES and conf >= 0.4:
                    color = (0, 0, 255)
                    tag = f"{label} (WEAPON {conf:.2f})"
                    weapon_this_frame = True
                else:
                    color = (255, 0, 0)
                    tag = f"{label} ({conf:.2f})"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    tag,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if seen_labels:
            print("Detected:", ", ".join(sorted(seen_labels)))

        # -------- TEMPORAL WEAPON CONFIRMATION -------- #
        if weapon_this_frame:
            weapon_frame_count += 1
        else:
            weapon_frame_count = max(weapon_frame_count - 1, 0)

        weapon_confirmed = weapon_frame_count >= WEAPON_CONFIRM_FRAMES

        if weapon_confirmed:
            cv2.putText(
                annotated,
                "POTENTIAL WEAPON CONFIRMED",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
        # ---------------------------------------------- #

        cv2.imshow("Weapon Debug View - YOLO Classes + Knife Check", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
