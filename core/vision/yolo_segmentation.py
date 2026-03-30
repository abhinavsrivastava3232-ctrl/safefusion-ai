import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ---------------- #

YOLO_MODEL_PATH = "yolov8n-seg.pt"  # segmentation model
HIGH_THREAT_THRESHOLD = 70
MODERATE_THREAT_THRESHOLD = 40

# Weapon-like classes (COCO labels) to treat as weapons
WEAPON_CLASSES = [
    "knife",
    "scissors",
    "bottle",
    "baseball bat",
    "hammer",
    "wrench",
    "sports ball",  # proxy for stone / throwable
]

# ---------------------------------------- #


def estimate_distress(person_boxes):
    """Detect proximity-based distress from person bounding boxes."""
    if len(person_boxes) < 2:
        return False

    centers = []
    for (x1, y1, x2, y2) in person_boxes:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy))

    min_dist = float("inf")
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            d = (dx * dx + dy * dy) ** 0.5
            if d < min_dist:
                min_dist = d

    return min_dist < 120  # tuned for 640x480


def compute_threat_score(weapon_detected, distress_detected, num_males, num_females):
    """Weighted threat scoring logic."""
    score = 0

    if weapon_detected:
        score += 60

    if distress_detected:
        score += 30

    # 1 female + >=2 males = extra risk
    if num_females >= 1 and num_males >= 2:
        score += 20

    return min(score, 100)


def draw_status_overlay(
    frame,
    threat_score,
    weapon_detected,
    distress_detected,
    num_males,
    num_females,
):
    """Draws top status bar and bottom danger banner."""
    if threat_score >= HIGH_THREAT_THRESHOLD:
        status = "HIGH RISK"
        color = (0, 0, 255)
    elif threat_score >= MODERATE_THREAT_THRESHOLD:
        status = "MODERATE RISK"
        color = (0, 165, 255)
    else:
        status = "SAFE"
        color = (0, 255, 0)

    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), thickness=-1)

    text1 = f"Status: {status} | Threat Score: {threat_score}"
    text2 = (
        f"Weapon: {weapon_detected} | Distress: {distress_detected} "
        f"| M: {num_males} F: {num_females}"
    )

    cv2.putText(
        frame,
        text1,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text2,
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Bottom danger banner
    if threat_score >= HIGH_THREAT_THRESHOLD:
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 255), thickness=-1)
        cv2.putText(
            frame,
            "DANGER DETECTED",
            (10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )


def main():
    print("[INFO] Loading YOLOv8 SEGMENTATION model...")
    model = YOLO(YOLO_MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not access webcam.")
        return

    print("[INFO] Press 'q' to quit.")

    frame_idx = 0
    prev_threat = 0
    prev_males = 0
    prev_females = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.resize(frame, (640, 480))

        # Run segmentation model
        results = model(frame, verbose=False)[0]

        person_boxes = []
        weapon_detected = False

        # Use boxes for logic
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = results.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0]

                if label == "person":
                    person_boxes.append((x1, y1, x2, y2))

                if label in WEAPON_CLASSES:
                    weapon_detected = True

        # -------- GENDER HEURISTIC + SMOOTHING -------- #

        if len(person_boxes) == 0:
            raw_females = 0
            raw_males = 0
        elif len(person_boxes) == 1:
            raw_females = 0
            raw_males = 1
        else:
            raw_females = 1
            raw_males = len(person_boxes) - 1

        frame_idx += 1

        if frame_idx % 5 == 0:
            num_males = raw_males
            num_females = raw_females
            prev_males, prev_females = num_males, num_females
        else:
            num_males, num_females = prev_males, prev_females

        # ---------------------------------------------- #

        distress_detected = estimate_distress(person_boxes)

        raw_threat = compute_threat_score(
            weapon_detected,
            distress_detected,
            num_males,
            num_females,
        )

        threat_score = int(0.7 * prev_threat + 0.3 * raw_threat)
        prev_threat = threat_score

        # --------- CUSTOM SOFT SEGMENTATION VISUALIZATION --------- #
        segmented = frame.copy()
        overlay = frame.copy()

        if results.masks is not None:
            masks = results.masks.data.cpu().numpy()  # shape: [N, H, W]
            boxes = results.boxes

            for mask, box in zip(masks, boxes):
                cls_id = int(box.cls[0])
                label = results.names[cls_id]

                # Only color person + weapon classes
                if label == "person":
                    color = (0, 255, 0)  # green mask for people
                elif label in WEAPON_CLASSES:
                    color = (0, 0, 255)  # red mask for weapons
                else:
                    continue

                # mask is [H, W], values 0..1
                mask_bool = mask > 0.5

                # Apply color to overlay where mask is True
                overlay[mask_bool] = (
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                )

        # Blend overlay with original frame
        alpha = 0.4  # transparency
        segmented = cv2.addWeighted(overlay, alpha, segmented, 1 - alpha, 0)

        # Draw bounding boxes + labels only for persons/weapons
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = results.names[cls_id]

                if label not in ["person"] + WEAPON_CLASSES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)

                if label == "person":
                    color = (0, 200, 0)
                    text = "Person"
                else:
                    color = (0, 0, 200)
                    text = label.capitalize()

                cv2.rectangle(segmented, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                cv2.putText(
                    segmented,
                    text,
                    (x1_i, y1_i - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Draw threat overlay
        draw_status_overlay(
            segmented,
            threat_score,
            weapon_detected,
            distress_detected,
            num_males,
            num_females,
        )

        cv2.imshow("Women Safety AI - Segmentation (Soft Masks)", segmented)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
