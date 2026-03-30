import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
import clip

# ---------------- CONFIG ---------------- #

YOLO_MODEL_PATH = "yolov8n.pt"   # detection model
HIGH_THREAT_THRESHOLD = 70
MODERATE_THREAT_THRESHOLD = 40

# required to update 
WEAPON_CLASSES = [
    "knife",
    "scissors",
    "bottle",
    "baseball bat",
    "hammer",
    "wrench",
   
]

# required to update 
TEXT_CLASSES = [
    "a woman is being attacked",
    "a violent fight in a public place",
    "a person holding a weapon in a threatening way",
    "a dangerous and unsafe situation",
    "a normal safe public scene",
]

# ---------------- CLIP SETUP ---------------- #

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
text_tokens = clip.tokenize(TEXT_CLASSES).to(device)


def get_clip_unsafe_score(frame_bgr):
    """
    Returns an unsafe score in range [0, 100] using CLIP.
    Higher = more unsafe.
    """
    # Convert BGR (OpenCV) -> RGB (PIL)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    image_input = clip_preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_per_image, _ = clip_model(image_input, text_tokens)
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    # unsafe = sum of first 4 prompts
    unsafe_score = probs[0] + probs[1] + probs[2] + probs[3]
    return float(unsafe_score * 100.0)  # convert to 0–100 scale


# ---------------- CLASSIC THREAT MODULE ---------------- #


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
            min_dist = min(min_dist, d)

    return min_dist < 120  # tuned for 640x480


def compute_yolo_threat(weapon_detected, distress_detected, num_males, num_females):
    """
    Classic threat scoring (YOLO side):
    - Weapon present: +60
    - Distress (people very close): +30
    - Female outnumbered by males (>=2 males and >=1 female): +20
    """
    score = 0

    if weapon_detected:
        score += 60

    if distress_detected:
        score += 30

    if num_females >= 1 and num_males >= 2:
        score += 20

    return min(score, 100)


def draw_status_overlay(
    frame,
    final_threat,
    yolo_threat,
    clip_threat,
    weapon_detected,
    distress_detected,
    num_males,
    num_females,
):
    """Draws top status bar and bottom danger banner."""
    if final_threat >= HIGH_THREAT_THRESHOLD:
        status = "HIGH RISK"
        color = (0, 0, 255)
    elif final_threat >= MODERATE_THREAT_THRESHOLD:
        status = "MODERATE RISK"
        color = (0, 165, 255)
    else:
        status = "SAFE"
        color = (0, 255, 0)

    h, w = frame.shape[:2]

    # Top black bar
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), thickness=-1)

    text1 = f"Status: {status} | Final Threat: {final_threat}"
    text2 = (
        f"YOLO: {yolo_threat}  CLIP unsafe: {int(clip_threat)}  "
        f"Weapon: {weapon_detected}  Distress: {distress_detected}"
    )
    text3 = f"Males: {num_males}  Females: {num_females}"

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
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text3,
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Bottom danger banner
    if final_threat >= HIGH_THREAT_THRESHOLD:
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
    print("[INFO] Loading YOLOv8 model (detection)...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam not accessible.")
        return

    print("[INFO] Press 'q' to quit.")

    frame_idx = 0
    prev_final_threat = 0
    prev_clip_threat = 0
    prev_males = 0
    prev_females = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.resize(frame, (640, 480))

        # ---------- YOLO INFERENCE ---------- #
        yolo_results = yolo_model(frame, verbose=False)[0]

        person_boxes = []
        weapon_detected = False

        if yolo_results.boxes is not None:
            for box in yolo_results.boxes:
                cls_id = int(box.cls[0])
                label = yolo_results.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0]

                if label == "person":
                    person_boxes.append((x1, y1, x2, y2))

                if label in WEAPON_CLASSES:
                    weapon_detected = True

        # ---------- GENDER HEURISTIC (same logic) ---------- #
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

        # Smooth M/F (update every 5 frames)
        if frame_idx % 5 == 0:
            num_males = raw_males
            num_females = raw_females
            prev_males, prev_females = num_males, num_females
        else:
            num_males, num_females = prev_males, prev_females

        # ---------- DISTRESS ---------- #
        distress_detected = estimate_distress(person_boxes)

        # ---------- YOLO THREAT ---------- #
        yolo_threat = compute_yolo_threat(
            weapon_detected,
            distress_detected,
            num_males,
            num_females,
        )

        # ---------- CLIP THREAT (semantic) ---------- #
        # Run CLIP every 10 frames to save compute
        if frame_idx % 10 == 0:
            clip_threat = get_clip_unsafe_score(frame)
            prev_clip_threat = clip_threat
        else:
            clip_threat = prev_clip_threat

        # ---------- COMBINE THREATS ---------- #
        # raw_final = 60% YOLO + 40% CLIP
        raw_final = int(0.6 * yolo_threat + 0.4 * clip_threat)

        # Temporal smoothing
        final_threat = int(0.7 * prev_final_threat + 0.3 * raw_final)
        prev_final_threat = final_threat

        # ---------- DRAW VISUALS ---------- #
        annotated = frame.copy()

        # Draw only relevant YOLO boxes (persons + weapons)
        if yolo_results.boxes is not None:
            for box in yolo_results.boxes:
                cls_id = int(box.cls[0])
                label = yolo_results.names[cls_id]
                if label not in ["person"] + WEAPON_CLASSES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)

                if label == "person":
                    color = (255, 0, 0)
                    text = "Person"
                else:
                    color = (0, 0, 255)
                    text = label.capitalize()

                cv2.rectangle(annotated, (x1_i, y1_i), (x2_i, y2_i), color, 2)
                cv2.putText(
                    annotated,
                    text,
                    (x1_i, y1_i - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        draw_status_overlay(
            annotated,
            final_threat,
            yolo_threat,
            clip_threat,
            weapon_detected,
            distress_detected,
            num_males,
            num_females,
        )

        cv2.imshow("Women Safety AI - YOLO + CLIP Multimodal", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
