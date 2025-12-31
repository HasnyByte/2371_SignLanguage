import cv2
import torch
import time
import numpy as np
import os

from torchvision import transforms
from train import SignCNN

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


# ======================
# LABEL MAP (SIGN MNIST)
# ======================
LABEL_MAP = {
    0: 'A',  1: 'B',  2: 'C',  3: 'D',  4: 'E',
    5: 'F',  6: 'G',  7: 'H',  8: 'I',
    10: 'K', 11: 'L', 12: 'U', 13: 'N',
    14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'M', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y'
}


# ======================
# PREPROCESS (MATCH TRAINING)
# ======================
def preprocess_hand(hand_img):
    gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    tensor = transform(resized).unsqueeze(0)
    return tensor


# ======================
# SAVE WEBCAM DATASET
# ======================
def save_webcam_dataset(hand_img, label):
    base_dir = "dataset_webcam"
    label_dir = os.path.join(base_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))

    timestamp = int(time.time() * 1000)
    filename = os.path.join(label_dir, f"{label}_{timestamp}.png")

    cv2.imwrite(filename, resized)
    print(f"✓ Saved dataset: {filename}")


# ======================
# LOAD MODEL
# ======================
def load_model(device):
    model = SignCNN(num_classes=25).to(device)
    model.load_state_dict(
        torch.load("model/model.pth", map_location=device)
    )
    model.eval()
    print("✓ CNN model loaded")
    return model


# ======================
# PREDICTION
# ======================
def predict_sign(model, tensor, device):
    with torch.no_grad():
        outputs = model(tensor.to(device))
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = LABEL_MAP.get(pred.item(), "?")
    return label, conf.item()


# ======================
# MAIN
# ======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = load_model(device)

    # MediaPipe Hand Landmarker
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="model/hand_landmarker.task"
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )

    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("✗ Webcam tidak bisa dibuka")
        return

    print("Q = keluar | S = simpan dataset (confidence > 0.9)")

    last_pred = ""
    stable_count = 0
    stable_threshold = 8

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        timestamp = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        hand_img = None
        pred, conf = "", 0.0

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            xs = [lm.x for lm in hand]
            ys = [lm.y for lm in hand]

            x1 = max(0, int(min(xs) * w) - 40)
            y1 = max(0, int(min(ys) * h) - 40)
            x2 = min(w, int(max(xs) * w) + 40)
            y2 = min(h, int(max(ys) * h) + 40)

            hand_img = frame[y1:y2, x1:x2]

            if hand_img.size > 0:
                tensor = preprocess_hand(hand_img)
                pred, conf = predict_sign(model, tensor, device)

                # Stabilization
                if pred == last_pred and conf > 0.8:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_pred = pred

                display = pred if stable_count >= stable_threshold else "..."

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{display} ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Sign Language Recognition (Hand Only)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s") and hand_img is not None and conf > 0.9:
            save_webcam_dataset(hand_img, pred)

    cap.release()
    cv2.destroyAllWindows()


# ======================
# ENTRY POINT
# ======================
if __name__ == "__main__":
    main()