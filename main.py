import cv2
import torch
import numpy as np
import time

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
    10: 'K', 11: 'L', 12: 'M', 13: 'N',
    14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y'
}


# ======================
# PREPROCESS HAND IMAGE
# ======================
def preprocess_hand(hand_img):
    gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    resized = cv2.resize(thresh, (28, 28))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    tensor = transform(resized).unsqueeze(0)
    return tensor


# ======================
# LOAD CNN MODEL
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
        out = model(tensor.to(device))
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)

    label = LABEL_MAP.get(pred.item(), "?")
    return label, conf.item()


# ======================
# MAIN PROGRAM
# ======================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = load_model(device)

    # ===== MediaPipe Hand Landmarker (Tasks API) =====
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

    print("Q = keluar")

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

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            xs = [lm.x for lm in hand]
            ys = [lm.y for lm in hand]

            # Bounding box tangan (lebih besar)
            x1 = int(min(xs) * w) - 60
            y1 = int(min(ys) * h) - 60
            x2 = int(max(xs) * w) + 60
            y2 = int(max(ys) * h) + 60

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            hand_img = frame[y1:y2, x1:x2]

            if hand_img.size != 0:
                tensor = preprocess_hand(hand_img)
                pred, conf = predict_sign(model, tensor, device)

                # Stabilization
                if pred == last_pred and conf > 0.85:
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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ======================
# ENTRY POINT
# ======================
if __name__ == "__main__":
    main()
