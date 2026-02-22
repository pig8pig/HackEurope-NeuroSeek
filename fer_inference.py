"""
fer_inference.py — Facial Expression Recognition for NeuroSeek

Loads the trained FERNet model and classifies facial expressions
in webcam frames. Designed to be called from client.py every 10 seconds.

Usage:
    from fer_inference import FERInference
    fer = FERInference("fer_model_best.pt")

    # Every 10 seconds, pass the current frame:
    result = fer.analyze(frame)
    # → {"emotion": "happy", "confidence": 0.87, "all_probs": {...}}
    # → None if no face detected
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Model Architecture (must match training) ──────────────────────────

class SqueezeExcite(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        mid = max(ch // r, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)

class ResBlock(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(oc)
        self.conv2 = nn.Conv2d(oc, oc, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(oc)
        self.se = SqueezeExcite(oc)
        self.shortcut = nn.Sequential(
            nn.Conv2d(ic, oc, 1, stride=stride, bias=False), nn.BatchNorm2d(oc)
        ) if stride != 1 or ic != oc else nn.Identity()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.se(self.bn2(self.conv2(out)))
        return F.relu(out + self.shortcut(x), inplace=True)

class FERNet(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2, bias=False), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            ResBlock(32, 64, stride=2), ResBlock(64, 64),
            ResBlock(64, 128, stride=2), ResBlock(128, 128),
            ResBlock(128, 256, stride=2), ResBlock(256, 256),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(256, 128),
            nn.ReLU(inplace=True), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes))
    def forward(self, x):
        return self.net(x)


# ── Emotion labels ────────────────────────────────────────────────────

EMOTION_LABELS = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy",
    4: "sad", 5: "surprise", 6: "neutral",
}


# ── Main inference class ──────────────────────────────────────────────

class FERInference:
    """
    Facial expression recognition from a single frame.

    1. Detects faces using OpenCV Haar Cascade (fast, CPU, no extra files)
    2. Crops the largest face
    3. Converts to 96×96 grayscale
    4. Runs through FERNet
    5. Returns emotion + confidence + full probability distribution
    """

    def __init__(self, model_path: str, device: str = ""):
        # Device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        print(f"[FER] Loading model from {model_path} on {self.device}...")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        self.img_size = ckpt.get("img_size", 96)
        num_classes = ckpt.get("num_classes", 7)

        self.model = FERNet(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        val_acc = ckpt.get("val_acc", 0)
        print(f"[FER] Model loaded — val acc: {val_acc:.1%}")

        # Face detector (ships with OpenCV, no download needed)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def analyze(self, frame: np.ndarray) -> dict | None:
        """
        Analyze a BGR frame for facial expression using Haar Cascade.

        Args:
            frame: BGR numpy array from cv2.VideoCapture

        Returns:
            dict with emotion, confidence, all_probs, box
            None if no face detected
        """
        # Detect faces
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray_full, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        if len(faces) == 0:
            return None

        # Take the largest face (closest to camera)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Crop with margin
        margin = int(0.1 * max(w, h))
        y1 = max(0, y - margin)
        y2 = min(frame.shape[0], y + h + margin)
        x1 = max(0, x - margin)
        x2 = min(frame.shape[1], x + w + margin)

        face_gray = gray_full[y1:y2, x1:x2]
        if face_gray.size == 0:
            return None

        return self._classify_face(face_gray, x, y, w, h)

    def analyze_from_keypoints(self, frame: np.ndarray, keypoints_norm: list) -> dict | None:
        """
        Analyze facial expression using YOLO pose keypoints to crop the face.

        Uses nose (0), L eye (1), R eye (2), L ear (3), R ear (4) to compute
        a square bounding box, crops from the BGR frame, converts to grayscale,
        and runs through FERNet.

        Args:
            frame: BGR numpy array from cv2.VideoCapture
            keypoints_norm: list of 17 [x, y] normalised (0-1) COCO keypoints

        Returns:
            dict with emotion, confidence, all_probs, box
            None if not enough face keypoints are visible
        """
        h_frame, w_frame = frame.shape[:2]

        # Gather valid face keypoints (indices 0-4: nose, L eye, R eye, L ear, R ear)
        face_kp_indices = [0, 1, 2, 3, 4]
        valid_pts = []
        for i in face_kp_indices:
            kp = keypoints_norm[i]
            if kp[0] > 0.0 or kp[1] > 0.0:  # not [0, 0]
                valid_pts.append((kp[0] * w_frame, kp[1] * h_frame))

        # Need at least 2 face keypoints to form a reasonable box
        if len(valid_pts) < 2:
            return None

        # Compute bounding box around valid face keypoints
        xs = [p[0] for p in valid_pts]
        ys = [p[1] for p in valid_pts]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2

        # Square side = max span with generous margin (1.8x for forehead/chin)
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        side = max(int(span * 1.8), 60)  # at least 60px
        half = side // 2

        # Square crop coordinates (clamped to frame)
        x1 = max(0, int(cx - half))
        y1 = max(0, int(cy - half))
        x2 = min(w_frame, int(cx + half))
        y2 = min(h_frame, int(cy + half))

        # Convert to grayscale and crop
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_gray = gray_full[y1:y2, x1:x2]
        if face_gray.size == 0:
            return None

        box_x, box_y = x1, y1
        box_w, box_h = x2 - x1, y2 - y1

        return self._classify_face(face_gray, box_x, box_y, box_w, box_h)

    def _classify_face(self, face_gray: np.ndarray, x: int, y: int, w: int, h: int) -> dict | None:
        """Shared classification: resize grayscale face crop → FERNet → result dict."""
        # Preprocess: resize to 96×96, normalize
        face_resized = cv2.resize(face_gray, (self.img_size, self.img_size))
        face_float = face_resized.astype(np.float32) / 255.0
        face_norm = (face_float - 0.5) / 0.5

        # To tensor: (1, 1, 96, 96)
        tensor = torch.FloatTensor(face_norm).unsqueeze(0).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        emotion_idx = int(probs.argmax())
        emotion_name = EMOTION_LABELS[emotion_idx]
        confidence = float(probs[emotion_idx])

        all_probs = {
            EMOTION_LABELS[i]: round(float(probs[i]), 3)
            for i in range(len(EMOTION_LABELS))
        }

        return {
            "emotion": emotion_name,
            "confidence": confidence,
            "all_probs": all_probs,
            "box": (int(x), int(y), int(w), int(h)),
        }

    def get_state_string(self, result: dict) -> str | None:
        """
        Convert a FER result into a body-language-style state string
        that can be appended to _active_states for Claude.

        Returns None if confidence is too low to be meaningful.
        """
        if result is None:
            return None

        emotion = result["emotion"]
        conf = result["confidence"]
        probs = result["all_probs"]

        # Skip low-confidence detections
        if conf < 0.35:
            return None

        # Skip neutral (not interesting on its own)
        if emotion == "neutral" and conf > 0.5:
            return None

        # Map emotions to state strings
        state_map = {
            "happy":    "The person is smiling — they appear happy and engaged.",
            "angry":    "The person's face shows anger or frustration.",
            "sad":      "The person looks sad or dejected.",
            "fear":     "The person looks anxious or fearful — they may be uncomfortable.",
            "surprise": "The person looks surprised.",
            "disgust":  "The person's expression shows displeasure or disapproval.",
        }

        state = state_map.get(emotion)
        if state is None:
            return None

        # Detect compound expressions from probability distribution

        # Confused: surprise + fear both elevated
        if probs.get("surprise", 0) > 0.25 and probs.get("fear", 0) > 0.15:
            state = "The person looks confused — consider explaining more clearly."

        # Polite/fake smile: low-confidence happy + high neutral
        if emotion == "happy" and conf < 0.55 and probs.get("neutral", 0) > 0.3:
            state = "The person is giving a polite smile — they may not be genuinely engaged."

        # Masking: happy face but fear/sad undertones
        if emotion == "happy" and (probs.get("fear", 0) > 0.15 or probs.get("sad", 0) > 0.15):
            state = "The person is smiling but may be masking discomfort."

        return state

    def draw_on_frame(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw the face bounding box and emotion label on the frame."""
        if result is None:
            return frame

        x, y, w, h = result["box"]
        emotion = result["emotion"]
        conf = result["confidence"]

        # Colour per emotion
        colours = {
            "angry": (0, 0, 255), "disgust": (0, 140, 0), "fear": (0, 100, 255),
            "happy": (0, 255, 100), "sad": (255, 150, 50),
            "surprise": (0, 255, 255), "neutral": (200, 200, 200),
        }
        colour = colours.get(emotion, (255, 255, 255))

        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        label = f"{emotion.upper()} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 4, y), colour, -1)
        cv2.putText(frame, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame
