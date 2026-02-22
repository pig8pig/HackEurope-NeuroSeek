import cv2
import requests
import math
import threading
import time

import anthropic
from elevenlabs.client import ElevenLabs
from elevenlabs import play


# ━━━━━━━━━━━━━━━━━━━━━━ CONFIG ━━━━━━━━━━━━━━━━━━━━━━

# Which phases to run:
#   1 = Phase 1 only (vision + body-language overlay, no API keys needed)
#   2 = Phase 1 + 2 (adds Claude LLM advice, needs ANTHROPIC_API_KEY)
#   3 = All phases  (adds ElevenLabs TTS audio, needs ELEVEN_API_KEY too)
PHASE = 2

# Your live Cloudflare URL pointing to the A10G
API_URL = "https://wars-emerald-ratio-jvc.trycloudflare.com/analyze-pose"

# Thresholds (normalised 0-1 coords — tune for your camera distance)
CROSSED_ARMS_DIST   = 0.05   # max wrist-to-wrist distance to count as "crossed"
DISENGAGE_THRESHOLD = 0.08   # min nose-above-shoulder gap; below this → "looking down"

# Cooldown so we don't spam the LLM for the same gesture (seconds)
STATE_COOLDOWN = 10.0


# ━━━━━━━━━━━━━ PHASE 1 · SPATIAL TRANSLATOR ━━━━━━━━━━━━━

def _valid(kp):
    """Return True if the keypoint was actually detected (not [0, 0])."""
    return kp[0] > 0.0 or kp[1] > 0.0


def analyze_body_language(keypoints):
    """Take 17 COCO keypoints (normalised) for one person → list of state strings."""
    states = []

    nose       = keypoints[0]
    l_shoulder = keypoints[5]
    r_shoulder = keypoints[6]
    l_wrist    = keypoints[9]
    r_wrist    = keypoints[10]

    # ── Crossed Arms ──
    if _valid(l_wrist) and _valid(r_wrist):
        dist = _dist(l_wrist, r_wrist)
        if dist < CROSSED_ARMS_DIST:
            states.append("The person just crossed their arms.")

    # ── Open / Expansive Posture ──
    if _valid(l_wrist) and _valid(r_wrist):
        dist = _dist(l_wrist, r_wrist)
        if dist > OPEN_ARMS_DIST:
            states.append("The person has an open, expansive posture — arms spread wide.")

    # ── Hand Touching Face (thinking, anxious, or deceptive cue) ──
    if _valid(nose):
        if _valid(l_wrist) and _dist(l_wrist, nose) < HAND_TO_FACE_DIST:
            states.append("The person is touching their face with their left hand.")
        if _valid(r_wrist) and _dist(r_wrist, nose) < HAND_TO_FACE_DIST:
            states.append("The person is touching their face with their right hand.")

    # ── Hand Raised Above Shoulder (wants to speak / waving) ──
    if _valid(l_wrist) and _valid(l_shoulder):
        if l_wrist[1] < l_shoulder[1] - 0.05:
            states.append("The person has their left hand raised above their shoulder.")
    if _valid(r_wrist) and _valid(r_shoulder):
        if r_wrist[1] < r_shoulder[1] - 0.05:
            states.append("The person has their right hand raised above their shoulder.")

    # ── Fidgeting (rapid wrist movement between frames) ──
    if _valid(l_wrist) and _valid(r_wrist):
        if _prev_l_wrist is not None and _prev_r_wrist is not None:
            l_move = _dist(l_wrist, _prev_l_wrist)
            r_move = _dist(r_wrist, _prev_r_wrist)
            if l_move > FIDGET_THRESHOLD or r_move > FIDGET_THRESHOLD:
                states.append("The person appears to be fidgeting with their hands.")
        _prev_l_wrist = l_wrist
        _prev_r_wrist = r_wrist

    # ═══════════════════════════════════════════════════════════
    # HEAD STATES
    # ═══════════════════════════════════════════════════════════

    # ── Disengaged / Looking Down ──
    if _valid(nose) and _valid(l_shoulder) and _valid(r_shoulder):
        avg_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        gap = avg_shoulder_y - nose[1]
        if gap < DISENGAGE_THRESHOLD:
            states.append("The person is looking down and seems disengaged.")

    # ── Head Tilted (curiosity, confusion, or interest) ──
    if _valid(l_ear) and _valid(r_ear):
        tilt = _angle_deg(l_ear, r_ear)
        if tilt > HEAD_TILT_DEGREES:
            states.append("The person is tilting their head to the right — possibly curious or confused.")
        elif tilt < -HEAD_TILT_DEGREES:
            states.append("The person is tilting their head to the left — possibly curious or confused.")

    # ── Nodding (nose drops below eye midpoint) ──
    if _valid(nose) and _valid(l_eye) and _valid(r_eye):
        eye_mid_y = (l_eye[1] + r_eye[1]) / 2
        if nose[1] - eye_mid_y > 0.04:
            states.append("The person appears to be nodding — a sign of agreement.")

    # ── Looking Away / No Eye Contact (face turned sideways) ──
    if _valid(l_ear) and _valid(r_ear) and _valid(nose):
        # If one ear is much closer to the nose than the other, face is turned
        l_ear_to_nose = abs(l_ear[0] - nose[0])
        r_ear_to_nose = abs(r_ear[0] - nose[0])
        if l_ear_to_nose > 0.01 and r_ear_to_nose > 0.01:
            ratio = min(l_ear_to_nose, r_ear_to_nose) / max(l_ear_to_nose, r_ear_to_nose)
            if ratio < 0.4:
                states.append("The person is looking away — they may have lost interest or are thinking.")

    # ═══════════════════════════════════════════════════════════
    # SHOULDER / TORSO STATES
    # ═══════════════════════════════════════════════════════════

    # ── Shoulders Raised (tension, stress, nervousness) ──
    if _valid(l_shoulder) and _valid(r_shoulder) and _valid(l_ear) and _valid(r_ear):
        shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
        if shoulder_width > 0.01:
            l_gap = abs(l_shoulder[1] - l_ear[1])
            r_gap = abs(r_shoulder[1] - r_ear[1])
            avg_gap = (l_gap + r_gap) / 2
            ratio = avg_gap / shoulder_width
            if ratio < SHOULDER_RAISE_RATIO:
                states.append("The person's shoulders are raised — they may be tense or stressed.")

    # ── Asymmetric Shoulders (discomfort, shrug) ──
    if _valid(l_shoulder) and _valid(r_shoulder):
        angle = _angle_deg(l_shoulder, r_shoulder)
        if abs(angle) > SHOULDER_ASYM_DEGREES:
            states.append("The person has uneven shoulders — possibly shrugging or uncomfortable.")

    # ── Leaning Forward (engaged, interested) ──
    if (_valid(l_shoulder) and _valid(r_shoulder) and
            _valid(l_hip) and _valid(r_hip)):
        sh_mid_y = (l_shoulder[1] + r_shoulder[1]) / 2
        hip_mid_y = (l_hip[1] + r_hip[1]) / 2
        torso_len = abs(hip_mid_y - sh_mid_y)
        if torso_len > 0.01:
            vertical = (sh_mid_y - hip_mid_y) / torso_len
            if vertical < LEAN_FORWARD_RATIO:
                states.append("The person is leaning forward — they seem engaged and interested.")

    # ── Leaning Sideways (bored, shifting away) ──
    if (_valid(l_shoulder) and _valid(r_shoulder) and
            _valid(l_hip) and _valid(r_hip)):
        sh_mid_x  = (l_shoulder[0] + r_shoulder[0]) / 2
        hip_mid_x = (l_hip[0] + r_hip[0]) / 2
        torso_len = abs((l_hip[1] + r_hip[1]) / 2 - (l_shoulder[1] + r_shoulder[1]) / 2)
        if torso_len > 0.01:
            lateral = (sh_mid_x - hip_mid_x) / torso_len
            if lateral > LEAN_LATERAL_RATIO:
                states.append("The person is leaning to their right — they may be shifting away.")
            elif lateral < -LEAN_LATERAL_RATIO:
                states.append("The person is leaning to their left — they may be shifting away.")

    # ── Turned Away (shoulders rotated away from camera) ──
    if _valid(l_shoulder) and _valid(r_shoulder):
        shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
        if shoulder_width < 0.08:
            states.append("The person has turned their body away from you.")

    return states


# ━━━━━━━━━━━━━ PHASE 2 · LLM SOCIAL CO-PILOT ━━━━━━━━━━━━━

claude_client = anthropic.Anthropic() if PHASE >= 2 else None

SYSTEM_PROMPT = (
    "You are a real-time social coach for a neurodivergent user who is in a live "
    "conversation. Respond with exactly one short, actionable sentence. "
    "Be warm and concise."
)


def get_social_advice(state: str) -> str:
    """Send the detected body-language state to Claude and return 1-sentence advice."""
    msg = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"The person I am talking to just exhibited the following body language: "
                f"{state} What should I do?"
            ),
        }],
    )
    return msg.content[0].text


# ━━━━━━━━━━━━━ PHASE 3 · AUDIO OUTPUT ━━━━━━━━━━━━━

eleven_client = ElevenLabs() if PHASE >= 3 else None


def _speak(text: str):
    """Generate TTS via ElevenLabs and play it (blocking)."""
    try:
        audio = eleven_client.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",       # "George" — calm & clear
            model_id="eleven_turbo_v2_5",
        )
        play(audio)                                  # requires mpv or ffplay on PATH
    except Exception as e:
        print(f"  [Audio Error] {e}")


def speak_async(text: str):
    """Fire-and-forget TTS in a daemon thread so the webcam loop is never blocked."""
    threading.Thread(target=_speak, args=(text,), daemon=True).start()


# ━━━━━━━━━━━━━ VISUALISATION · KEYPOINT OVERLAY ━━━━━━━━━━━━━

COCO_KEYPOINT_NAMES = [
    "Nose", "L Eye", "R Eye", "L Ear", "R Ear",
    "L Shoulder", "R Shoulder", "L Elbow", "R Elbow",
    "L Wrist", "R Wrist", "L Hip", "R Hip",
    "L Knee", "R Knee", "L Ankle", "R Ankle",
]

# Pairs of keypoint indexes that form the skeleton
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),           # arms
    (5, 11), (6, 12), (11, 12),                 # torso
    (11, 13), (13, 15), (12, 14), (14, 16),     # legs
]

# Highlight the nodes we actively use for body-language detection
ACTIVE_NODES = {0, 5, 6, 9, 10}


def draw_keypoints(frame, keypoints_norm):
    """Draw COCO keypoints + skeleton onto the frame.

    `keypoints_norm` is the list of 17 [x, y] values normalised to 0-1.
    """
    h, w = frame.shape[:2]
    pts = []  # pixel coords

    for i, (nx, ny) in enumerate(keypoints_norm):
        if nx == 0.0 and ny == 0.0:
            pts.append(None)
            continue
        px, py = int(nx * w), int(ny * h)
        pts.append((px, py))

        # Larger green dot + label for active nodes; small blue dot for the rest
        if i in ACTIVE_NODES:
            cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(frame, COCO_KEYPOINT_NAMES[i], (px + 8, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.circle(frame, (px, py), 4, (255, 180, 0), -1)

    # Draw skeleton lines
    for (a, b) in SKELETON:
        if pts[a] and pts[b]:
            cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2)


# ━━━━━━━━━━━━━━━━━━━━━━ MAIN LOOP ━━━━━━━━━━━━━━━━━━━━━━

cap = cv2.VideoCapture(0)
print(f"NeuroSeek v0.2 [PHASE {PHASE} of 3] — Connecting to Red Hat OpenShift Cluster...")

# ── Shared state between the main (display) thread and the API thread ──
_lock = threading.Lock()
_latest_jpg: bytes | None = None         # most recent compressed frame for the API
_latest_kps: list | None = None          # most recent keypoints from the GPU
_active_states: list[str] = []
_state_display_until = 0.0
_latest_advice = ""
_last_triggered: dict[str, float] = {}


def _api_worker():
    """Background thread: grabs the latest JPEG, sends it to the GPU, updates shared state."""
    global _latest_kps, _active_states, _state_display_until, _latest_advice

    while cap.isOpened():
        # Grab the most recent frame (skip stale ones automatically)
        with _lock:
            jpg = _latest_jpg
        if jpg is None:
            time.sleep(0.01)
            continue

        try:
            resp = requests.post(API_URL, files={"file": jpg}, timeout=5)
            if resp.status_code != 200:
                print(f"  [Debug] Server returned HTTP {resp.status_code}: {resp.text[:200]}")
                time.sleep(1)
                continue
            data = resp.json()

            if data['status'] == 'success':
                kps = data['keypoints'][0]
                nose_x, nose_y = kps[0]

                with _lock:
                    _latest_kps = kps

                # ── Phase 1: detect body-language states ──
                states = analyze_body_language(kps)

                # Live terminal readout every frame
                if states:
                    print(f"  [State] {' | '.join(states)}")
                else:
                    print(f"  [State] Neutral — no body language signals detected")

                if states:
                    with _lock:
                        _active_states = states
                        _state_display_until = time.time() + 3.0

                for state in states:
                    now = time.time()
                    if now - _last_triggered.get(state, 0) > STATE_COOLDOWN:
                        _last_triggered[state] = now
                        print(f"  [Body Language] {state}")

                        if PHASE >= 2:
                            try:
                                advice = get_social_advice(state)
                                with _lock:
                                    _latest_advice = advice
                                print(f"  [Coach] {advice}")
                                if PHASE >= 3:
                                    speak_async(advice)
                            except Exception as llm_err:
                                print(f"  [LLM Error] {llm_err}")
            else:
                print("No person detected.")
                with _lock:
                    _latest_kps = None

        except Exception as e:
            print(f"Network lag or server unavailable... {e}")

        # Small sleep to avoid hammering the API faster than it can respond
        time.sleep(0.03)


# Start the API worker as a daemon thread
threading.Thread(target=_api_worker, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Compress the frame and hand it to the API thread (non-blocking)
    small = cv2.resize(frame, (640, 480))
    _, jpg = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    with _lock:
        _latest_jpg = jpg.tobytes()

    # 2. Draw the most recent keypoints we have (even if a few frames old)
    with _lock:
        kps = _latest_kps
        states = list(_active_states)
        show_states_until = _state_display_until
        advice = _latest_advice

    if kps is not None:
        draw_keypoints(frame, kps)

    # OSD: overlay detected body-language states
    if time.time() < show_states_until and states:
        for i, st in enumerate(states):
            y_pos = 30 + i * 30
            cv2.putText(frame, f">> {st}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # OSD: overlay the latest coach advice
    if advice and PHASE >= 2:
        cv2.putText(frame, advice[:90], (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.imshow('NeuroSeek Client Feed', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()