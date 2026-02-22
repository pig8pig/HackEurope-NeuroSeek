import io
import os
import sys
import queue
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import cv2
import requests
import math
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from dotenv import load_dotenv
load_dotenv()

import anthropic
from elevenlabs.client import ElevenLabs
from fer_inference import FERInference

pygame.mixer.init()


# ━━━━━━━━━━━━━━━━━━━━━━ CONFIG ━━━━━━━━━━━━━━━━━━━━━━

# Which phases to run:
#   1 = Phase 1 only (vision + body-language overlay, no API keys needed)
#   2 = Phase 1 + 2 (adds Claude LLM advice, needs ANTHROPIC_API_KEY)
#   3 = All phases  (adds ElevenLabs TTS audio, needs ELEVEN_API_KEY too)
PHASE = 3

# Your live Cloudflare URL pointing to the A10G
API_URL = "https://wars-emerald-ratio-jvc.trycloudflare.com/analyze-pose"

# Thresholds (normalised 0-1 coords — tune for your camera distance)
CROSSED_ARMS_DIST      = 0.12    # max wrist-to-wrist distance for "crossed" (relaxed — torso check gates it)
OPEN_ARMS_DIST         = 0.35    # min wrist-to-wrist distance for "open posture"
HAND_TO_FACE_DIST      = 0.07    # max wrist-to-nose distance for "touching face"
DISENGAGE_THRESHOLD    = 0.08    # min nose-above-shoulder gap; below this → "looking down"
FIDGET_THRESHOLD       = 0.06    # min frame-to-frame wrist movement to count as fidgeting
HEAD_TILT_DEGREES      = 20.0    # min ear-to-ear angle for "head tilt"
SHOULDER_RAISE_RATIO   = 0.25    # ear-shoulder gap / shoulder width — below = raised
SHOULDER_ASYM_DEGREES  = 18.0    # min shoulder angle for "uneven shoulders"
LEAN_FORWARD_RATIO     = 0.85    # shoulder-hip vertical ratio — below = leaning forward
LEAN_LATERAL_RATIO     = 0.15    # shoulder-hip lateral offset ratio — above = leaning sideways

# Minimum gap between Claude API calls (seconds)
CLAUDE_COOLDOWN = 15.0

# ── Facial Expression Recognition Config ──
FER_MODEL_PATH = "fer_model_best.pt"
FER_INTERVAL = 5.0                # seconds between FER analyses (3 per Claude cycle)

# ── Audio / STT Config ──
SR = 16000                  # sampling rate for microphone capture
CHANNELS = 1                # mono
WINDOW_SECONDS = 15         # STT window (seconds)
BLOCK_SECONDS = 0.5         # streaming block size (seconds) for sounddevice callback

# Shared audio queue — sounddevice callback pushes blocks here
_audio_queue: queue.Queue = queue.Queue()


# ━━━━━━━━━━━━━ AUDIO HELPERS ━━━━━━━━━━━━━

def int16_to_float32(x: np.ndarray) -> np.ndarray:
    """Convert int16 audio to float32 in [-1, 1]."""
    if x.dtype == np.int16:
        return x.astype(np.float32) / 32768.0
    return x.astype(np.float32)


def _audio_callback(indata, frames, time_info, status):
    """sounddevice callback: push incoming audio into the shared queue."""
    if status:
        print(f"  [Mic Status] {status}", file=sys.stderr)
    _audio_queue.put(indata.copy())


def _collect_audio_segment(duration_s: float) -> np.ndarray:
    """Drain *duration_s* seconds of audio from _audio_queue → 1-D float32 array."""
    needed = int(round(duration_s * SR))
    collected = np.zeros((0, CHANNELS), dtype=np.float32)
    while collected.shape[0] < needed:
        try:
            block = _audio_queue.get(timeout=5.0)
        except queue.Empty:
            raise RuntimeError("No audio received from microphone (timeout).")
        block = int16_to_float32(block) if block.dtype.kind == "i" else block.astype(np.float32)
        collected = np.vstack((collected, block))
    collected = collected[:needed, :]
    return collected[:, 0] if CHANNELS == 1 else np.mean(collected, axis=1)


# ━━━━━━━━━━━━━ PHASE 1 · SPATIAL TRANSLATOR ━━━━━━━━━━━━━

def _valid(kp):
    """Return True if the keypoint was actually detected (not [0, 0])."""
    return kp[0] > 0.0 or kp[1] > 0.0


def _dist(a, b):
    """Euclidean distance between two normalised keypoints."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _angle_deg(a, b):
    """Tilt angle in degrees from horizontal. 0° = level.
    Uses abs(dx) so L/R ordering of COCO keypoints doesn't matter.
    Positive = b is lower than a, negative = b is higher."""
    dx = abs(b[0] - a[0])
    dy = b[1] - a[1]
    if dx < 0.001:
        return 0.0
    return math.degrees(math.atan2(dy, dx))



# Global state for fidget detection (rolling buffer of last 5 frame movements)
_fidget_l_history = [0.0] * 5
_fidget_r_history = [0.0] * 5
_prev_l_wrist = None
_prev_r_wrist = None


def analyze_body_language(keypoints):
    """Take 17 COCO keypoints (normalised) for one person → list of state strings."""
    global _prev_l_wrist, _prev_r_wrist
    states = []

    nose       = keypoints[0]
    l_eye      = keypoints[1]
    r_eye      = keypoints[2]
    l_ear      = keypoints[3]
    r_ear      = keypoints[4]
    l_shoulder = keypoints[5]
    r_shoulder = keypoints[6]
    l_elbow    = keypoints[7]
    r_elbow    = keypoints[8]
    l_wrist    = keypoints[9]
    r_wrist    = keypoints[10]
    l_hip      = keypoints[11]
    r_hip      = keypoints[12]

    # ── Crossed Arms / Closed Position ──
    # Robust detection that handles YOLO swapping L/R wrist labels.
    # We treat both wrists as a pair ("wrist A" and "wrist B") without caring
    # which one YOLO labelled as left or right.
    if (_valid(l_wrist) and _valid(r_wrist) and
            _valid(l_shoulder) and _valid(r_shoulder)):
        wrist_a, wrist_b = l_wrist, r_wrist

        # Shoulder bounding box (horizontal + vertical)
        sh_left_x  = min(l_shoulder[0], r_shoulder[0])
        sh_right_x = max(l_shoulder[0], r_shoulder[0])
        sh_top_y   = min(l_shoulder[1], r_shoulder[1])

        # Hip midpoint Y (if available) — otherwise estimate as shoulder + 0.25
        if _valid(l_hip) and _valid(r_hip):
            hip_mid_y = (l_hip[1] + r_hip[1]) / 2
        else:
            hip_mid_y = sh_top_y + 0.25

        # 1) Both wrists must be between the shoulders horizontally
        #    (i.e., in front of the torso, not behind the back or to the sides)
        margin = 0.05  # small margin outside shoulder line
        a_in_torso_x = (sh_left_x - margin) < wrist_a[0] < (sh_right_x + margin)
        b_in_torso_x = (sh_left_x - margin) < wrist_b[0] < (sh_right_x + margin)

        # 2) Both wrists must be in the upper 2/3 of the torso (not low at the waist)
        #    That is: wrist_y < (hip_mid_y - (torso_len * 1/3))
        torso_len = hip_mid_y - sh_top_y
        min_y = sh_top_y - 0.03
        max_y = hip_mid_y - (torso_len * (1/3))
        a_at_chest = min_y < wrist_a[1] < max_y
        b_at_chest = min_y < wrist_b[1] < max_y

        in_front_and_chest = a_in_torso_x and b_in_torso_x and a_at_chest and b_at_chest

        if in_front_and_chest:
            wrist_dist = _dist(wrist_a, wrist_b)

            # Check A: wrists are close together at chest level
            close_together = wrist_dist < CROSSED_ARMS_DIST

            # Check B: wrists have crossed over each other (left wrist on
            # the right side and vice versa). YOLO may swap the labels, so
            # we also check relative to the elbows if available.
            crossover = False
            if _valid(l_elbow) and _valid(r_elbow):
                # If a wrist is on the opposite side of its own elbow, it crossed
                l_crossed = (l_wrist[0] > l_elbow[0] + 0.03) if l_shoulder[0] < r_shoulder[0] else (l_wrist[0] < l_elbow[0] - 0.03)
                r_crossed = (r_wrist[0] < r_elbow[0] - 0.03) if l_shoulder[0] < r_shoulder[0] else (r_wrist[0] > r_elbow[0] + 0.03)
                crossover = l_crossed and r_crossed

            if close_together or crossover:
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

    # ── Fidgeting (rapid wrist movement over several frames) ──
    if _valid(l_wrist) and _valid(r_wrist):
        global _fidget_l_history, _fidget_r_history
        if _prev_l_wrist is not None and _prev_r_wrist is not None:
            l_move = _dist(l_wrist, _prev_l_wrist)
            r_move = _dist(r_wrist, _prev_r_wrist)
            # Update rolling history (last 5 frames)
            _fidget_l_history = _fidget_l_history[1:] + [l_move]
            _fidget_r_history = _fidget_r_history[1:] + [r_move]
            # Count how many of the last 5 frames had big movement
            l_fidget_count = sum(1 for m in _fidget_l_history if m > FIDGET_THRESHOLD)
            r_fidget_count = sum(1 for m in _fidget_r_history if m > FIDGET_THRESHOLD)
            # Only trigger if 3+ out of 5 frames had big movement for either hand
            if l_fidget_count >= 3 or r_fidget_count >= 3:
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
    # Neutral zone: -HEAD_TILT_DEGREES to +HEAD_TILT_DEGREES is considered straight
    if _valid(l_ear) and _valid(r_ear):
        tilt = _angle_deg(l_ear, r_ear)
        if tilt > HEAD_TILT_DEGREES:
            states.append("The person is tilting their head to the right — possibly curious or confused.")
        elif tilt < -HEAD_TILT_DEGREES:
            states.append("The person is tilting their head to the left — possibly curious or confused.")
        # else: within ±HEAD_TILT_DEGREES → neutral, no state

    # ── Nodding (nose drops below eye midpoint) ──
    if _valid(nose) and _valid(l_eye) and _valid(r_eye):
        eye_mid_y = (l_eye[1] + r_eye[1]) / 2
        if nose[1] - eye_mid_y > 0.06:
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
    # Neutral zone: -SHOULDER_ASYM_DEGREES to +SHOULDER_ASYM_DEGREES is considered even
    if _valid(l_shoulder) and _valid(r_shoulder):
        angle = _angle_deg(l_shoulder, r_shoulder)
        if abs(angle) > SHOULDER_ASYM_DEGREES:
            states.append("The person has uneven shoulders — possibly shrugging or uncomfortable.")
        # else: within ±SHOULDER_ASYM_DEGREES → neutral, no state

    # ── Turned Away (shoulders rotated away from camera) ──
    if _valid(l_shoulder) and _valid(r_shoulder):
        shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
        if shoulder_width < 0.08:
            states.append("The person has turned their body away from you.")

    return states


# ━━━━━━━━━━━━━ PHASE 2 · LLM SOCIAL CO-PILOT ━━━━━━━━━━━━━

claude_client = anthropic.Anthropic() if PHASE >= 2 else None

# ── Facial Expression Recognition (safe load) ──
fer_engine = None
if os.path.exists(FER_MODEL_PATH) and os.path.getsize(FER_MODEL_PATH) > 0:
    try:
        fer_engine = FERInference(FER_MODEL_PATH)
        print("[FER] Facial expression recognition active.")
    except Exception as e:
        print(f"[FER] Model file exists but failed to load (likely still training/empty): {e}")
else:
    print(f"[FER] Model empty or not found at {FER_MODEL_PATH} — FER disabled for now.")

SYSTEM_PROMPT = (
    "You are NeuroCue, a real-time multi-modal social cue interpreter and coach "
    "designed for neurodivergent individuals and employees in workplace settings.\n\n"
    "You receive three types of signals: body language from pose estimation, facial "
    "expressions from emotion recognition, and a transcript of what was said. "
    "When body language and facial expression conflict (e.g. open posture but fearful "
    "face), note the discrepancy — the mismatch itself is informative. Facial "
    "expressions are harder to fake than body posture.\n\n"
    "Every 15 seconds you receive:\n"
    "  • [Latest Body Language] — visual pose / gesture observations from a camera.\n"
    "  • [Facial Expression] — emotion detected from face recognition (may be empty "
    "if no face found or model not loaded).\n"
    "  • [Transcript] — what they actually said, transcribed from audio.\n\n"
    "Synthesise ALL available streams into a single, coherent picture. "
    "Respond with EXACTLY two parts:\n"
    "1. **What's happening:** A single short sentence summarising what the other person "
    "is likely feeling or communicating socially.\n"
    "2. **What to do:** One or two brief, concrete, actionable sentences the user can "
    "act on right now.\n\n"
    "Keep the total response under 50 words. Be warm, direct, and non-judgemental. "
    "Never use jargon. Assume the user is in a live conversation and needs to glance "
    "at your advice quickly."
)


def get_social_advice(body_states: list[str], fer_state: str, transcript: str) -> str:
    """Send multi-modal context (vision + FER + transcript) to Claude → advice."""
    body_text = " ".join(body_states) if body_states else "No body language signals detected."
    fer_text = fer_state if fer_state else "No facial expression detected."
    user_content = (
        f"[Latest Body Language]: {body_text}\n"
        f"[Facial Expression]: {fer_text}\n"
        f'[Transcript]: "{transcript}"\n\n'
        f"What's happening and what should I do?"
    )
    msg = claude_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=150,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return msg.content[0].text


# ━━━━━━━━━━━━━ PHASE 3 · AUDIO OUTPUT ━━━━━━━━━━━━━

eleven_client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY")) if PHASE >= 3 else None


def _speak(text: str):
    """Generate TTS via ElevenLabs and play it using Pygame."""
    try:
        # 1. Generate the raw audio byte stream
        audio_generator = eleven_client.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_turbo_v2_5",
        )
        
        # 2. Collect bytes into memory
        audio_bytes = b"".join(list(audio_generator))
        
        # 3. Stream directly to speakers
        audio_stream = io.BytesIO(audio_bytes)
        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()
        
        # 4. Keep thread alive until audio finishes
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

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
ACTIVE_NODES = {0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12}


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
print(f"NeuroCue v0.3 Multi-Modal [PHASE {PHASE} of 3] — Vision + Transcription")

# ── Shared state between the main (display) thread and the API thread ──
_lock = threading.Lock()
_latest_jpg: bytes | None = None         # most recent compressed frame for the API
_latest_kps: list | None = None          # most recent keypoints from the GPU
_active_states: list[str] = []
_state_display_until = 0.0
_latest_advice = ""
_latest_fer_result = None             # most recent FER analysis dict (for drawing)
_fer_state_history: list[str] = []    # accumulates FER states between Claude calls
_last_fer_time = 0.0                  # timestamp of last FER run


def _api_worker():
    """Background thread (vision only): grabs the latest JPEG, sends it to the
    GPU for pose estimation, and updates _active_states.  Does NOT call Claude —
    the new _audio_and_llm_worker is solely responsible for that."""
    global _latest_kps, _active_states, _state_display_until

    while cap.isOpened():
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

                with _lock:
                    _latest_kps = kps

                # ── Phase 1: detect body-language states ──
                states = analyze_body_language(kps)

                if states:
                    print(f"  [State] {' | '.join(states)}")
                else:
                    print(f"  [State] Neutral — no body language signals detected")

                with _lock:
                    _active_states = states
                    _state_display_until = time.time() + 3.0 if states else 0.0
            else:
                print("No person detected.")
                with _lock:
                    _latest_kps = None

        except Exception as e:
            print(f"Network lag or server unavailable... {e}")

        time.sleep(0.03)


# ━━━━━━━━━━ AUDIO + LLM HEARTBEAT THREAD ━━━━━━━━━━

def _audio_and_llm_worker():
    """Daemon thread — the primary heartbeat for the LLM.

    Opens a sounddevice InputStream and collects WINDOW_SECONDS of audio.
    Every cycle it:
      a) Exports segment to an in-memory WAV → ElevenLabs STT → transcript
      b) Reads _active_states from the vision thread
      c) Calls Claude with both modalities (vision + transcript)
      d) Pushes advice to _latest_advice and optionally speaks it
    """
    global _latest_advice

    blocksize = int(BLOCK_SECONDS * SR)

    print(f"  [Audio] Opening microphone (SR={SR}, block={blocksize}) …")
    with sd.InputStream(channels=CHANNELS, samplerate=SR,
                        blocksize=blocksize, callback=_audio_callback):
        while cap.isOpened():
            try:
                # ── 1. Collect WINDOW_SECONDS of audio ──
                y = _collect_audio_segment(WINDOW_SECONDS)

                # ── 2. Export WAV to memory → ElevenLabs STT ──
                transcript = ""
                if PHASE >= 3 and eleven_client is not None:
                    try:
                        wav_buf = io.BytesIO()
                        sf.write(wav_buf, y, SR, format="WAV", subtype="PCM_16")
                        wav_buf.seek(0)
                        stt_resp = eleven_client.speech_to_text.convert(
                            model_id="scribe_v1",
                            file=wav_buf,
                            language_code="en",
                        )
                        transcript = stt_resp.text if hasattr(stt_resp, "text") else str(stt_resp)
                    except Exception as stt_err:
                        print(f"  [STT Error] {stt_err}")

                # ── Pretty-print transcript ──
                print()
                print("  ┌─────────────────── TRANSCRIPT ───────────────────┐")
                if transcript.strip():
                    words = transcript.strip().split()
                    line = "  │  "
                    for w in words:
                        if len(line) + len(w) + 1 > 55:
                            print(line)
                            line = "  │  " + w
                        else:
                            line = line + " " + w if line.strip() != "│" else line + w
                    if line.strip():
                        print(line)
                else:
                    print("  │  (no speech detected)")
                print("  └─────────────────────────────────────────────────────┘")
                print()

                # ── 3. Snapshot the latest vision + drain FER history ──
                with _lock:
                    current_body_states = list(_active_states)
                    # Drain all FER states accumulated since the last Claude call
                    fer_history = list(_fer_state_history)
                    _fer_state_history.clear()

                # Deduplicate while preserving order (shows emotional progression)
                seen = set()
                unique_fer = []
                for s in fer_history:
                    if s not in seen:
                        seen.add(s)
                        unique_fer.append(s)
                current_fer_state = " → ".join(unique_fer) if unique_fer else ""

                # ── 4. Call Claude (Phase 2+) ──
                if PHASE >= 2 and claude_client is not None:
                    for s in current_body_states:
                        print(f"  [Body Language → Claude] {s}")
                    if unique_fer:
                        for i, fs in enumerate(unique_fer, 1):
                            print(f"  [FER {i}/{len(unique_fer)} → Claude] {fs}")
                    else:
                        print("  [FER → Claude] (no expressions captured)")
                    print(f'  [Transcript → Claude] "{transcript}"')
                    try:
                        advice = get_social_advice(
                            current_body_states, current_fer_state, transcript
                        )
                        with _lock:
                            _latest_advice = advice
                        print(f"  [Claude Response] {advice}")

                        # ── 5. Speak (Phase 3) ──
                        if PHASE >= 3:
                            speak_async(advice)
                    except Exception as llm_err:
                        print(f"  [LLM Error] {llm_err}")

            except RuntimeError as mic_err:
                print(f"  [Audio Error] {mic_err}")
                time.sleep(1)
            except Exception as exc:
                print(f"  [Audio Worker Error] {exc}")
                time.sleep(1)


# Start both daemon threads
threading.Thread(target=_api_worker, daemon=True).start()
if PHASE >= 2:
    threading.Thread(target=_audio_and_llm_worker, daemon=True).start()

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

    # OSD: overlay the latest coach advice (word-wrapped, bottom of screen)
    if advice and PHASE >= 2:
        max_chars = 70  # chars per line at this font size
        words = advice.split()
        lines = []
        current = ""
        for w in words:
            if current and len(current) + 1 + len(w) > max_chars:
                lines.append(current)
                current = w
            else:
                current = f"{current} {w}".strip() if current else w
        if current:
            lines.append(current)
        # Draw lines bottom-up so the last line sits near the frame bottom
        for i, line in enumerate(reversed(lines)):
            y_pos = frame.shape[0] - 20 - i * 28
            cv2.putText(frame, line, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # ── FER: run facial expression recognition every FER_INTERVAL seconds ──
    if fer_engine is not None:
        now = time.time()
        if now - _last_fer_time >= FER_INTERVAL:
            _last_fer_time = now
            try:
                # Use YOLO keypoints to crop face (much more reliable than Haar)
                # Falls back to Haar cascade if no keypoints available
                if kps is not None:
                    fer_result = fer_engine.analyze_from_keypoints(frame, kps)
                else:
                    fer_result = fer_engine.analyze(frame)
                fer_state = fer_engine.get_state_string(fer_result) if fer_result else ""
                with _lock:
                    _latest_fer_result = fer_result
                    if fer_state:
                        _fer_state_history.append(fer_state)
                if fer_result:
                    emo = fer_result['emotion']
                    conf = fer_result['confidence']
                    probs = fer_result['all_probs']
                    print()
                    print("  ┌─────────────────── FER ───────────────────────┐")
                    print(f"  │  Detected: {emo.upper()} ({conf:.0%})")
                    for e, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
                        bar = "█" * int(p * 30)
                        marker = " ◀" if e == emo else ""
                        print(f"  │    {e:10s}  {p:.3f}  {bar}{marker}")
                    if fer_state:
                        print(f"  │  → {fer_state}")
                    else:
                        print("  │  → (below confidence threshold)")
                    print("  └─────────────────────────────────────────────────┘")
                else:
                    print("  [FER] No face detected.")
            except Exception as fer_err:
                print(f"  [FER Error] {fer_err}")

    # Draw FER bounding box + label if we have a recent result
    with _lock:
        fer_draw = _latest_fer_result
    if fer_engine is not None and fer_draw is not None:
        fer_engine.draw_on_frame(frame, fer_draw)

    cv2.imshow('NeuroCue Client Feed', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()