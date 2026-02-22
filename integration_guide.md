# Integrating Facial Expression Recognition into NeuroSeek

## Files to Add

Place these two files in your repo root (`HackEurope-NeuroSeek/`):

| File                  | What it is                                 | How to get it                      |
| --------------------- | ------------------------------------------ | ---------------------------------- |
| `fer_inference.py`  | Inference module (face detection + FERNet) | Already provided                   |
| `fer_model_best.pt` | Trained model weights (~11MB)              | Download from Colab after training |

## Changes to `client.py`

There are **5 edits** to make. Each one is small.

---

### Change 1 — Import

At the top of `client.py`, near the other imports:

```python
from fer_inference import FERInference
```

---

### Change 2 — Config Constants

Near your other constants (`CLAUDE_COOLDOWN`, `CROSSED_ARMS_DIST`, etc.):

```python
FER_MODEL_PATH = "fer_model_best.pt"       # trained model file
FER_INTERVAL   = 10.0                       # seconds between FER snapshots
```

---

### Change 3 — Initialisation

Near where you create `claude_client`, `eleven_client`, etc.:

```python
import os

fer_engine = None
if os.path.exists(FER_MODEL_PATH):
    fer_engine = FERInference(FER_MODEL_PATH)
else:
    print(f"[FER] Model not found at {FER_MODEL_PATH} — facial expression recognition disabled.")
```

Also add these shared-state variables near `_active_states`, `_latest_advice`, etc.:

```python
_latest_fer_result = None        # latest FER detection result dict
_latest_fer_state  = ""          # latest FER state string for Claude
_last_fer_time     = 0.0         # timestamp of last FER analysis
```

---

### Change 4 — FER Snapshot in the Main Loop

In your main `while True` loop, **before** the `cv2.imshow()` call, add:

```python
# ── Facial Expression Recognition (every FER_INTERVAL seconds) ──
if fer_engine is not None:
    now = time.time()
    if now - _last_fer_time >= FER_INTERVAL:
        _last_fer_time = now

        # Analyze the current frame
        fer_result = fer_engine.analyze(frame)

        with _lock:
            _latest_fer_result = fer_result

            # Convert to a state string
            fer_state = fer_engine.get_state_string(fer_result)
            _latest_fer_state = fer_state or ""

            if fer_state:
                print(f"  [FER] {fer_result['emotion'].upper()} "
                      f"({fer_result['confidence']:.0%}) → {fer_state}")

    # Draw face box + emotion label on every frame (uses cached result)
    with _lock:
        if _latest_fer_result is not None:
            fer_engine.draw_on_frame(frame, _latest_fer_result)
```

---

### Change 5 — Feed FER to Claude

Find where you build the prompt for Claude in `_audio_and_llm_worker`. Currently it combines `_active_states` and the transcript. Update it to also include the facial expression:

```python
# Grab all signals under the lock
with _lock:
    body_states = list(_active_states)
    fer_state = _latest_fer_state

# Build text for each signal
body_language_text = "; ".join(body_states) if body_states else "No notable body language detected."
facial_expression_text = fer_state if fer_state else "No clear facial expression detected."

# Updated prompt — add the [Facial Expression] line
content = (
    f"[Latest Body Language]: {body_language_text}\n"
    f"[Facial Expression]: {facial_expression_text}\n"
    f"[Transcript]: \"{transcript}\"\n"
    f"What should I do?"
)
```

---

## Optional — Improve Claude's System Prompt

Add this to your existing `SYSTEM_PROMPT` so Claude understands the new signal:

```
You receive three types of signals: body language from pose estimation,
facial expressions from emotion recognition, and a transcript of what
was said. When body language and facial expression conflict (e.g. open
posture but fearful face), note the discrepancy — the mismatch itself
is informative. Facial expressions are harder to fake than body posture.
```

---

## How It Works

Every 10 seconds, the main thread:

1. Grabs the current webcam frame
2. Runs OpenCV Haar Cascade to detect the largest face
3. Crops, converts to 96x96 grayscale, normalises
4. Runs through FERNet (~15ms on CPU)
5. Stores the result as a state string (e.g. "The person looks anxious or fearful")
6. Draws a bounding box + emotion label on the video feed

Every 15 seconds (your existing `CLAUDE_COOLDOWN`), the audio+LLM thread picks up the latest facial expression state alongside body language and transcript, and sends all three to Claude.

```
Webcam ─► GPU Pose Estimation ─► Body Language ─────────────┐
                                                             │
Webcam ─► Haar Face Detect ─► FERNet ─► Facial Expression ──├─► Claude ─► TTS
                                                             │
Microphone ─► ElevenLabs STT ─► Transcript ─────────────────┘
```

---

## Detected Facial Expressions

| Emotion  | State String                                                | When It Triggers |
| -------- | ----------------------------------------------------------- | ---------------- |
| Happy    | "The person is smiling — they appear happy and engaged."   | confidence > 60% |
| Angry    | "The person's face shows anger or frustration."             | confidence > 35% |
| Sad      | "The person looks sad or dejected."                         | confidence > 35% |
| Fear     | "The person looks anxious or fearful..."                    | confidence > 35% |
| Surprise | "The person looks surprised."                               | confidence > 35% |
| Disgust  | "The person's expression shows displeasure or disapproval." | confidence > 35% |
| Neutral  | *(not reported)*                                          | filtered out     |

**Compound expressions** detected from probability distributions:

| Pattern                             | State String                                                                |
| ----------------------------------- | --------------------------------------------------------------------------- |
| Surprise + Fear both elevated       | "The person looks confused — consider explaining more clearly."            |
| Low-confidence happy + high neutral | "The person is giving a polite smile — they may not be genuinely engaged." |
| Happy + fear/sad undertones         | "The person is smiling but may be masking discomfort."                      |
