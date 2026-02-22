# NeuroCue

> **Hack Europe 2026** — An enterprise-grade, real-time multi-modal social co-pilot for neurodivergent individuals.

NeuroCue combines **computer vision** (YOLOv11 pose estimation) with **live speech transcription** (ElevenLabs Scribe) to read body language *and* listen to what's being said. Every 15 seconds, it feeds both modalities into **Claude** and whispers real-time social advice into your ear via **ElevenLabs TTS** — all in near-real-time.

---

## How It Works

```
Webcam ─► GPU Pose Estimation ─► Body Language Geometry ─┐
                                                          ├─► Claude LLM ─► ElevenLabs TTS ─► Your Ear
Microphone ─► ElevenLabs Speech-to-Text ─────────────────┘
```

| Layer | What it does | Where it runs |
|-------|-------------|---------------|
| **Vision Engine** | YOLOv11 Nano Pose → 17 COCO keypoints | Red Hat OpenShift (NVIDIA A10G) |
| **Network** | Cloudflare Quick Tunnel | Cloud |
| **Spatial Translator** | Geometry on keypoints → body-language states | Local laptop |
| **Speech-to-Text** | ElevenLabs Scribe v1 → transcript of last 15 s | ElevenLabs API |
| **Social Co-Pilot** | Claude synthesises vision + transcript → coaching advice | Anthropic API |
| **Audio Output** | ElevenLabs TTS plays advice through speakers | Local laptop |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        LOCAL LAPTOP                          │
│                                                              │
│  ┌─────────────────┐          ┌──────────────────────────┐   │
│  │  Vision Thread   │  JPEG   │  Red Hat OpenShift (GPU)  │   │
│  │  (_api_worker)   │ ──────► │  vision_api.py            │   │
│  │  webcam frames   │ ◄────── │  YOLOv11-pose on A10G    │   │
│  │                  │ JSON    └──────────────────────────┘   │
│  │  Pose → body-    │                                        │
│  │  language states  │─── _active_states ──┐                 │
│  └─────────────────┘       (via _lock)     │                 │
│                                            ▼                 │
│  ┌─────────────────┐    ┌─────────────────────────────────┐  │
│  │  Audio Thread    │    │  LLM Call (every 15 s)          │  │
│  │  (_audio_and_    │───►│                                 │  │
│  │   llm_worker)    │    │  [Latest Body Language]: ...    │  │
│  │                  │    │  [Transcript]: "..."            │  │
│  │  sounddevice     │    │                                 │  │
│  │  → WAV → 11Labs  │    │  ──► Claude API ──► advice      │  │
│  │    Scribe STT    │    │  ──► ElevenLabs TTS ──► 🔊      │  │
│  └─────────────────┘    └─────────────────────────────────┘  │
│                                                              │
│  ┌─────────────────┐                                         │
│  │  Main Thread     │  OpenCV display + keypoint overlay     │
│  │  (UI loop)       │  + on-screen advice text               │
│  └─────────────────┘                                         │
└──────────────────────────────────────────────────────────────┘
```

### Thread Model

| Thread | Role | Writes to | Reads from |
|--------|------|-----------|------------|
| **Main (UI)** | Webcam capture, keypoint overlay, OSD text | `_latest_jpg` | `_latest_kps`, `_active_states`, `_latest_advice` |
| **Vision (`_api_worker`)** | Sends frames to GPU, runs body-language geometry | `_latest_kps`, `_active_states` | `_latest_jpg` |
| **Audio + LLM (`_audio_and_llm_worker`)** | Records mic → STT → calls Claude → TTS | `_latest_advice` | `_active_states` |

All shared state is protected by a single `threading.Lock`.

## Detected Body-Language States

| State | Logic | Keypoints Used |
|-------|-------|----------------|
| **Crossed Arms** | Wrists close together at chest height + torso gate | 5, 6, 7, 8, 9, 10, 11, 12 |
| **Open / Expansive Posture** | Wrist-to-wrist distance > threshold | 9, 10 |
| **Touching Face** | Wrist-to-nose distance < threshold | 0, 9, 10 |
| **Hand Raised** | Wrist above shoulder | 5, 6, 9, 10 |
| **Fidgeting** | Rapid wrist movement over 5+ frames | 9, 10 |
| **Disengaged / Looking Down** | Nose Y near shoulder avg Y | 0, 5, 6 |
| **Head Tilt** | Ear-to-ear angle > threshold | 3, 4 |
| **Nodding** | Nose drops below eye midpoint | 0, 1, 2 |
| **Looking Away** | Asymmetric ear-to-nose distance | 0, 3, 4 |
| **Shoulders Raised** | Ear-shoulder gap / shoulder width too small | 3, 4, 5, 6 |
| **Uneven Shoulders** | Shoulder angle > threshold | 5, 6 |
| **Turned Away** | Shoulder width < threshold | 5, 6 |

## Quick Start

### Prerequisites

- Python 3.10+
- A webcam and microphone
- API keys for [Anthropic](https://console.anthropic.com/) and [ElevenLabs](https://elevenlabs.io/)

### 1. Clone & set up

```bash
git clone https://github.com/pig8pig/HackEurope-NeuroSeek.git
cd HackEurope-NeuroSeek
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure environment

```bash
copy .env.example .env
```

Open `.env` and paste your API keys:

```
ANTHROPIC_API_KEY=sk-ant-...
ELEVEN_API_KEY=...
```

### 3. Run the client

```bash
python client.py
```

Press **q** to quit the webcam feed.

### Server Side (Red Hat OpenShift)

`vision_api.py` runs on the remote GPU container:

```bash
pip install fastapi uvicorn ultralytics opencv-python-headless numpy
uvicorn vision_api:app --host 0.0.0.0 --port 8000
```

Expose via Cloudflare Quick Tunnel and update `API_URL` in `client.py`.

## Configuration

Tunable constants at the top of `client.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `PHASE` | `3` | 1 = vision only, 2 = + Claude, 3 = + TTS & STT |
| `CROSSED_ARMS_DIST` | `0.12` | Max normalised wrist-to-wrist distance to detect crossed arms |
| `OPEN_ARMS_DIST` | `0.35` | Min wrist-to-wrist distance for open posture |
| `HAND_TO_FACE_DIST` | `0.07` | Max wrist-to-nose distance for face touch |
| `DISENGAGE_THRESHOLD` | `0.08` | Min nose-above-shoulder gap; below = disengaged |
| `FIDGET_THRESHOLD` | `0.06` | Min frame-to-frame wrist movement for fidget |
| `HEAD_TILT_DEGREES` | `20.0` | Min ear-to-ear angle for head tilt |
| `SHOULDER_RAISE_RATIO` | `0.25` | Ear-shoulder gap / shoulder width; below = raised |
| `SHOULDER_ASYM_DEGREES` | `18.0` | Min shoulder angle for asymmetry |
| `CLAUDE_COOLDOWN` | `15.0` | Seconds between Claude API calls (= audio window) |
| `SR` | `16000` | Microphone sampling rate (Hz) |
| `WINDOW_SECONDS` | `15` | Audio window for STT transcription |

## Tech Stack

- **Computer Vision:** YOLOv11 Nano Pose (Ultralytics)
- **GPU Compute:** NVIDIA A10G on Red Hat OpenShift AI
- **Backend API:** FastAPI + Uvicorn
- **Tunnel:** Cloudflare Quick Tunnel
- **Speech-to-Text:** ElevenLabs Scribe v1
- **LLM:** Anthropic Claude 3 Haiku
- **TTS:** ElevenLabs Turbo v2.5
- **Audio Capture:** sounddevice + soundfile
- **Client:** Python, OpenCV, Pygame, NumPy

## Team

Built at **Hack Europe 2026**.

## License

MIT
