# NeuroCue

> **Hack Europe 2026** — An enterprise-grade, real-time social co-pilot for neurodivergent individuals.

NeuroCue uses computer vision to read body language, translates it into psychological intent with an LLM, and whispers real-time social advice into your ear — all in under a second.

---

## How It Works

```
Webcam → GPU Pose Estimation → Body Language Geometry → Claude LLM → ElevenLabs TTS → Your Ear
```

| Layer | What it does | Where it runs |
|-------|-------------|---------------|
| **Vision Engine** | YOLOv8 Nano Pose → 17 COCO keypoints | Red Hat OpenShift (NVIDIA A10G) |
| **Network** | Cloudflare Quick Tunnel | Cloud |
| **Spatial Translator** | Geometry on keypoints → body-language states | Local laptop |
| **Social Co-Pilot** | Claude generates 1-sentence coaching advice | Anthropic API |
| **Audio Output** | ElevenLabs TTS plays advice through speakers | Local laptop |

## Architecture

```
┌──────────────┐   JPEG/POST   ┌──────────────────────────┐
│  Laptop      │ ────────────► │  Red Hat OpenShift (GPU)  │
│  client.py   │ ◄──────────── │  vision_api.py            │
│  webcam      │   JSON 17-kp  │  YOLOv8-pose on A10G     │
└──────┬───────┘               └──────────────────────────┘
       │
       ▼
  Phase 1: Spatial Translator
  (crossed arms? disengaged?)
       │
       ▼
  Phase 2: Claude API
  ("Try asking an open-ended question.")
       │
       ▼
  Phase 3: ElevenLabs TTS
  🔊 spoken through laptop speakers
```

## Detected Body-Language States

| State | Logic | Keypoints Used |
|-------|-------|----------------|
| **Crossed Arms** | Wrist-to-wrist distance < threshold | 9 (L Wrist), 10 (R Wrist) |
| **Disengaged** | Nose Y drops near/below shoulder avg Y | 0 (Nose), 5 (L Shoulder), 6 (R Shoulder) |
| *More coming…* | | |

## Quick Start

### Prerequisites

- Python 3.10+
- A webcam
- [mpv](https://mpv.io/) or ffplay on PATH (for ElevenLabs audio playback)
- API keys for [Anthropic](https://console.anthropic.com/) and [ElevenLabs](https://elevenlabs.io/)

### 1. Clone & set up

```bash
git clone https://github.com/<your-org>/NeuroCue.git
cd NeuroCue
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
| `CROSSED_ARMS_DIST` | `0.05` | Max normalised wrist-to-wrist distance to detect crossed arms |
| `DISENGAGE_THRESHOLD` | `0.08` | Min nose-above-shoulder gap; below = disengaged |
| `STATE_COOLDOWN` | `10.0` | Seconds before re-triggering the LLM for the same state |

## Tech Stack

- **Computer Vision:** YOLOv8 Nano Pose (Ultralytics)
- **GPU Compute:** NVIDIA A10G on Red Hat OpenShift AI
- **Backend API:** FastAPI + Uvicorn
- **Tunnel:** Cloudflare Quick Tunnel
- **LLM:** Anthropic Claude
- **TTS:** ElevenLabs (Turbo v2.5)
- **Client:** Python, OpenCV, Requests

## Team

Built at **Hack Europe 2026**.

## License

MIT
