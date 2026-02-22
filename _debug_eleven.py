"""Quick debug script: checks ElevenLabs key, STT, and TTS."""
import os, io, sys
import numpy as np
from dotenv import load_dotenv
load_dotenv()

key = os.getenv("ELEVEN_API_KEY")
print(f"1. ELEVEN_API_KEY loaded: {bool(key)}")
print(f"   Length: {len(key) if key else 0}")
if key:
    print(f"   Starts with: {key[:8]}...")
else:
    print("   ERROR: No key found in .env — ElevenLabs will not work.")
    sys.exit(1)

from elevenlabs.client import ElevenLabs
client = ElevenLabs(api_key=key)
print("2. ElevenLabs client created OK.")

# Test STT with a short silent WAV
import soundfile as sf
print("3. Testing Speech-to-Text (sending 1s of silence)...")
try:
    silence = np.zeros(16000, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, silence, 16000, format="WAV", subtype="PCM_16")
    buf.seek(0)
    resp = client.speech_to_text.convert(model_id="scribe_v1", file=buf)
    print(f"   STT response type: {type(resp).__name__}")
    text = resp.text if hasattr(resp, "text") else str(resp)
    print(f"   STT text: '{text}'")
    print("   STT: OK")
except Exception as e:
    print(f"   STT ERROR: {e}")

# Test TTS
print("4. Testing Text-to-Speech...")
try:
    gen = client.text_to_speech.convert(
        text="Hello",
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_turbo_v2_5",
    )
    audio_bytes = b"".join(list(gen))
    print(f"   TTS returned {len(audio_bytes)} bytes of audio.")
    if len(audio_bytes) > 0:
        print("   TTS: OK")
    else:
        print("   TTS ERROR: 0 bytes returned.")
except Exception as e:
    print(f"   TTS ERROR: {e}")

print("\nDone.")
