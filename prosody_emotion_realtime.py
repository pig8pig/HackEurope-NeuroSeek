#!/usr/bin/env python3
"""
prosody_emotion_realtime.py

Capture live microphone audio and estimate emotion every 15 seconds from prosodic features.
Rule-based arousal/valence mapping -> emotion label.

Usage:
    python prosody_emotion_realtime.py

Press Ctrl+C to stop.
"""

import queue
import sys
import threading
import time
import datetime
import os
from typing import Dict, Tuple, List

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from scipy import stats

# PARAMETERS
SR = 16000                  # sampling rate
CHANNELS = 1                # mono
WINDOW_SECONDS = 15         # how often to produce an estimate
BLOCK_SECONDS = 0.5         # streaming block size (seconds) for callback
SAVE_SEGMENTS = False       # set True to write each segment to disk
OUTPUT_DIR = "recorded_segments"

# Emotion set (you can change or expand)
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "surprised"]

# Make output dir
if SAVE_SEGMENTS and not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

audio_queue = queue.Queue()  # queue of audio blocks from callback


def int16_to_float32(x: np.ndarray) -> np.ndarray:
    """Convert int16 audio to float32 in [-1, 1]."""
    if x.dtype == np.int16:
        return (x.astype(np.float32) / 32768.0)
    return x.astype(np.float32)


def audio_callback(indata, frames, time_info, status):
    """sounddevice callback: push incoming audio into queue."""
    if status:
        # print status but don't crash
        print(f"Input stream status: {status}", file=sys.stderr)
    # copy is necessary because sounddevice re-uses buffer
    audio_queue.put(indata.copy())


def collect_segment(duration_s: float) -> np.ndarray:
    """Collect 'duration_s' seconds of audio from audio_queue and return 1-D float32 array."""
    needed_frames = int(round(duration_s * SR))
    collected = np.zeros((0, CHANNELS), dtype=np.float32)
    while collected.shape[0] < needed_frames:
        try:
            block = audio_queue.get(timeout=5.0)  # block until a chunk arrives
        except queue.Empty:
            raise RuntimeError("No audio received from microphone (timeout).")
        # convert to float32 and append
        block = int16_to_float32(block) if block.dtype.kind == 'i' else block.astype(np.float32)
        collected = np.vstack((collected, block))
    collected = collected[:needed_frames, :]
    # flatten to mono 1-D
    if CHANNELS > 1:
        collected = np.mean(collected, axis=1)
    else:
        collected = collected[:, 0]
    return collected


def compute_prosodic_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Compute a set of prosodic features on audio y."""
    # Pre-emphasis (optional)
    if len(y) == 0:
        return {k: 0.0 for k in [
            "f0_median", "f0_std", "energy_rms", "energy_std", "speech_ratio",
            "onset_rate", "zcr_mean", "spectral_centroid_mean"
        ]}

    # RMS energy and its std
    hop_length = 512
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    energy_rms = float(np.mean(rms))
    energy_std = float(np.std(rms))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr_mean = float(np.mean(zcr))

    # Spectral centroid (brightness proxy)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
    spectral_centroid_mean = float(np.mean(cent))

    # Estimate voiced F0 with librosa.yin (works fairly well and is light)
    # yin may throw warnings if too short; guard length
    f0 = np.array([])
    try:
        f0 = librosa.yin(y, fmin=50, fmax=600, sr=sr, frame_length=frame_length, hop_length=hop_length)
        # librosa.yin returns np.nan for unvoiced frames
        voiced = ~np.isnan(f0)
        if np.any(voiced):
            f0_voiced = f0[voiced]
            f0_median = float(np.median(f0_voiced))
            f0_std = float(np.std(f0_voiced))
            voiced_ratio = float(np.mean(voiced))
        else:
            f0_median, f0_std, voiced_ratio = 0.0, 0.0, 0.0
    except Exception:
        # fallback: zeros if estimation fails
        f0_median, f0_std, voiced_ratio = 0.0, 0.0, 0.0

    # Simple speech-activity ratio: fraction of frames with energy above a threshold
    energy_thresh = max(1e-6, np.percentile(rms, 20))
    speech_ratio = float(np.mean(rms > energy_thresh))

    # Onset rate as rough speaking rate proxy (onsets per second)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Count onsets
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration_s = max(1e-6, len(y) / sr)
    onset_rate = float(len(onsets) / duration_s)

    features = {
        "f0_median": f0_median,
        "f0_std": f0_std,
        "voiced_ratio": voiced_ratio,
        "energy_rms": energy_rms,
        "energy_std": energy_std,
        "speech_ratio": speech_ratio,
        "onset_rate": onset_rate,
        "zcr_mean": zcr_mean,
        "spectral_centroid_mean": spectral_centroid_mean,
        "duration_s": duration_s
    }
    return features


def map_prosody_to_arousal_valence(feat: Dict[str, float]) -> Tuple[float, float]:
    """
    Heuristic mapping from prosody to (arousal, valence) in [-1, 1].
    - Arousal mostly driven by energy, onset_rate, pitch variability
    - Valence partly correlated with pitch height & spectral centroid and low pitch variability for positive valence
    This is a simple, interpretable linear-ish mapping — tune weights to your dataset.
    """
    # Extract with safe defaults
    energy = feat.get("energy_rms", 0.0)
    energy_std = feat.get("energy_std", 0.0)
    onset_rate = feat.get("onset_rate", 0.0)
    f0 = feat.get("f0_median", 0.0)
    f0_std = feat.get("f0_std", 0.0)
    voiced = feat.get("voiced_ratio", 0.0)
    spectral_centroid = feat.get("spectral_centroid_mean", 0.0)

    # Normalize things heuristically (we choose ranges that work reasonably at SR=16k)
    # Avoid divide-by-zero; add small eps
    eps = 1e-9
    # energy scale: typical rms for human speech at 16k is within [1e-4, 1e-1] depending on mic/gain
    energy_norm = np.tanh((energy - 1e-4) / (1e-3 + eps))  # roughly -1..1 mapped by tanh
    onset_norm = np.tanh(onset_rate / 3.0)  # onsets per sec: 0..6 typical -> normalize
    f0_norm = np.tanh((f0 - 150.0) / 100.0)  # median pitch center ~150 Hz
    f0var_norm = np.tanh(f0_std / (40.0 + eps))  # pitch variability
    speccent_norm = np.tanh((spectral_centroid - 2000.0) / 2000.0)  # brightness

    # Arousal: energy, onset rate, pitch variability
    arousal = 0.6 * energy_norm + 0.25 * onset_norm + 0.15 * f0var_norm
    # Valence: higher pitch and brightness sometimes correlate with positive valence; stable pitch maybe positive
    valence = 0.5 * f0_norm + 0.3 * speccent_norm - 0.2 * f0var_norm

    # Clip to [-1, 1]
    arousal = float(np.clip(arousal, -1.0, 1.0))
    valence = float(np.clip(valence, -1.0, 1.0))
    return arousal, valence


def arousal_valence_to_emotion_probs(arousal: float, valence: float) -> Dict[str, float]:
    """
    Map (arousal, valence) to a simple probability distribution over EMOTIONS.
    Uses Gaussian-like similarity to prototypical points in A/V space.
    """
    # Prototypical positions for each emotion in (arousal, valence)
    prototypes = {
        "neutral": (0.0, 0.0),
        "happy": (0.6, 0.7),
        "sad": (-0.5, -0.6),
        "angry": (0.8, -0.3),
        "fearful": (0.9, -0.5),
        "surprised": (0.9, 0.4)
    }
    # compute similarity via gaussian kernel on distance
    sigma = 0.5
    scores = {}
    total = 0.0
    for e, (a_p, v_p) in prototypes.items():
        d2 = (arousal - a_p) ** 2 + (valence - v_p) ** 2
        s = np.exp(-d2 / (2 * sigma * sigma))
        scores[e] = s
        total += s
    # normalize to make sum 1 (prob-like)
    if total <= 0:
        # fallback uniform
        probs = {e: 1.0 / len(prototypes) for e in prototypes}
    else:
        probs = {e: float(scores[e] / total) for e in prototypes}
    return probs


def pretty_print(features: Dict[str, float], arousal: float, valence: float, probs: Dict[str, float]):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_emotion = max(probs, key=probs.get)
    print("\n" + "=" * 60)
    print(f"[{ts}] Segment duration: {features.get('duration_s', 0):.1f}s | Best emotion: {best_emotion}")
    print(f" Arousal: {arousal:.3f}  Valence: {valence:.3f}")
    print(" Top emotion probs:")
    for k, v in sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:6]:
        print(f"  {k:10s}: {v:.3f}")
    print(" Key prosodic features:")
    print(f"  f0_median: {features.get('f0_median'):.1f} Hz   f0_std: {features.get('f0_std'):.1f} Hz  voiced_ratio: {features.get('voiced_ratio'):.2f}")
    print(f"  energy_rms: {features.get('energy_rms'):.6f}   energy_std: {features.get('energy_std'):.6f}   speech_ratio: {features.get('speech_ratio'):.2f}")
    print(f"  onset_rate: {features.get('onset_rate'):.2f} per s   zcr_mean: {features.get('zcr_mean'):.4f}")
    print("=" * 60 + "\n")


def save_segment_wav(y: np.ndarray, sr: int, prefix: str = "segment"):
    fname = f"{prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    sf.write(os.path.join(OUTPUT_DIR, fname), y, sr)
    return fname


def main():
    print("Starting microphone stream (SR={} Hz) ...".format(SR))
    print("Will produce an emotion estimate every {} seconds.".format(WINDOW_SECONDS))
    print("Press Ctrl+C to stop.\n")
    blocksize = int(BLOCK_SECONDS * SR)
    try:
        with sd.InputStream(channels=CHANNELS, samplerate=SR, blocksize=blocksize, callback=audio_callback):
            # Main loop: every WINDOW_SECONDS, pull audio and analyze
            while True:
                # Collect exact window
                y = collect_segment(WINDOW_SECONDS)  # returns 1-D float32
                # optional save
                if SAVE_SEGMENTS:
                    filename = save_segment_wav(y, SR)
                    print(f"Saved segment to {filename}")
                # Compute features
                feats = compute_prosodic_features(y, SR)
                # Map to arousal/valence
                arousal, valence = map_prosody_to_arousal_valence(feats)
                # Map to emotion probabilities
                probs = arousal_valence_to_emotion_probs(arousal, valence)
                # Print nicely
                pretty_print(feats, arousal, valence, probs)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()