#!/usr/bin/env python3
"""
extract_emotion_features_no_parselmouth.py

Feature extractor for time-varying acoustic features useful for emotion estimation.
This version DOES NOT use parselmouth / Praat.

Outputs JSON with:
 - global: overall stats
 - contours: time-series (times, pitch_hz, rms, mfcc)
 - voiced_segments: list of (start_s, end_s)
 - pauses: list of pause durations
 - windows: sliding-window functionals

Dependencies:
  pip install librosa numpy soundfile

Usage:
  python extract_emotion_features_no_parselmouth.py path/to/audio.wav --out features.json
"""

import argparse
import json
import math
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import librosa
import soundfile as sf


def load_audio(path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    y, orig_sr = librosa.load(path, sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y, sr


def detect_voiced_regions(y: np.ndarray, sr: int,
                          top_db: int = 30,
                          min_duration_s: float = 0.05) -> List[Tuple[float, float]]:
    """
    Use librosa.effects.split to find non-silent (voiced) regions.
    Returns list of (start_s, end_s).
    """
    intervals = librosa.effects.split(y, top_db=top_db)
    voiced = []
    for start, end in intervals:
        dur = (end - start) / sr
        if dur >= min_duration_s:
            voiced.append((start / sr, end / sr))
    return voiced


def compute_pauses_from_voiced(voiced: List[Tuple[float, float]], total_dur: float) -> List[float]:
    pauses = []
    if len(voiced) == 0:
        return [total_dur]
    # leading
    if voiced[0][0] > 0:
        pauses.append(voiced[0][0])
    # between
    for (s1, e1), (s2, e2) in zip(voiced, voiced[1:]):
        gap = max(0.0, s2 - e1)
        if gap > 0:
            pauses.append(gap)
    # trailing
    if voiced[-1][1] < total_dur:
        pauses.append(total_dur - voiced[-1][1])
    return pauses


def extract_contours(y: np.ndarray, sr: int,
                     hop_length: int = 512,
                     n_mfcc: int = 13) -> Dict[str, Any]:
    """
    Extract contours:
      - times (frame centers)
      - rms per frame
      - mfcc per frame
      - pitch per frame via librosa.pyin (NaN for unvoiced -> converted to 0.0)
    """
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=frame_length, hop_length=hop_length)
    n_frames = rms.shape[0]
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr,
                                   hop_length=hop_length, n_fft=frame_length).tolist()

    # Pitch via librosa.pyin. If it fails, produce zeros.
    pitch_hz = None
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        # pyin returns array length ~= n_frames; fill NaNs with 0.0 (unvoiced)
        f0_clean = np.nan_to_num(f0, nan=0.0)
        if len(f0_clean) != n_frames:
            f0_clean = librosa.util.fix_length(f0_clean, n_frames)
        pitch_hz = f0_clean.tolist()
    except Exception:
        # fallback: simple autocorrelation-based estimate per frame (slow & crude) or zeros
        # We'll produce zeros to keep it simple and reliable across environments
        pitch_hz = [0.0] * n_frames

    contours = {
        "times": times,
        "rms": rms.tolist(),
        "mfcc": mfcc.T.tolist(),  # shape (n_frames, n_mfcc)
        "pitch_hz": pitch_hz
    }
    return contours


def window_functionals(contours: Dict[str, Any],
                       window_s: float = 2.0,
                       step_s: float = 0.5) -> List[Dict[str, Any]]:
    times = np.array(contours["times"])
    rms = np.array(contours["rms"])
    pitch = np.array(contours["pitch_hz"])
    mfcc = np.array(contours["mfcc"])
    n_mfcc = mfcc.shape[1] if mfcc.ndim == 2 else 0

    if len(times) == 0:
        return []

    t0 = float(times[0])
    t_end = float(times[-1])
    windows = []
    start = t0
    while start < t_end:
        end = start + window_s
        center = (start + end) / 2.0
        idx = np.where((times >= start) & (times < end))[0]
        feat = {}
        if len(idx) > 0:
            pitch_sel = pitch[idx]
            pitch_voiced = pitch_sel[pitch_sel > 0]
            feat["pitch_mean"] = float(np.mean(pitch_voiced)) if pitch_voiced.size > 0 else 0.0
            feat["pitch_std"] = float(np.std(pitch_voiced)) if pitch_voiced.size > 0 else 0.0
            feat["rms_mean"] = float(np.mean(rms[idx]))
            feat["rms_std"] = float(np.std(rms[idx]))
            for k in range(min(6, n_mfcc)):
                feat[f"mfcc{k+1}_mean"] = float(np.mean(mfcc[idx, k]))
                feat[f"mfcc{k+1}_std"] = float(np.std(mfcc[idx, k]))
            # simple slopes
            if len(idx) >= 2:
                dt = np.diff(times[idx])
                if dt.size > 0:
                    pitch_diff = np.diff(pitch_sel) / np.maximum(dt, 1e-6)
                    rms_diff = np.diff(rms[idx]) / np.maximum(dt, 1e-6)
                    pitch_diff_voiced = pitch_diff[pitch_sel[1:] > 0]
                    feat["pitch_slope_mean"] = float(np.mean(pitch_diff_voiced)) if pitch_diff_voiced.size > 0 else 0.0
                    feat["rms_slope_mean"] = float(np.mean(rms_diff))
                else:
                    feat["pitch_slope_mean"] = 0.0
                    feat["rms_slope_mean"] = 0.0
            else:
                feat["pitch_slope_mean"] = 0.0
                feat["rms_slope_mean"] = 0.0
        else:
            feat["pitch_mean"] = 0.0
            feat["pitch_std"] = 0.0
            feat["rms_mean"] = 0.0
            feat["rms_std"] = 0.0
            for k in range(6):
                feat[f"mfcc{k+1}_mean"] = 0.0
                feat[f"mfcc{k+1}_std"] = 0.0
            feat["pitch_slope_mean"] = 0.0
            feat["rms_slope_mean"] = 0.0

        windows.append({
            "center_s": center,
            "start_s": start,
            "end_s": end,
            "features": feat
        })
        start += step_s
    return windows


def compute_global_stats(contours: Dict[str, Any], voiced_segments: List[Tuple[float, float]], pauses: List[float]) -> Dict[str, Any]:
    """
    Robust global stats. Clips voiced_ratio to [0,1] to avoid tiny rounding artifacts.
    """
    rms = np.array(contours["rms"])
    pitch = np.array(contours["pitch_hz"])
    mfcc = np.array(contours["mfcc"])

    # prefer a robust total duration: use audio frame times if available
    if len(contours.get("times", [])) > 0:
        total_dur = float(contours["times"][-1] - contours["times"][0])
        # if total_dur is tiny (or zero) fall back to voiced segments or 0.0
        if total_dur <= 1e-6:
            total_dur = sum([e - s for s, e in voiced_segments]) if voiced_segments else 0.0
    else:
        total_dur = sum([e - s for s, e in voiced_segments]) if voiced_segments else 0.0

    voiced_time = sum([e - s for s, e in voiced_segments]) if len(voiced_segments) else 0.0
    # clip to 0..1 to avoid >1.0 due to rounding
    voiced_ratio = (voiced_time / total_dur) if total_dur > 0 else 0.0
    voiced_ratio = max(0.0, min(1.0, voiced_ratio))

    pitch_voiced = pitch[pitch > 0]
    global_stats = {
        "duration_s": float(total_dur),
        "voiced_time_s": float(voiced_time),
        "voiced_ratio": float(voiced_ratio),
        "total_pause_time_s": float(sum(pauses)),
        "pause_count": int(len(pauses)),
        "rms_mean": float(np.mean(rms)) if rms.size > 0 else 0.0,
        "rms_std": float(np.std(rms)) if rms.size > 0 else 0.0,
        "pitch_mean_hz": float(np.mean(pitch_voiced)) if pitch_voiced.size > 0 else 0.0,
        "pitch_std_hz": float(np.std(pitch_voiced)) if pitch_voiced.size > 0 else 0.0,
        "mfcc1_mean": float(np.mean(mfcc[:, 0])) if (mfcc.ndim == 2 and mfcc.shape[0] > 0) else 0.0
    }
    return global_stats


def extract_features_from_file(path: str,
                               sr: int = 22050,
                               hop_length: int = 512,
                               n_mfcc: int = 13,
                               window_s: float = 2.0,
                               step_s: float = 0.5) -> Dict[str, Any]:
    y, sr = load_audio(path, sr=sr)
    total_dur = float(len(y) / sr)
    voiced_segments = detect_voiced_regions(y, sr, top_db=30, min_duration_s=0.05)
    pauses = compute_pauses_from_voiced(voiced_segments, total_dur)
    contours = extract_contours(y, sr, hop_length=hop_length, n_mfcc=n_mfcc)
    windows = window_functionals(contours, window_s=window_s, step_s=step_s)
    global_stats = compute_global_stats(contours, voiced_segments, pauses)

    result = {
        "file": os.path.abspath(path),
        "global": global_stats,
        "contours": contours,
        "voiced_segments": voiced_segments,
        "pauses": pauses,
        "windows": windows
    }
    return result

import math
import numpy as np
from statistics import median

# -------------------------
# Simple rule-based window classifier
# -------------------------
def softmax(d):
    exps = np.exp(np.array(list(d.values())) - np.max(list(d.values())))
    probs = exps / exps.sum() if exps.sum() > 0 else np.ones_like(exps) / len(exps)
    return dict(zip(list(d.keys()), probs.tolist()))

def classify_window_rule(window_features: dict, global_stats: dict) -> dict:
    """
    Return a dict: {'label': str, 'confidence': float, 'probs': {emotion:prob,...}}
    Heuristics use:
      - rms_mean (relative to global rms_mean)
      - pitch_mean (relative to global pitch_mean)
      - pitch_std
      - pitch_slope_mean and rms_slope_mean (dynamics)
    """
    # emotions considered
    emotions = ["happy", "sad", "angry", "neutral", "fearful", "surprised"]

    # prepare normalized features (guard against zero globals)
    g_rms = max(1e-9, global_stats.get("rms_mean", 1e-9))
    g_pitch = max(1e-9, global_stats.get("pitch_mean_hz", 1e-9))

    rms = float(window_features.get("rms_mean", 0.0))
    rms_norm = rms / g_rms
    pitch = float(window_features.get("pitch_mean", 0.0))
    # if pitch==0 (unvoiced) set a small baseline
    pitch_norm = (pitch / g_pitch) if pitch > 0 else 0.5 * (1.0 if g_pitch > 0 else 0.5)
    pitch_std = float(window_features.get("pitch_std", 0.0))
    pitch_slope = float(window_features.get("pitch_slope_mean", 0.0))
    rms_slope = float(window_features.get("rms_slope_mean", 0.0))

    # initialize scores
    scores = {e: 0.0 for e in emotions}

    # Neutral baseline
    scores["neutral"] += 0.2

    # Happy: moderately higher energy, slightly higher pitch, positive pitch slope (rising)
    if rms_norm > 1.05 and pitch_norm > 1.05:
        scores["happy"] += 0.6
    if pitch_slope > 2 and rms_slope > 0:
        scores["happy"] += 0.3

    # Angry / high arousal: high energy and high pitch variability or strong positive slopes
    if rms_norm > 1.3:
        scores["angry"] += 0.6
    if pitch_std > 30:
        scores["angry"] += 0.4
    if pitch_slope > 8 or rms_slope > 0.01:
        scores["angry"] += 0.3

    # Sad: low energy, low pitch, negative slopes, low variability
    if rms_norm < 0.85 and (pitch_norm < 0.9 or pitch == 0):
        scores["sad"] += 0.7
    if pitch_slope < -2 and rms_slope <= 0:
        scores["sad"] += 0.4

    # Fearful: higher pitch but low-to-moderate energy, high pitch variability, sometimes faster changes
    if pitch_norm > 1.1 and rms_norm < 1.1 and pitch_std > 25:
        scores["fearful"] += 0.5
    if pitch_slope > 5 and rms_slope < 0.02:
        scores["fearful"] += 0.3

    # Surprised: sudden spike in pitch and/or energy (high positive slope) and short-lived
    if pitch_slope > 15 or rms_slope > 0.02:
        scores["surprised"] += 0.8

    # Small smoothing: prefer more neutral unless other evidence strong
    # (already have neutral baseline; we'll leave this as-is)

    # Ensure non-negativity
    for k in scores:
        if scores[k] < 0:
            scores[k] = 0.0

    # Convert to probabilities via softmax-like normalization
    probs = softmax(scores)

    # choose top emotion and its confidence
    label = max(probs.items(), key=lambda kv: kv[1])[0]
    confidence = float(probs[label])

    return {"label": label, "confidence": round(confidence, 3), "probs": probs}


def classify_all_windows_and_smooth(windows: list, global_stats: dict, smooth_radius: int = 1):
    """
    Classify each window, then smooth using majority vote over (2*smooth_radius+1) window.
    Returns list of dicts with original window keys plus 'emotion' entry.
    """
    classified = []
    for w in windows:
        feat = w.get("features", {})
        c = classify_window_rule(feat, global_stats)
        new_w = dict(w)
        new_w["emotion"] = c
        classified.append(new_w)

    # optional smoothing: majority vote in neighbor window radius
    if smooth_radius > 0 and len(classified) > 0:
        labels = [c["emotion"]["label"] for c in classified]
        smoothed_labels = []
        n = len(labels)
        for i in range(n):
            lo = max(0, i - smooth_radius)
            hi = min(n - 1, i + smooth_radius)
            window_labels = labels[lo:hi+1]
            # majority label (tie -> original label)
            counts = {}
            for lab in window_labels:
                counts[lab] = counts.get(lab, 0) + 1
            # pick label with max count; break ties by original label preference
            maj_label = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            smoothed_labels.append(maj_label)
        # attach smoothed label and keep original confidence/probs
        for i, lab in enumerate(smoothed_labels):
            classified[i]["emotion"]["smoothed_label"] = lab

    return classified

def main():
    parser = argparse.ArgumentParser(description="Extract features for emotion estimation (no parselmouth).")
    parser.add_argument("audio", help="Path to audio file (wav, mp3, etc.)")
    parser.add_argument("--sr", type=int, default=22050, help="Target sampling rate")
    parser.add_argument("--hop", type=int, default=512, help="Hop length (samples) for frame features")
    parser.add_argument("--mfcc", type=int, default=13, help="Number of MFCCs")
    parser.add_argument("--window", type=float, default=2.0, help="Window size (seconds) for functionals")
    parser.add_argument("--step", type=float, default=0.5, help="Step (seconds) between windows")
    parser.add_argument("--out", type=str, default="features.json", help="Output JSON filename")
    args = parser.parse_args()

    print(f"Loading and extracting from: {args.audio}")
    feats = extract_features_from_file(args.audio, sr=args.sr, hop_length=args.hop,
                                       n_mfcc=args.mfcc, window_s=args.window, step_s=args.step)



    classified_windows = classify_all_windows_and_smooth(
        feats["windows"],
        feats["global"],
        smooth_radius=1
    )

    feats["windows"] = classified_windows

    out_path = "features_with_emotion.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feats, f, indent=2)

    print(f"Wrote classified features to {out_path}")





    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(feats, f, indent=2)
    print(f"Wrote features to {args.out}")
    print("Summary:")
    g = feats["global"]
    print(f"  duration {g['duration_s']:.2f}s, voiced_ratio {g['voiced_ratio']:.2f}, pauses {g['pause_count']} (total {g['total_pause_time_s']:.2f}s)")
    print(f"  rms_mean {g['rms_mean']:.5f}, pitch_mean {g['pitch_mean_hz']:.1f} Hz (voiced frames)")


if __name__ == "__main__":
    main()

