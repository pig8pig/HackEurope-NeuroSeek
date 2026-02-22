#!/usr/bin/env python3
"""
extract_emotion_live_full.py

Single-file real-time microphone emotion feature extractor + optional live plotting.

Usage examples:
  # Quick live run (no plotting), write snapshots into live_features/
  python extract_emotion_live_full.py --analysis-length 3.0 --analysis-step 1.0 --out-folder live_features --disable-pyin

  # Full run with live plotting GUI
  python extract_emotion_live_full.py --analysis-length 3.0 --analysis-step 1.0 --plot --disable-pyin

Requirements:
  pip install numpy librosa soundfile sounddevice matplotlib
"""

import argparse
import json
import os
import time
from queue import Queue, Empty
import threading
from typing import Dict, Any, List, Tuple

import numpy as np
import sounddevice as sd
import librosa

# -------------------------
# Feature extraction / classifier (adapted from your code)
# -------------------------

def load_audio(path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    y, orig_sr = librosa.load(path, sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    return y, sr


def detect_voiced_regions(y: np.ndarray, sr: int,
                          top_db: int = 30,
                          min_duration_s: float = 0.05) -> List[Tuple[float, float]]:
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
    if voiced[0][0] > 0:
        pauses.append(voiced[0][0])
    for (s1, e1), (s2, e2) in zip(voiced, voiced[1:]):
        gap = max(0.0, s2 - e1)
        if gap > 0:
            pauses.append(gap)
    if voiced[-1][1] < total_dur:
        pauses.append(total_dur - voiced[-1][1])
    return pauses


def extract_contours(y: np.ndarray, sr: int,
                     hop_length: int = 512,
                     n_mfcc: int = 13) -> Dict[str, Any]:
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=frame_length, hop_length=hop_length)
    n_frames = rms.shape[0]
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr,
                                   hop_length=hop_length, n_fft=frame_length).tolist()

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
        f0_clean = np.nan_to_num(f0, nan=0.0)
        if len(f0_clean) != n_frames:
            f0_clean = librosa.util.fix_length(f0_clean, n_frames)
        pitch_hz = f0_clean.tolist()
    except Exception:
        pitch_hz = [0.0] * n_frames

    contours = {
        "times": times,
        "rms": rms.tolist(),
        "mfcc": mfcc.T.tolist(),
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
    rms = np.array(contours["rms"])
    pitch = np.array(contours["pitch_hz"])
    mfcc = np.array(contours["mfcc"])

    if len(contours.get("times", [])) > 0:
        total_dur = float(contours["times"][-1] - contours["times"][0])
        if total_dur <= 1e-6:
            total_dur = sum([e - s for s, e in voiced_segments]) if voiced_segments else 0.0
    else:
        total_dur = sum([e - s for s, e in voiced_segments]) if voiced_segments else 0.0

    voiced_time = sum([e - s for s, e in voiced_segments]) if len(voiced_segments) else 0.0
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


# classifier
def softmax(d):
    exps = np.exp(np.array(list(d.values())) - np.max(list(d.values())))
    probs = exps / exps.sum() if exps.sum() > 0 else np.ones_like(exps) / len(exps)
    return dict(zip(list(d.keys()), probs.tolist()))


def classify_window_rule(window_features: dict, global_stats: dict) -> dict:
    emotions = ["happy", "sad", "angry", "neutral", "fearful", "surprised"]
    g_rms = max(1e-9, global_stats.get("rms_mean", 1e-9))
    g_pitch = max(1e-9, global_stats.get("pitch_mean_hz", 1e-9))

    rms = float(window_features.get("rms_mean", 0.0))
    rms_norm = rms / g_rms
    pitch = float(window_features.get("pitch_mean", 0.0))
    pitch_norm = (pitch / g_pitch) if pitch > 0 else 0.5 * (1.0 if g_pitch > 0 else 0.5)
    pitch_std = float(window_features.get("pitch_std", 0.0))
    pitch_slope = float(window_features.get("pitch_slope_mean", 0.0))
    rms_slope = float(window_features.get("rms_slope_mean", 0.0))

    scores = {e: 0.0 for e in emotions}
    scores["neutral"] += 0.2

    if rms_norm > 1.05 and pitch_norm > 1.05:
        scores["happy"] += 0.6
    if pitch_slope > 2 and rms_slope > 0:
        scores["happy"] += 0.3

    if rms_norm > 1.3:
        scores["angry"] += 0.6
    if pitch_std > 30:
        scores["angry"] += 0.4
    if pitch_slope > 8 or rms_slope > 0.01:
        scores["angry"] += 0.3

    if rms_norm < 0.85 and (pitch_norm < 0.9 or pitch == 0):
        scores["sad"] += 0.7
    if pitch_slope < -2 and rms_slope <= 0:
        scores["sad"] += 0.4

    if pitch_norm > 1.1 and rms_norm < 1.1 and pitch_std > 25:
        scores["fearful"] += 0.5
    if pitch_slope > 5 and rms_slope < 0.02:
        scores["fearful"] += 0.3

    if pitch_slope > 15 or rms_slope > 0.02:
        scores["surprised"] += 0.8

    for k in scores:
        if scores[k] < 0:
            scores[k] = 0.0

    probs = softmax(scores)
    label = max(probs.items(), key=lambda kv: kv[1])[0]
    confidence = float(probs[label])
    return {"label": label, "confidence": round(confidence, 3), "probs": probs}


def classify_all_windows_and_smooth(windows: list, global_stats: dict, smooth_radius: int = 1):
    classified = []
    for w in windows:
        feat = w.get("features", {})
        c = classify_window_rule(feat, global_stats)
        new_w = dict(w)
        new_w["emotion"] = c
        classified.append(new_w)

    if smooth_radius > 0 and len(classified) > 0:
        labels = [c["emotion"]["label"] for c in classified]
        smoothed_labels = []
        n = len(labels)
        for i in range(n):
            lo = max(0, i - smooth_radius)
            hi = min(n - 1, i + smooth_radius)
            window_labels = labels[lo:hi+1]
            counts = {}
            for lab in window_labels:
                counts[lab] = counts.get(lab, 0) + 1
            maj_label = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            smoothed_labels.append(maj_label)
        for i, lab in enumerate(smoothed_labels):
            classified[i]["emotion"]["smoothed_label"] = lab

    return classified

# -------------------------
# Mic capture + realtime analyzer
# -------------------------

def mic_stream_to_queue(q: Queue, samplerate: int, channels: int, blocksize: int, device=None):
    def callback(indata, frames, time_info, status):
        if status:
            # Non-fatal I/O status messages
            pass
        if channels > 1:
            data = np.mean(indata, axis=1)
        else:
            data = indata[:, 0] if indata.ndim > 1 else indata
        q.put(data.copy())

    stream = sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        blocksize=blocksize,
        callback=callback,
        dtype='float32',
        device=device
    )
    stream.start()
    return stream


def realtime_analyzer(queue: Queue,
                      sr: int = 22050,
                      analysis_length_s: float = 3.0,
                      analysis_step_s: float = 1.0,
                      hop_length: int = 512,
                      n_mfcc: int = 13,
                      out_folder: str = None,
                      disable_pyin: bool = False,
                      plot_queue: Queue = None,
                      emotions: list = None):
    """
    Consume microphone blocks from queue and run analysis on a rolling buffer.
    Push compact plot updates into plot_queue if provided.
    """
    buffer = np.zeros(0, dtype=np.float32)
    last_analysis_time = 0.0
    analysis_frame_count = int(analysis_length_s * sr)
    session_results = []

    if emotions is None:
        emotions = ["happy", "sad", "angry", "neutral", "fearful", "surprised"]

    # Initialize a session-level running baseline (EMA) to compare each window against.
    # This baseline is intentionally slow-moving so short bursts change the normalized features.
    session_baseline = {
        "rms_mean": None,
        "pitch_mean_hz": None,
        # you can add more aggregated entries if desired (rms_std, pitch_std, etc.)
    }
    ema_alpha = 0.2  # smoothing factor: 0.0 (very slow) .. 1.0 (no smoothing). Tune this.
    try:
        while True:
            try:
                block = queue.get(timeout=0.5)
                buffer = np.concatenate((buffer, block))
            except Empty:
                pass

            cur_time = time.time()
            if len(buffer) >= analysis_frame_count and (cur_time - last_analysis_time) >= analysis_step_s:
                y = buffer[-analysis_frame_count:]
                y32 = y.astype(np.float32)

                # optionally monkeypatch pyin to speed up
                pyin_original = getattr(librosa, 'pyin', None)
                if disable_pyin:
                    def _pyin_dummy(*args, **kwargs):
                        # compute approximate frame count using hop_length & frame_length assumptions
                        frame_length = kwargs.get('frame_length', 1024)
                        hop = kwargs.get('hop_length', hop_length)
                        n_frames = int(np.ceil((len(y32) - frame_length + 1) / float(hop))) if len(y32) > 0 else 0
                        f0 = np.full(max(1, n_frames), np.nan)
                        return f0, np.zeros_like(f0, dtype=bool), np.zeros_like(f0, dtype=float)
                    librosa.pyin = _pyin_dummy

                contours = extract_contours(y32, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)
                total_dur = float(len(y32) / sr)
                voiced_segments = detect_voiced_regions(y32, sr, top_db=30, min_duration_s=0.05)
                pauses = compute_pauses_from_voiced(voiced_segments, total_dur)
                windows = window_functionals(contours, window_s=analysis_length_s, step_s=analysis_step_s)
                # compute global_stats for this chunk (existing)
                global_stats = compute_global_stats(contours, voiced_segments, pauses)

                # ----------------------------
                # UPDATE session-level EMA baseline
                # ----------------------------
                # On first update, initialize baseline to this window's stats
                if session_baseline["rms_mean"] is None:
                    session_baseline["rms_mean"] = global_stats.get("rms_mean", 0.0)
                else:
                    session_baseline["rms_mean"] = (ema_alpha * global_stats.get("rms_mean", 0.0)
                                                    + (1.0 - ema_alpha) * session_baseline["rms_mean"])

                if session_baseline["pitch_mean_hz"] is None:
                    session_baseline["pitch_mean_hz"] = global_stats.get("pitch_mean_hz", 0.0)
                else:
                    session_baseline["pitch_mean_hz"] = (ema_alpha * global_stats.get("pitch_mean_hz", 0.0)
                                                        + (1.0 - ema_alpha) * session_baseline["pitch_mean_hz"])

                # create a 'baseline_stats' dict matching the shape expected by the classifier
                baseline_stats = {
                    "rms_mean": float(session_baseline["rms_mean"] or 0.0),
                    "pitch_mean_hz": float(session_baseline["pitch_mean_hz"] or 0.0),
                    # keep other fields from this-chunk global_stats if classifier uses them:
                    "rms_std": global_stats.get("rms_std", 0.0),
                    "pitch_std_hz": global_stats.get("pitch_std_hz", 0.0),
                    "duration_s": global_stats.get("duration_s", 0.0),
                }

                # ----------------------------
                # CLASSIFY using the session baseline (not the local per-window global_stats)
                # ----------------------------
                classified_windows = classify_all_windows_and_smooth(windows, baseline_stats, smooth_radius=1)

                # keep the result structure using the *local* global_stats as well
                result = {
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "global": global_stats,
                    "contours": contours,
                    "voiced_segments": voiced_segments,
                    "pauses": pauses,
                    "windows": classified_windows
                }

                

                # create compact plot message
                if classified_windows:
                    latest_win = classified_windows[-1]
                    probs = latest_win["emotion"]["probs"]
                    label = latest_win["emotion"]["label"]
                    conf = latest_win["emotion"]["confidence"]
                else:
                    probs = {e: 0.0 for e in emotions}
                    label = "none"
                    conf = 0.0

                plot_msg = {
                    "time": time.time(),
                    "probs": probs,
                    "label": label,
                    "confidence": conf
                }

                if plot_queue is not None:
                    try:
                        plot_queue.put_nowait(plot_msg)
                    except:
                        pass

                session_results.append(result)

                if out_folder:
                    os.makedirs(out_folder, exist_ok=True)
                    fname = os.path.join(out_folder, f"features_{int(time.time())}.json")
                    try:
                        with open(fname, "w", encoding="utf-8") as f:
                            json.dump(result, f, indent=2)
                        print(f"[{time.strftime('%H:%M:%S')}] Wrote {fname}")
                    except Exception as ex:
                        print("Failed to write snapshot:", ex)

                g = global_stats
                print(f"[{time.strftime('%H:%M:%S')}] analyzed {g['duration_s']:.2f}s | voiced_ratio {g['voiced_ratio']:.2f} | rms {g['rms_mean']:.5f} | pitch_mean {g['pitch_mean_hz']:.1f}")

                if disable_pyin and pyin_original is not None:
                    librosa.pyin = pyin_original

                last_analysis_time = cur_time

                keep_samples = int(sr * max(analysis_length_s * 2, 10))
                if len(buffer) > keep_samples:
                    buffer = buffer[-keep_samples:]

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Realtime analyzer received KeyboardInterrupt, exiting loop.")
        return session_results


# -------------------------
# Live plotting helper (uses matplotlib)
# -------------------------
def start_live_plotter(plot_queue: Queue,
                       emotions,
                       history_len: int = 120,
                       interval_ms: int = 500,
                       title: str = "Live emotion probabilities"):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    times = []
    prob_history = {e: [] for e in emotions}
    latest_label = ""
    latest_conf = 0.0

    fig, (ax_timeline, ax_bar) = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={'height_ratios':[2,1]})
    fig.canvas.manager.set_window_title(title)

    lines = {}
    for emo in emotions:
        line, = ax_timeline.plot([], [], label=emo, linewidth=1.5)
        lines[emo] = line

    ax_timeline.set_xlim(0, history_len)
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_xlabel("Window index (time)")
    ax_timeline.set_ylabel("Probability")
    ax_timeline.set_title("Emotion probabilities over time")
    ax_timeline.legend(loc='upper right', fontsize='small')

    bars = ax_bar.bar(emotions, [0.0] * len(emotions))
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlabel("Emotion")
    ax_bar.set_ylabel("Probability")
    bar_title = ax_bar.text(0.01, 0.95, "", transform=ax_bar.transAxes, va='top')

    def init():
        for line in lines.values():
            line.set_data([], [])
        for b in bars:
            b.set_height(0.0)
        bar_title.set_text("")
        return list(lines.values()) + list(bars) + [bar_title]

    def update(frame):
        nonlocal latest_label, latest_conf
        got_any = False
        newest = None
        while True:
            try:
                item = plot_queue.get_nowait()
                newest = item
                got_any = True
            except Empty:
                break

        if not got_any:
            return list(lines.values()) + list(bars) + [bar_title]

        t = newest.get("time", time.time())
        probs = newest.get("probs", {e: 0.0 for e in emotions})
        latest_label = newest.get("label", "")
        latest_conf = newest.get("confidence", 0.0)

        times.append(t)
        for e in emotions:
            prob_history[e].append(probs.get(e, 0.0))

        if len(times) > history_len:
            excess = len(times) - history_len
            times[:] = times[excess:]
            for e in emotions:
                prob_history[e][:] = prob_history[e][excess:]

        xs = list(range(len(times)))
        for e, line in lines.items():
            line.set_data(xs, prob_history[e])
        ax_timeline.set_xlim(0, max(history_len, len(times)))

        for b, e in zip(bars, emotions):
            height = prob_history[e][-1] if prob_history[e] else 0.0
            b.set_height(height)

        bar_title.set_text(f"Latest: {latest_label} (conf={latest_conf:.2f})")
        return list(lines.values()) + list(bars) + [bar_title]

    ani = FuncAnimation(fig, update, init_func=init, interval=interval_ms, blit=False)
    plt.tight_layout()
    plt.show(block=True)
    return ani


# -------------------------
# Main: glue everything together
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Live microphone emotion feature extractor (with optional plotting)")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--mfcc", type=int, default=13)
    parser.add_argument("--analysis-length", type=float, default=3.0, help="seconds of audio per analysis")
    parser.add_argument("--analysis-step", type=float, default=1.0, help="seconds between analyses")
    parser.add_argument("--blocksize", type=int, default=2048, help="sounddevice blocksize (samples)")
    parser.add_argument("--out-folder", type=str, default=None, help="where to write JSON snapshots (optional)")
    parser.add_argument("--out-session", type=str, default=None, help="write aggregated session JSON on stop")
    parser.add_argument("--device", type=str, default=None, help="sounddevice input device (optional: id or name)")
    parser.add_argument("--disable-pyin", action="store_true", help="disable librosa.pyin (speedup; pitch will be zeros)")
    parser.add_argument("--plot", action="store_true", help="show live plotting GUI")
    args = parser.parse_args()

    emotions = ["happy", "sad", "angry", "neutral", "fearful", "surprised"]

    audio_q = Queue()
    plot_q = Queue(maxsize=8) if args.plot else None

    sr = args.sr
    channels = 1
    blocksize = args.blocksize

    try:
        print(f"Starting microphone input at {sr} Hz, blocksize={blocksize}...")
        stream = mic_stream_to_queue(audio_q, samplerate=sr, channels=channels, blocksize=blocksize, device=args.device)
    except Exception as ex:
        print("Failed to open audio input stream. Try listing devices with:")
        print('  python -c "import sounddevice as sd; print(sd.query_devices())"')
        print("Error:", ex)
        return

    analyzer_kwargs = {
        "queue": audio_q,
        "sr": sr,
        "analysis_length_s": args.analysis_length,
        "analysis_step_s": args.analysis_step,
        "hop_length": args.hop,
        "n_mfcc": args.mfcc,
        "out_folder": args.out_folder,
        "disable_pyin": args.disable_pyin,
        "plot_queue": plot_q,
        "emotions": emotions
    }

    analyzer_thread = threading.Thread(target=realtime_analyzer, kwargs=analyzer_kwargs, daemon=True)
    analyzer_thread.start()

    try:
        if args.plot:
            # run plotter in main thread (required by many GUI backends)
            start_live_plotter(plot_q, emotions, history_len=120, interval_ms=500)
        else:
            print("Running without plotting. Press Ctrl+C to stop.")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("Main received KeyboardInterrupt, shutting down...")
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        # Gather analyzer results if needed: not blocking (daemon thread)
        # Optionally write session results if analyzer returned them (not implemented here)
        print("Exited. Goodbye.")


if __name__ == "__main__":
    main()