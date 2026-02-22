"""
live_plotter.py

Provides start_live_plotter(plot_queue, emotions, history_len=120, interval_ms=500)

plot_queue: a queue.Queue instance where elements are dicts:
  {
    "time": float (timestamp or window index),
    "probs": {"happy":0.1, "sad":0.2, ...},   # must include the same keys as `emotions`
    "label": "happy",
    "confidence": 0.72
  }

emotions: ordered list of emotion names (e.g. ["happy","sad","angry","neutral","fearful","surprised"])

The function blocks (shows a matplotlib window) and returns the FuncAnimation object.
"""

from queue import Empty
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

def start_live_plotter(plot_queue,
                       emotions,
                       history_len: int = 120,
                       interval_ms: int = 500,
                       title: str = "Live emotion probabilities"):
    """
    Start a live plot that reads updates from plot_queue.

    This call should be made from the main thread (matplotlib GUI mainloop).
    """

    # History containers
    times = []                       # list of floats / ints
    prob_history = {e: [] for e in emotions}
    latest_label = ""
    latest_conf = 0.0

    # Create figure + axes
    fig, (ax_timeline, ax_bar) = plt.subplots(2, 1, figsize=(11, 6), gridspec_kw={'height_ratios':[2,1]})
    fig.canvas.manager.set_window_title(title)

    # Prepare timeline lines
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

    # Prepare bars for current probs
    bars = ax_bar.bar(emotions, [0.0] * len(emotions))
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlabel("Emotion")
    ax_bar.set_ylabel("Probability")
    bar_title = ax_bar.text(0.01, 0.95, "", transform=ax_bar.transAxes, va='top')

    # Initialization function for FuncAnimation
    def init():
        for line in lines.values():
            line.set_data([], [])
        for b in bars:
            b.set_height(0.0)
        bar_title.set_text("")
        return list(lines.values()) + list(bars) + [bar_title]

    # Update function called periodically by FuncAnimation
    def update(frame):
        nonlocal latest_label, latest_conf

        # Drain plot_queue (we'll process all available messages but only show the newest)
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
            # nothing new; leave plot as is
            return list(lines.values()) + list(bars) + [bar_title]

        # newest should be dict as described; handle missing keys gracefully
        t = newest.get("time", time.time())
        probs = newest.get("probs", {e: 0.0 for e in emotions})
        latest_label = newest.get("label", "")
        latest_conf = newest.get("confidence", 0.0)

        # append to histories
        times.append(t)
        for e in emotions:
            prob_history[e].append(probs.get(e, 0.0))

        # trim history length
        if len(times) > history_len:
            excess = len(times) - history_len
            times[:] = times[excess:]
            for e in emotions:
                prob_history[e][:] = prob_history[e][excess:]

        # X axis: simple integer index for windows (0..n-1)
        xs = list(range(len(times)))
        for e, line in lines.items():
            line.set_data(xs, prob_history[e])
        ax_timeline.set_xlim(0, max(history_len, len(times)))
        # auto-scale left/back end? keep y fixed to 0..1

        # update bars with latest probs
        for b, e in zip(bars, emotions):
            height = prob_history[e][-1] if prob_history[e] else 0.0
            b.set_height(height)

        # update bar title text
        bar_title.set_text(f"Latest: {latest_label} (conf={latest_conf:.2f})")

        return list(lines.values()) + list(bars) + [bar_title]

    ani = FuncAnimation(fig, update, init_func=init, interval=interval_ms, blit=False)
    plt.tight_layout()
    plt.show(block=True)   # block True so the GUI stays open while animation runs

    return ani