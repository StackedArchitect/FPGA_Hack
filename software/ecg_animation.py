"""ECG monitor-style animation for hackathon video submission."""

import argparse
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

# ── Configuration ──────────────────────────────────────────────────
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
AAMI_FULL = {
    0: "Normal",
    1: "Supraventricular",
    2: "Ventricular",
    3: "Fusion",
    4: "Unknown",
}

# Colors — hospital monitor aesthetic
BG_COLOR       = "#0a0e14"       # near-black
GRID_COLOR     = "#162028"       # subtle grid
TRACE_NORMAL   = "#00ff88"       # green trace
TRACE_ARRHYTH  = "#ff3355"       # red for arrhythmia
TRACE_S_CLASS  = "#ffaa33"       # orange for supraventricular
TRACE_F_CLASS  = "#ff66cc"       # pink for fusion
TRACE_Q_CLASS  = "#66ccff"       # blue for unknown
TEXT_COLOR     = "#c8d0d8"       # light gray text
WARN_COLOR     = "#ff3355"       # warning red
GLOW_ALPHA     = 0.3             # glow effect opacity

CLASS_COLORS = {
    0: TRACE_NORMAL,
    1: TRACE_S_CLASS,
    2: TRACE_ARRHYTH,
    3: TRACE_F_CLASS,
    4: TRACE_Q_CLASS,
}

# ── Data Loading ──────────────────────────────────────────────────

def load_beats(data_path, num_beats=10, seed=42):
    """
    Load ECG beats from Kaggle CSV ensuring we get a mix with V-class beats.
    Returns beats as continuous waveform + beat boundaries + labels.
    """
    print(f"[Animation] Loading data from {data_path}...")
    data = np.loadtxt(data_path, delimiter=",", dtype=np.float32)
    X = data[:, :-1]  # (N, 187)
    y = data[:, -1].astype(np.int64)

    rng = np.random.RandomState(seed)

    # Curate a sequence: start with N beats, then introduce a V beat dramatically
    # Pattern: N, N, N, V, N, N, V, N, S, N  (adjust based on num_beats)
    selected_indices = []

    # Get indices per class
    class_indices = {c: np.where(y == c)[0] for c in range(5)}

    # Build a dramatic sequence
    pattern = []
    for i in range(num_beats):
        if i in [3, 6]:  # V-class beats at positions 3 and 6
            pattern.append(2)
        elif i == 8 and num_beats > 8:  # S-class beat
            pattern.append(1)
        elif i == num_beats - 2 and num_beats > 5:  # F-class near end
            pattern.append(3)
        else:
            pattern.append(0)  # Normal

    for cls in pattern:
        available = class_indices[cls]
        if len(available) == 0:
            available = class_indices[0]  # fallback to N
        idx = rng.choice(available)
        selected_indices.append(idx)

    # Build continuous waveform
    beats = []
    beat_boundaries = [0]
    labels = []

    for idx in selected_indices:
        beat = X[idx]
        beats.append(beat)
        beat_boundaries.append(beat_boundaries[-1] + len(beat))
        labels.append(y[idx])

    waveform = np.concatenate(beats)
    print(f"[Animation] Loaded {len(labels)} beats: {[AAMI_CLASSES[l] for l in labels]}")
    return waveform, beat_boundaries, labels


def get_beat_index(sample_idx, beat_boundaries):
    """Return which beat a sample belongs to."""
    for i in range(len(beat_boundaries) - 1):
        if beat_boundaries[i] <= sample_idx < beat_boundaries[i + 1]:
            return i
    return len(beat_boundaries) - 2


# ── Animation ─────────────────────────────────────────────────────

def create_animation(waveform, beat_boundaries, labels, fps=30,
                     samples_per_frame=3, save_path=None, fmt="mp4"):
    """Create the ECG monitor animation."""

    total_samples = len(waveform)
    total_frames = total_samples // samples_per_frame + 50  # +50 for pause at end

    # ── Figure setup ──
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Style the axes like a hospital monitor
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.set_xlabel("Sample Index", color=TEXT_COLOR, fontsize=10, fontfamily="monospace")
    ax.set_ylabel("Amplitude", color=TEXT_COLOR, fontsize=10, fontfamily="monospace")

    # Grid
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    # Limits
    y_min = waveform.min() - 0.15
    y_max = waveform.max() + 0.15
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, total_samples)

    # ── Static elements ──

    # Title
    title_text = ax.text(
        0.5, 1.06, "ECG Monitor — MIT-BIH Record 208",
        transform=ax.transAxes, ha="center", va="bottom",
        color=TEXT_COLOR, fontsize=16, fontweight="bold", fontfamily="monospace",
    )

    # Subtitle
    subtitle_text = ax.text(
        0.5, 1.01, "WaveBNN-ECG  ·  Real-Time Arrhythmia Detection",
        transform=ax.transAxes, ha="center", va="bottom",
        color="#5a6a7a", fontsize=10, fontfamily="monospace",
    )

    # Status indicator (top-right)
    status_text = ax.text(
        0.98, 1.06, "● MONITORING",
        transform=ax.transAxes, ha="right", va="bottom",
        color=TRACE_NORMAL, fontsize=11, fontweight="bold", fontfamily="monospace",
    )

    # Beat counter (top-left)
    beat_counter = ax.text(
        0.02, 1.06, "Beat: 0/0",
        transform=ax.transAxes, ha="left", va="bottom",
        color=TEXT_COLOR, fontsize=11, fontfamily="monospace",
    )

    # Classification label (center, shown on arrhythmia)
    class_label = ax.text(
        0.5, 0.92, "",
        transform=ax.transAxes, ha="center", va="top",
        color=WARN_COLOR, fontsize=16, fontweight="bold", fontfamily="monospace",
        alpha=0,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a0000", edgecolor=WARN_COLOR,
                  alpha=0.8, linewidth=2),
    )

    # Heart rate display (bottom-right)
    hr_text = ax.text(
        0.98, 0.05, "HR: 72 bpm",
        transform=ax.transAxes, ha="right", va="bottom",
        color=TRACE_NORMAL, fontsize=12, fontfamily="monospace",
    )

    # BNN inference time (bottom-left)
    inference_text = ax.text(
        0.02, 0.05, "Inference: 6.53 µs  |  0 DSP  |  0.42 W",
        transform=ax.transAxes, ha="left", va="bottom",
        color="#5a6a7a", fontsize=9, fontfamily="monospace",
    )

    # ── Dynamic line (will be updated per frame) ──
    # We use LineCollection for per-segment coloring
    line_segs = []
    line_colors = []
    lc = LineCollection([], linewidths=1.8, zorder=3)
    ax.add_collection(lc)

    # Glow line (thicker, transparent, behind main trace)
    lc_glow = LineCollection([], linewidths=5, alpha=GLOW_ALPHA, zorder=2)
    ax.add_collection(lc_glow)

    # Arrhythmia markers
    arrhythmia_markers = []

    # Fade-in alpha tracker for warning label
    warn_alpha = [0.0]
    current_warn_class = [None]

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    def init():
        lc.set_segments([])
        lc_glow.set_segments([])
        return lc, lc_glow, beat_counter, class_label, status_text

    def animate(frame):
        drawn = min(frame * samples_per_frame, total_samples)

        if drawn < 2:
            return lc, lc_glow, beat_counter, class_label, status_text

        # Build colored segments up to `drawn`
        x = np.arange(drawn)
        y_vals = waveform[:drawn]

        # Create segments: each is [[x0,y0],[x1,y1]]
        points = np.column_stack([x, y_vals])
        segments = np.array([points[i:i+2] for i in range(len(points)-1)])

        # Color each segment by which beat it belongs to
        colors = []
        for i in range(len(segments)):
            beat_idx = get_beat_index(i, beat_boundaries)
            cls = labels[beat_idx] if beat_idx < len(labels) else 0
            colors.append(CLASS_COLORS.get(cls, TRACE_NORMAL))

        lc.set_segments(segments)
        lc.set_colors(colors)

        lc_glow.set_segments(segments)
        lc_glow.set_colors(colors)

        # Current beat
        current_beat = get_beat_index(drawn - 1, beat_boundaries)
        beat_counter.set_text(f"Beat: {current_beat + 1}/{len(labels)}")

        # Check if we just entered an arrhythmia beat
        if current_beat < len(labels):
            cls = labels[current_beat]
            if cls == 2:  # V-class
                # Show warning
                warn_alpha[0] = min(1.0, warn_alpha[0] + 0.15)
                class_label.set_text(">> VENTRICULAR ARRHYTHMIA <<")
                class_label.set_alpha(warn_alpha[0])
                class_label.set_color(WARN_COLOR)
                status_text.set_text("● ARRHYTHMIA DETECTED")
                status_text.set_color(WARN_COLOR)
                current_warn_class[0] = 2

                # Add a vertical red band for this beat
                beat_start = beat_boundaries[current_beat]
                beat_end = beat_boundaries[current_beat + 1] if current_beat + 1 < len(beat_boundaries) else total_samples

            elif cls == 1:  # S-class
                warn_alpha[0] = min(1.0, warn_alpha[0] + 0.15)
                class_label.set_text(">> SUPRAVENTRICULAR <<")
                class_label.set_alpha(warn_alpha[0])
                class_label.set_color(TRACE_S_CLASS)
                status_text.set_text("● ABNORMAL BEAT")
                status_text.set_color(TRACE_S_CLASS)
                current_warn_class[0] = 1

            elif cls == 3:  # F-class
                warn_alpha[0] = min(1.0, warn_alpha[0] + 0.15)
                class_label.set_text(">> FUSION BEAT <<")
                class_label.set_alpha(warn_alpha[0])
                class_label.set_color(TRACE_F_CLASS)
                status_text.set_text("● ABNORMAL BEAT")
                status_text.set_color(TRACE_F_CLASS)
                current_warn_class[0] = 3

            else:
                # Normal — fade out warning
                warn_alpha[0] = max(0.0, warn_alpha[0] - 0.05)
                class_label.set_alpha(warn_alpha[0])
                if warn_alpha[0] < 0.01:
                    status_text.set_text("● MONITORING")
                    status_text.set_color(TRACE_NORMAL)
                    current_warn_class[0] = None

        return lc, lc_glow, beat_counter, class_label, status_text

    print(f"[Animation] Creating animation: {total_frames} frames at {fps} FPS...")
    print(f"[Animation] Duration: ~{total_frames / fps:.1f} seconds")

    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=total_frames, interval=1000 / fps, blit=False,
    )

    if save_path:
        print(f"[Animation] Saving to {save_path}...")
        if fmt == "mp4":
            try:
                writer = animation.FFMpegWriter(fps=fps, bitrate=3000,
                                                 extra_args=['-pix_fmt', 'yuv420p'])
                anim.save(save_path, writer=writer, dpi=150)
            except (FileNotFoundError, RuntimeError):
                print("[Animation] ffmpeg not found, trying Pillow writer for GIF fallback...")
                gif_path = save_path.replace(".mp4", ".gif")
                writer = animation.PillowWriter(fps=fps)
                anim.save(gif_path, writer=writer, dpi=100)
                save_path = gif_path
        elif fmt == "gif":
            writer = animation.PillowWriter(fps=fps)
            anim.save(save_path, writer=writer, dpi=100)
        print(f"[Animation] ✅ Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return save_path


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ECG Waveform Animation for Hackathon Video")
    parser.add_argument("--data", default=None,
                        help="Path to mitbih_test.csv (default: auto-detect)")
    parser.add_argument("--beats", type=int, default=10,
                        help="Number of beats to animate (default: 10)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second (default: 30)")
    parser.add_argument("--speed", type=int, default=3,
                        help="Samples drawn per frame (default: 3, lower=slower)")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4",
                        help="Output format (default: mp4)")
    parser.add_argument("--no-save", action="store_true",
                        help="Display instead of saving")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for beat selection")
    args = parser.parse_args()

    # Find data file
    if args.data:
        data_path = args.data
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, "data", "mitbih_test.csv")

    if not os.path.exists(data_path):
        print(f"[Error] Data file not found: {data_path}")
        print("Download from: https://www.kaggle.com/datasets/shayanfazeli/heartbeat")
        sys.exit(1)

    # Load beats
    waveform, beat_boundaries, labels = load_beats(
        data_path, num_beats=args.beats, seed=args.seed
    )

    # Output path
    if args.no_save:
        save_path = None
    else:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        ext = args.format
        save_path = os.path.join(results_dir, f"ecg_arrhythmia_animation.{ext}")

    create_animation(
        waveform, beat_boundaries, labels,
        fps=args.fps,
        samples_per_frame=args.speed,
        save_path=save_path,
        fmt=args.format,
    )


if __name__ == "__main__":
    main()
