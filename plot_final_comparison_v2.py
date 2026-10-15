# plot_final_comparison_v2.py
"""
Comparison plot: naive vs filtered policies, one color per pair, line style
per method (solid = naive, dashed = filtered).

Usage: python plot_final_comparison_v2.py \
    e1_final.json e1_safe_final.json e2_final.json e2_safe_final.json
"""
import json
import sys
import numpy as np
import matplotlib.pyplot as plt


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_series(path):
    with open(path) as f:
        d = json.load(f)
    items = sorted(d["summary"].items(), key=lambda kv: int(kv[0]))
    steps = np.array([int(k) for k, _ in items])
    fr = np.array([v["fall_rate"] for _, v in items])
    n = np.array([v["n"] for _, v in items])
    ci_lo = np.array([wilson_ci(int(round(fr[i] * n[i])), n[i])[0] for i in range(len(steps))])
    ci_hi = np.array([wilson_ci(int(round(fr[i] * n[i])), n[i])[1] for i in range(len(steps))])
    return steps, fr, ci_lo, ci_hi


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_final_comparison_v2.py naive1.json safe1.json [naive2.json safe2.json ...]")
        sys.exit(1)

    files = sys.argv[1:]
    # Pair up: (naive, safe) × N pairs
    pairs = []
    i = 0
    while i < len(files):
        naive = files[i]
        safe = files[i + 1] if i + 1 < len(files) else None
        pairs.append((naive, safe))
        i += 2

    fig, ax = plt.subplots(figsize=(11, 6))
    palette = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']

    for pi, (naive_path, safe_path) in enumerate(pairs):
        color = palette[pi % len(palette)]
        steps_n, fr_n, lo_n, hi_n = load_series(naive_path)
        label_n = naive_path.replace('.json', '').replace('_', ' ')
        ax.errorbar(steps_n, fr_n, yerr=[fr_n - lo_n, hi_n - fr_n], fmt='o-',
                    color=color, capsize=3, lw=2, markersize=5,
                    label=f"{label_n} (naive)", alpha=0.85)

        if safe_path:
            steps_s, fr_s, lo_s, hi_s = load_series(safe_path)
            label_s = safe_path.replace('.json', '').replace('_', ' ')
            ax.errorbar(steps_s, fr_s, yerr=[fr_s - lo_s, hi_s - fr_s], fmt='s--',
                        color=color, capsize=3, lw=2, markersize=5,
                        label=f"{label_s} (filtered)", alpha=0.85)

    ax.axhline(0.5, ls=':', color='gray', alpha=0.4, lw=1)
    ax.set_xlabel("Switch step (× 20 ms @ 50 Hz)")
    ax.set_ylabel("Fall rate")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Naive vs filtered policy switching (Wilson 95% CI)")
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig("final_comparison_v2.png", dpi=150)
    print("Saved final_comparison_v2.png")
    plt.show()


if __name__ == "__main__":
    main()