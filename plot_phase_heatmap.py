# plot_phase_heatmap.py
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load(fp):
    p = Path(fp)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p.resolve()}")
    with open(p) as f:
        return json.load(f)

def get_series(d, valid_max_step=90):
    """Return (valid_steps, valid_fall, invalid_steps, invalid_fall)."""
    items = sorted(d["summary"].items(), key=lambda kv: int(kv[0]))
    steps = np.array([int(k) for k, _ in items])
    fall = np.array([v["fall_rate"] for _, v in items])
    valid = steps <= valid_max_step
    invalid = steps > valid_max_step
    return steps[valid], fall[valid], steps[invalid], fall[invalid]

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_phase_heatmap.py e1.json e2.json e3.json")
        sys.exit(1)

    files = sys.argv[1:]
    n = len(files)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.2 * n), squeeze=False)
    axes = axes[:, 0]

    for i, fp in enumerate(files):
        d = load(fp)
        v_steps, v_fall, i_steps, i_fall = get_series(d)
        name = f"{d['config']['from_policy']}→{d['config']['to_policy']} @ {d['config']['walk_vel']} m/s"

        ax = axes[i]
        # Valid region
        ax.bar(v_steps, v_fall, width=6, color='steelblue', edgecolor='k',
               label='valid (walking stable)')
        # Invalid region (confound)
        if len(i_steps):
            ax.bar(i_steps, i_fall, width=6, color='lightgray', edgecolor='k',
                   hatch='//', label='confound: walking fell on its own')

        ax.axhline(0.5, ls='--', color='orange', lw=1, alpha=0.7)
        ax.axvline(90, ls=':', color='red', lw=1, alpha=0.7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Fall rate")
        ax.set_title(name)
        ax.set_xticks(list(v_steps) + list(i_steps))
        ax.tick_params(axis='x', rotation=45)
        if i == 0:
            ax.legend(loc='upper left', fontsize=9)

    axes[-1].set_xlabel("Switch step (× 20 ms at 50 Hz)")
    plt.tight_layout()
    out = "naive_switch_phase.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()

if __name__ == "__main__":
    main()