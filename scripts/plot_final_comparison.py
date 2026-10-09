# plot_final_comparison.py
import json, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def wilson_ci(k, n, z=1.96):
    if n == 0: return 0, 0
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0, centre-half), min(1, centre+half)

def load_series(path):
    with open(path) as f: d = json.load(f)
    items = sorted(d["summary"].items(), key=lambda kv: int(kv[0]))
    steps = np.array([int(k) for k,_ in items])
    fr = np.array([v["fall_rate"] for _,v in items])
    n = np.array([v["n"] for _,v in items])
    ci_lo = np.array([wilson_ci(int(round(fr[i]*n[i])), n[i])[0] for i in range(len(steps))])
    ci_hi = np.array([wilson_ci(int(round(fr[i]*n[i])), n[i])[1] for i in range(len(steps))])
    return steps, fr, ci_lo, ci_hi

files = sys.argv[1:]
fig, ax = plt.subplots(figsize=(10, 5.5))
colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd']
for i, fp in enumerate(files):
    steps, fr, lo, hi = load_series(fp)
    label = fp.replace('.json','').replace('_', ' ')
    ax.errorbar(steps, fr, yerr=[fr-lo, hi-fr], fmt='o-',
                color=colors[i % 4], capsize=4, lw=2, markersize=6, label=label)

ax.axhline(0.5, ls='--', color='gray', alpha=0.5, lw=1)
ax.set_xlabel("Switch step (× 20 ms @ 50 Hz)")
ax.set_ylabel("Fall rate")
ax.set_ylim(-0.02, 1.02)
ax.set_title("Naive policy switching: fall rate vs switch phase (30 trials, Wilson 95% CI)")
ax.grid(alpha=0.3)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig("final_comparison.png", dpi=150)
print("Saved final_comparison.png")