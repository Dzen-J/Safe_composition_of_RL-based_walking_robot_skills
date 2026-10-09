# reanalyze_falls.py
"""Recompute fall_rate with a stricter, physically-correct criterion:
   fell = (max_tilt_deg > TILT_DEG) OR (min_height_m < HEIGHT_M)
"""
import json
import sys
from pathlib import Path

TILT_DEG = 75.0
HEIGHT_M = 0.060

def reanalyze(path):
    with open(path) as f:
        d = json.load(f)

    # Recompute per-episode
    for e in d["episodes"]:
        real_fell = (e["max_tilt_deg"] > TILT_DEG) or (e["min_height_m"] < HEIGHT_M)
        e["old_fell"] = e["fell"]
        e["fell"] = real_fell

    # Recompute summary
    for k, v in d["summary"].items():
        ep = [e for e in d["episodes"] if str(e["switch_step"]) == k]
        n = len(ep)
        v["old_fall_rate"] = v["fall_rate"]
        v["fall_rate"] = sum(e["fell"] for e in ep) / n if n else 0.0

    return d

def main():
    if len(sys.argv) < 2:
        print("Usage: reanalyze_falls.py <file.json> [<file2.json> ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        d = reanalyze(path)
        out = path.replace(".json", "_strict.json")
        with open(out, "w") as f:
            json.dump(d, f, indent=2)

        # Print comparison
        print(f"\n=== {path} → {out} ===")
        print(f"{'switch':>8} | {'old fall_rate':>13} | {'new fall_rate':>13} | Δ")
        for k in sorted(d["summary"], key=int):
            v = d["summary"][k]
            delta = v["fall_rate"] - v["old_fall_rate"]
            print(f"{k:>8} | {v['old_fall_rate']:>13.2f} | {v['fall_rate']:>13.2f} | {delta:+.2f}")

if __name__ == "__main__":
    main()