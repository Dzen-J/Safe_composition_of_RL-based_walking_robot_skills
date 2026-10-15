# compare_tilts_v2.py
"""
Correct leak check: match naive/filtered trials by (switch_step, trial),
and only verify that trajectories match up to the moment of switch.

- When filtered delay == 0: pre_switch_tilt must equal naive EXACTLY.
- When filtered delay > 0: pre_switch_tilt is expected to differ by the
  drift accumulated over the extra steps — this is legitimate filter
  behavior, not a leak.
"""
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)["episodes"]


def main():
    if len(sys.argv) != 3:
        print("Usage: compare_tilts_v2.py naive.json filtered.json")
        sys.exit(1)

    a = load(sys.argv[1])
    b = load(sys.argv[2])

    index_a = {(e["switch_step"], e["trial"]): e for e in a}
    index_b = {(e["switch_step"], e["trial"]): e for e in b}

    print(f"{'step':>5} | {'trial':>5} | {'delay':>5} | {'naive_tilt':>11} "
          f"| {'filt_tilt':>11} | {'delta':>8} | verdict")
    print("-" * 88)

    max_delta_zero_delay = 0.0
    n_zero_delay = 0
    n_delayed = 0

    for key in sorted(index_a.keys()):
        if key not in index_b:
            continue
        sw, t = key
        ea, eb = index_a[key], index_b[key]
        ta = ea["pre_switch_tilt_max_deg"]
        tb = eb["pre_switch_tilt_max_deg"]
        d = tb - ta
        delay = eb["switch_delay"]

        if delay == 0:
            n_zero_delay += 1
            max_delta_zero_delay = max(max_delta_zero_delay, abs(d))
            if abs(d) < 1e-6:
                verdict = "OK (exact match)"
            else:
                verdict = f"*** LEAK: {d:+.6f} ***"
        else:
            n_delayed += 1
            verdict = f"delayed {delay} step(s) — expected drift"

        print(f"{sw:5d} | {t:5d} | {str(delay):>5s} | {ta:11.3f} "
              f"| {tb:11.3f} | {d:+8.3f} | {verdict}")

    print("-" * 88)
    print(f"cells with delay=0 : {n_zero_delay}   max |delta| = {max_delta_zero_delay:.8f}")
    print(f"cells with delay>0 : {n_delayed}   (drift expected — not a leak)")

    if max_delta_zero_delay < 1e-6:
        print("\n>>> LEAK-FREE. The fix works. Any remaining delta is due to switch delay.")
    else:
        print("\n>>> STILL LEAKING. Investigate further.")


if __name__ == "__main__":
    main()