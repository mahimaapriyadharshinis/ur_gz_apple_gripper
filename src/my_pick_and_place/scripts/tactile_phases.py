#!/usr/bin/env python3
"""Turn a fingertip-sensor CSV into sustained force per grasp phase, per attempt.

A peak over a run is dominated by impact spikes (91N index, 107N thumb measured, against
about 7N of apple weight), so this reads the MEDIAN force over two windows of each
attempt, located with the simulation clock that pocket_grasp_test.py now prints:

  hold  -- from "squeeze finished" to "lift started": what the hand is really squeezing
  lift  -- the first LIFT_WINDOW_S of simulated time after the lift starts, which is
           exactly where apple_10 loses the apple when it fails

and lines each attempt up with its outcome (held / not held), so picks and misses can be
compared finger by finger.

Usage:
    python3 tactile_phases.py /tmp/tactile.csv /tmp/grasp.log
"""

import csv
import re
import statistics
import sys

FINGERS = ["index", "middle", "ring", "pinky", "thumb"]
LIFT_WINDOW_S = 0.5


def read_phases(log_path):
    """Per attempt: (squeeze finished, lift started, held) from the grasp log."""
    attempts, cur = [], {}
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = re.search(r"\[phase\] squeeze finished at sim t=([0-9.]+)s", line)
            if m:
                cur = {"squeeze": float(m.group(1))}
                continue
            m = re.search(r"\[phase\] lift started at sim t=([0-9.]+)s", line)
            if m and cur:
                cur["lift"] = float(m.group(1))
                continue
            if "apple height change after lift" in line and cur:
                cur["held"] = "(HELD)" in line
                attempts.append(cur)
                cur = {}
    return attempts


def read_csv(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                sim = float(r["sim_s"])
            except (TypeError, ValueError):
                continue
            vals = {}
            for f in FINGERS:
                try:
                    vals[f] = (float(r[f"{f}_N"]), float(r[f"{f}_Nm"]))
                except (TypeError, ValueError, KeyError):
                    vals[f] = None
            rows.append((sim, vals))
    return rows


def window_median(rows, t0, t1):
    out = {}
    for f in FINGERS:
        n = [v[f][0] for s, v in rows if t0 <= s <= t1 and v.get(f)]
        e = [v[f][1] for s, v in rows if t0 <= s <= t1 and v.get(f)]
        out[f] = (statistics.median(n) if n else None,
                  statistics.median(e) if e else None, len(n))
    return out


def fmt(cell):
    n, e, count = cell
    if n is None:
        return "     -     "
    return f"{n:5.1f}N/{e:4.2f}"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    rows = read_csv(sys.argv[1])
    attempts = read_phases(sys.argv[2])
    if not rows:
        print("No usable rows in the CSV (was tactile_check.py run with --csv?)")
        return 1
    if not attempts:
        print("No [phase] lines in the grasp log (it must come from the updated "
              "pocket_grasp_test.py)")
        return 1

    head = "  ".join(f"{f:>12}" for f in FINGERS)
    print(f"sustained (median) fingertip force N / joint effort Nm, "
          f"{len(attempts)} attempts, {len(rows)} sensor samples\n")
    for i, a in enumerate(attempts, 1):
        verdict = "HELD" if a.get("held") else "NOT HELD"
        print(f"attempt {i}: {verdict}")
        print(f"  {'phase':<6}{head}")
        if "squeeze" in a and "lift" in a:
            hold = window_median(rows, a["squeeze"], a["lift"])
            print(f"  {'hold':<6}" + "  ".join(f"{fmt(hold[f]):>12}" for f in FINGERS))
        if "lift" in a:
            lift = window_median(rows, a["lift"], a["lift"] + LIFT_WINDOW_S)
            print(f"  {'lift':<6}" + "  ".join(f"{fmt(lift[f]):>12}" for f in FINGERS))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
