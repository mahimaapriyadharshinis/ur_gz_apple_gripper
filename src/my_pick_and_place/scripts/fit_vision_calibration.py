#!/usr/bin/env python3
"""Decide from data how much to trust each camera-based fragility estimate.

Two camera estimates are calibrated here, each against the fragility world's truth file:

  vision   the vision model's answer (its ripeness/firmness words on the 0-10 scale)
           -> vision_calibration.json
  colour   the apple's hue measured from the image pixels
           -> colour_calibration.json

For each, a straight line from the estimate to true fragility is fitted. It only gets a
weight if the data genuinely supports it:

    weight = R^2   if n >= MIN_SAMPLES on >= MIN_APPLES apples, R^2 >= MIN_R2, and its
                   LEAVE-ONE-OUT error beats always guessing 5
    weight = 0     otherwise, with the reason written into the file

Leave-one-out: every attempt is predicted by a line fitted WITHOUT that attempt, so a fit
cannot look good just by memorising a few points. With ~20 attempts this matters.

Also prints a per-attempt table (votes, agreement, hue, image age), so any bad estimate
can be traced to its cause.

Usage:
    python3 fit_vision_calibration.py            # uses ~/fragility_records.jsonl
    python3 fit_vision_calibration.py records.jsonl
"""

import json
import sys

import fragility_grasp as fg
from fit_touch_calibration import fit_line

MIN_SAMPLES = 8
MIN_APPLES = 4
MIN_R2 = 0.5


def leave_one_out_mae(xs, ys):
    errs = []
    for i in range(len(xs)):
        fit = fit_line(xs[:i] + xs[i + 1:], ys[:i] + ys[i + 1:])
        if fit is None:
            return None
        a, b, _ = fit
        errs.append(abs(fg._clamp(a + b * xs[i], 0.0, 10.0) - ys[i]))
    return sum(errs) / len(errs)


def fit_source(label, pairs):
    """pairs: list of (value, true_fragility, apple)."""
    n = len(pairs)
    apples = len({p[2] for p in pairs})
    if n < 3:
        return {"confidence": 0.0, "n": n, "note": f"only {n} usable records"}
    xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
    base = sum(abs(5.0 - y) for y in ys) / n
    fit = fit_line(xs, ys)
    if fit is None:
        return {"confidence": 0.0, "n": n, "apples": apples, "baseline_mae": base,
                "note": f"{label} gave the same value every time -- cannot tell apples apart"}
    a, b, r2 = fit
    loo = leave_one_out_mae(xs, ys)
    out = {"a": a, "b": b, "r2": r2, "n": n, "apples": apples,
           "loo_mae": loo, "baseline_mae": base}
    print(f"  {label}: true = {a:.2f} + {b:.4f} * value, R^2 {r2:.2f}, leave-one-out error "
          f"{'-' if loo is None else f'{loo:.2f}'} vs always-5 {base:.2f} (n={n}, {apples} apples)")
    if n < MIN_SAMPLES or apples < MIN_APPLES:
        out.update(confidence=0.0, note=f"too little data (need {MIN_SAMPLES} attempts on "
                                        f"{MIN_APPLES} apples)")
    elif r2 < MIN_R2:
        out.update(confidence=0.0, note=f"R^2 {r2:.2f} below {MIN_R2}")
    elif loo is None or loo >= base:
        out.update(confidence=0.0, note=f"leave-one-out error {loo} does not beat guessing 5 "
                                        f"({base:.2f})")
    else:
        out.update(confidence=round(r2, 3), note="fitted")
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else fg.RECORDS_PATH
    truth = fg.load_json(fg.TRUTH_PATH, {})
    if not truth:
        print(f"No truth file at {fg.TRUTH_PATH}. Calibrate in the fragility world "
              f"(bash start_everything.sh tactile fragility).")
        return 1
    try:
        with open(path, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    except OSError:
        print(f"No records at {path}. Run full_layer_grasp.py ... fragility=log first.")
        return 1

    vision_pairs, colour_pairs = [], []
    print(f"{'apple':<9} {'true':>4} {'vision':>6} {'hue':>6} {'agree':>5} {'age s':>6}  votes")
    for r in sorted(records, key=lambda r: r.get("apple", "")):
        true_f = (truth.get(r.get("apple")) or {}).get("fragility")
        if true_f is None:
            continue
        v = r.get("vision") or {}
        vis = v.get("prior_fragility", v.get("fragility"))
        hue = (r.get("colour_features") or {}).get("hue_from_green_deg")
        if vis is not None:
            vision_pairs.append((vis, true_f, r["apple"]))
        if hue is not None:
            colour_pairs.append((hue, true_f, r["apple"]))
        age = (r.get("frame") or {}).get("age_sim_s")
        agree = v.get("agreement")
        vis_s = "-" if vis is None else f"{vis:.1f}"
        hue_s = "-" if hue is None else f"{hue:.0f}"
        agree_s = "-" if agree is None else f"{agree:.2f}"
        age_s = "-" if age is None else f"{age:.2f}"
        votes = (r.get("vision_raw") or {}).get("votes")
        print(f"{r['apple']:<9} {true_f:>4} {vis_s:>6} {hue_s:>6} {agree_s:>5} {age_s:>6}  {votes}")

    print("\nfits:")
    for label, pairs, target in (("vision", vision_pairs, fg.VISION_CALIBRATION_PATH),
                                 ("colour", colour_pairs, fg.COLOUR_CALIBRATION_PATH)):
        out = fit_source(label, pairs)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"  -> {target}: weight {out['confidence']} -- {out['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
