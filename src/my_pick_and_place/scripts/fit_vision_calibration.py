#!/usr/bin/env python3
"""Decide from data how much to trust the vision model's fragility estimate.

Reads the per-attempt records from fragility_grasp.py (fragility=log or on) and the
fragility world's truth file, and fits a straight line from what vision said (its
estimate before any calibration) to true fragility. Vision gets a weight only if that fit
genuinely explains the data AND vision beats simply guessing the middle of the scale:

    weight = R^2   if n >= MIN_SAMPLES on >= MIN_APPLES apples, R^2 >= MIN_R2,
                   and vision's error is below the always-guess-5 error
    weight = 0     otherwise, with the reason written into vision_calibration.json

Without this, the model's own confidence (0.8-0.9 every time, even when it called a green
apple overripe) would drive the grip.

Also prints a per-apple table and each attempt's vote agreement and image age, so a bad
answer can be traced to its cause (a stale image, an ambiguous picture, a model that
cannot tell).

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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else fg.RECORDS_PATH
    truth = fg.load_json(fg.TRUTH_PATH, {})
    if not truth:
        print(f"No truth file at {fg.TRUTH_PATH}. Vision can only be calibrated in the "
              f"fragility world (bash start_everything.sh tactile fragility).")
        return 1
    try:
        with open(path, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    except OSError:
        print(f"No records at {path}. Run full_layer_grasp.py ... fragility=log first.")
        return 1

    rows = []
    for r in records:
        v = r.get("vision") or {}
        est = v.get("prior_fragility", v.get("fragility"))
        true_f = (truth.get(r.get("apple")) or {}).get("fragility")
        if est is None or true_f is None:
            continue
        raw = r.get("vision_raw") or {}
        frame = r.get("frame") or {}
        rows.append({"apple": r["apple"], "true": true_f, "est": est,
                     "votes": raw.get("votes"), "agreement": v.get("agreement"),
                     "age": frame.get("age_sim_s"), "frame": frame.get("path")})

    print(f"{'apple':<9} {'true':>4} {'vision':>6} {'error':>6}  {'agree':>5}  "
          f"{'image age (sim s)':>17}  votes")
    for x in sorted(rows, key=lambda x: x["apple"]):
        age = "-" if x["age"] is None else f"{x['age']:.2f}"
        agree = "-" if x["agreement"] is None else f"{x['agreement']:.2f}"
        print(f"{x['apple']:<9} {x['true']:>4} {x['est']:>6.1f} {x['est'] - x['true']:>+6.1f}  "
              f"{agree:>5}  {age:>17}  {x['votes']}")

    if len(rows) < 2:
        out = {"confidence": 0.0, "note": f"only {len(rows)} usable records"}
    else:
        n = len(rows)
        apples = len({x["apple"] for x in rows})
        mae = sum(abs(x["est"] - x["true"]) for x in rows) / n
        base = sum(abs(5.0 - x["true"]) for x in rows) / n
        fit = fit_line([x["est"] for x in rows], [x["true"] for x in rows])
        print(f"\nvision mean abs error {mae:.2f}  vs  always guessing 5: {base:.2f}")
        if fit is None:
            out = {"confidence": 0.0, "n": n, "apples": apples, "mae": mae,
                   "baseline_mae": base,
                   "note": "vision gave the same estimate every time -- it cannot tell apples apart"}
        else:
            a, b, r2 = fit
            print(f"fit: true fragility = {a:.2f} + {b:.3f} * vision estimate, R^2 = {r2:.2f} "
                  f"(n={n} on {apples} apples)")
            out = {"a": a, "b": b, "r2": r2, "n": n, "apples": apples, "mae": mae,
                   "baseline_mae": base}
            if n < MIN_SAMPLES or apples < MIN_APPLES:
                out.update(confidence=0.0, note=f"too little data (need {MIN_SAMPLES} "
                                                f"attempts on {MIN_APPLES} apples)")
            elif r2 < MIN_R2:
                out.update(confidence=0.0, note=f"R^2 {r2:.2f} below {MIN_R2}: vision does "
                                                f"not track fragility well enough")
            elif mae >= base:
                out.update(confidence=0.0, note=f"vision error {mae:.2f} is not better than "
                                                f"guessing 5 ({base:.2f})")
            else:
                out.update(confidence=round(r2, 3), note="fitted")

    with open(fg.VISION_CALIBRATION_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {fg.VISION_CALIBRATION_PATH}: vision weight {out['confidence']} -- "
          f"{out['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
