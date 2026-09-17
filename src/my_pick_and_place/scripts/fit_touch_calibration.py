#!/usr/bin/env python3
"""Fit touch readings to true fragility, and decide from the fit how much to trust touch.

Reads the per-attempt records written by fragility_grasp.py (fragility=log or on) and the
fragility world's truth file, fits a straight line from each touch feature to true
fragility, keeps the better one, and writes touch_calibration.json. Touch only gets a
weight if the fit genuinely explains the data:

    weight = R^2   if at least MIN_SAMPLES attempts on at least MIN_APPLES apples and R^2 >= MIN_R2
    weight = 0     otherwise, with the reason written into the file

So if the simulator does not really make soft apples feel different -- which is possible,
since the physics engine may ignore the apples' contact softness -- touch stays switched
off, and the file says why, instead of the grip being driven by noise.

Usage:
    python3 fit_touch_calibration.py            # uses ~/fragility_records.jsonl
    python3 fit_touch_calibration.py records.jsonl
"""

import json
import os
import sys

import fragility_grasp as fg

MIN_SAMPLES = 8
MIN_APPLES = 4
MIN_R2 = 0.5
FEATURES = ("sink_ratio", "stiffness_nm_per_rad")


def fit_line(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot
    return a, b, r2


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else fg.RECORDS_PATH
    truth = fg.load_json(fg.TRUTH_PATH, {})
    if not truth:
        print(f"No truth file at {fg.TRUTH_PATH}. Touch can only be calibrated in the "
              f"fragility world (bash start_everything.sh tactile fragility).")
        return 1
    records = []
    try:
        with open(path, encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
    except OSError:
        print(f"No records at {path}. Run full_layer_grasp.py ... fragility=log first.")
        return 1

    best = None
    report = []
    for feature in FEATURES:
        pairs = [((r.get("touch_features") or {}).get(feature),
                  (truth.get(r.get("apple")) or {}).get("fragility"), r.get("apple"))
                 for r in records]
        pairs = [(x, y, a) for x, y, a in pairs if x is not None and y is not None]
        apples = {a for _, _, a in pairs}
        if len(pairs) < 2:
            report.append(f"  {feature}: only {len(pairs)} usable readings")
            continue
        fit = fit_line([x for x, _, _ in pairs], [y for _, y, _ in pairs])
        if fit is None:
            report.append(f"  {feature}: every reading identical -- touch cannot tell "
                          f"apples apart with this feature")
            continue
        a, b, r2 = fit
        report.append(f"  {feature}: n={len(pairs)} on {len(apples)} apples, "
                      f"fragility = {a:.2f} + {b:.3f} * value, R^2 = {r2:.2f}")
        cand = {"feature": feature, "a": a, "b": b, "r2": r2, "n": len(pairs),
                "apples": len(apples)}
        if best is None or r2 > best["r2"]:
            best = cand

    print("touch features against true fragility:")
    print("\n".join(report))

    if best is None:
        out = {"confidence": 0.0, "note": "no feature could be fitted"}
    elif best["n"] < MIN_SAMPLES or best["apples"] < MIN_APPLES:
        out = dict(best, confidence=0.0,
                   note=f"too little data (need {MIN_SAMPLES} attempts on {MIN_APPLES} apples)")
    elif best["r2"] < MIN_R2:
        out = dict(best, confidence=0.0,
                   note=(f"R^2 {best['r2']:.2f} below {MIN_R2}: touch does not track "
                         f"fragility well enough to be trusted"))
    else:
        out = dict(best, confidence=round(best["r2"], 3), note="fitted")

    with open(fg.CALIBRATION_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {fg.CALIBRATION_PATH}: touch weight {out['confidence']} -- {out['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
