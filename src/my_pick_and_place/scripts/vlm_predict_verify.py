#!/usr/bin/env python3
"""Ask the vision model to PREDICT before the grasp and VERIFY after the lift, then score
both against what the run actually measured.

Deliberately observation-only: nothing here changes a single command sent to the robot.
The grasp already picks all ten apples with one measured configuration, so the way to add
vision reasoning without risking that is to let the model commit to answers that can be
checked, and keep a scorecard. If the model turns out accurate, its answers can later be
trusted for control (see VISION_SETS_GRIP); if not, the scorecard is the evidence.

Two questions per attempt:

  predict_before -- at the grasp pose, before the fingers move: what is it, how big, how
                    heavy, is the hand well placed, will this grasp hold?
  verify_after   -- after the lift: is the object still in the hand, does it look damaged?

Every field is one the run can check for itself:
  diameter/mass -> the apple's radius and its own model.sdf
  hand placement -> the measured spread of the four fingertip gaps
  will_hold      -> whether the apple actually rose
  still_held     -> the same outcome, so the model's after-the-fact reading can be
                    compared with the physical result (a vision model that can verify its
                    own grasps is worth more than one that only guesses beforehand).
"""

import base64
import json
import os
import re
import time

PREDICT_PROMPT = """You are inspecting a robot hand about to grasp the object in the image.
Respond with ONLY a valid JSON object (no markdown, no extra text) with these exact keys:
{
  "object_name": "best guess at what the object is",
  "estimated_diameter_cm": number, the object's width in centimetres,
  "estimated_mass_g": number, the object's mass in grams,
  "hand_placement": "centred" or "off_to_one_side",
  "will_hold": "yes" or "no", will this grasp lift the object without dropping it,
  "confidence": 0.0-1.0 float,
  "notes": "one short sentence"
}"""

VERIFY_PROMPT = """You are checking a robot hand AFTER it tried to lift the object.
Respond with ONLY a valid JSON object (no markdown, no extra text) with these exact keys:
{
  "still_held": "yes" or "no", is the object still gripped by the hand,
  "looks_damaged": "yes" or "no", does the object look crushed or bruised,
  "confidence": 0.0-1.0 float,
  "notes": "one short sentence"
}"""

# Per-attempt records are appended here, one JSON object per line.
DEFAULT_LOG = os.path.expanduser("~/vlm_predict_verify.jsonl")

# The hand counts as "centred" when the four fingertip gaps agree within this (the run
# already prints exactly this spread, and calls 8mm or less "even").
CENTRED_SPREAD_M = 0.008


def encode_frame(frame):
    """BGR image -> base64 JPEG, or None if it cannot be encoded."""
    try:
        import cv2
    except ImportError:
        return None
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return base64.b64encode(buf).decode("utf-8")


def ask_ollama(prompt, image_b64, url, model, timeout=240.0):
    """One JSON-formatted vision-model call. Returns a dict, or {"error": ...} -- never
    raises, so a model that is slow or down cannot fail an attempt."""
    import requests
    try:
        r = requests.post(url, json={"model": model, "prompt": prompt,
                                     "images": [image_b64] if image_b64 else [],
                                     "stream": False, "format": "json"}, timeout=timeout)
        r.raise_for_status()
        return json.loads(r.json().get("response", "").strip())
    except Exception as e:
        return {"error": str(e)}


def predict_before(frame, url, model, ask=ask_ollama):
    started = time.time()
    out = ask(PREDICT_PROMPT, encode_frame(frame), url, model)
    if not isinstance(out, dict):
        out = {"error": f"model returned {type(out).__name__}, not an object"}
    out["seconds"] = round(time.time() - started, 1)
    return out


def verify_after(frame, url, model, ask=ask_ollama):
    started = time.time()
    out = ask(VERIFY_PROMPT, encode_frame(frame), url, model)
    if not isinstance(out, dict):
        out = {"error": f"model returned {type(out).__name__}, not an object"}
    out["seconds"] = round(time.time() - started, 1)
    return out


def apple_truth(target_name, radius_m, models_root=None):
    """Real diameter and mass: the radius comes from the caller, the mass is read out of
    the apple's own model.sdf rather than kept as a second hardcoded table."""
    truth = {"diameter_cm": None if radius_m is None else round(radius_m * 200.0, 1),
             "mass_g": None}
    root = models_root or os.path.expanduser("~/ur_gz_ws/src/apple_gripper_sim/models")
    try:
        with open(os.path.join(root, target_name, "model.sdf"), encoding="utf-8") as fh:
            m = re.search(r"<mass>\s*([0-9.eE+-]+)\s*</mass>", fh.read())
        if m:
            truth["mass_g"] = round(float(m.group(1)) * 1000.0, 1)
    except OSError:
        pass
    return truth


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _yes(value):
    return str(value).strip().lower() in ("yes", "true", "1")


def score(pred, ver, truth, measured):
    """Compare the model's answers with the run's own measurements.

    measured: {"held": bool, "gap_spread_m": float or None}
    """
    held = bool(measured.get("held"))
    spread = _num(measured.get("gap_spread_m"))
    d_pred = _num(pred.get("estimated_diameter_cm"))
    m_pred = _num(pred.get("estimated_mass_g"))
    d_true = _num(truth.get("diameter_cm"))
    m_true = _num(truth.get("mass_g"))
    centred_measured = None if spread is None else spread <= CENTRED_SPREAD_M
    placement = str(pred.get("hand_placement", "")).strip().lower() or None
    will_hold = None if pred.get("will_hold") is None else _yes(pred.get("will_hold"))
    still_held = None if ver.get("still_held") is None else _yes(ver.get("still_held"))

    return {
        "diameter_error_cm": None if None in (d_pred, d_true) else round(d_pred - d_true, 1),
        "mass_error_g": None if None in (m_pred, m_true) else round(m_pred - m_true, 1),
        "will_hold_pred": will_hold,
        "will_hold_correct": None if will_hold is None else will_hold == held,
        "placement_pred": placement,
        "placement_correct": (None if (placement is None or centred_measured is None)
                              else (placement == "centred") == centred_measured),
        "still_held_pred": still_held,
        "verify_correct": None if still_held is None else still_held == held,
        "held_measured": held,
    }


def log_record(record, path=DEFAULT_LOG):
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return path
    except OSError:
        return None


def one_line(target_name, pred, ver, scored):
    """The per-attempt line printed into the run log."""
    def mark(ok):
        return "?" if ok is None else ("correct" if ok else "WRONG")

    def err(value, unit):
        return "-" if value is None else f"{value:+.1f}{unit}"

    return (f"  [VLM predict/verify] {target_name}: size {err(scored['diameter_error_cm'], 'cm')}, "
            f"mass {err(scored['mass_error_g'], 'g')}, placement {mark(scored['placement_correct'])}, "
            f"will-hold {mark(scored['will_hold_correct'])}, after-lift check "
            f"{mark(scored['verify_correct'])} (said {pred.get('will_hold')}/"
            f"{ver.get('still_held')}, actually "
            f"{'held' if scored['held_measured'] else 'not held'})")


def summary(records):
    """A short accuracy table over one run's attempts."""
    if not records:
        return "No VLM predict/verify records in this run."

    def rate(key):
        vals = [r["score"][key] for r in records if r["score"].get(key) is not None]
        return None if not vals else (sum(1 for v in vals if v), len(vals))

    def spread_of(key, unit):
        vals = [r["score"][key] for r in records if r["score"].get(key) is not None]
        if not vals:
            return "-"
        return (f"mean {sum(vals) / len(vals):+.1f}{unit}, "
                f"worst {max(vals, key=abs):+.1f}{unit}")

    lines = ["",
             "VLM PREDICT / VERIFY SCORECARD (observation only -- no effect on the grasp)",
             f"{'question':<24}  {'correct':>22}"]
    for label, key in (("hand placement", "placement_correct"),
                       ("will it hold (before)", "will_hold_correct"),
                       ("still held (after)", "verify_correct")):
        got = rate(key)
        lines.append(f"{label:<24}  {('-' if got is None else f'{got[0]} / {got[1]}'):>22}")
    lines.append(f"{'size error':<24}  {spread_of('diameter_error_cm', 'cm'):>22}")
    lines.append(f"{'mass error':<24}  {spread_of('mass_error_g', 'g'):>22}")
    lines.append(f"{len(records)} attempts; records appended to {DEFAULT_LOG}")
    return "\n".join(lines)
