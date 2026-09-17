#!/usr/bin/env python3
"""Estimate an object's fragility from what the robot SEES and what it FEELS, and choose
how hard to grip from that estimate -- instead of using one fixed grip for everything.

    1. SEE     gripper camera -> vision model -> fragility guess and its confidence
    2. TOUCH   after first contact, press the fingers a little further (a probe) and
               measure how far they sink and how much their load rises
    3. FUSE    combine the two, each weighted by how much it has earned trust
    4. DECIDE  squeeze past first contact, chosen inside a MEASURED safe band
    5. RECORD  one line per attempt; truth is only ever used afterwards, for scoring

Nothing in the decision is a per-object constant. The only fixed numbers are the safe
band and the probe size (fragility_config.json), and each records where it came from.

Trust is earned, not assumed:
  * vision's weight is the model's own stated confidence, scaled by vision_weight;
  * touch's weight is ZERO until fit_touch_calibration.py has fitted touch readings to
    known fragility with a good enough fit. If the simulation does not actually make soft
    objects feel soft, the fit will be poor, touch keeps zero weight, and the records say
    why -- rather than touch quietly driving the grip on noise.
If neither source has any weight, the grasp uses the tested squeeze, unchanged.

Modes (full_layer_grasp.py fragility=...):
  off  default -- none of this runs; the grasp is the tested one
  log  probe, estimate and decide, but grip with the tested squeeze; record everything
  on   grip with the decided squeeze
"""

import json
import os
import statistics
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "fragility_config.json")
CALIBRATION_PATH = os.path.join(HERE, "touch_calibration.json")
RECORDS_PATH = os.path.expanduser("~/fragility_records.jsonl")
TRUTH_PATH = os.path.join(HERE, "..", "..", "apple_gripper_sim", "fragility_truth.json")

DEFAULT_CONFIG = {
    "squeeze_min_rad": 0.08,
    "squeeze_max_rad": 0.12,
    "probe_rad": 0.02,
    "probe_settle_sim_s": 0.3,
    "vision_weight": 1.0,
    "effort_saturation_nm": 1.45,
}

# Finger segments that carry a contact sensor when the robot is built with tactile:=true.
SEGMENT_TOPICS = [f"/tactile/{finger}_{seg}"
                  for finger in ("index", "middle", "ring", "pinky", "thumb")
                  for seg in ("proximal", "middle", "tip")]


def load_json(path, fallback):
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else fallback
    except (OSError, ValueError):
        return fallback


def load_config(path=CONFIG_PATH):
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({k: v for k, v in load_json(path, {}).items() if not k.startswith("_")})
    if cfg["squeeze_min_rad"] > cfg["squeeze_max_rad"]:
        raise ValueError("fragility_config.json: squeeze_min_rad is above squeeze_max_rad")
    if cfg["probe_rad"] >= cfg["squeeze_min_rad"]:
        raise ValueError("fragility_config.json: the probe must be smaller than the "
                         "gentlest squeeze, or probing alone would over-squeeze")
    return cfg


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# --- 1. SEE --------------------------------------------------------------------------------
def vision_estimate(vlm_result, cfg):
    """The vision model's fragility (0-10) and a weight for it."""
    vlm_result = vlm_result or {}
    try:
        frag = float(vlm_result.get("fragility_score"))
    except (TypeError, ValueError):
        return {"source": "vision", "fragility": None, "confidence": 0.0,
                "note": "no usable fragility_score from the vision model"}
    try:
        conf = float(vlm_result.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    # A model that could not see (fallback answer) reports confidence 0 -> no weight.
    return {"source": "vision", "fragility": _clamp(frag, 0.0, 10.0),
            "confidence": _clamp(conf, 0.0, 1.0) * float(cfg["vision_weight"]),
            "label": vlm_result.get("object_name")}


# --- 2. TOUCH -------------------------------------------------------------------------------
def touch_features(probe_report, cfg):
    """From a probe: how far fingers sank per rad commanded (1 = moved freely, 0 = rigid),
    and how much load rose per rad they actually moved. Fingers already at the effort
    cap tell nothing and are skipped."""
    sink, stiff, used = [], [], []
    for finger, r in (probe_report or {}).items():
        moved, cmd = r.get("moved_rad"), r.get("commanded_rad")
        e0, e1 = r.get("effort_before_nm"), r.get("effort_after_nm")
        if moved is None or not cmd or e0 is None or e1 is None:
            continue
        if e0 >= cfg["effort_saturation_nm"]:
            continue
        sink.append(_clamp(moved / cmd, 0.0, 1.5))
        if moved > 1e-4:
            stiff.append((e1 - e0) / moved)
        used.append(finger)
    return {"sink_ratio": statistics.median(sink) if sink else None,
            "stiffness_nm_per_rad": statistics.median(stiff) if stiff else None,
            "fingers_used": used}


def touch_estimate(features, calibration):
    """Fragility from touch, using a fitted calibration. No calibration, or a poor fit,
    means zero weight."""
    calibration = calibration or {}
    feature = calibration.get("feature")
    value = (features or {}).get(feature) if feature else None
    if value is None or calibration.get("confidence", 0.0) <= 0.0:
        return {"source": "touch", "fragility": None, "confidence": 0.0,
                "note": ("not calibrated yet" if not calibration else
                         f"calibration unusable: {calibration.get('note', 'low confidence')}")}
    frag = calibration["a"] + calibration["b"] * value
    return {"source": "touch", "fragility": _clamp(frag, 0.0, 10.0),
            "confidence": float(calibration["confidence"]), "feature": feature,
            "value": value}


# --- 3. FUSE --------------------------------------------------------------------------------
def fuse(*estimates):
    usable = [e for e in estimates
              if e and e.get("fragility") is not None and e.get("confidence", 0.0) > 0.0]
    if not usable:
        return {"fragility": None, "confidence": 0.0, "sources": [],
                "note": "no source had any weight -- using the tested grip"}
    w = sum(e["confidence"] for e in usable)
    frag = sum(e["fragility"] * e["confidence"] for e in usable) / w
    spread = (max(e["fragility"] for e in usable) - min(e["fragility"] for e in usable)
              if len(usable) > 1 else 0.0)
    return {"fragility": frag, "confidence": min(1.0, w),
            "sources": [e["source"] for e in usable],
            "disagreement": spread}


# --- 4. DECIDE ------------------------------------------------------------------------------
def decide(fused, cfg, tested_squeeze):
    """Squeeze past first contact. More fragile -> gentler, always inside the band.

    The band's top is the tested squeeze: squeezing harder dropped the 0.70kg apple
    (0/2 at 0.16). Its bottom is the gentlest squeeze measured to still pick both a light
    and the heaviest apple. So no estimate, however wrong, can push the grip outside
    what has been shown to pick."""
    lo, hi = float(cfg["squeeze_min_rad"]), float(cfg["squeeze_max_rad"])
    if fused.get("fragility") is None:
        return {"squeeze": float(tested_squeeze), "fallback": True,
                "why": fused.get("note", "no estimate")}
    squeeze = hi - (hi - lo) * fused["fragility"] / 10.0
    return {"squeeze": _clamp(squeeze, lo, hi), "fallback": False,
            "why": (f"fragility {fused['fragility']:.1f} from "
                    f"{'+'.join(fused['sources'])}")}


# --- contact sensors (optional) -------------------------------------------------------------
class ContactMonitor:
    """Listens to the finger-segment contact sensors if they exist (tactile:=true), and
    summarises a time window: which segments touched the apple and how deep."""

    def __init__(self, node):
        self.node = node
        self.available = False
        self._window = None
        try:
            from ros_gz_interfaces.msg import Contacts
        except ImportError:
            return
        for topic in SEGMENT_TOPICS:
            node.create_subscription(Contacts, topic,
                                     lambda msg, t=topic: self._cb(t, msg), 10)
        self.available = True

    def _cb(self, topic, msg):
        if self._window is None:
            return
        seg = self._window.setdefault(topic.rsplit("/", 1)[-1],
                                      {"samples": 0, "apple_contacts": 0, "max_depth_m": 0.0})
        seg["samples"] += 1
        for c in msg.contacts:
            names = f"{c.collision1.name} {c.collision2.name}"
            if "apple" in names:
                seg["apple_contacts"] += 1
                for d in c.depths:
                    seg["max_depth_m"] = max(seg["max_depth_m"], float(d))

    def start(self):
        self._window = {}

    def stop(self):
        window, self._window = self._window or {}, None
        return window


# --- 5. RECORD and score --------------------------------------------------------------------
def record(rec, path=RECORDS_PATH):
    rec = dict(rec)
    rec["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return path
    except OSError:
        return None


def score(records, truth):
    """Accuracy of each estimate against the world's true fragility. Truth is read only
    here, after every decision has already been made."""
    rows = {"vision": [], "touch": [], "fused": []}
    picked = 0
    for r in records:
        true_f = (truth.get(r.get("apple")) or {}).get("fragility")
        picked += 1 if r.get("held") else 0
        if true_f is None:
            continue
        for key in rows:
            est = (r.get(key) or {}).get("fragility")
            if est is not None:
                rows[key].append(est - true_f)
    lines = ["", "FRAGILITY SCORECARD (truth used only for scoring)",
             f"attempts {len(records)}, picked {picked}"]
    for key, errs in rows.items():
        if errs:
            mae = sum(abs(e) for e in errs) / len(errs)
            lines.append(f"  {key:<7} n={len(errs):<3} mean abs error {mae:.1f} "
                         f"(bias {sum(errs) / len(errs):+.1f})")
        else:
            lines.append(f"  {key:<7} no estimates" +
                         (" (no truth file -- run the fragility world)" if not truth else ""))
    return "\n".join(lines)
