#!/usr/bin/env python3
"""Estimate an object's fragility from what the robot SEES and what it FEELS, and choose
how hard to grip from that estimate -- instead of using one fixed grip for everything.

    1. SEE     gripper camera -> vision model -> fragility guess and its confidence
    2. TOUCH   watch the first part of the squeeze the grasp already does: how far each
               finger actually sinks per rad it is commanded, and how its load rises.
               Read only -- no extra motion. (An earlier version pressed an extra
               0.02 rad and paused 0.3s; that dropped 3 of 10 apples which then picked
               6 of 6 without it, so touch no longer moves anything.)
    3. FUSE    combine the two, each weighted by how much it has earned trust
    4. DECIDE  squeeze past first contact, chosen inside a MEASURED safe band
    5. RECORD  one line per attempt; truth is only ever used afterwards, for scoring

Nothing in the decision is a per-object constant. The only fixed numbers are the safe
band and the probe size (fragility_config.json), and each records where it came from.

Trust is earned, not assumed:
  * vision is asked vision_votes times and the answers are voted on; its weight is ZERO
    until fit_vision_calibration.py has fitted its estimates to known fragility with a
    good enough fit (then: fit quality x how much the votes agreed). A 3B model called the
    same green apple under-ripe twice and overripe once, so its own stated confidence
    (0.8-0.9 every time) is not a usable weight;
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
VISION_CALIBRATION_PATH = os.path.join(HERE, "vision_calibration.json")
FRAMES_DIR = os.path.expanduser("~/fragility_frames")
RECORDS_PATH = os.path.expanduser("~/fragility_records.jsonl")
TRUTH_PATH = os.path.join(HERE, "..", "..", "apple_gripper_sim", "fragility_truth.json")

DEFAULT_CONFIG = {
    "squeeze_min_rad": 0.08,
    "squeeze_max_rad": 0.12,
    "observe_rad": 0.03,
    "vision_weight": 1.0,
    "vision_votes": 3,
    "effort_saturation_nm": 1.45,
}

# Finger segments that carry a contact sensor when the robot is built with tactile:=true:
# short name -> link. Ignition Fortress ignores the <topic> set in the xacro and publishes
# each on its default path (confirmed with ign topic -l), so that path is what is used.
SEGMENTS = {
    "index_proximal": "Index_Knuckle_1", "index_middle": "Index_Middle_1",
    "index_tip": "Index_Tip_1",
    "middle_proximal": "Middle_Knuckle_1", "middle_middle": "Middle_Middle_1",
    "middle_tip": "Midle_Tip_1",
    "ring_proximal": "Ring_Knuckle_1", "ring_middle": "Ring_Middle_1", "ring_tip": "Ring_Tip_1",
    "pinky_proximal": "Pinky_Knuckle_1", "pinky_middle": "Pinky_Middle_1",
    "pinky_tip": "Pinky_Tip_1",
    "thumb_proximal": "Thumb_Knuckle_1", "thumb_middle": "Thumb_Middle_1",
    "thumb_tip": "Thumb_Tip_1",
}


def contact_topic(short, link, world="apple_world", model="ur"):
    return f"/world/{world}/model/{model}/link/{link}/sensor/{short}_contact/contact"


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
    if cfg["observe_rad"] >= cfg["squeeze_min_rad"]:
        raise ValueError("fragility_config.json: observe_rad must be below the gentlest "
                         "squeeze, or the decision would come after the grip is already set")
    return cfg


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# --- 1. SEE --------------------------------------------------------------------------------
RIPENESS_PROMPT = """You are a fruit-handling assistant looking at one fruit in a robot hand.
First judge its ripeness from colour and skin: under-ripe fruit is green or pale and firm,
ripe fruit is fully coloured, overripe fruit is dark, brown or dull and soft.
Then rate how fragile it is to grip.
Respond with ONLY a valid JSON object (no markdown) with these exact keys:
{
  "object_name": "best guess at what the object is",
  "ripeness": "under-ripe", "ripe" or "overripe",
  "firmness": "firm", "medium" or "soft",
  "fragility_score": 0-10 integer, 0 = very firm, 10 = very easily bruised,
  "confidence": 0.0-1.0 float,
  "notes": "one short sentence on what you saw"
}"""


def vision_query(frame, url, model, ask=None):
    """Ask the vision model about ripeness and fragility (fragility mode only; the
    pipeline's own Layer 1 prompt is left exactly as it was).

    Why a separate prompt: asked directly for fragility, the 3B model answered 5 for all
    ten apples, including clearly green and clearly brown ones -- an error of 2.5, exactly
    what always guessing 5 gives. Ripeness from colour and skin is the cue a person uses,
    so the model is asked for that first. This is judged by the scorecard, not assumed."""
    import vlm_predict_verify as pv
    ask = ask or pv.ask_ollama
    started = time.time()
    out = ask(RIPENESS_PROMPT, pv.encode_frame(frame), url, model)
    if not isinstance(out, dict):
        out = {"error": "model did not return an object"}
    out["seconds"] = round(time.time() - started, 1)
    return out


# Word -> fragility, as the middle of the low / middle / high third of the 0-10 scale. These
# only translate the model's own three-way answers onto the scale; they say nothing about
# any particular object.
RIPENESS_SCALE = {"under-ripe": 1.7, "unripe": 1.7, "ripe": 5.0, "overripe": 8.3}
FIRMNESS_SCALE = {"firm": 1.7, "medium": 5.0, "soft": 8.3}


def save_frame(frame, apple, directory=None):
    """Save exactly the image given to the vision model, so a wrong answer can be checked
    against what it actually saw. Returns the path, or None."""
    if frame is None:
        return None
    try:
        import cv2
        directory = directory or FRAMES_DIR
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, time.strftime("%Y%m%d_%H%M%S") + f"_{apple}.jpg")
        return path if cv2.imwrite(path, frame) else None
    except Exception:
        return None


def frame_info(frame_stamp, sim_now):
    """How old the image is, in simulated seconds. Apples 04-09 were all called 'overripe'
    straight after the brown apple_03, which is what a stale image would produce."""
    age = None if (frame_stamp is None or sim_now is None) else round(sim_now - frame_stamp, 3)
    return {"frame_sim_s": frame_stamp, "sim_now_s": sim_now, "age_sim_s": age}


def _majority(values):
    values = [str(v).strip().lower() for v in values if v not in (None, "")]
    if not values:
        return None, 0.0
    best = max(set(values), key=values.count)
    return best, values.count(best) / len(values)


def vision_query_consistent(frame, url, model, n=3, ask=None):
    """Ask n times and vote. agreement = share of answers matching the winning ripeness
    (and firmness); disagreement lowers vision's weight automatically."""
    answers = [vision_query(frame, url, model, ask=ask) for _ in range(max(1, n))]
    good = [a for a in answers if "error" not in a]
    ripeness, r_agree = _majority([a.get("ripeness") for a in good])
    firmness, f_agree = _majority([a.get("firmness") for a in good])
    scores = sorted(float(a["fragility_score"]) for a in good
                    if isinstance(a.get("fragility_score"), (int, float)))
    confs = [float(a["confidence"]) for a in good
             if isinstance(a.get("confidence"), (int, float))]
    agree_parts = [x for x, v in ((r_agree, ripeness), (f_agree, firmness)) if v is not None]
    return {
        "ripeness": ripeness, "firmness": firmness,
        "fragility_score": scores[len(scores) // 2] if scores else None,
        "confidence": sum(confs) / len(confs) if confs else 0.0,
        "agreement": round(min(agree_parts), 3) if agree_parts else 0.0,
        "object_name": _majority([a.get("object_name") for a in good])[0],
        "votes": [f"{a.get('ripeness')}/{a.get('firmness')}" for a in good],
        "errors": len(answers) - len(good),
        "answers": answers,
    }


def vision_estimate(vlm_result, cfg, calibration=None):
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
    # The model's WORDS track what it sees better than its number: on a green apple (true
    # fragility 1) it answered "under-ripe", "firm" -- correct -- and fragility_score 5,
    # the same 5 it gave all ten apples. So when it gives ripeness/firmness, those set the
    # estimate (on the 0-10 scale: the low, middle and high thirds), and its number is
    # kept only as a fallback. The scorecard judges whether this helps.
    words = []
    for key, table in (("ripeness", RIPENESS_SCALE), ("firmness", FIRMNESS_SCALE)):
        value = str(vlm_result.get(key, "")).strip().lower()
        if value in table:
            words.append(table[value])
    model_score = _clamp(frag, 0.0, 10.0)
    if words:
        frag = sum(words) / len(words)
    prior = _clamp(frag, 0.0, 10.0)
    agreement = float(vlm_result.get("agreement", 1.0))
    calibration = calibration or {}
    cal_weight = float(calibration.get("confidence", 0.0))
    if cal_weight > 0.0 and conf > 0.0:
        estimate = _clamp(calibration["a"] + calibration["b"] * prior, 0.0, 10.0)
        weight = cal_weight * agreement * float(cfg["vision_weight"])
        note = f"calibrated (R^2 {calibration.get('r2', 0):.2f}), votes agreed {agreement:.2f}"
    else:
        estimate, weight = prior, 0.0
        note = ("not calibrated yet -- recorded, but given no weight" if not calibration
                else f"calibration unusable: {calibration.get('note', 'low fit')}")
    return {"source": "vision", "fragility": estimate, "prior_fragility": prior,
            "from": "ripeness/firmness words" if words else "model score",
            "model_score": model_score, "agreement": agreement, "note": note,
            "confidence": weight,
            "label": vlm_result.get("object_name"),
            "ripeness": vlm_result.get("ripeness"), "firmness": vlm_result.get("firmness")}


# --- 2. TOUCH -------------------------------------------------------------------------------
FINGERS_FOR_TOUCH = ("R_Index", "R_Middle", "R_Ring", "R_Pinky")


def touch_features_from_squeeze(samples, cfg):
    """From the squeeze samples: for the four fingers (the thumb is preloaded separately and
    behaves differently), how far each finger actually moved per rad it was commanded
    (sink: near 0 = it met something rigid, higher = the object gave way), and how much
    load rose per rad it moved (stiffness, only while below the effort cap).

    Position is not capped, so sink stays informative even when load has saturated --
    which is what left the earlier probe blind on 9 of 10 apples."""
    sink, stiff, used = [], [], set()
    for a, b in zip(samples or [], (samples or [])[1:]):
        for g in FINGERS_FOR_TOUCH:
            fa, fb = a["fingers"].get(g), b["fingers"].get(g)
            if not fa or not fb or fa.get("pos") is None or fb.get("pos") is None:
                continue
            dcmd = fb["cmd"] - fa["cmd"]
            if dcmd <= 1e-6:
                continue            # this finger was not advanced on this step
            dpos = fb["pos"] - fa["pos"]
            sink.append(_clamp(dpos / dcmd, -0.5, 1.5))
            used.add(g)
            if fa["effort"] < cfg["effort_saturation_nm"] and dpos > 1e-4:
                stiff.append((fb["effort"] - fa["effort"]) / dpos)
    return {"sink_ratio": statistics.median(sink) if sink else None,
            "stiffness_nm_per_rad": statistics.median(stiff) if stiff else None,
            "fingers_used": sorted(used), "steps": max(0, len(samples or []) - 1)}


def touch_features_from_closing(closing, cfg, load_threshold_nm=0.10):
    """Touch from the whole closing, for the four fingers, whatever the contact flag says.

    For each finger, find the first step its strongest-joint load rises above
    load_threshold_nm (it has met the apple), then over the following steps in which it
    was still being commanded forward, measure how far it actually moved per rad
    commanded (sink) and how its load rose per rad moved (stiffness, below the cap).
    A finger that never loads contributes nothing."""
    sink, stiff, used, first_touch = [], [], set(), {}
    rows = closing or []
    for g in FINGERS_FOR_TOUCH:
        touched_at = None
        for i, row in enumerate(rows):
            f = row["fingers"].get(g)
            if f and f.get("load", 0.0) > load_threshold_nm:
                touched_at = i
                break
        if touched_at is None:
            continue
        first_touch[g] = touched_at
        for a, b in zip(rows[touched_at:], rows[touched_at + 1:]):
            fa, fb = a["fingers"].get(g), b["fingers"].get(g)
            if not fa or not fb or fa.get("pos") is None or fb.get("pos") is None:
                continue
            dcmd = fb["cmd"] - fa["cmd"]
            if dcmd <= 1e-6:
                continue
            dpos = fb["pos"] - fa["pos"]
            sink.append(_clamp(dpos / dcmd, -0.5, 1.5))
            used.add(g)
            if fa["load"] < cfg["effort_saturation_nm"] and dpos > 1e-4:
                stiff.append((fb["load"] - fa["load"]) / dpos)
    return {"sink_ratio": statistics.median(sink) if sink else None,
            "stiffness_nm_per_rad": statistics.median(stiff) if stiff else None,
            "fingers_used": sorted(used), "steps": len(sink),
            "first_touch_step": first_touch}


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
    summarises a time window: which segments touched the apple and how deep.

    It runs on its OWN node, spun by its own executor in a background thread, and must
    never subscribe on the grasp node. The grasp waits a fixed number of incoming messages
    per closing step; 15 contact topics at 50Hz on that node made each step 3.1ms of
    simulated time instead of 8.5ms, so the fingers closed about 2.7x faster than in the
    tested grasp and read stale positions -- and apples that pick without the monitor
    dropped with it."""

    def __init__(self, node=None):
        self.available = False
        self._window = None
        self._lock = None
        try:
            import threading
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from ros_gz_interfaces.msg import Contacts
        except ImportError:
            return
        self._lock = threading.Lock()
        self._node = Node("fragility_contact_monitor")
        for short, link in SEGMENTS.items():
            self._node.create_subscription(Contacts, contact_topic(short, link),
                                           lambda msg, n=short: self._cb(n, msg), 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._spin, daemon=True,
                                        name="fragility_contact_monitor")
        self._thread.start()
        self.available = True

    def _spin(self):
        try:
            self._executor.spin()
        except Exception:
            # rclpy shutting down underneath the executor at program exit; nothing to do.
            pass

    def close(self):
        """Stop the background thread before rclpy shuts down."""
        if not self.available:
            return
        self.available = False
        try:
            self._executor.shutdown(timeout_sec=2.0)
            self._node.destroy_node()
        except Exception:
            pass
        self._thread.join(timeout=3.0)

    def _cb(self, short, msg):
        with self._lock:
            if self._window is None:
                return
            seg = self._window.setdefault(short, {"samples": 0, "apple_contacts": 0,
                                                  "max_depth_m": 0.0})
            seg["samples"] += 1
            for c in msg.contacts:
                names = f"{c.collision1.name} {c.collision2.name}"
                if "apple" in names:
                    seg["apple_contacts"] += 1
                    for d in c.depths:
                        seg["max_depth_m"] = max(seg["max_depth_m"], float(d))

    def start(self):
        if self._lock is None:
            return
        with self._lock:
            self._window = {}

    def stop(self):
        if self._lock is None:
            return {}
        with self._lock:
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
    rows = {"vision": [], "touch": [], "fused": [], "baseline": []}
    picked = 0
    for r in records:
        true_f = (truth.get(r.get("apple")) or {}).get("fragility")
        picked += 1 if r.get("held") else 0
        if true_f is None:
            continue
        # "baseline" = always guessing the middle of the scale; vision must beat it.
        rows["baseline"].append(5.0 - true_f)
        for key in ("vision", "touch", "fused"):
            block = r.get(key) or {}
            est = block.get("prior_fragility", block.get("fragility")) if key == "vision" \
                else block.get("fragility")
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
