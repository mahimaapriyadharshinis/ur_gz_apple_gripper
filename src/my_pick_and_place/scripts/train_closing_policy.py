#!/usr/bin/env python3
"""
Small verification run for learned finger-closing.

This is deliberately NOT a full RL training run -- it's a small, fast check that
the whole loop actually works end to end before committing to a long one: does the
environment (one real grasp attempt in Gazebo) reset cleanly each time, does the
reward function score attempts the way we'd expect by eye, and does even a simple
search show ANY improvement signal over a handful of tries.

Method: a basic (1+lambda) evolution strategy, entirely dependency-free (just
numpy, already a dependency). This is a legitimate, standard black-box
optimization technique -- CMA-ES's simpler cousin -- well suited to a LOW-dimensional
parameter search (10 numbers here) with very few, expensive-to-evaluate samples,
which is exactly this situation: each "sample" is a real ~60-120 second grasp
attempt in the simulator, not a cheap function call. A full deep RL algorithm
(SAC/PPO) would show no learning signal at all in this few attempts -- those need
thousands of episodes, which isn't realistic to verify in one sitting.

Policy representation: for each of the 5 finger groups, two numbers --
  speed:            multiplies CLOSE_STEP_SIZE for that finger (how fast it closes)
  threshold_offset: added to the base contact-force threshold for that finger
Starting point is the NEUTRAL vector (speed=1.0, threshold_offset=0.0 for every
finger), which reproduces the existing fixed-schedule behavior exactly -- so the
very first evaluation in this run doubles as a sanity check that threading
policy_params through the pipeline didn't change anything when it shouldn't.

Usage:
  python3 train_closing_policy.py [target_name] [--generations N] [--lambda N]

Every attempt's params and reward are appended to closing_policy_runs.json as it
runs, so progress survives even if the run is interrupted.
"""
import json
import os
import sys
import time

import numpy as np
import rclpy

from full_layer_grasp import FINGER_GROUPS, FullLayerGraspNode

RUN_LOG = os.path.expanduser("~/ur_gz_ws/closing_policy_runs.json")

# (low, high) search bounds per parameter, and the neutral (= current fixed-schedule
# behavior) starting value.
SPEED_BOUNDS = (0.3, 2.5)
SPEED_NEUTRAL = 1.0
THRESH_BOUNDS = (-0.08, 0.15)
THRESH_NEUTRAL = 0.0

PARAM_KEYS = []
for _g in FINGER_GROUPS:
    PARAM_KEYS.append((_g, 'speed', SPEED_BOUNDS, SPEED_NEUTRAL))
    PARAM_KEYS.append((_g, 'threshold_offset', THRESH_BOUNDS, THRESH_NEUTRAL))


def neutral_vector():
    return np.array([neutral for _, _, _, neutral in PARAM_KEYS], dtype=float)


def vector_to_policy_params(vec):
    params = {g: {} for g in FINGER_GROUPS}
    for (g, key, bounds, _), value in zip(PARAM_KEYS, vec):
        lo, hi = bounds
        params[g][key] = float(np.clip(value, lo, hi))
    return params


def mutate(vec, sigma):
    ranges = np.array([hi - lo for _, _, (lo, hi), _ in PARAM_KEYS])
    noise = np.random.normal(0.0, sigma, size=vec.shape) * ranges
    mutated = vec + noise
    for i, (_, _, (lo, hi), _) in enumerate(PARAM_KEYS):
        mutated[i] = np.clip(mutated[i], lo, hi)
    return mutated


def append_run_log(entry):
    history = []
    if os.path.exists(RUN_LOG):
        try:
            with open(RUN_LOG) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(entry)
    with open(RUN_LOG, "w") as f:
        json.dump(history, f, indent=2)


def evaluate(node, target_name, vec, label):
    params = vector_to_policy_params(vec)
    print(f"\n--- {label}: params={ {g: {k: round(v, 3) for k, v in d.items()} for g, d in params.items()} } ---")
    start = time.time()
    result = node.run_for_target(target_name, closing_policy_params=params)
    elapsed = time.time() - start
    reward = result.get("reward", float("-inf"))
    print(f"--- {label}: reward={reward:.2f}  success={result.get('success')}  "
          f"({elapsed:.1f}s) ---")
    append_run_log({
        "label": label,
        "target": target_name,
        "params": params,
        "reward": reward,
        "success": result.get("success"),
        "elapsed_sec": elapsed,
        "timestamp": time.time(),
    })
    return reward


def main():
    args = sys.argv[1:]
    target_name = "apple_06"
    generations = 3
    lam = 3
    i = 0
    positional = []
    while i < len(args):
        if args[i] == "--generations":
            generations = int(args[i + 1]); i += 2
        elif args[i] == "--lambda":
            lam = int(args[i + 1]); i += 2
        else:
            positional.append(args[i]); i += 1
    if positional:
        target_name = positional[0]

    total_attempts = 1 + generations * lam
    print(f"Small verification run: {total_attempts} total attempts "
          f"(1 baseline + {generations} generations x {lam} candidates) "
          f"on {target_name}. Logging to {RUN_LOG}")

    rclpy.init()
    node = FullLayerGraspNode()

    best_vec = neutral_vector()
    best_reward = evaluate(node, target_name, best_vec, "baseline (neutral params)")

    sigma = 0.25
    for gen in range(generations):
        print(f"\n=== Generation {gen + 1}/{generations} (sigma={sigma:.3f}) ===")
        for c in range(lam):
            candidate = mutate(best_vec, sigma)
            reward = evaluate(node, target_name, candidate,
                               f"gen{gen + 1} candidate{c + 1}")
            if reward > best_reward:
                print(f"  -> new best! {reward:.2f} > {best_reward:.2f}")
                best_reward, best_vec = reward, candidate
        sigma *= 0.8  # narrow the search a bit each generation

    node.destroy_node()
    rclpy.shutdown()

    print("\n=== VERIFICATION SUMMARY ===")
    with open(RUN_LOG) as f:
        history = json.load(f)
    baseline_reward = history[0]["reward"]
    print(f"Baseline reward:        {baseline_reward:.2f}")
    print(f"Best reward found:      {best_reward:.2f}")
    print(f"Improvement over baseline: {best_reward - baseline_reward:+.2f}")
    print(f"Best params: {vector_to_policy_params(best_vec)}")
    print(f"\nFull history in {RUN_LOG}")


if __name__ == "__main__":
    main()
