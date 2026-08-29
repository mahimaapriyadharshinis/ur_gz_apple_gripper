#!/usr/bin/env python3
"""
One-off diagnostic: is a stable 3+ finger grasp physically achievable at all with
the current hand geometry, given the most forgiving closing behavior we can
reasonably justify -- BEFORE trusting an automated search (train_closing_policy.py)
to find good parameters for it.

Across every attempt so far this project, 3+ finger contact has never happened once.
If it still doesn't happen here, that's evidence the blocker isn't closing speed or
contact-threshold tuning (which is all the ES search in train_closing_policy.py can
adjust) -- it points to something structural: finger span vs. apple size, contact
detection miscalibrated, or the approach position/pre-shape not actually surrounding
the apple before closing starts. If it DOES work here, that tells us positioning is
fine and the small ES run just needs to find its way to similar values.

"Generous" here means, concretely:
  speed=0.4            -- much slower than the default (1.0), so a leading finger's
                           momentum doesn't shove the apple away before the other
                           fingers catch up.
  threshold_offset=-0.05 -- lower than default (0.0), so a finger reports "contact"
                           and stops advancing as soon as it feels a SMALL amount of
                           resistance, instead of continuing to push through it.
Both are within the same bounds train_closing_policy.py searches (speed 0.3-2.5,
threshold_offset -0.08 to 0.15) -- this is the generous end of that same range, not
a separate untested regime.

Usage: python3 diagnostic_grasp_attempt.py [target_name]
"""
import sys

import rclpy

from full_layer_grasp import FINGER_GROUPS, FullLayerGraspNode

GENEROUS_PARAMS = {g: {'speed': 0.4, 'threshold_offset': -0.05} for g in FINGER_GROUPS}


def main():
    target_name = sys.argv[1] if len(sys.argv) > 1 else "apple_06"
    print(f"Diagnostic: one generous, slow, forgiving grasp attempt on {target_name}.")
    print(f"Params: {GENEROUS_PARAMS}")

    rclpy.init()
    node = FullLayerGraspNode()
    result = node.run_for_target(target_name, closing_policy_params=GENEROUS_PARAMS)
    node.destroy_node()
    rclpy.shutdown()

    print("\n=== DIAGNOSTIC RESULT ===")
    print(result)
    n_contacted = sum(result.get("contacted", {}).values())
    if result.get("reason"):
        print(f"\nAttempt aborted: {result['reason']} -- fix the environment and re-run "
              f"before drawing any conclusion.")
    elif n_contacted >= 3:
        print(f"\n{n_contacted}/5 fingers contacted -- a stable grasp IS achievable "
              f"with this geometry. Positioning is fine; the small ES search just "
              f"needs to find values like these on its own.")
    else:
        print(f"\nOnly {n_contacted}/5 fingers contacted even with generous, forgiving "
              f"settings -- this points to a structural problem (finger span vs. "
              f"apple size, contact threshold, or approach position/pre-shape), not "
              f"a closing-schedule tuning problem. Closing-speed search alone won't "
              f"fix this.")


if __name__ == '__main__':
    main()
