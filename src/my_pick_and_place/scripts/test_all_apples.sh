#!/bin/bash
# Run pocket_grasp_test.py on several apples one after another, keep each full log, and
# print a one-line-per-apple summary at the end.
#
# Usage (from this scripts directory, with the simulation already running):
#   bash test_all_apples.sh                          # apples 02-05 and 07-09, 2 attempts each
#   bash test_all_apples.sh 4 apple_02 apple_03      # chosen apples, 4 attempts each
#
# Full logs: /tmp/pocket_<apple>.log

ATTEMPTS=${1:-2}
shift 2>/dev/null
APPLES=("$@")
if [ ${#APPLES[@]} -eq 0 ]; then
    APPLES=(apple_02 apple_03 apple_04 apple_05 apple_07 apple_08 apple_09)
fi

cd "$(dirname "$0")"
for apple in "${APPLES[@]}"; do
    echo "=================== $apple ($ATTEMPTS attempts) ==================="
    # -u: unbuffered, so the script's own lines stay in order with the ROS log lines
    python3 -u pocket_grasp_test.py "$apple" "$ATTEMPTS" 2>&1 \
        | grep -v "UserWarning\|warnings.warn" | tee "/tmp/pocket_${apple}.log"
done

echo ""
echo "=================== SUMMARY ==================="
for apple in "${APPLES[@]}"; do
    picked=$(grep -h "^PICKED" "/tmp/pocket_${apple}.log" | tail -1)
    lifts=$(grep -h "grasp attempts lifted" "/tmp/pocket_${apple}.log" | tail -1 | sed 's/^ *//')
    # Match only attempt verdicts ("  2. [thumb cap on] SIM FROZE: ...") and the abort line,
    # not the legend, which names every verdict and made every apple look affected.
    frozen=$(grep -cE "\] (SIM FROZE|SIM TOO SLOW|DEAD SIM):|^  SIM FROZE: " "/tmp/pocket_${apple}.log")
    echo "$apple: ${picked:-NO RESULT (see /tmp/pocket_${apple}.log)}"
    [ -n "$lifts" ] && echo "    $lifts"
    [ "$frozen" -gt 0 ] && echo "    !! simulation problem reported -- restart Gazebo and rerun this apple"
done
