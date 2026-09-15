#!/bin/bash
# Run the grasp test, or the full pick-and-place, on several apples one after another,
# keep each full log, and print a one-line-per-apple summary at the end.
#
# Usage (from this scripts directory, with the simulation already running):
#   bash test_all_apples.sh                          # grasp test: apples 02-05, 07-09, 2 attempts each
#   bash test_all_apples.sh 4 apple_02 apple_03      # grasp test: chosen apples, 4 attempts each
#   bash test_all_apples.sh full                     # full pick-and-place: all 10 apples, 2 attempts each
#   bash test_all_apples.sh full 1 apple_06          # full pick-and-place: chosen apples
#
# Full logs: /tmp/pocket_<apple>.log (grasp test) or /tmp/pickplace_<apple>.log (full)

MODE=grasp
if [ "$1" = "full" ]; then
    MODE=full
    shift
fi

ATTEMPTS=${1:-2}
shift 2>/dev/null
APPLES=("$@")
if [ ${#APPLES[@]} -eq 0 ]; then
    if [ "$MODE" = full ]; then
        APPLES=(apple_01 apple_02 apple_03 apple_04 apple_05 apple_06 apple_07 apple_08 apple_09 apple_10)
    else
        APPLES=(apple_02 apple_03 apple_04 apple_05 apple_07 apple_08 apple_09)
    fi
fi

# Match only attempt verdicts ("  2. [thumb cap on] SIM FROZE: ...") and the abort line,
# not the legend, which names every verdict and made every apple look affected.
FROZEN_RE='\] (SIM FROZE|SIM TOO SLOW|DEAD SIM):|^  SIM FROZE: '

cd "$(dirname "$0")"
RAN=()
STOPPED_AT=""
for apple in "${APPLES[@]}"; do
    echo "=================== $apple ($MODE, $ATTEMPTS attempts) ==================="
    # -u: unbuffered, so the script's own lines stay in order with the ROS log lines
    if [ "$MODE" = full ]; then
        python3 -u full_layer_grasp.py "$apple" "$ATTEMPTS" 2>&1 \
            | grep -v "UserWarning\|warnings.warn" | tee "/tmp/pickplace_${apple}.log"
    else
        python3 -u pocket_grasp_test.py "$apple" "$ATTEMPTS" 2>&1 \
            | grep -v "UserWarning\|warnings.warn" | tee "/tmp/pocket_${apple}.log"
    fi
    RAN+=("$apple")
    # A frozen simulation stays broken: on 15 Sep apple_03 and apple_04 ran straight after
    # apple_02's freeze and failed (apple knocked 29-37mm, rolled 39-160mm), then picked
    # 4 of 4 each after a restart. Stop here rather than record meaningless results.
    log="/tmp/pocket_${apple}.log"
    [ "$MODE" = full ] && log="/tmp/pickplace_${apple}.log"
    if [ "$(grep -cE "$FROZEN_RE" "$log")" -gt 0 ]; then
        STOPPED_AT="$apple"
        break
    fi
done

echo ""
echo "=================== SUMMARY ==================="
for apple in "${RAN[@]}"; do
    if [ "$MODE" = full ]; then
        line=$(grep -h "^${apple}: picked" "/tmp/pickplace_${apple}.log" | tail -1)
        echo "${line:-$apple: NO RESULT (see /tmp/pickplace_${apple}.log)}"
    else
        picked=$(grep -h "^PICKED" "/tmp/pocket_${apple}.log" | tail -1)
        lifts=$(grep -h "grasp attempts lifted" "/tmp/pocket_${apple}.log" | tail -1 | sed 's/^ *//')
        frozen=$(grep -cE "$FROZEN_RE" "/tmp/pocket_${apple}.log")
        echo "$apple: ${picked:-NO RESULT (see /tmp/pocket_${apple}.log)}"
        [ -n "$lifts" ] && echo "    $lifts"
        [ "$frozen" -gt 0 ] && echo "    !! simulation problem reported -- restart Gazebo and rerun this apple"
    fi
done

if [ -n "$STOPPED_AT" ]; then
    NOT_RUN=()
    seen=0
    for apple in "${APPLES[@]}"; do
        [ "$seen" = 1 ] && NOT_RUN+=("$apple")
        [ "$apple" = "$STOPPED_AT" ] && seen=1
    done
    echo ""
    echo "!! STOPPED after $STOPPED_AT: the simulation froze. Restart Gazebo (Terminal 1), then rerun:"
    if [ "$MODE" = full ]; then
        echo "   bash test_all_apples.sh full $ATTEMPTS $STOPPED_AT ${NOT_RUN[*]}"
    else
        echo "   bash test_all_apples.sh $ATTEMPTS $STOPPED_AT ${NOT_RUN[*]}"
    fi
fi
