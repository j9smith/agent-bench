#!/usr/bin/env bash
# Cheap check, before you spend a sweep's worth of GPU-hours.
#
# The likeliest way the delegation experiment fails is not a bug: it's the model
# simply never calling DelegateTool. If it doesn't, the "on" and "off" arms are the
# same workload and you'd be sweeping noise. Same for the condenser: if tasks are
# short, it never fires.
#
# Run 2 tasks with both toggles on and look at the counts. If delegations==0, either
# raise --condenser-max-size / pick a stronger model, or accept that this model does
# not delegate and report THAT.
set -euo pipefail
python -m drivers.openhands.run_openhands \
  --num-tasks "${NUM_TASKS:-2}" --concurrency 2 \
  --delegation --condenser --condenser-max-size "${MAX_SIZE:-40}" "$@"

echo
echo "--- if 'delegate calls' is 0, do NOT sweep the delegation arm."
echo "--- if 'condensation events' is 0, lower --condenser-max-size."
