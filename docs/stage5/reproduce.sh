#!/usr/bin/env bash
set -euo pipefail
# Reference commands only. Existing run directories are protected against overwrite.
# For independent reproduction, first adapt ROOT/output paths in a separate checkout.
/root/research/docs/stage5/run_env.sh freeze
/root/research/docs/stage5/python_env.sh -m unittest discover -s /root/research/tgppo/experiments/stage5/tests -v
for group in R0 Rprod Rjoint; do
  /root/research/docs/stage5/run_env.sh bridge --group "$group" --seed 11
done
/root/research/docs/stage5/python_env.sh /root/research/docs/stage5/bridge_audit.py
# Main scheduler executes all 9 trainings and 225 evaluations, then stops.
/root/anaconda3/bin/python /root/research/docs/stage5/orchestrate.py
# Finalizer audits, produces only the stage5 report, and stops before stage6.
/root/anaconda3/bin/python /root/research/docs/stage5/finish_stage5.py
