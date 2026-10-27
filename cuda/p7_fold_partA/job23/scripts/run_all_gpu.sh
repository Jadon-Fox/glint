#!/bin/bash
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
job23/run_crank.sh; echo "crank done rc=$?"
job23/run_ctrl_layer.sh; echo "ctrl done rc=$?"
job23/run_smoke.sh; echo "smoke done rc=$?"
echo ALLDONE
