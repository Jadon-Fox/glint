#!/bin/bash
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
J=job22
: > $J/vram_samples_gf.txt
$J/res.sh gf-ref30-before
( while true; do echo "gf_ref30 $(date -u +%T) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $J/vram_samples_gf.txt; sleep 0.5; done ) &
SAMP=$!
/usr/bin/time -v -o $J/logs/act_ref30.time $J/runpy.sh $J/src/act_ref30.py 1e-3 > $J/logs/act_ref30.out 2>&1
echo "rc=$?" >> $J/logs/act_ref30.out
kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
$J/res.sh gf-ref30-after
