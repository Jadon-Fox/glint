#!/bin/bash
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
job22/res.sh build-fwd30-before
CMD="nvcc -arch=sm_86 -ccbin g++-12 -O2 -std=c++17 -o job22/fwd30 job22/src/fwd30.cu -lcublas -lcusolver -lcrypto"
{ echo "+ $CMD"; $CMD; echo "build_rc=$?"; } > job22/logs/build_fwd30.log 2>&1
cat job22/logs/build_fwd30.log
job22/res.sh build-fwd30-after
grep -q build_rc=0 job22/logs/build_fwd30.log
