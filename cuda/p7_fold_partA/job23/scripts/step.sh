#!/bin/bash
# usage: step.sh <label> <cmd...> : res before, run under /usr/bin/time -v, res after
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
L=$1; shift
job23/res.sh "$L-before"
/usr/bin/time -v -o job23/logs/$L.time "$@" > job23/logs/$L.out 2>&1
rc=$?
echo "rc=$rc" >> job23/logs/$L.out
job23/res.sh "$L-after"
exit $rc
