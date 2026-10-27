#!/bin/bash
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
job22/res.sh build-p7fold2-before
H=$(sha256sum src/p7fold.cu | cut -d' ' -f1)
echo "src/p7fold.cu sha256=$H" > job22/logs/build_p7fold2.log
[ "$H" = 04bfd293c1e22b54a79661aa2a0782844f9b780e715a7f19274f37d20f4c36e1 ] || { echo "HASH MISMATCH" >> job22/logs/build_p7fold2.log; exit 9; }
CMD="nvcc -arch=sm_86 -ccbin g++-12 -O2 -std=c++17 -o job22/p7fold2 job22/src/p7fold2.cu -lcublas -lcusolver -lcrypto"
{ echo "+ $CMD"; $CMD; echo "build_rc=$?"; } >> job22/logs/build_p7fold2.log 2>&1
grep -v -E "deprecated|SHA256_|note: declared|OSSL_DEPR|~~|\^~" job22/logs/build_p7fold2.log | head -60
job22/res.sh build-p7fold2-after
grep -q build_rc=0 job22/logs/build_p7fold2.log
