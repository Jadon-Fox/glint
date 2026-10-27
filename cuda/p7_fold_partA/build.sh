#!/bin/bash
# Build p7fold inside 40-Runtime-Deps (run via lxd_exec.py). Logs the exact nvcc command and its output to build.log.
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
res() { { echo "== $1 $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/gpu_mem_used,total_MiB: /'; free -m | sed -n 2p | sed 's/^/ram_MB: /'; } >> resources.log; }
res before-build
CMD="nvcc -arch=sm_86 -ccbin g++-12 -O2 -std=c++17 -o p7fold src/p7fold.cu -lcublas -lcusolver -lcrypto"
{ echo "+ $CMD"; $CMD; echo "build_rc=$?"; } > build.log 2>&1
rc=$(grep -o 'build_rc=[0-9]*' build.log | cut -d= -f2)
res after-build
exit "${rc:-1}"
