#!/bin/bash
# usage: res.sh <label>  (run inside 40-RD). Appends step label, UTC time, GPU mem, free -m to job23/resources.log
{ echo "== $1 $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/gpu_mem_used,total_MiB: /'; free -m | sed -n 1,2p | sed 's/^/free: /'; } >> /home/workspace/supervisor/scratch/p7-fold-harness/job23/resources.log
