#!/bin/bash
# Re-run the NF4 parity step with resource snapshots (same inputs; parity.json is regenerated).
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
res() { { echo "== $1 $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/gpu_mem_used,total_MiB: /'; free -m | sed -n 2p | sed 's/^/ram_MB: /'; } >> resources.log; }
res before-parity
/opt/venv/bin/python src/parity_prep.py make > out/parity_make.txt 2>&1 && \
./p7fold parity data/parity_in.f64 data/parity_cuda_packed.bin data/parity_cuda_absmax.f64 >> out/parity_make.txt 2>&1 && \
/opt/venv/bin/python src/parity_prep.py cmp >> out/parity_make.txt 2>&1
rc=$?
res after-parity
exit $rc
