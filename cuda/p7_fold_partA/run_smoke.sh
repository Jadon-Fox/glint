#!/bin/bash
# One SMOKE run of p7fold inside 40-Runtime-Deps. Samples VRAM every 0.5 s and records harness peak RSS (/usr/bin/time -v).
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
WEIGHTS=/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083/model-00001-of-00002.safetensors
res() { { echo "== $1 $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/gpu_mem_used,total_MiB: /'; free -m | sed -n 2p | sed 's/^/ram_MB: /'; } >> resources.log; }
res before-smoke
: > vram_samples.txt
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> vram_samples.txt; sleep 0.5; done ) &
SAMP=$!
/usr/bin/time -v ./p7fold smoke --weights "$WEIGHTS" --xcal data/x_cal_f32.bin --xeval data/x_eval_f32.bin \
  --cal-ids data/cal_ids.i32 --eval-ids data/eval_ids.i32 --arms A0,A2,A1,A3 --set-id smoke-eval-0 \
  --input-kind SMOKE-PY-REFERENCE-X --src src/p7fold.cu --out out/smoke.json --smoke > smoke_run.log 2>&1
rc=$?
kill $SAMP 2>/dev/null
wait $SAMP 2>/dev/null
res after-smoke
echo "smoke_rc=$rc" >> smoke_run.log
echo "peak_vram_MiB=$(sort -n vram_samples.txt | tail -1)" >> resources.log
exit $rc
