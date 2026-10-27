#!/bin/bash
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
J=job23
: > $J/vram_samples_ga.txt
( while true; do echo "ga $(date -u +%T) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $J/vram_samples_ga.txt; sleep 0.5; done ) &
SAMP=$!
$J/step.sh ga $J/p7fold3 ga --st1 /opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083/model-00001-of-00002.safetensors --st2 /opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083/model-00002-of-00002.safetensors --outdir $J/ga2 --out $J/ga_apparatus.json --srcs job23/src/p7fold3.cu,job22/src/fwd30.cu
kill $SAMP 2>/dev/null
