#!/bin/bash
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
J=job22
mkdir -p $J/crank
: > $J/vram_samples_crank.txt
( while true; do echo "crank $(date -u +%T) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $J/vram_samples_crank.txt; sleep 0.5; done ) &
SAMP=$!
for L in 3 16 29; do
 for K in qkv o gu down; do
  N=3072; [ $K = down ] && N=8192
  $J/step.sh crank_L${L}_$K $J/p7fold2 crank --xcal $J/x/real_cal16k_L${L}_$K.f32 --seqlens $J/inputs/cal16k_seqlens.txt --n $N --label cal16k_L${L}_$K --out $J/crank/L${L}_$K.json --srcs job22/src/p7fold2.cu,job22/src/fwd30.cu
 done
done
kill $SAMP 2>/dev/null
echo finished > $J/crank/DONE
