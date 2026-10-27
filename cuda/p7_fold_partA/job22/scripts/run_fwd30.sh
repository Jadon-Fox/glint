#!/bin/bash
# runs fwd30 for the real path (captures at 3,16,29, residual dump) and the three negative controls, one GPU job at a time.
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
J=job22
S=/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083
SETS="cal16k:$J/inputs/cal16k.i32:$J/inputs/cal16k_seqlens.txt"
for i in 1 2 3 4 5; do SETS="$SETS;eval_$i:job20/inputs/eval_$i.i32:job20/inputs/eval_${i}_seqlens.txt"; done
mkdir -p $J/x $J/ctrl $J/resid
: > $J/vram_samples.txt
run() { # label, extra args...
  L=$1; shift
  $J/res.sh "fwd30-$L-before"
  ( while true; do echo "$L $(date -u +%T) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $J/vram_samples.txt; sleep 0.5; done ) &
  SAMP=$!
  /usr/bin/time -v -o $J/logs/fwd30-$L.time $J/fwd30 --st1 $S/model-00001-of-00002.safetensors --st2 $S/model-00002-of-00002.safetensors \
     --sets "$SETS" --src $J/src/fwd30.cu "$@" > $J/logs/fwd30-$L.out 2>&1
  rc=$?
  kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
  echo "$L rc=$rc" >> $J/logs/fwd30_rc.txt
  $J/res.sh "fwd30-$L-after"
}
: > $J/logs/fwd30_rc.txt
run real --variant real --out-prefix $J/x/real --resid-prefix $J/resid/real --capture-layers 3,16,29 --capture-kinds qkv,o,gu,down
run rope128 --variant rope128 --out-prefix $J/ctrl/rope128 --capture-layers 3,16,29 --capture-kinds o
run gqamod --variant gqamod --out-prefix $J/ctrl/gqamod --capture-layers 3,16,29 --capture-kinds o
run gateup_swap --variant gateup_swap --out-prefix $J/ctrl/gateup_swap --capture-layers 3,16,29 --capture-kinds down
echo finished >> $J/logs/fwd30_rc.txt
