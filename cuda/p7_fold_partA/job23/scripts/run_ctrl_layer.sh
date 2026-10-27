#!/bin/bash
# Step 5: layer-only negative-control captures (Amendment B-3). Existing job22/fwd30 (sha 6f4f043b), no rebuild. One GPU job at a time.
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
J=job23
S=/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083
SETS="cal16k:job22/inputs/cal16k.i32:job22/inputs/cal16k_seqlens.txt"
for i in 1 2 3 4 5; do SETS="$SETS;eval_$i:job20/inputs/eval_$i.i32:job20/inputs/eval_${i}_seqlens.txt"; done
mkdir -p $J/ctrl_layer
: > $J/vram_samples_ctrl.txt
: > $J/logs/ctrl_layer_rc.txt
run() { # variant kind layer
  V=$1; K=$2; L=$3; T=$V-L$L
  $J/res.sh "ctrl-$T-before"
  ( while true; do echo "$T $(date -u +%T) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $J/vram_samples_ctrl.txt; sleep 0.5; done ) &
  SAMP=$!
  /usr/bin/time -v -o $J/logs/ctrl-$T.time job22/fwd30 --st1 $S/model-00001-of-00002.safetensors --st2 $S/model-00002-of-00002.safetensors \
     --sets "$SETS" --src job22/src/fwd30.cu --variant $V --variant-layer $L --nlayers $((L+1)) --capture-layers $L --capture-kinds $K \
     --out-prefix $J/ctrl_layer/$V > $J/logs/ctrl-$T.out 2>&1
  rc=$?
  kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
  echo "$T rc=$rc" >> $J/logs/ctrl_layer_rc.txt
  $J/res.sh "ctrl-$T-after"
}
for L in 3 16 29; do
  run rope128 o $L
  run gqamod o $L
  run gateup_swap down $L
done
echo finished >> $J/logs/ctrl_layer_rc.txt
