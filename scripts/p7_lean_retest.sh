#!/bin/bash
# Build the lean GLINT decoder once, prove it against the recorded 35% teacher-forced result, then run the given configs.
# Usage: scripts/p7_lean_retest.sh OUTDIR "config ..." [steps]
#   config = percent[:roll]   e.g. "35 25 35:roll"   (roll = free-running rollout, 12 prompts x 32 greedy steps)
#   GLINT_* switches pass through from the caller's environment: CLASS_PCT GB FORCE_FILE ROLL_OFFSET MARGIN ALT_FILE TIME_ABL ONE_STREAM.
#   BIN overrides the output binary (default /tmp/p7_lean); SKIP_BUILD=1 reuses it.
# Each run goes through the GPU lease; logs and summary.txt go to OUTDIR. Exit 5 if the identity check fails (configs are not run).
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?outdir}"; CFGS="${2:-}"; STEPS="${3:-2047}"
PIN="${PIN:-/home/workspace/pins/Phi-4-mini-glint-from-bf16}"
M="${MODEL_DIR:-/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083}"
A="$R/docs/p7_mode3_fold/artifacts"
FL="${GPU_LEASE:-/home/workspace/fractal-deployment/training_orchestrator/scripts/with_gpu_flock.sh}"
BIN="${BIN:-/tmp/p7_lean}"
MODE=fast+attn+dsel+kmaj+vgat+skel+ksplit+fix+kgemv+ahead
RMODE=fast+attn+dsel+kmaj+vgat+roll+skel+ksplit+fix+kgemv+ahead
IDENT="MOJO_GLINT_SELECT_DECODE ppl 74.34960059115652 oracle_top1 120 / 128"
mkdir -p "$OUT"
echo "# lean retest $(date -Is) host $(hostname) bin $BIN" >> "$OUT/summary.txt"
if [ "${SKIP_BUILD:-0}" != 1 ]; then
  s=$(date +%s)
  lxc exec 40-Runtime-Deps -- bash -c "export MODULAR_HOME=/root/.pixi/envs/modular/share/max; export PATH=/root/.pixi/envs/modular/bin:\$PATH; cd $R && mojo build -I mojo -o $BIN mojo/p7_glint_lean.mojo" > "$OUT/build.log" 2>&1 || { echo "build failed" | tee -a "$OUT/summary.txt"; exit 3; }
  echo "build_s $(( $(date +%s) - s ))" >> "$OUT/summary.txt"
fi
run() {  # tag steps percent mode
  bash "$FL" -- lxc exec 40-Runtime-Deps -- env MODULAR_HOME=/root/.pixi/envs/modular/share/max GLINT_EPHEMERAL=2 \
    GLINT_CLASS_PCT="${GLINT_CLASS_PCT:-}" GLINT_GB="${GLINT_GB:-}" GLINT_FORCE_FILE="${GLINT_FORCE_FILE:-}" \
    GLINT_ROLL_OFFSET="${GLINT_ROLL_OFFSET:-0}" GLINT_MARGIN="${GLINT_MARGIN:-0}" GLINT_ALT_FILE="${GLINT_ALT_FILE:-}" \
    GLINT_TIME_ABL="${GLINT_TIME_ABL:-0}" GLINT_ONE_STREAM="${GLINT_ONE_STREAM:-0}" \
    "$BIN" "$PIN" "$M" "$A" "$2" "$3" 0 "$4" 0 0 0 > "$OUT/$1.log" 2>&1
  echo "== $1 exit $?" >> "$OUT/summary.txt"
  grep -E "^MOJO_(GLINT_SELECT_(CONFIG|DECODE steps|DECODE ppl)|ENQ)" "$OUT/$1.log" | cut -c1-230 >> "$OUT/summary.txt"
}
( unset GLINT_CLASS_PCT GLINT_GB GLINT_FORCE_FILE GLINT_ROLL_OFFSET GLINT_MARGIN GLINT_ALT_FILE GLINT_TIME_ABL GLINT_ONE_STREAM; run identity_p35_s128 128 35 "$MODE" )
if ! grep -qxF "$IDENT" "$OUT/identity_p35_s128.log"; then
  echo "IDENTITY FAILED: expected '$IDENT'" | tee -a "$OUT/summary.txt"; exit 5
fi
echo "identity ok" >> "$OUT/summary.txt"
for c in $CFGS; do
  IFS=: read -r pct r <<< "$c"
  if [ "${r:-}" = roll ]; then run "p${pct}_roll" "$STEPS" "$pct" "$RMODE"; else run "p${pct}_s${STEPS}" "$STEPS" "$pct" "$MODE"; fi
done
echo ALLDONE >> "$OUT/summary.txt"
