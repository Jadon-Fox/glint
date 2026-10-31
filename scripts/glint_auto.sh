#!/usr/bin/env bash
# GLINT auto: BF16 LLM → CUDA encode pin → orch course train.
# One kernel for every model. Pins change; PTX does not.
# Requires 3060 + nvcc + orch tree.
#
# Usage:
# bash scripts/glint_auto.sh --src /path/to/bf16 --course tdc_v2 --steps 8
# bash scripts/glint_auto.sh --src ./phi4-bf16 --course /abs/course.jsonl --orch /path/to/training_orchestrator
set -euo pipefail
GLINT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC=""
OUT_PIN="${GLINT_PIN_OUT:-$GLINT_ROOT/pins/last}"
COURSE="${COURSE:-tdc_v2}"
STEPS="${STEPS:-8}"
ORCH="${ORCH_ROOT:-}"
N_REC="${N_REC:-48}"
PRODUCT_LAYERS="${PRODUCT_LAYERS:-4}"
ALLOW_CPU="${ALLOW_CPU:-0}"
usage() {
  sed -n '2,12p' "$0" | sed 's/^# //'
  echo "courses: tdc_v2 tdc_v1 a b or a path to .jsonl"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="$2"; shift 2 ;;
    --out) OUT_PIN="$2"; shift 2 ;;
    --course) COURSE="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --orch) ORCH="$2"; shift 2 ;;
    --n-rec) N_REC="$2"; shift 2 ;;
    --layers) PRODUCT_LAYERS="$2"; shift 2 ;;
    --allow-cpu) ALLOW_CPU=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$SRC" ]] || { echo "--src BF16 dir or safetensors required" >&2; exit 2; }
if [[ -z "$ORCH" ]]; then
  for d in "$GLINT_ROOT/../training_orchestrator" /workspace/training_orchestrator; do
    if [[ -x "$d/scripts/run_tdc_muon_course.sh" ]]; then ORCH="$d"; break; fi
  done
fi
[[ -n "$ORCH" && -d "$ORCH" ]] || { echo "set --orch /path/to/training_orchestrator" >&2; exit 2; }
resolve_course() {
  local c="$1"
  local seeds="$ORCH/docs/curriculum/courses/seeds"
  case "$c" in
    tdc_v2|tdc|depth|tdc_depth_v2) echo "$seeds/course_tdc_depth_v2.jsonl" ;;
    tdc_v1|muon_fit) echo "$seeds/course_tdc_muon_model_fit.jsonl" ;;
    a|logic) echo "$seeds/course_a_semantic_integrity.jsonl" ;;
    b|energy) echo "$seeds/course_b_education_energy.jsonl" ;;
    *.jsonl) echo "$c" ;;
    *) echo "$c" ;;
  esac
}
SEED="$(resolve_course "$COURSE")"
[[ -f "$SEED" ]] || { echo "course not found: $SEED" >&2; exit 2; }
# --- 1. kernel (once per machine, not per model) ---
SO="$GLINT_ROOT/libglint_encode.so"
if [[ ! -f "$SO" ]]; then
  echo "=== build GLINT CUDA encode/expand (sm_86) ==="
  bash "$GLINT_ROOT/scripts/build_glint_cuda.sh"
fi
export GLINT_ENCODE_SO="$SO"
# --- 2. convert BF16 → GLINT pin (GPU) ---
echo "=== GLINT pin src=$SRC out=$OUT_PIN ==="
PIN_ARGS=( --src "$SRC" --out "$OUT_PIN" )
if [[ "$ALLOW_CPU" == 1 ]]; then PIN_ARGS+=( --allow-cpu ); fi
python3 "$GLINT_ROOT/glint_pin.py" "${PIN_ARGS[@]}"
[[ -f "$OUT_PIN/pin.json" ]] || { echo "pin.json missing" >&2; exit 4; }
python3 - <<PY
import json,sys
p=json.load(open("$OUT_PIN/pin.json"))
assert p.get("schema")=="glint_pin_v1", p
print("pin_ok", p.get("n_converted"), "backend", p.get("backend"))
PY
# --- 2b. verify the pin. VERIFY=layout (default) | parity (host re-encode byte gate + GPU expand) | off ---
VERIFY="${VERIFY:-layout}"
if [[ "$VERIFY" != off ]]; then
  V_ARGS=( --pin "$OUT_PIN" --quiet --out "$OUT_PIN/VERIFY.json" )
  if [[ "$VERIFY" == parity ]]; then V_ARGS+=( --src "$SRC" --parity --cuda ); fi
  python3 "$GLINT_ROOT/glint_verify.py" "${V_ARGS[@]}" || { echo "pin verify failed: $OUT_PIN/VERIFY.json" >&2; exit 5; }
fi
# --- 3. place + train (orch). Loader must honor ORCH_GLINT_PIN (L0 prompt). ---
export ORCH_GLINT_PIN="$OUT_PIN"
export SEED
export STEPS
export N_REC
export PRODUCT_LAYERS
export ROOT="$ORCH"
echo "=== orch course SEED=$SEED STEPS=$STEPS ORCH_GLINT_PIN=$ORCH_GLINT_PIN ==="
echo "NOTE: if orch has not wired ORCH_GLINT_PIN yet, this run is still NF4. Paste GROK_BUILD_ORCH_GLINT_PROMPT.md first."
cd "$ORCH"
bash "$ORCH/scripts/run_tdc_muon_course.sh" "$STEPS" "$N_REC" "$PRODUCT_LAYERS"
