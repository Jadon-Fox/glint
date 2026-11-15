#!/bin/bash
# Build and run the Mojo GLINT decoders (resident and streamed) in 40-Runtime-Deps under the GPU lease.
# Usage: scripts/p7_mojo_glint_decode.sh [steps]   (default 32; 2047 = full teacher-forced gate, about 5-7 min each)
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
STEPS="${1:-32}"
PIN="${PIN:-/home/workspace/pins/Phi-4-mini-glint-from-bf16}"
M="${MODEL_DIR:-/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083}"
A="$R/docs/p7_mode3_fold/artifacts"
FL="${GPU_LEASE:-/home/workspace/fractal-deployment/training_orchestrator/scripts/with_gpu_flock.sh}"
for k in glint_decode glint_stream_decode; do
  lxc exec 40-Runtime-Deps -- bash -c "export MODULAR_HOME=/root/.pixi/envs/modular/share/max; export PATH=/root/.pixi/envs/modular/bin:\$PATH; cd $R && mojo build -I mojo -o /tmp/p7_$k mojo/p7_$k.mojo" || exit 3
  bash "$FL" -- lxc exec 40-Runtime-Deps -- env MODULAR_HOME=/root/.pixi/envs/modular/share/max /tmp/p7_$k "$PIN" "$M" "$A" "$STEPS" | grep -E "^MOJO_GLINT_(STREAM_)?DECODE( |_POSITION)"
done
