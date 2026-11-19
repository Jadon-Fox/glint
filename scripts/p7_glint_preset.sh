#!/bin/bash
# GLINT decode preset (PROPOSED, pending the operator's lock of the landing margin delta = 0.25 logits):
# two-tier plug 12/28 with every bit-identical speed path on. Passes the landing-with-margin gate on prompt sets A, B and C
# (docs/p7_mode3_fold/GFOLD_SELECTION_2026-11-15.md); full plug 23% and 25% do not. About 17 tok/s on the RTX 3060.
# Usage: scripts/p7_glint_preset.sh BIN STEPS [MODE]   (BIN = lean decoder build, e.g. /tmp/p7_lean; MODE default teacher-forced)
set -u
BIN="${1:?bin}"; STEPS="${2:-128}"; MODE="${3:-fast+attn+dsel+kmaj+vgat+skel+ksplit+fix+kgemv+ahead}"
R="$(cd "$(dirname "$0")/.." && pwd)"
PIN="${PIN:-/home/workspace/pins/Phi-4-mini-glint-from-bf16}"
M="${MODEL_DIR:-/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083}"
FL="${GPU_LEASE:-/home/workspace/fractal-deployment/training_orchestrator/scripts/with_gpu_flock.sh}"
exec bash "$FL" -- lxc exec 40-Runtime-Deps -- env MODULAR_HOME=/root/.pixi/envs/modular/share/max GLINT_EPHEMERAL=2 \
  GLINT_PLUG_BITS=2 GLINT_TIER2_REAL=1 GLINT_TIER2_PCT=28 GLINT_TIER2_FUSED=m \
  GLINT_CORR_R=2,2,4,2 GLINT_SEL_FAST=1 GLINT_GEMV_U=4 GLINT_BIRD=1 \
  "$BIN" "$PIN" "$M" "$R/docs/p7_mode3_fold/artifacts" "$STEPS" 12 0 "$MODE" 0 0 0
