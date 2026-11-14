#!/bin/bash
# Retest the Mojo activation-selected decoder on an IDLE GPU. Refuses to start if the GPU is busy.
# Usage: scripts/p7_mojo_select_retest.sh OUTDIR "percent[:fixed_n[:mode[:ends:endpct]]] ..." [steps]   (ends/endpct: first and last `ends` layers select endpct percent)   (mode "ref" = reference kernels, "fast" = parallel kernels)
#   e.g. scripts/p7_mojo_select_retest.sh /home/workspace/lakes/p7_glint/mojo_select_retest "0 10 5 0:1 10:0:fast" 2047
# Each config runs once under the GPU lease; per-position logs and a summary table go to OUTDIR.
set -u
R="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?outdir}"; CFGS="${2:?configs}"; STEPS="${3:-2047}"
PIN="${PIN:-/home/workspace/pins/Phi-4-mini-glint-from-bf16}"
M="${MODEL_DIR:-/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083}"
A="$R/docs/p7_mode3_fold/artifacts"
FL="${GPU_LEASE:-/home/workspace/fractal-deployment/training_orchestrator/scripts/with_gpu_flock.sh}"
mkdir -p "$OUT"
# Idle guard. This is a desktop machine, so ambient graphics load (Xorg, gnome-shell, browsers) of a few to ~60% on
# short spikes is normal and is recorded, not refused. Refuse if (a) any other compute process holds the GPU,
# (b) a qemu VM is running (VM GPU use is invisible to nvidia-smi), or (c) mean utilization over 10 s exceeds 40%.
sum=0; n=0
for _ in 1 2 3 4 5 6 7 8 9 10; do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  sum=$((sum + ${u:-0})); n=$((n + 1)); sleep 1
done
mean=$((sum / n))
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
vms=$(pgrep -af qemu-system | wc -l)
if [ "$apps" -gt 0 ] || [ "$vms" -gt 0 ] || [ "$mean" -gt 40 ]; then
  echo "REFUSING TO RUN: compute_processes=$apps qemu_vms=$vms mean_util_10s=${mean}%. Nothing was run." | tee "$OUT/REFUSED.txt"
  exit 4
fi
{ echo "# retest $(date -Is) host $(hostname) driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"; echo "# idle guard passed: compute_processes=$apps qemu_vms=$vms mean_util_10s=${mean}% (ambient desktop graphics included)"; } >> "$OUT/summary.txt"
lxc exec 40-Runtime-Deps -- bash -c "export MODULAR_HOME=/root/.pixi/envs/modular/share/max; export PATH=/root/.pixi/envs/modular/bin:\$PATH; cd $R && mojo build -I mojo -o /tmp/p7_select_retest mojo/p7_glint_select_decode.mojo" > "$OUT/build.log" 2>&1 || { echo "build failed" | tee -a "$OUT/summary.txt"; exit 3; }
for c in $CFGS; do
  IFS=: read -r pct fx mode ends endpct <<< "$c"; fx="${fx:-0}"; mode="${mode:-ref}"; ends="${ends:-0}"; endpct="${endpct:-0}"
  tag="p${pct}_n${fx}_${mode}_e${ends}x${endpct}_s${STEPS}"
  bash "$FL" -- lxc exec 40-Runtime-Deps -- env MODULAR_HOME=/root/.pixi/envs/modular/share/max /tmp/p7_select_retest "$PIN" "$M" "$A" "$STEPS" "$pct" "$fx" "$mode" "$ends" "$endpct" > "$OUT/$tag.log" 2>&1
  echo "== $tag exit $?" >> "$OUT/summary.txt"
  grep -E "^MOJO_GLINT_SELECT_(CONFIG|DECODE steps|DECODE ppl|STATS)" "$OUT/$tag.log" | cut -c1-230 >> "$OUT/summary.txt"
done
echo ALLDONE >> "$OUT/summary.txt"
