#!/bin/bash
# usage: runpy.sh <script> [args...]  (inside 40-RD). Offline, no bytecode, caches under job22/
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX=$PWD/job22/pycache HF_HOME=$PWD/job22/cache XDG_CACHE_HOME=$PWD/job22/cache
exec /opt/venv/bin/python -B -s "$@"
