#!/bin/bash
# usage: lx.sh <outfile> <timeout_s> "<bash cmd>"  -> runs in 40-RD from the harness cwd
cd /home/workspace/supervisor/scratch/p7-fold-harness
python3 lxd_exec.py /home/workspace/supervisor/scratch/p7-fold-harness "$1" "$2" -- "$3"
