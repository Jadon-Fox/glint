#!/bin/bash
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
J=job22
$J/step.sh ga1dump $J/p7fold2 ga1dump --st1 /opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083/model-00001-of-00002.safetensors --st2 /opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083/model-00002-of-00002.safetensors --layers 3,16,29 --outdir $J/ga1 --out $J/ga1_harness.json --srcs job22/src/p7fold2.cu,job22/src/fwd30.cu
$J/step.sh ga1check $J/runpy.sh $J/src/ga1_check.py
