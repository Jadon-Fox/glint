#!/bin/bash
# SYNTHETIC weights + synthetic X only (job22/smoke inputs, read-only). Regression job22/p7fold2 vs job23/p7fold3 for A0/A1/A3/A5 and a measure-mode smoke of A4.
# Not a measurement, not round-2 data.
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
J=job23; IN=job22/smoke; SM=$J/smoke
mkdir -p $SM/old_reg $SM/new_reg $SM/dec_old $SM/dec_new $SM/old_k $SM/new_k $SM/dec_old_k $SM/dec_new_k $SM/new_d $SM/dec_new_d
: > $J/vram_samples_smoke.txt
( while true; do echo "smoke $(date -u +%T) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $J/vram_samples_smoke.txt; sleep 0.5; done ) &
SAMP=$!
N="--note SMOKE:synthetic_weights+X_regression_not_a_measurement"
OLDS=job22/src/p7fold2.cu,job22/src/fwd30.cu; NEWS=job23/src/p7fold3.cu,job22/src/fwd30.cu
$J/step.sh reg_old job22/p7fold2 measure --st1 $IN/reg_syn.safetensors --st2 $IN/reg_syn.safetensors --layer 0 --type o --pcts 2,5,10 --xcal $IN/reg_xcal.f32 --xeval $IN/reg_xeval.f32 --set-ids reg --arms A0,A1,A3,A5 --out-dir $SM/old_reg --decode-dir $SM/dec_old $N --srcs $OLDS
$J/step.sh reg_new $J/p7fold3 measure --st1 $IN/reg_syn.safetensors --st2 $IN/reg_syn.safetensors --layer 0 --type o --pcts 2,5,10 --xcal $IN/reg_xcal.f32 --xeval $IN/reg_xeval.f32 --set-ids reg --arms A0,A1,A3,A5 --out-dir $SM/new_reg --decode-dir $SM/dec_new $N --srcs $NEWS
$J/step.sh k_old job22/p7fold2 measure --synthetic-w 11 --layer 0 --type k --pcts 2,5,10 --xcal $IN/sk_xcal.f32 --xeval $IN/sk_xe1.f32,$IN/sk_xe2.f32 --set-ids s1,s2 --arms A0,A1,A3,A5 --out-dir $SM/old_k --decode-dir $SM/dec_old_k $N --srcs $OLDS
$J/step.sh k_new $J/p7fold3 measure --synthetic-w 11 --layer 0 --type k --pcts 2,5,10 --xcal $IN/sk_xcal.f32 --xeval $IN/sk_xe1.f32,$IN/sk_xe2.f32 --set-ids s1,s2 --arms A0,A1,A3,A5 --out-dir $SM/new_k --decode-dir $SM/dec_new_k $N --srcs $NEWS
$J/step.sh smoke_down_A4 $J/p7fold3 measure --synthetic-w 12 --layer 0 --type down --pcts 2,5,10 --xcal $IN/sd_xcal.f32 --xeval $IN/sd_xe1.f32 --set-ids s1 --arms A0,A1,A4,A3,A5 --out-dir $SM/new_d --decode-dir $SM/dec_new_d $N --srcs $NEWS
kill $SAMP 2>/dev/null
echo finished > $SM/DONE
