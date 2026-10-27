#!/bin/bash
# SYNTHETIC weights + synthetic X only. Lineage regression (old 04bfd293 binary vs p7fold2) and measure-mode smoke.
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
J=job22; SM=$J/smoke; SRCS=job22/src/p7fold2.cu,job22/src/fwd30.cu
mkdir -p $SM/out_k $SM/dec_k $SM/out_d $SM/dec_d $SM/out_reg $SM/dec_reg
: > $J/vram_samples_smoke.txt
( while true; do echo "smoke $(date -u +%T) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> $J/vram_samples_smoke.txt; sleep 0.5; done ) &
SAMP=$!
$J/step.sh reg_old job20/p7fold measure --weights $SM/reg_syn.safetensors --xcal $SM/reg_xcal.f32 --xeval $SM/reg_xeval.f32 --cal-ids $SM/reg_cal.i32 --eval-ids $SM/reg_eval.i32 --arms A0,A1,A3 --set-id reg --input-kind SYNTHETIC --src src/p7fold.cu --out $SM/reg_old.json --smoke
$J/step.sh reg_new $J/p7fold2 measure --st1 $SM/reg_syn.safetensors --st2 $SM/reg_syn.safetensors --layer 0 --type o --pcts 5 --xcal $SM/reg_xcal.f32 --xeval $SM/reg_xeval.f32 --set-ids reg --arms A0,A1,A3 --out-dir $SM/out_reg --decode-dir $SM/dec_reg --note "SMOKE: synthetic weights+X, lineage regression, not a measurement" --srcs $SRCS
$J/step.sh smoke_k $J/p7fold2 measure --synthetic-w 11 --layer 0 --type k --pcts 2,5,10 --xcal $SM/sk_xcal.f32 --xeval $SM/sk_xe1.f32,$SM/sk_xe2.f32 --set-ids s1,s2 --arms A0,A1,A4,A3,A5 --out-dir $SM/out_k --decode-dir $SM/dec_k --note "SMOKE: synthetic weights+X, not a measurement" --srcs $SRCS
$J/step.sh smoke_down $J/p7fold2 measure --synthetic-w 12 --layer 0 --type down --pcts 2,5,10 --xcal $SM/sd_xcal.f32 --xeval $SM/sd_xe1.f32 --set-ids s1 --arms A0,A1,A4,A3,A5 --out-dir $SM/out_d --decode-dir $SM/dec_d --note "SMOKE: synthetic weights+X, not a measurement" --srcs $SRCS
kill $SAMP 2>/dev/null
echo finished > $SM/DONE
