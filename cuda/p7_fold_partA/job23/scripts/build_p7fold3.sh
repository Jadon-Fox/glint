#!/bin/bash
# runs inside 40-RD. Logs to job23/build.log
cd /home/workspace/supervisor/scratch/p7-fold-harness || exit 1
job23/res.sh build-p7fold3-before
B=job23/build.log
{
echo "== build p7fold3 $(date -u +%FT%TZ)"
echo "source job22/src/p7fold2.cu sha256=$(sha256sum job22/src/p7fold2.cu | cut -d' ' -f1)"
echo "source job23/src/p7fold3.cu sha256=$(sha256sum job23/src/p7fold3.cu | cut -d' ' -f1)"
# quantizer block: from 'static const int kBlock' through end of nf4_quantize_device (inclusive)
for f in job22/src/p7fold2.cu job23/src/p7fold3.cu; do
  awk '/^static const int kBlock/{p=1} p{print} /^static void nf4_quantize_device/{q=1} q&&/^}/{exit}' $f > job23/logs/quant_$(basename $f).txt
done
echo "quantizer block lines: $(wc -l < job23/logs/quant_p7fold2.cu.txt) / $(wc -l < job23/logs/quant_p7fold3.cu.txt)"
echo "quantizer block sha256: $(sha256sum job23/logs/quant_p7fold2.cu.txt | cut -d' ' -f1) / $(sha256sum job23/logs/quant_p7fold3.cu.txt | cut -d' ' -f1)"
diff job23/logs/quant_p7fold2.cu.txt job23/logs/quant_p7fold3.cu.txt > job23/diffs/quantizer_block.diff; echo "quantizer_block_diff_rc=$? (0 = empty)"
CMD="nvcc -arch=sm_86 -ccbin g++-12 -O2 -std=c++17 -o job23/p7fold3 job23/src/p7fold3.cu -lcublas -lcusolver -lcrypto"
echo "+ $CMD"
$CMD > job23/logs/build_raw.txt 2>&1; echo "build_rc=$?"
grep -v -E "deprecated|SHA256_|note: declared|OSSL_DEPR|~~|\^~" job23/logs/build_raw.txt | head -60
echo "binary job23/p7fold3 sha256=$(sha256sum job23/p7fold3 | cut -d' ' -f1)"
} >> $B 2>&1
job23/res.sh build-p7fold3-after
tail -5 $B
