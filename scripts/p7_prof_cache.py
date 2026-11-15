#!/usr/bin/env python3
"""Hot-channel cache sizing from a decoder selection-frequency profile (verification tooling only, not on any train/serve/eval path).

Input: a log written by p7_glint_select_decode with the `prof` mode (MOJO_PROF lines: per (half, layer, linear) the number of times
each input channel was selected; half 0 = prompts 0-5, half 1 = prompts 6-11, 95 tokens per prompt).
Method: choose cache entries (layer, linear, channel) greedily by TRAIN frequency per byte until the budget is full, then report the
share of the TEST half's gathered plug bytes that those entries would have served. Train and test are different prompts, so the
hit rate is held out. Usage: p7_prof_cache.py LOG
"""
import sys

H, INTER, QKV, LAYERS = 3072, 8192, 5120, 32
ROW_BYTES = [QKV // 2, H // 2, INTER, H // 2]  # plug bytes per selected channel for qkv, o, gate+up, down
K = [H, H, H, INTER]

counts = {}
for line in open(sys.argv[1]):
    if line.startswith("MOJO_PROF "):
        f = line.split()
        counts[int(f[1])] = list(map(int, f[2:]))
assert len(counts) == 2 * LAYERS * 4, len(counts)
TOK = 6 * 95

def g(half, layer, lin):
    return counts[(half * LAYERS + layer) * 4 + lin]

total_plug = sum(LAYERS * K[j] * ROW_BYTES[j] for j in range(4))
test_total = sum(sum(g(1, l, j)) * ROW_BYTES[j] for l in range(LAYERS) for j in range(4))
print(f"full plug {total_plug/1e9:.3f} GB; selected plug bytes per token (test half) {test_total/TOK/1e6:.1f} MB")

entries = []  # (train_freq / row_bytes, row_bytes, test_count)
for l in range(LAYERS):
    for j in range(4):
        tr, te = g(0, l, j), g(1, l, j)
        for c in range(K[j]):
            if tr[c]:
                entries.append((tr[c] / ROW_BYTES[j], ROW_BYTES[j], te[c]))
entries.sort(key=lambda e: -e[0])
print("budget MB | hit rate on held-out prompts | remaining MB/token")
cum, hit, i = 0, 0, 0
for budget in (50, 100, 200, 300, 400, 600, 800, 1200):
    while i < len(entries) and cum + entries[i][1] <= budget * 1e6:
        cum += entries[i][1]
        hit += entries[i][2] * entries[i][1]
        i += 1
    print(f"{budget:9d} | {100*hit/test_total:6.1f}% | {(test_total-hit)/TOK/1e6:8.1f}")
