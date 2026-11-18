# Project 7 Mojo/MAX activation-selected plug decode: parent (fold) GEMV + current-|x| top-k plug correction. GLINT #117.
# Reads the K-major plug rows of the selected input channels straight from pinned host memory (zero-copy gather kernel),
# applies y += sum_c (cell[h][p]-parent[h]) * absmax * x[c], summing in selected-channel order (deterministic).
# Usage: mojo build -I mojo -o p7_glint_select_decode mojo/p7_glint_select_decode.mojo
#        ./p7_glint_select_decode GLINT_PIN_DIR MODEL_DIR ARTIFACTS_DIR [steps] [percent] [fixed_n]
#        percent 0..100 of input channels per linear; fixed_n > 0 overrides it with that many channels per linear
#        7th arg "fast" (or "fast+attn", which also uses the position-parallel attention kernel) selects the parallel norm / logits / cross-entropy and vectorised GEMV kernels (summation order differs from the
#        reference kernels; validate on the recorded gate). Default = reference kernels.
# percent 0 = fold only; 100 = every channel (differs from the resident full-cell GEMV only in summation order).
# Selection rule mirrors the CUDA act rule: n = ceil(percent*K/100); keep every channel whose |x| key >= the n-th largest key
# (ties are all kept). Selection runs on the host after a per-linear device-to-host copy of the linear's input (a reference
# implementation, NOT the fast path). Teacher-forced scoring only; not M6.
from std.math import ceildiv, exp, log, sqrt, cos, sin, pow
from std.sys import argv
from std.os import getenv
from std.time import perf_counter_ns
from std.gpu import global_idx, thread_idx, block_idx
from std.memory import stack_allocation
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext, DeviceBuffer
from p7_safetensors import parse_header, STEntry
from p7_forward import bf16r, embed_gather, rms_norm_bf16, rope, residual_add, swiglu, ce_argmax, read_range
from p7_glint_linear import glint_gemv_warp
from p7_glint_gemv_fast import glint_gemv_fast
from p7_glint_gemv_km import glint_gemv_km, gemv_km_combine, GTB
from p7_decode_fast_kernels import rms_norm_block, logits_warp, ce_argmax_block, margin_block
from p7_attention_split import attention_split_partial, attention_split_combine, NCH
from p7_select_device import select_topk, gather_rows_d, apply_corr_d, apply_corr_k, transpose_scales, gather_rows_d4, apply_corr_ks, combine_corr, KS, accum_hist, gather_rows_p4, GB, overlap_count, overlap_probe, overlap_age, gather_rows_fix, apply_corr_ksc, apply_corr_kse, pool_lookup, pool_free, pool_assign, apply_corr_ksp

comptime CTX = 2048
comptime NT = CTX - 1
comptime H = 3072
comptime INTER = 8192
comptime NQ = 24
comptime NKV = 8
comptime HD = 128
comptime ROT = 96
comptime QKV = 5120
comptime V = 200064
comptime LAYERS = 32
comptime AHEAD_LAG = 2
comptime AHEAD_RING = 4
comptime SCALE = Float32(1.1902380714238083)

# K/V are canonical state for this layer. Each position is written exactly once
# before any attention kernel consumes it. The early-preview/prefetch path will
# have separate scratch storage and never write to these buffers.
def kv_write[
    ro: ImmOrigin, ko: MutOrigin, vo: MutOrigin
](
    rotated: ImmPointer[Float32, ro],
    cache_k: MutPointer[Float32, ko],
    cache_v: MutPointer[Float32, vo],
    position: Int32,
):
    var i = global_idx.x
    if i >= NKV * HD:
        return
    var dst = Int(position) * NKV * HD + i
    cache_k[unsafe_offset=dst] = rotated[unsafe_offset=NQ * HD + i]
    cache_v[unsafe_offset=dst] = rotated[unsafe_offset=(NQ + NKV) * HD + i]

# Same online-softmax reduction order as p7_forward.attention, but consumes
# past K/V from a persistent per-layer cache rather than a full T x QKV buffer.
def attention_one[
    ro: ImmOrigin, ko: ImmOrigin, vo: ImmOrigin, ao: MutOrigin, bo: MutOrigin
](
    rotated: ImmPointer[Float32, ro],
    cache_k: ImmPointer[Float32, ko],
    cache_v: ImmPointer[Float32, vo],
    dst: MutPointer[Float32, ao],
    out_bf16: MutPointer[BFloat16, bo],
    position: Int32,
):
    var hq = block_idx.y
    var d = thread_idx.x
    var hkv = hq // 3
    var s = stack_allocation[HD, Float32, address_space=AddressSpace.SHARED]()
    var qv = rotated[unsafe_offset=hq * HD + d] * (Float32(1) / sqrt(Float32(HD)))
    var mv = Float32(-1e30)
    var dv = Float32(0)
    var acc = Float32(0)
    for tk in range(Int(position) + 1):
        s[unsafe_offset=d] = qv * cache_k[unsafe_offset=(tk * NKV + hkv) * HD + d]
        barrier()
        if d < 64:
            s[unsafe_offset=d] = s[unsafe_offset=d] + s[unsafe_offset=d + 64]
        barrier()
        if d < 32:
            s[unsafe_offset=d] = s[unsafe_offset=d] + s[unsafe_offset=d + 32]
        barrier()
        comptime for step in range(5):
            comptime off = 16 >> step
            if d < off:
                s[unsafe_offset=d] = s[unsafe_offset=d] + s[unsafe_offset=d + off]
            barrier()
        var score = s[unsafe_offset=0]
        var mn = score if score > mv else mv
        var alpha = exp(mv - mn)
        var beta = exp(score - mn)
        dv = dv * alpha + beta
        mv = mn
        acc = acc * alpha + bf16r(beta) * cache_v[unsafe_offset=(tk * NKV + hkv) * HD + d]
        barrier()
    var o = acc / dv if dv > Float32(0) else Float32(0)
    dst[unsafe_offset=hq * HD + d] = o
    out_bf16[unsafe_offset=hq * HD + d] = o.cast[DType.bfloat16]()

# A single-token output row; identical expression to p7_forward.logits_gemm.
def logits_one[
    xo: ImmOrigin, eo: ImmOrigin, lo: MutOrigin
](
    hidden: ImmPointer[BFloat16, xo],
    emb: ImmPointer[BFloat16, eo],
    logits: MutPointer[Float32, lo],
):
    var v = global_idx.x
    if v >= V:
        return
    var acc = Float32(0)
    for k in range(H):
        acc += hidden[unsafe_offset=k].cast[DType.float32]() * emb[unsafe_offset=v * H + k].cast[DType.float32]()
    logits[unsafe_offset=v] = acc

def transpose_plug[
    io: ImmOrigin, oo: MutOrigin
](src: ImmPointer[UInt8, io], dst: MutPointer[UInt8, oo], n: Int32, kdim: Int32):
    # src: row-major nibbles, flattened j = row*K + k, element j at byte j>>1, low nibble first.
    # dst: K-major, row c holds the n plug nibbles of input channel c; element r at byte r>>1, low nibble first.
    var g = global_idx.x
    var half = Int(n) // 2
    if g >= Int(kdim) * half:
        return
    var c = g // half
    var b = g - c * half
    var r0 = 2 * b
    var j0 = r0 * Int(kdim) + c
    var j1 = (r0 + 1) * Int(kdim) + c
    var n0 = (src[unsafe_offset=j0 >> 1] >> UInt8(4 * (j0 & 1))) & UInt8(15)
    var n1 = (src[unsafe_offset=j1 >> 1] >> UInt8(4 * (j1 & 1))) & UInt8(15)
    dst[unsafe_offset=g] = n0 | (n1 << UInt8(4))

def make_tables[
    co: ImmOrigin, po: ImmOrigin, to: MutOrigin, dq: MutOrigin
](cells: ImmPointer[Float32, co], parent: ImmPointer[Float32, po], t0: MutPointer[Float32, to], dlt: MutPointer[Float32, dq]):
    var i = global_idx.x
    if i >= 256:
        return
    var h = i // 16
    t0[unsafe_offset=i] = parent[unsafe_offset=h]
    dlt[unsafe_offset=i] = cells[unsafe_offset=i] - parent[unsafe_offset=h]

def gather_rows[
    po: ImmOrigin, io: ImmOrigin, so: MutOrigin
](plugk: ImmPointer[UInt32, po], idx: ImmPointer[Int32, io], stage: MutPointer[UInt32, so], cnt: Int32, rowwords: Int32):
    var g = global_idx.x
    var rw = Int(rowwords)
    if g >= Int(cnt) * rw:
        return
    var i = g // rw
    var w = g - i * rw
    stage[unsafe_offset=g] = plugk[unsafe_offset=Int(idx[unsafe_offset=i]) * rw + w]

def apply_corr[
    ho: ImmOrigin, so: ImmOrigin, dt: ImmOrigin, xo: ImmOrigin, io: ImmOrigin, go: ImmOrigin, yo: MutOrigin
](
    hole: ImmPointer[UInt8, ho],
    scales: ImmPointer[Float32, so],
    delta: ImmPointer[Float32, dt],
    x: ImmPointer[BFloat16, xo],
    idx: ImmPointer[Int32, io],
    stage: ImmPointer[UInt8, go],
    y: MutPointer[Float32, yo],
    cnt: Int32, n: Int32, kdim: Int32,
):
    var r = global_idx.x
    if r >= Int(n):
        return
    var kk = Int(kdim)
    var half = Int(n) // 2
    var acc = Float32(0)
    for i in range(Int(cnt)):
        var c = Int(idx[unsafe_offset=i])
        var j = r * kk + c
        var h = Int((hole[unsafe_offset=j >> 1] >> UInt8(4 * (j & 1))) & UInt8(15))
        var pb = stage[unsafe_offset=i * half + (r >> 1)]
        var p = Int((pb >> UInt8(4 * (r & 1))) & UInt8(15))
        acc += (delta[unsafe_offset=h * 16 + p] * scales[unsafe_offset=j // 64]) * x[unsafe_offset=c].cast[DType.float32]()
    y[unsafe_offset=r] = y[unsafe_offset=r] + acc

def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error("Usage: p7_decode PIN MODEL_DIR ARTIFACTS_DIR [steps (1..2047)]")
    var pin_dir = String(args[1])
    var pin_path = pin_dir + "/model-00001-of-00002.safetensors"
    var pin_path2 = pin_dir + "/model-00002-of-00002.safetensors"
    var model_dir = String(args[2])
    var art_dir = String(args[3])
    var steps = 1
    if len(args) > 4:
        steps = 0
        for b in String(args[4]).as_bytes():
            if b < 48 or b > 57:
                raise Error("steps must be a decimal integer")
            steps = steps * 10 + Int(b) - 48
    if steps < 1 or steps > NT:
        raise Error("steps outside 1..2047")
    var pct = 10
    if len(args) > 5:
        pct = 0
        for b in String(args[5]).as_bytes():
            if b < 48 or b > 57:
                raise Error("percent must be a decimal integer 0..100")
            pct = pct * 10 + Int(b) - 48
    if pct < 0 or pct > 100:
        raise Error("percent must be 0..100")
    # Optional 6th arg: fixed channel count per linear (overrides percent), e.g. 1 = the single largest-|x| channel.
    var fast = False
    var attn_split = False
    var dsel = False
    var kmaj = False
    var vgat = False
    var roll = False
    var stale = False
    var tier = False
    var pct2 = 0
    if len(args) > 10:
        for b in String(args[10]).as_bytes():
            pct2 = pct2 * 10 + Int(b) - 48
    var ksplit = False
    var prof = False
    var skel = False
    var ovl = False
    var fix = False
    var pool = False
    var kgemv = False
    var ahead = False
    var capfrac = 60
    var lyr = False
    var skg = False
    var ska = False
    var skn = False
    var skl = False
    var abl = 0
    if len(args) > 7:
        var mode = String(args[7])
        fast = mode.startswith("fast")
        attn_split = "attn" in mode
        dsel = "dsel" in mode
        kmaj = "kmaj" in mode
        vgat = "vgat" in mode
        roll = "roll" in mode
        stale = "stale" in mode
        tier = "tier" in mode
        ksplit = "ksplit" in mode
        prof = "prof" in mode
        skel = "skel" in mode
        ovl = "ovl" in mode
        fix = "fix" in mode
        pool = "pool" in mode
        kgemv = "kgemv" in mode
        ahead = "ahead" in mode
        if ahead and not (fast and attn_split and dsel and kmaj and kgemv and skel and ksplit and fix):
            raise Error("ahead needs fast+attn+dsel+kmaj+kgemv+skel+ksplit+fix")
        if kgemv and not kmaj:
            raise Error("kgemv needs kmaj")
        if pool and not fix:
            raise Error("pool needs fix")
        var mb = mode.as_bytes()
        for bi in range(len(mb) - 3):
            if mb[bi] == 99 and mb[bi + 1] == 97 and mb[bi + 2] == 112 and mb[bi + 3] >= 48 and mb[bi + 3] <= 57:
                capfrac = 0
                var bj = bi + 3
                while bj < len(mb) and mb[bj] >= 48 and mb[bj] <= 57:
                    capfrac = capfrac * 10 + Int(mb[bj]) - 48
                    bj += 1
        lyr = "lyr" in mode
        # timing-only class ablations of the fold-only path (results are NOT valid): skipg = GEMVs, skipa = attention+rope+kv, skipn = norms/residual/swiglu, skipl = logits+CE
        skg = "skipg" in mode
        ska = "skipa" in mode
        skn = "skipn" in mode
        skl = "skipl" in mode
        if fix and not (skel and ksplit):
            raise Error("fix needs skel and ksplit")
        if skel:
            stale = True
        # timing-only ablations of the device-selection path (results are NOT valid): abl1 = select only, abl2 = select + gather, abl3 = select + correction on stale staging
        if "abl1" in mode:
            abl = 1
        elif "abl2" in mode:
            abl = 2
        elif "abl3" in mode:
            abl = 3
    if roll:
        steps = 12 * 95
    # Optional 9th/10th args (selection control): layers [0, ends) and [LAYERS - ends, LAYERS) select endpct percent instead of percent.
    var ends = 0
    var endpct = 0
    if len(args) > 9:
        for b in String(args[8]).as_bytes():
            ends = ends * 10 + Int(b) - 48
        for b in String(args[9]).as_bytes():
            endpct = endpct * 10 + Int(b) - 48
    var fixed_n = 0
    if len(args) > 6:
        for b in String(args[6]).as_bytes():
            if b < 48 or b > 57:
                raise Error("fixed channel count must be a decimal integer")
            fixed_n = fixed_n * 10 + Int(b) - 48
    var ctx = DeviceContext()
    var tf = open(art_dir + "/eval_tokens_2048.txt", "r").read()
    var toks = List[Int32]()
    var cur = 0
    var have = False
    for b in tf.as_bytes():
        if b >= 48 and b <= 57:
            cur = cur * 10 + Int(b) - 48
            have = True
        elif have:
            toks.append(Int32(cur))
            cur = 0
            have = False
    if have:
        toks.append(Int32(cur))
    if len(toks) != CTX:
        raise Error("expected 2048 pinned eval tokens")
    var ob = open(art_dir + "/oracle_top1_2048.bin", "r").read_bytes()
    if len(ob) != NT * 4:
        raise Error("oracle_top1_2048.bin length mismatch")

    def read_header(path: String) raises -> Tuple[List[STEntry], Int]:
        var f = open(path, "r")
        var h8 = f.read_bytes(8)
        var hl = 0
        for k in range(8):
            hl |= Int(h8[k]) << (8 * k)
        var hb = f.read_bytes(hl)
        return (parse_header(Span(hb)), 8 + hl)

    var pin_h = read_header(pin_path)
    var pin_h2 = read_header(pin_path2)
    var s1_path = model_dir + "/model-00001-of-00002.safetensors"
    var s2_path = model_dir + "/model-00002-of-00002.safetensors"
    var s1_h = read_header(s1_path)
    var s2_h = read_header(s2_path)

    def find_entry(hd: Tuple[List[STEntry], Int], name: String) -> Int:
        for i in range(len(hd[0])):
            if hd[0][i].name == name:
                return i
        return -1

    # Compatible with the existing forward's loader; improvement to bulk pinned
    # host loads is a separate independently benchmarked task.
    def load_u8(ctx: DeviceContext, path: String, hd: Tuple[List[STEntry], Int], name: String) raises -> type_of(ctx.enqueue_create_buffer[DType.uint8](1)):
        var ix = find_entry(hd, name)
        if ix < 0:
            raise Error("missing tensor " + name + " in " + path)
        var e = hd[0][ix].copy()
        var n = e.end - e.begin
        var bytes = read_range(path, hd[1] + e.begin, n)
        var host = ctx.enqueue_create_host_buffer[DType.uint8](n)
        for i in range(n):
            host[i] = bytes[i]
        var dev = ctx.enqueue_create_buffer[DType.uint8](n)
        ctx.enqueue_copy(dst_buf=dev, src_buf=host)
        ctx.synchronize()
        return dev^

    def bf16_from(ctx: DeviceContext, name: String, h1: Tuple[List[STEntry], Int], h2: Tuple[List[STEntry], Int], p1: String, p2: String) raises -> type_of(ctx.enqueue_create_buffer[DType.uint8](1)):
        if find_entry(h1, name) >= 0:
            return load_u8(ctx, p1, h1, name)
        return load_u8(ctx, p2, h2, name)

    def load_glint(
        ctx: DeviceContext, name: String, expected: Int, dtype: String,
        h1: Tuple[List[STEntry], Int], h2: Tuple[List[STEntry], Int],
        p1: String, p2: String
    ) raises -> type_of(ctx.enqueue_create_buffer[DType.uint8](1)):
        if find_entry(h1,name)>=0:
            var e=h1[0][find_entry(h1,name)].copy()
            if e.dtype!=dtype or e.end-e.begin!=expected:
                raise Error("invalid GLINT tensor format "+name)
            return load_u8(ctx,p1,h1,name)
        if find_entry(h2,name)>=0:
            var e=h2[0][find_entry(h2,name)].copy()
            if e.dtype!=dtype or e.end-e.begin!=expected:
                raise Error("invalid GLINT tensor format "+name)
            return load_u8(ctx,p2,h2,name)
        raise Error("missing GLINT tensor "+name)

    # Retain plug bytes as pinned host memory, NOT permanent device allocations.
    # DeviceContext.enqueue_copy from each source is ordered with DeviceEvent.
    def load_glint_host(
        ctx: DeviceContext, name: String, expected: Int, dtype: String,
        h1: Tuple[List[STEntry], Int], h2: Tuple[List[STEntry], Int],
        p1: String, p2: String
    ) raises -> type_of(ctx.enqueue_create_host_buffer[DType.uint8](1)):
        var ent=STEntry(name)
        var path=String("")
        var offset=0
        if find_entry(h1,name)>=0:
            ent=h1[0][find_entry(h1,name)].copy()
            path=p1
            offset=h1[1]
        elif find_entry(h2,name)>=0:
            ent=h2[0][find_entry(h2,name)].copy()
            path=p2
            offset=h2[1]
        else:
            raise Error("missing host plug "+name)
        if ent.dtype!=dtype or ent.end-ent.begin!=expected:
            raise Error("invalid host plug layout "+name)
        var data=read_range(path,offset+ent.begin,expected)
        var buf=ctx.enqueue_create_host_buffer[DType.uint8](expected)
        for i in range(expected):
            buf[i]=data[i]
        return buf^

    var load_t0 = perf_counter_ns()
    var k_tr = ctx.compile_function[transpose_plug[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_trs0 = ctx.compile_function[transpose_scales[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var hk_early = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var sk_early = List[type_of(ctx.enqueue_create_buffer[DType.float32](1))]()
    var emb_u8 = bf16_from(ctx, "model.embed_tokens.weight", s1_h, s2_h, s1_path, s2_path)
    var fnorm_u8 = bf16_from(ctx, "model.norm.weight", s1_h, s2_h, s1_path, s2_path)
    var in_norm = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var post_norm = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var lq = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var plugk_host = List[type_of(ctx.enqueue_create_host_buffer[DType.uint8](1))]()
    var ls = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var names = List[String]()
    names.append("self_attn.qkv_proj")
    names.append("self_attn.o_proj")
    names.append("mlp.gate_up_proj")
    names.append("mlp.down_proj")
    var kin = List[Int]()
    kin.append(H)
    kin.append(H)
    kin.append(H)
    kin.append(INTER)
    var nout = List[Int]()
    nout.append(QKV)
    nout.append(H)
    nout.append(2 * INTER)
    nout.append(H)
    for l in range(LAYERS):
        in_norm.append(bf16_from(ctx, "model.layers." + String(l) + ".input_layernorm.weight", s1_h, s2_h, s1_path, s2_path))
        post_norm.append(bf16_from(ctx, "model.layers." + String(l) + ".post_attention_layernorm.weight", s1_h, s2_h, s1_path, s2_path))
        for j in range(4):
            var nm = "model.layers." + String(l) + "." + names[j] + ".weight"
            var nbytes=nout[j]*kin[j]//2
            if kgemv:
                var rm_h = load_glint(ctx,nm,nbytes,"U8",pin_h,pin_h2,pin_path,pin_path2)
                var rm_s = load_glint(ctx,nm+".absmax",nout[j]*kin[j]//64*4,"F32",pin_h,pin_h2,pin_path,pin_path2)
                var rm_sv = DeviceBuffer[DType.float32](ctx, rm_s.unsafe_ptr().unsafe_bitcast[Float32](), nout[j] * kin[j] // 64, owning=False)
                var hb0 = ctx.enqueue_create_buffer[DType.uint8](nbytes)
                var sb0 = ctx.enqueue_create_buffer[DType.float32](nout[j] * kin[j] // 64)
                ctx.enqueue_function(k_tr, rm_h, hb0, Int32(nout[j]), Int32(kin[j]), grid_dim=ceildiv(kin[j] * (nout[j] // 2), 256), block_dim=256)
                ctx.enqueue_function(k_trs0, rm_sv, sb0, Int32(nout[j]), Int32(kin[j]), grid_dim=ceildiv(nout[j] * (kin[j] // 64), 256), block_dim=256)
                ctx.synchronize()
                _ = rm_s.unsafe_ptr()
                _ = rm_h.unsafe_ptr()
                hk_early.append(hb0^)
                sk_early.append(sb0^)
            else:
                lq.append(load_glint(ctx,nm,nbytes,"U8",pin_h,pin_h2,pin_path,pin_path2))
            var rowmaj = load_glint_host(ctx,nm+".glint_plug",nbytes,"U8",pin_h,pin_h2,pin_path,pin_path2)
            var src_dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
            var dst_dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
            ctx.enqueue_copy(dst_buf=src_dev, src_buf=rowmaj)
            ctx.enqueue_function(k_tr, src_dev, dst_dev, Int32(nout[j]), Int32(kin[j]), grid_dim=ceildiv(kin[j] * (nout[j] // 2), 256), block_dim=256)
            var pk = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
            ctx.enqueue_copy(dst_buf=pk, src_buf=dst_dev)
            ctx.synchronize()
            plugk_host.append(pk^)
            if not kgemv:
                ls.append(load_glint(ctx,nm+".absmax",nout[j]*kin[j]//64*4,"F32",pin_h,pin_h2,pin_path,pin_path2))
    var table_ent=pin_h[0][find_entry(pin_h,"_glint_cells")].copy()
    if table_ent.dtype!="F32" or len(table_ent.shape)!=2 or table_ent.shape[0]!=16 or table_ent.shape[1]!=16:
        raise Error("invalid pin _glint_cells shape")
    var table_u8=load_glint(ctx,"_glint_cells",1024,"F32",pin_h,pin_h2,pin_path,pin_path2)
    print("GLINT weights loaded in", Float64(perf_counter_ns() - load_t0) / 1e9, "s")

    var emb = DeviceBuffer[DType.bfloat16](ctx, emb_u8.unsafe_ptr().unsafe_bitcast[BFloat16](), V * H, owning=False)
    var fnorm = DeviceBuffer[DType.bfloat16](ctx, fnorm_u8.unsafe_ptr().unsafe_bitcast[BFloat16](), H, owning=False)
    var wq_v = List[DeviceBuffer[DType.uint8]]()
    var sw_v = List[DeviceBuffer[DType.float32]]()
    var table = DeviceBuffer[DType.float32](ctx,table_u8.unsafe_ptr().unsafe_bitcast[Float32](),256,owning=False)
    var innorm_v = List[DeviceBuffer[DType.bfloat16]]()
    var postnorm_v = List[DeviceBuffer[DType.bfloat16]]()
    for l in range(LAYERS):
        innorm_v.append(DeviceBuffer[DType.bfloat16](ctx, in_norm[l].unsafe_ptr().unsafe_bitcast[BFloat16](), H, owning=False))
        postnorm_v.append(DeviceBuffer[DType.bfloat16](ctx, post_norm[l].unsafe_ptr().unsafe_bitcast[BFloat16](), H, owning=False))
        for j in range(4):
            if kgemv:
                continue
            wq_v.append(DeviceBuffer[DType.uint8](ctx, lq[l * 4 + j].unsafe_ptr(), nout[j] * kin[j] // 2, owning=False))
            sw_v.append(DeviceBuffer[DType.float32](ctx, ls[l * 4 + j].unsafe_ptr().unsafe_bitcast[Float32](), nout[j] * kin[j] // 64, owning=False))

    var cth = ctx.enqueue_create_host_buffer[DType.float32](CTX * (ROT // 2))
    var sth = ctx.enqueue_create_host_buffer[DType.float32](CTX * (ROT // 2))
    for pos in range(CTX):
        for i in range(ROT // 2):
            var inv64 = Float64(1) / pow(Float64(10000.0), (Float64(2) * Float64(i)) / Float64(ROT))
            var a64 = Float64(pos) * inv64
            cth[pos * (ROT // 2) + i] = bf16r(Float32(cos(a64)) * SCALE)
            sth[pos * (ROT // 2) + i] = bf16r(Float32(sin(a64)) * SCALE)
    var ct = ctx.enqueue_create_buffer[DType.float32](CTX * (ROT // 2))
    var st = ctx.enqueue_create_buffer[DType.float32](CTX * (ROT // 2))
    ctx.enqueue_copy(dst_buf=ct, src_buf=cth)
    ctx.enqueue_copy(dst_buf=st, src_buf=sth)
    var tokh = ctx.enqueue_create_host_buffer[DType.int32](1)
    var tokd = ctx.enqueue_create_buffer[DType.int32](1)
    var tgt = ctx.enqueue_create_buffer[DType.int32](NT)
    var tgth = ctx.enqueue_create_host_buffer[DType.int32](NT)
    for i in range(NT):
        tgth[i] = toks[i + 1]
    ctx.enqueue_copy(dst_buf=tgt, src_buf=tgth)

    var x = ctx.enqueue_create_buffer[DType.float32](H)
    var hbf = ctx.enqueue_create_buffer[DType.bfloat16](H)
    var qkv = ctx.enqueue_create_buffer[DType.float32](QKV)
    var rp = ctx.enqueue_create_buffer[DType.float32](QKV)
    var attn = ctx.enqueue_create_buffer[DType.float32](H)
    var attnb = ctx.enqueue_create_buffer[DType.bfloat16](H)
    var att_pm = ctx.enqueue_create_buffer[DType.float32](NQ * NCH)
    var att_pd = ctx.enqueue_create_buffer[DType.float32](NQ * NCH)
    var att_pa = ctx.enqueue_create_buffer[DType.float32](NQ * NCH * HD)
    var proj = ctx.enqueue_create_buffer[DType.float32](H)
    var gu = ctx.enqueue_create_buffer[DType.float32](2 * INTER)
    var actb = ctx.enqueue_create_buffer[DType.bfloat16](INTER)
    var act = ctx.enqueue_create_buffer[DType.float32](INTER)
    var down = ctx.enqueue_create_buffer[DType.float32](H)
    var xq = ctx.enqueue_create_buffer[DType.int8](INTER)
    var sx = ctx.enqueue_create_buffer[DType.float32](INTER // 64)
    var lg = ctx.enqueue_create_buffer[DType.float32](V)
    var loss = ctx.enqueue_create_buffer[DType.float32](steps)
    var top1 = ctx.enqueue_create_buffer[DType.int32](steps)
    var cache_len = LAYERS * CTX * NKV * HD
    var kvk = ctx.enqueue_create_buffer[DType.float32](cache_len)
    var kvv = ctx.enqueue_create_buffer[DType.float32](cache_len)
    var kv_k = List[DeviceBuffer[DType.float32]]()
    var kv_v = List[DeviceBuffer[DType.float32]]()
    for l in range(LAYERS):
        kv_k.append(DeviceBuffer[DType.float32](ctx, kvk.unsafe_ptr().unsafe_offset(l * CTX * NKV * HD), CTX * NKV * HD, owning=False))
        kv_v.append(DeviceBuffer[DType.float32](ctx, kvv.unsafe_ptr().unsafe_offset(l * CTX * NKV * HD), CTX * NKV * HD, owning=False))
    ctx.synchronize()

    var k_embed = ctx.compile_function[embed_gather[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_norm = ctx.compile_function[rms_norm_bf16[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_rope = ctx.compile_function[rope[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_res = ctx.compile_function[residual_add[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_swi = ctx.compile_function[swiglu[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_gemv = ctx.compile_function[glint_gemv_warp[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_kv = ctx.compile_function[kv_write[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_attn = ctx.compile_function[attention_one[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_attp = ctx.compile_function[attention_split_partial[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_attc = ctx.compile_function[attention_split_combine[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_lg = ctx.compile_function[logits_one[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_ce = ctx.compile_function[ce_argmax[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()

    var ks=ctx.create_stream()
    var emb_done=ctx.create_event()
    var ev_in=ctx.create_event()
    var ev_gat=ctx.create_event()
    # ---- selection machinery ----
    var stage_bytes = 3072 * 8192
    var stage_u8 = ctx.enqueue_create_buffer[DType.uint8](stage_bytes)
    var stage_w = DeviceBuffer[DType.uint32](ctx, stage_u8.unsafe_ptr().unsafe_bitcast[UInt32](), stage_bytes // 4, owning=False)
    var idx_dev = ctx.enqueue_create_buffer[DType.int32](8192)
    var idxh = ctx.enqueue_create_host_buffer[DType.int32](8192)
    var xh_h = ctx.enqueue_create_host_buffer[DType.uint16](H)
    var xh_i = ctx.enqueue_create_host_buffer[DType.uint16](INTER)
    var hbf16 = DeviceBuffer[DType.uint16](ctx, hbf.unsafe_ptr().unsafe_bitcast[UInt16](), H, owning=False)
    var attnb16 = DeviceBuffer[DType.uint16](ctx, attnb.unsafe_ptr().unsafe_bitcast[UInt16](), H, owning=False)
    var actb16 = DeviceBuffer[DType.uint16](ctx, actb.unsafe_ptr().unsafe_bitcast[UInt16](), INTER, owning=False)
    var plugk_w = List[DeviceBuffer[DType.uint32]]()
    for i in range(LAYERS * 4):
        var jj = i % 4
        plugk_w.append(DeviceBuffer[DType.uint32](ctx, plugk_host[i].unsafe_ptr().unsafe_bitcast[UInt32](), kin[jj] * nout[jj] // 8, owning=False))
    var hist = List[Int32]()
    for _ in range(32768):
        hist.append(0)
    # parent (NF4) table and cell-minus-parent delta table, built on the device from the pin's own cells
    var parent_h = ctx.enqueue_create_host_buffer[DType.float32](16)
    parent_h[0] = Float32(-1.0)
    parent_h[1] = Float32(-0.6961928009986877)
    parent_h[2] = Float32(-0.5250730514526367)
    parent_h[3] = Float32(-0.39491748809814453)
    parent_h[4] = Float32(-0.28444138169288635)
    parent_h[5] = Float32(-0.18477343022823334)
    parent_h[6] = Float32(-0.09105003625154495)
    parent_h[7] = Float32(0.0)
    parent_h[8] = Float32(0.07958029955625534)
    parent_h[9] = Float32(0.16093020141124725)
    parent_h[10] = Float32(0.24611230194568634)
    parent_h[11] = Float32(0.33791524171829224)
    parent_h[12] = Float32(0.44070982933044434)
    parent_h[13] = Float32(0.5626170039176941)
    parent_h[14] = Float32(0.7229568362236023)
    parent_h[15] = Float32(1.0)
    var parent_b = ctx.enqueue_create_buffer[DType.float32](16)
    ctx.enqueue_copy(dst_buf=parent_b, src_buf=parent_h)
    var t0b = ctx.enqueue_create_buffer[DType.float32](256)
    var dltb = ctx.enqueue_create_buffer[DType.float32](256)
    var k_mk = ctx.compile_function[make_tables[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    ctx.enqueue_function(k_mk, table, parent_b, t0b, dltb, grid_dim=1, block_dim=256)
    ctx.synchronize()
    # Tiered plug (emulation of reconstruction only; bytes are analytic): channels in the top `pct` percent keep the full 4-bit plug,
    # the next `pct2` percent get a 3-bit code in which adjacent child cells (p = 2m, 2m+1) share their mean. With T3 the merged table,
    # delta3 = T3 - parent is applied to every selected channel and (delta4 - delta3) is added on the top slice only, so the top slice
    # ends up with the exact 4-bit delta and the second slice with the merged one.
    var dltD = ctx.enqueue_create_buffer[DType.float32](256)
    if tier:
        var cells_h = ctx.enqueue_create_host_buffer[DType.float32](256)
        var d3_h = ctx.enqueue_create_host_buffer[DType.float32](256)
        var dD_h = ctx.enqueue_create_host_buffer[DType.float32](256)
        var d4_h = ctx.enqueue_create_host_buffer[DType.float32](256)
        ctx.enqueue_copy(dst_buf=cells_h, src_buf=table)
        ctx.enqueue_copy(dst_buf=d4_h, src_buf=dltb)
        ctx.synchronize()
        for h in range(16):
            for m in range(8):
                var avg = (cells_h[h * 16 + 2 * m] + cells_h[h * 16 + 2 * m + 1]) * Float32(0.5)
                var par = cells_h[h * 16 + 2 * m] - d4_h[h * 16 + 2 * m]
                for u in range(2):
                    var pi = 2 * m + u
                    d3_h[h * 16 + pi] = avg - par
                    dD_h[h * 16 + pi] = d4_h[h * 16 + pi] - (avg - par)
        ctx.enqueue_copy(dst_buf=dltb, src_buf=d3_h)
        ctx.enqueue_copy(dst_buf=dltD, src_buf=dD_h)
        ctx.synchronize()
    var idx_b = ctx.enqueue_create_buffer[DType.int32](8192)
    var cnt_b = ctx.enqueue_create_buffer[DType.int32](1)
    var dummy_log = ctx.enqueue_create_buffer[DType.int32](4)
    var stage_b8 = ctx.enqueue_create_buffer[DType.uint8](stage_bytes)
    var stage_b32 = DeviceBuffer[DType.uint32](ctx, stage_b8.unsafe_ptr().unsafe_bitcast[UInt32](), stage_bytes // 4, owning=False)
    var ev_gat2 = ctx.create_event()
    # Optional K-major copies of the hole nibbles and block scales (extra VRAM ~ the size of the resident hole + scales) so the
    # selected-channel correction reads are coalesced across rows.
    var hk_u8 = hk_early^
    var sk_f = sk_early^
    var hk_v = List[DeviceBuffer[DType.uint8]]()
    var sk_v = List[DeviceBuffer[DType.float32]]()
    var hk32 = List[DeviceBuffer[DType.uint32]]()
    if kmaj:
        var k_trs = ctx.compile_function[transpose_scales[ImmUntrackedOrigin, MutUntrackedOrigin]]()
        if not kgemv:
            for i in range(LAYERS * 4):
                var jj = i % 4
                var hb = ctx.enqueue_create_buffer[DType.uint8](nout[jj] * kin[jj] // 2)
                var sb = ctx.enqueue_create_buffer[DType.float32](nout[jj] * kin[jj] // 64)
                ctx.enqueue_function(k_tr, wq_v[i], hb, Int32(nout[jj]), Int32(kin[jj]), grid_dim=ceildiv(kin[jj] * (nout[jj] // 2), 256), block_dim=256)
                ctx.enqueue_function(k_trs, sw_v[i], sb, Int32(nout[jj]), Int32(kin[jj]), grid_dim=ceildiv(nout[jj] * (kin[jj] // 64), 256), block_dim=256)
                hk_u8.append(hb^)
                sk_f.append(sb^)
            ctx.synchronize()
        for i in range(LAYERS * 4):
            var jj = i % 4
            hk_v.append(DeviceBuffer[DType.uint8](ctx, hk_u8[i].unsafe_ptr(), nout[jj] * kin[jj] // 2, owning=False))
            sk_v.append(DeviceBuffer[DType.float32](ctx, sk_f[i].unsafe_ptr(), nout[jj] * kin[jj] // 64, owning=False))
            hk32.append(DeviceBuffer[DType.uint32](ctx, hk_u8[i].unsafe_ptr().unsafe_bitcast[UInt32](), nout[jj] * kin[jj] // 8, owning=False))
    var k_corrks = ctx.compile_function[apply_corr_ks[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_comb = ctx.compile_function[combine_corr[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var part_buf = ctx.enqueue_create_buffer[DType.float32](KS * 16384)
    var k_corrk = ctx.compile_function[apply_corr_k[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_gatv = ctx.compile_function[gather_rows_d4[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_gather = ctx.compile_function[gather_rows[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_corr = ctx.compile_function[apply_corr[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_sel = ctx.compile_function[select_topk[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_gatd = ctx.compile_function[gather_rows_d[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_corrd = ctx.compile_function[apply_corr_d[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var cnt_dev = ctx.enqueue_create_buffer[DType.int32](1)
    var k_gatp = ctx.compile_function[gather_rows_p4[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_gatx = ctx.compile_function[gather_rows_fix[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_corrksc = ctx.compile_function[apply_corr_ksc[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_corrkse = ctx.compile_function[apply_corr_kse[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var lyr_flags = ctx.enqueue_create_buffer[DType.int32](4 * INTER)
    var lyr_hits = ctx.enqueue_create_buffer[DType.int32](steps * LAYERS * 4)
    var lyr_zero = ctx.enqueue_create_host_buffer[DType.int32](4 * INTER)
    for i in range(4 * INTER):
        lyr_zero[i] = 0
    ctx.enqueue_copy(dst_buf=lyr_flags, src_buf=lyr_zero)
    var k_age = ctx.compile_function[overlap_age[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var age_hits = ctx.enqueue_create_buffer[DType.int32](steps * LAYERS * 4 * 16)
    var age_zero = ctx.enqueue_create_host_buffer[DType.int32](steps * LAYERS * 4 * 16)
    for i in range(steps * LAYERS * 4 * 16):
        age_zero[i] = 0
    ctx.enqueue_copy(dst_buf=age_hits, src_buf=age_zero)
    var k_prb = ctx.compile_function[overlap_probe[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var prb_hits = ctx.enqueue_create_buffer[DType.int32](steps * LAYERS * 4 * 2)
    var k_ovl = ctx.compile_function[overlap_count[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var ovl_flags = ctx.enqueue_create_buffer[DType.int32](LAYERS * 4 * INTER)
    var ovl_hits = ctx.enqueue_create_buffer[DType.int32](steps * LAYERS * 4)
    var ovl_zero = ctx.enqueue_create_host_buffer[DType.int32](LAYERS * 4 * INTER)
    for i in range(LAYERS * 4 * INTER):
        ovl_zero[i] = 0
    ctx.enqueue_copy(dst_buf=ovl_flags, src_buf=ovl_zero)
    var k_acc = ctx.compile_function[accum_hist[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var hist_d = ctx.enqueue_create_buffer[DType.int32](2 * LAYERS * 4 * INTER)
    var hist_z = ctx.enqueue_create_host_buffer[DType.int32](2 * LAYERS * 4 * INTER)
    for i in range(2 * LAYERS * 4 * INTER):
        hist_z[i] = 0
    ctx.enqueue_copy(dst_buf=hist_d, src_buf=hist_z)
    var cnt_log = ctx.enqueue_create_buffer[DType.int32](steps * LAYERS * 4)
    # Prefetch-timing control "stale": the channels fetched for token t are the ones selected on token t-1 (kept per linear, double
    # buffered), so the plug rows for a whole token can be gathered on a second stream while the compute stream runs. The values
    # multiplied are still the current token's activations. Needs dsel + kmaj.
    if stale and not (dsel and kmaj):
        raise Error("stale needs dsel and kmaj")
    # Ring depth: AHEAD_RING layers for the skeleton path; one slot per linear for transient staging (GLINT_EPHEMERAL=2).
    var rslots = AHEAD_RING
    if getenv("GLINT_EPHEMERAL", "0") == "2":
        rslots = 1
    var stg_total = 0
    var stg_off = List[Int]()
    if stale and not pool:
        for i in range((rslots if ahead else LAYERS) * 4):
            var jj = i % 4
            stg_off.append(stg_total)
            stg_total += kin[jj] * nout[jj] // 2
    # Ephemeral plug: nothing is staged, so no staging memory is allocated (the views below are then never dereferenced).
    var stg_all = ctx.enqueue_create_buffer[DType.uint8]((0 if getenv("GLINT_EPHEMERAL", "0") == "1" else stg_total) + 16)
    var sel_idx_all = ctx.enqueue_create_buffer[DType.int32]((2 * LAYERS * 4 * 8192 if stale else 1) + 16)
    var sel_cnt_all = ctx.enqueue_create_buffer[DType.int32]((2 * LAYERS * 4 if stale else 1) + 16)
    var stg8 = List[DeviceBuffer[DType.uint8]]()
    var stg32 = List[DeviceBuffer[DType.uint32]]()
    var sidx = List[DeviceBuffer[DType.int32]]()
    var scnt = List[DeviceBuffer[DType.int32]]()
    var gs = ctx.create_stream()
    var ev_pref = List[type_of(ev_gat)]()
    var ev_sel = List[type_of(ev_gat)]()
    var ev_ldone = List[type_of(ev_gat)]()
    var ev_ph = ctx.create_event()
    var zero_h = ctx.enqueue_create_host_buffer[DType.int32](1)
    zero_h[0] = 0
    if stale:
        for i in range((rslots if ahead else LAYERS) * 4):
            var jj = i % 4
            if pool:
                break
            stg8.append(DeviceBuffer[DType.uint8](ctx, stg_all.unsafe_ptr().unsafe_offset(stg_off[i]), kin[jj] * nout[jj] // 2, owning=False))
            stg32.append(DeviceBuffer[DType.uint32](ctx, stg_all.unsafe_ptr().unsafe_offset(stg_off[i]).unsafe_bitcast[UInt32](), kin[jj] * nout[jj] // 8, owning=False))
        for i in range(2 * LAYERS * 4):
            sidx.append(DeviceBuffer[DType.int32](ctx, sel_idx_all.unsafe_ptr().unsafe_offset(i * 8192), 8192, owning=False))
            scnt.append(DeviceBuffer[DType.int32](ctx, sel_cnt_all.unsafe_ptr().unsafe_offset(i), 1, owning=False))
        for _ in range(LAYERS):
            ev_pref.append(ctx.create_event())
            ev_ldone.append(ctx.create_event())
        for _ in range(LAYERS * 4):
            ev_sel.append(ctx.create_event())
        for i in range(2 * LAYERS * 4):
            ctx.enqueue_copy(dst_buf=scnt[i], src_buf=zero_h)
        ctx.synchronize()
    var cap_j = List[Int]()
    for j in range(4):
        var cj = (kin[j] * capfrac + 99) // 100
        if cj > kin[j]:
            cj = kin[j]
        cap_j.append(cj)
    var pl_total = 0
    var so_total = 0
    var cp_total = 0
    if pool:
        for i in range(LAYERS * 4):
            var j = i % 4
            pl_total += cap_j[j] * nout[j] // 2
            so_total += kin[j]
            cp_total += cap_j[j]
    var pool_all = ctx.enqueue_create_buffer[DType.uint8](pl_total + 16)
    var so_all = ctx.enqueue_create_buffer[DType.int32](so_total + 16)
    var sc_all = ctx.enqueue_create_buffer[DType.int32](cp_total + 16)
    var st_all = ctx.enqueue_create_buffer[DType.int32](cp_total + 16)
    var ms_all = ctx.enqueue_create_buffer[DType.int32](so_total + 16)
    var fr_all = ctx.enqueue_create_buffer[DType.int32](cp_total + 16)
    var cn_all = ctx.enqueue_create_buffer[DType.int32](2 * LAYERS * 4 + 16)
    var pool_log = ctx.enqueue_create_buffer[DType.int32](steps * LAYERS * 8 + 16)
    var pool_ovf = ctx.enqueue_create_buffer[DType.int32](4)
    var pool8 = List[DeviceBuffer[DType.uint8]]()
    var pool32 = List[DeviceBuffer[DType.uint32]]()
    var so_v = List[DeviceBuffer[DType.int32]]()
    var sc_v = List[DeviceBuffer[DType.int32]]()
    var st_v = List[DeviceBuffer[DType.int32]]()
    var ms_v = List[DeviceBuffer[DType.int32]]()
    var fr_v = List[DeviceBuffer[DType.int32]]()
    var mcn_v = List[DeviceBuffer[DType.int32]]()
    var fcn_v = List[DeviceBuffer[DType.int32]]()
    if pool:
        var pl_o = 0
        var so_o = 0
        var cp_o = 0
        for i in range(LAYERS * 4):
            var j = i % 4
            pool8.append(DeviceBuffer[DType.uint8](ctx, pool_all.unsafe_ptr().unsafe_offset(pl_o), cap_j[j] * nout[j] // 2, owning=False))
            pool32.append(DeviceBuffer[DType.uint32](ctx, pool_all.unsafe_ptr().unsafe_offset(pl_o).unsafe_bitcast[UInt32](), cap_j[j] * nout[j] // 8, owning=False))
            so_v.append(DeviceBuffer[DType.int32](ctx, so_all.unsafe_ptr().unsafe_offset(so_o), kin[j], owning=False))
            sc_v.append(DeviceBuffer[DType.int32](ctx, sc_all.unsafe_ptr().unsafe_offset(cp_o), cap_j[j], owning=False))
            st_v.append(DeviceBuffer[DType.int32](ctx, st_all.unsafe_ptr().unsafe_offset(cp_o), cap_j[j], owning=False))
            ms_v.append(DeviceBuffer[DType.int32](ctx, ms_all.unsafe_ptr().unsafe_offset(so_o), kin[j], owning=False))
            fr_v.append(DeviceBuffer[DType.int32](ctx, fr_all.unsafe_ptr().unsafe_offset(cp_o), cap_j[j], owning=False))
            mcn_v.append(DeviceBuffer[DType.int32](ctx, cn_all.unsafe_ptr().unsafe_offset(2 * i), 1, owning=False))
            fcn_v.append(DeviceBuffer[DType.int32](ctx, cn_all.unsafe_ptr().unsafe_offset(2 * i + 1), 1, owning=False))
            pl_o += cap_j[j] * nout[j] // 2
            so_o += kin[j]
            cp_o += cap_j[j]
        var h_neg = ctx.enqueue_create_host_buffer[DType.int32](so_total + 16)
        for i in range(so_total + 16):
            h_neg[i] = -1
        ctx.enqueue_copy(dst_buf=so_all, src_buf=h_neg)
        var h_negc = ctx.enqueue_create_host_buffer[DType.int32](cp_total + 16)
        var h_zc = ctx.enqueue_create_host_buffer[DType.int32](cp_total + 16)
        for i in range(cp_total + 16):
            h_negc[i] = -1
            h_zc[i] = 0
        ctx.enqueue_copy(dst_buf=sc_all, src_buf=h_negc)
        ctx.enqueue_copy(dst_buf=st_all, src_buf=h_zc)
        var h_zo = ctx.enqueue_create_host_buffer[DType.int32](4)
        for i in range(4):
            h_zo[i] = 0
        ctx.enqueue_copy(dst_buf=pool_ovf, src_buf=h_zo)
        ctx.synchronize()
    var k_plk = ctx.compile_function[pool_lookup[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_pfr = ctx.compile_function[pool_free[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_pas = ctx.compile_function[pool_assign[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_cksp = ctx.compile_function[apply_corr_ksp[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var ss = ctx.create_stream()
    var xs = ctx.enqueue_create_buffer[DType.float32](H)
    var hbf_s = ctx.enqueue_create_buffer[DType.bfloat16](H)
    var qkv_s = ctx.enqueue_create_buffer[DType.float32](QKV)
    var rp_s = ctx.enqueue_create_buffer[DType.float32](QKV)
    var attn_s = ctx.enqueue_create_buffer[DType.float32](H)
    var attnb_s = ctx.enqueue_create_buffer[DType.bfloat16](H)
    var att_pm_s = ctx.enqueue_create_buffer[DType.float32](NQ * NCH)
    var att_pd_s = ctx.enqueue_create_buffer[DType.float32](NQ * NCH)
    var att_pa_s = ctx.enqueue_create_buffer[DType.float32](NQ * NCH * HD)
    var proj_s = ctx.enqueue_create_buffer[DType.float32](H)
    var gu_s = ctx.enqueue_create_buffer[DType.float32](2 * INTER)
    var actb_s = ctx.enqueue_create_buffer[DType.bfloat16](INTER)
    var act_s = ctx.enqueue_create_buffer[DType.float32](INTER)
    var down_s = ctx.enqueue_create_buffer[DType.float32](H)
    var part_gs = ctx.enqueue_create_buffer[DType.float32](32 * 16384)
    var hbf16_s = DeviceBuffer[DType.uint16](ctx, hbf_s.unsafe_ptr().unsafe_bitcast[UInt16](), H, owning=False)
    var attnb16_s = DeviceBuffer[DType.uint16](ctx, attnb_s.unsafe_ptr().unsafe_bitcast[UInt16](), H, owning=False)
    var actb16_s = DeviceBuffer[DType.uint16](ctx, actb_s.unsafe_ptr().unsafe_bitcast[UInt16](), INTER, owning=False)
    var k_normf = ctx.compile_function[rms_norm_block[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_lgf = ctx.compile_function[logits_warp[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_cef = ctx.compile_function[ce_argmax_block[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_gkm = ctx.compile_function[glint_gemv_km[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_gkc = ctx.compile_function[gemv_km_combine[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var part_g = ctx.enqueue_create_buffer[DType.float32](32 * 16384)
    var k_gemvf8 = ctx.compile_function[glint_gemv_fast[False, INTER, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_gemvf = ctx.compile_function[glint_gemv_fast[False, H, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var wq32 = List[DeviceBuffer[DType.uint32]]()
    for i in range(LAYERS * 4):
        if kgemv:
            break
        var jj = i % 4
        wq32.append(DeviceBuffer[DType.uint32](ctx, lq[i].unsafe_ptr().unsafe_bitcast[UInt32](), kin[jj] * nout[jj] // 8, owning=False))
    var nsel_h = (pct * H + 99) // 100
    var nsel_i = (pct * INTER + 99) // 100
    if pct > 0 and nsel_h < 1:
        nsel_h = 1
    if fixed_n > 0:
        nsel_h = fixed_n
        nsel_i = fixed_n
    var nh = List[Int]()
    var ni = List[Int]()
    var nh1 = List[Int]()
    var ni1 = List[Int]()
    for l in range(LAYERS):
        var pl = pct
        if ends > 0 and (l < ends or l >= LAYERS - ends):
            pl = endpct
        nh.append(((pl + pct2) * H + 99) // 100)
        ni.append(((pl + pct2) * INTER + 99) // 100)
        nh1.append((pl * H + 99) // 100)
        ni1.append((pl * INTER + 99) // 100)
    # G-fold price selection (docs/p7_mode3_fold/GFOLD_SELECTION_2026-11-15.md). GLINT_PRICE_E9 = lambda * 1e9 (integer) and
    # GLINT_PRICE_TABLE = text file of LAYERS*4 integers s_j * 1e9 (order: layer, then qkv, o, gate+up, down). Per-linear |x|
    # threshold T_j = lambda / s_j, turned into the bf16 key the selector compares (sign bit dropped, truncated mantissa).
    # The percent argument stays the per-linear cap. Unset or 0 = rank selection only (tk all zero, unchanged behaviour).
    var tk = List[Int]()
    for _ in range(LAYERS * 4):
        tk.append(0)
    var price_e9 = 0
    for b in getenv("GLINT_PRICE_E9", "0").as_bytes():
        price_e9 = price_e9 * 10 + Int(b) - 48
    if price_e9 > 0:
        var tl = open(getenv("GLINT_PRICE_TABLE", ""), "r").read()
        var vals = List[Int]()
        var cur = 0
        var have = False
        for b in tl.as_bytes():
            if b >= 48 and b <= 57:
                cur = cur * 10 + Int(b) - 48
                have = True
            elif have:
                vals.append(cur)
                cur = 0
                have = False
        if have:
            vals.append(cur)
        if len(vals) != LAYERS * 4:
            raise Error("GLINT_PRICE_TABLE must hold LAYERS*4 integers")
        for i in range(LAYERS * 4):
            var t = Float64(price_e9) / Float64(vals[i])
            var e = 0
            while t >= 2.0:
                t = t / 2.0
                e += 1
            while t < 1.0:
                t = t * 2.0
                e -= 1
            var key = ((e + 127) << 7) | Int((t - 1.0) * 128.0)
            if key < 1:
                key = 1
            if key > 32767:
                key = 32767
            tk[i] = key
        print("MOJO_GLINT_PRICE price_e9", price_e9, "tkey_l0", tk[0], tk[1], tk[2], tk[3], "tkey_l31", tk[124], tk[125], tk[126], tk[127])
    # Lead of the skeleton stream over the real pass, in layers (H-G4 lead-time test). Default AHEAD_LAG; 0..AHEAD_RING-1.
    var alag = 0
    for b in getenv("GLINT_AHEAD_LAG", "2").as_bytes():
        alag = alag * 10 + Int(b) - 48
    if alag > AHEAD_RING - 1:
        raise Error("GLINT_AHEAD_LAG must be 0..AHEAD_RING-1")
    print("MOJO_GLINT_AHEAD lag", alag, "ring", AHEAD_RING)
    # Ephemeral plug (operator, 2026-11-15): GLINT_EPHEMERAL=1 with `ahead`. No skeleton stream, no staging: the real pass selects
    # its true channels and the correction kernel reads the selected plug rows straight from pinned host memory (same K-major,
    # channel-addressed layout as the staging it replaces). Nothing of the plug is stored on the device.
    # GLINT_EPHEMERAL=2: transient staging. No skeleton; the true rows of ONE linear are fetched to a ring slot on the fetch
    # stream while the fold GEMV of that linear runs, applied, and the slot is overwritten by the next layer.
    var eph2 = getenv("GLINT_EPHEMERAL", "0") == "2"
    var eph = getenv("GLINT_EPHEMERAL", "0") == "1" or eph2
    if eph and not ahead:
        raise Error("GLINT_EPHEMERAL needs the ahead path")
    var cst = List[DeviceBuffer[DType.uint8]]()
    if ahead:
        for i in range(LAYERS * 4):
            var jj = i % 4
            if eph and not eph2:
                cst.append(DeviceBuffer[DType.uint8](ctx, plugk_host[i].unsafe_ptr(), kin[jj] * nout[jj] // 2, owning=False))
            else:
                cst.append(DeviceBuffer[DType.uint8](ctx, stg_all.unsafe_ptr().unsafe_offset(stg_off[((i // 4) % rslots) * 4 + jj]), kin[jj] * nout[jj] // 2, owning=False))
    if eph:
        alag = 0
    var ev_in2 = List[type_of(ev_gat)]()
    var ev_cd = List[type_of(ev_gat)]()
    if eph:
        for _ in range(LAYERS * 4):
            ev_in2.append(ctx.create_event())
            ev_cd.append(ctx.create_event())
    print("MOJO_GLINT_EPHEMERAL", eph)
    # Reuse test: GLINT_REUSE_TOKENS = W > 0 treats a staged row used within the last W tokens as still resident (not fetched
    # again). Only valid with whole-token, channel-addressed staging (not `ahead`, whose ring slots are shared between layers).
    var reuse_win = 0
    for b in getenv("GLINT_REUSE_TOKENS", "0").as_bytes():
        reuse_win = reuse_win * 10 + Int(b) - 48
    if reuse_win > 0 and ahead:
        raise Error("GLINT_REUSE_TOKENS needs whole-token staging; do not combine with ahead")
    if reuse_win > 100000:
        reuse_win = 100000
    reuse_win = reuse_win * LAYERS
    print("MOJO_GLINT_REUSE win_stamps", reuse_win)
    # Floor share per linear under a price threshold (GLINT_FLOOR_PCT percent of channels), and a second rollout prompt set
    # (GLINT_ROLL_OFFSET tokens added to each prompt's start; 64 gives 12 prompts disjoint from the default set).
    var floor_pct = 0
    for b in getenv("GLINT_FLOOR_PCT", "0").as_bytes():
        floor_pct = floor_pct * 10 + Int(b) - 48
    var nfh = (floor_pct * H + 99) // 100
    var nfi = (floor_pct * INTER + 99) // 100
    var roll_off = 0
    for b in getenv("GLINT_ROLL_OFFSET", "0").as_bytes():
        roll_off = roll_off * 10 + Int(b) - 48
    print("MOJO_GLINT_FLOOR pct", floor_pct, "roll_offset", roll_off)
    # Margin (bf16 logit gap top1 - top2) per token: GLINT_MARGIN=1 records it. GLINT_ESC_M100 > 0 with GLINT_ESC_PCT re-runs a
    # token at the larger plug share when its margin * 100 is below the threshold (transient ephemeral path only).
    var mrg_on = getenv("GLINT_MARGIN", "0") == "1"
    var esc_m100 = 0
    for b in getenv("GLINT_ESC_M100", "0").as_bytes():
        esc_m100 = esc_m100 * 10 + Int(b) - 48
    var esc_pct = 0
    for b in getenv("GLINT_ESC_PCT", "0").as_bytes():
        esc_pct = esc_pct * 10 + Int(b) - 48
    var esc_on = esc_m100 > 0 and esc_pct > 0
    if esc_on:
        mrg_on = True
    var nh_base = List[Int]()
    var ni_base = List[Int]()
    for l in range(LAYERS):
        nh_base.append(nh[l])
        ni_base.append(ni[l])
    var k_mrg = ctx.compile_function[margin_block[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var marg = ctx.enqueue_create_buffer[DType.float32](steps)
    var m1h = ctx.enqueue_create_host_buffer[DType.float32](1)
    var margs = List[Float64]()
    for _ in range(steps):
        margs.append(0.0)
    var esc_count = 0
    var redo = False
    var itn = 0
    # GLINT_ALT_FILE: integer triplets (prompt, generated step, token). At that step the run prints its own bf16 logit gap
    # between its chosen token and the listed token (how far below its choice this run scores another run's choice).
    var alt = List[Int]()
    var altp = getenv("GLINT_ALT_FILE", "")
    if altp != "":
        var cur2 = 0
        var have2 = False
        for b in open(altp, "r").read().as_bytes():
            if b >= 48 and b <= 57:
                cur2 = cur2 * 10 + Int(b) - 48
                have2 = True
            elif have2:
                alt.append(cur2)
                cur2 = 0
                have2 = False
        if have2:
            alt.append(cur2)
    var a1h = ctx.enqueue_create_host_buffer[DType.float32](1)
    var a2h = ctx.enqueue_create_host_buffer[DType.float32](1)
    # Attribution test switches. GLINT_FULL_CLASS = bit mask of linear classes given the full plug (1 qkv, 2 o, 4 gate+up,
    # 8 down) while the others keep the percent share. GLINT_FORCE_FILE = 12*32 integers: in rollout mode the generated
    # positions are fed these tokens instead of the run's own argmax, so every position is scored in the same context.
    var full_mask = 0
    for b in getenv("GLINT_FULL_CLASS", "0").as_bytes():
        full_mask = full_mask * 10 + Int(b) - 48
    # GLINT_BURN = "a-b": layers a..b-1 get the full plug in every class; all other layers keep the percent share (0 = coast on
    # the fold alone). Burn-window test of the coast/burn flight plan (GFOLD_SELECTION, "G-fold rebuilt to our intent").
    var burn_a = -1
    var burn_b = -1
    var bcur = 0
    for b in getenv("GLINT_BURN", "").as_bytes():
        if b == 45:
            burn_a = bcur
            bcur = 0
        else:
            bcur = bcur * 10 + Int(b) - 48
    if burn_a >= 0:
        burn_b = bcur
    print("MOJO_GLINT_BURN layers", burn_a, burn_b)
    var nc = List[Int]()
    for l in range(LAYERS):
        for c in range(4):
            var full = (full_mask >> c) & 1 == 1 or (l >= burn_a and l < burn_b)
            nc.append((INTER if c == 3 else H) if full else (ni[l] if c == 3 else nh[l]))
    var frc = List[Int]()
    var frcp = getenv("GLINT_FORCE_FILE", "")
    if frcp != "":
        var cur3 = 0
        var have3 = False
        for b in open(frcp, "r").read().as_bytes():
            if b >= 48 and b <= 57:
                cur3 = cur3 * 10 + Int(b) - 48
                have3 = True
            elif have3:
                frc.append(cur3)
                cur3 = 0
                have3 = False
        if have3:
            frc.append(cur3)
        if len(frc) != 12 * 32:
            raise Error("GLINT_FORCE_FILE must hold 12*32 integers")
    print("MOJO_GLINT_ATTR full_class_mask", full_mask, "forced", len(frc) > 0)
    # Overlap baseline (timing only, outputs invalid when set): GLINT_TIME_ABL=1 compute only (transient path skips its
    # host-to-slot plug copy), 2 copy only (per token, only the selection and the host-to-slot copy for every linear).
    var abl_t = 0
    for b in getenv("GLINT_TIME_ABL", "0").as_bytes():
        abl_t = abl_t * 10 + Int(b) - 48
    if abl_t > 0:
        print("MOJO_GLINT_TIME_ABL", abl_t, "outputs invalid")
    print("MOJO_GLINT_ESC margin_on", mrg_on, "m100", esc_m100, "pct", esc_pct)
    # Relational price (H-G2r): GLINT_TAU_Q10 = tau * 1024; every linear's threshold is tau * mean|x| of its own input, per token.
    var tau_q10 = 0
    for b in getenv("GLINT_TAU_Q10", "0").as_bytes():
        tau_q10 = tau_q10 * 10 + Int(b) - 48
    if tau_q10 > 0:
        for i in range(LAYERS * 4):
            tk[i] = -tau_q10
        print("MOJO_GLINT_PRICE tau_q10", tau_q10)
    var bytes_total = 0
    var frac_sum = Float64(0)
    print("MOJO_GLINT_SELECT_CONFIG fast", fast, "attn_split", attn_split, "dsel", dsel, "kmaj", kmaj, "vgat", vgat, "roll", roll, "stale", stale, "tier", tier, "pct2", pct2, "ksplit", ksplit, "abl", abl, "ends", ends, "endpct", endpct, "percent", pct, "fixed_n", fixed_n, "nsel_hidden", nsel_h, "nsel_inter", nsel_i)
    print("MOJO_GLINT_SELECT_DECODE_BEGIN", steps)
    var t0 = perf_counter_ns()
    var gen = List[Int]()
    for _ in range(12 * 32):
        gen.append(0)
    var t1h = ctx.enqueue_create_host_buffer[DType.int32](1)
    var enq_ns = 0
    var pos = -1
    for _it in range(steps * 2):
        var esc_now = redo
        redo = False
        if not esc_now:
            pos += 1
        if pos >= steps:
            break
        itn += 1
        if esc_on:
            for l in range(LAYERS):
                nh[l] = (esc_pct * H + 99) // 100 if esc_now else nh_base[l]
                ni[l] = (esc_pct * INTER + 99) // 100 if esc_now else ni_base[l]
                for c in range(4):
                    if (full_mask >> c) & 1 == 0 and not (l >= burn_a and l < burn_b):
                        nc[l * 4 + c] = ni[l] if c == 3 else nh[l]
        # Free-running rollout (Amendment 3 protocol): 12 prompts of 64 evaluation-text tokens at offsets 0,128,...,1408, then 32 greedy
        # steps; per prompt 95 forward passes (positions 0..94), generated token s is the argmax at position 63+s and is fed at 64+s.
        var q = pos
        var pi = 0
        if roll:
            q = pos % 95
            pi = pos // 95
            if q < 64:
                tokh[0] = toks[pi * 128 + roll_off + q]
            else:
                tokh[0] = Int32(frc[pi * 32 + q - 64]) if len(frc) > 0 else Int32(gen[pi * 32 + q - 64])
        else:
            tokh[0] = toks[pos]
        var t_tok0 = perf_counter_ns()
        ctx.enqueue_copy(dst_buf=tokd, src_buf=tokh)
        var cv = DeviceBuffer[DType.float32](ctx, ct.unsafe_ptr().unsafe_offset(q * (ROT // 2)), ROT // 2, owning=False)
        var sv = DeviceBuffer[DType.float32](ctx, st.unsafe_ptr().unsafe_offset(q * (ROT // 2)), ROT // 2, owning=False)
        if ahead and abl_t == 2:
            for lr in range(LAYERS):
                for c in range(4):
                    if nc[lr * 4 + c] == 0:
                        continue
                    var xin = hbf16 if c == 0 or c == 2 else (attnb16 if c == 1 else actb16)
                    gs.enqueue_function(k_sel, xin, idx_b, cnt_b, dummy_log, Int32(0), Int32(INTER if c == 3 else H), Int32(nc[lr * 4 + c]), Int32(tk[lr * 4 + c]), Int32(nfi if c == 3 else nfh), grid_dim=1, block_dim=1024)
                    gs.enqueue_function(k_gatx, plugk_w[lr * 4 + c], idx_b, cnt_b, stg32[(lr % rslots) * 4 + c], ovl_flags, Int32(nout[c] // 8), Int32(((lr % rslots) * 4 + c) * INTER), Int32((itn * LAYERS + lr) * 2 + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
            gs.synchronize()
            continue
        if ahead:
            var par = pos % 2
            ctx.enqueue_function(k_embed, emb, tokd, x, grid_dim=ceildiv(H, 256), block_dim=256)
            ctx.enqueue_function(k_embed, emb, tokd, xs, grid_dim=ceildiv(H, 256), block_dim=256)
            ctx.stream().record_event(emb_done)
            ks.enqueue_wait_for(emb_done)
            ss.enqueue_wait_for(emb_done)
            for step in range(LAYERS + alag):
                if step < LAYERS and not eph:
                    var l = step
                    if l >= AHEAD_RING:
                        gs.enqueue_wait_for(ev_ldone[l - AHEAD_RING])
                    ss.enqueue_function(k_normf, xs, innorm_v[l], hbf_s, grid_dim=1, block_dim=256)
                    ss.enqueue_function(k_gkm, hk32[l * 4 + 0], sk_v[l * 4 + 0], t0b, hbf_s, part_gs, Int32(QKV), Int32(H), Int32(24), grid_dim=(ceildiv(QKV // 8, GTB), 24), block_dim=GTB)
                    ss.enqueue_function(k_gkc, part_gs, qkv_s, Int32(QKV), Int32(24), grid_dim=ceildiv(QKV, 128), block_dim=128)
                    if nh[l] > 0:
                        ss.enqueue_function(k_sel, hbf16_s, sidx[par * 128 + l * 4 + 0], scnt[par * 128 + l * 4 + 0], cnt_log, Int32((pos * LAYERS + l) * 4 + 0), Int32(H), Int32(nc[l * 4 + 0]), Int32(tk[l * 4 + 0]), Int32(nfh), grid_dim=1, block_dim=1024)
                        ss.record_event(ev_sel[l * 4 + 0])
                        gs.enqueue_wait_for(ev_sel[l * 4 + 0])
                        gs.enqueue_function(k_gatx, plugk_w[l * 4 + 0], sidx[par * 128 + l * 4 + 0], scnt[par * 128 + l * 4 + 0], stg32[(l % AHEAD_RING) * 4 + 0], ovl_flags, Int32(nout[0] // 8), Int32(((l % AHEAD_RING) * 4 + 0) * INTER), Int32(pos * LAYERS + l + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                    ss.enqueue_function(k_rope, qkv_s, cv, sv, rp_s, grid_dim=ceildiv(QKV, 256), block_dim=256)
                    ss.enqueue_function(k_kv, rp_s, kv_k[l], kv_v[l], Int32(q), grid_dim=ceildiv(NKV * HD, 256), block_dim=256)
                    ss.enqueue_function(k_attp, rp_s, kv_k[l], kv_v[l], att_pm_s, att_pd_s, att_pa_s, Int32(q), grid_dim=(NCH, NQ), block_dim=128)
                    ss.enqueue_function(k_attc, att_pm_s, att_pd_s, att_pa_s, attn_s, attnb_s, grid_dim=NQ, block_dim=HD)
                    ss.enqueue_function(k_gkm, hk32[l * 4 + 1], sk_v[l * 4 + 1], t0b, attnb_s, part_gs, Int32(H), Int32(H), Int32(24), grid_dim=(ceildiv(H // 8, GTB), 24), block_dim=GTB)
                    ss.enqueue_function(k_gkc, part_gs, proj_s, Int32(H), Int32(24), grid_dim=ceildiv(H, 128), block_dim=128)
                    if nh[l] > 0:
                        ss.enqueue_function(k_sel, attnb16_s, sidx[par * 128 + l * 4 + 1], scnt[par * 128 + l * 4 + 1], cnt_log, Int32((pos * LAYERS + l) * 4 + 1), Int32(H), Int32(nc[l * 4 + 1]), Int32(tk[l * 4 + 1]), Int32(nfh), grid_dim=1, block_dim=1024)
                        ss.record_event(ev_sel[l * 4 + 1])
                        gs.enqueue_wait_for(ev_sel[l * 4 + 1])
                        gs.enqueue_function(k_gatx, plugk_w[l * 4 + 1], sidx[par * 128 + l * 4 + 1], scnt[par * 128 + l * 4 + 1], stg32[(l % AHEAD_RING) * 4 + 1], ovl_flags, Int32(nout[1] // 8), Int32(((l % AHEAD_RING) * 4 + 1) * INTER), Int32(pos * LAYERS + l + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                    ss.enqueue_function(k_res, proj_s, xs, grid_dim=ceildiv(H, 256), block_dim=256)
                    ss.enqueue_function(k_normf, xs, postnorm_v[l], hbf_s, grid_dim=1, block_dim=256)
                    ss.enqueue_function(k_gkm, hk32[l * 4 + 2], sk_v[l * 4 + 2], t0b, hbf_s, part_gs, Int32(2 * INTER), Int32(H), Int32(24), grid_dim=(ceildiv(2 * INTER // 8, GTB), 24), block_dim=GTB)
                    ss.enqueue_function(k_gkc, part_gs, gu_s, Int32(2 * INTER), Int32(24), grid_dim=ceildiv(2 * INTER, 128), block_dim=128)
                    if nh[l] > 0:
                        ss.enqueue_function(k_sel, hbf16_s, sidx[par * 128 + l * 4 + 2], scnt[par * 128 + l * 4 + 2], cnt_log, Int32((pos * LAYERS + l) * 4 + 2), Int32(H), Int32(nc[l * 4 + 2]), Int32(tk[l * 4 + 2]), Int32(nfh), grid_dim=1, block_dim=1024)
                        ss.record_event(ev_sel[l * 4 + 2])
                        gs.enqueue_wait_for(ev_sel[l * 4 + 2])
                        gs.enqueue_function(k_gatx, plugk_w[l * 4 + 2], sidx[par * 128 + l * 4 + 2], scnt[par * 128 + l * 4 + 2], stg32[(l % AHEAD_RING) * 4 + 2], ovl_flags, Int32(nout[2] // 8), Int32(((l % AHEAD_RING) * 4 + 2) * INTER), Int32(pos * LAYERS + l + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                    ss.enqueue_function(k_swi, gu_s, act_s, actb_s, grid_dim=ceildiv(INTER, 256), block_dim=256)
                    ss.enqueue_function(k_gkm, hk32[l * 4 + 3], sk_v[l * 4 + 3], t0b, actb_s, part_gs, Int32(H), Int32(INTER), Int32(32), grid_dim=(ceildiv(H // 8, GTB), 32), block_dim=GTB)
                    ss.enqueue_function(k_gkc, part_gs, down_s, Int32(H), Int32(32), grid_dim=ceildiv(H, 128), block_dim=128)
                    if ni[l] > 0:
                        ss.enqueue_function(k_sel, actb16_s, sidx[par * 128 + l * 4 + 3], scnt[par * 128 + l * 4 + 3], cnt_log, Int32((pos * LAYERS + l) * 4 + 3), Int32(INTER), Int32(nc[l * 4 + 3]), Int32(tk[l * 4 + 3]), Int32(nfi), grid_dim=1, block_dim=1024)
                        ss.record_event(ev_sel[l * 4 + 3])
                        gs.enqueue_wait_for(ev_sel[l * 4 + 3])
                        gs.enqueue_function(k_gatx, plugk_w[l * 4 + 3], sidx[par * 128 + l * 4 + 3], scnt[par * 128 + l * 4 + 3], stg32[(l % AHEAD_RING) * 4 + 3], ovl_flags, Int32(nout[3] // 8), Int32(((l % AHEAD_RING) * 4 + 3) * INTER), Int32(pos * LAYERS + l + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                    ss.enqueue_function(k_res, down_s, xs, grid_dim=ceildiv(H, 256), block_dim=256)
                    gs.record_event(ev_pref[l])
                if step >= alag:
                    var lr = step - alag
                    if not eph: ks.enqueue_wait_for(ev_pref[lr])
                    ks.enqueue_function(k_normf, x, innorm_v[lr], hbf, grid_dim=1, block_dim=256)
                    if eph and nh[lr] > 0:
                        ks.record_event(ev_in2[lr * 4 + 0])
                        gs.enqueue_wait_for(ev_in2[lr * 4 + 0])
                        gs.enqueue_function(k_sel, hbf16, idx_b, cnt_b, dummy_log, Int32(0), Int32(H), Int32(nc[lr * 4 + 0]), Int32(tk[lr * 4 + 0]), Int32(nfh), grid_dim=1, block_dim=1024)
                        if eph2 and abl_t != 1:
                            gs.enqueue_function(k_gatx, plugk_w[lr * 4 + 0], idx_b, cnt_b, stg32[(lr % rslots) * 4 + 0], ovl_flags, Int32(nout[0] // 8), Int32(((lr % rslots) * 4 + 0) * INTER), Int32((itn * LAYERS + lr) * 2 + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                        else:
                            gs.enqueue_function(k_corrkse, hk_v[lr * 4 + 0], sk_v[lr * 4 + 0], dltb, hbf, idx_b, cst[lr * 4 + 0], cnt_b, part_buf, Int32(QKV), Int32(H), grid_dim=(ceildiv(QKV, 128), KS), block_dim=128)
                        gs.record_event(ev_cd[lr * 4 + 0])
                    ks.enqueue_function(k_gkm, hk32[lr * 4 + 0], sk_v[lr * 4 + 0], t0b, hbf, part_g, Int32(QKV), Int32(H), Int32(24), grid_dim=(ceildiv(QKV // 8, GTB), 24), block_dim=GTB)
                    ks.enqueue_function(k_gkc, part_g, qkv, Int32(QKV), Int32(24), grid_dim=ceildiv(QKV, 128), block_dim=128)
                    if nh[lr] > 0:
                        if not eph: ks.enqueue_function(k_sel, hbf16, idx_b, cnt_b, dummy_log, Int32(0), Int32(H), Int32(nc[lr * 4 + 0]), Int32(tk[lr * 4 + 0]), Int32(nfh), grid_dim=1, block_dim=1024)
                        if not eph: ks.enqueue_function(k_gatx, plugk_w[lr * 4 + 0], idx_b, cnt_b, stg32[(lr % rslots) * 4 + 0], ovl_flags, Int32(nout[0] // 8), Int32(((lr % rslots) * 4 + 0) * INTER), Int32(pos * LAYERS + lr + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                        if eph:
                            ks.enqueue_wait_for(ev_cd[lr * 4 + 0])
                            if eph2: ks.enqueue_function(k_corrksc, hk_v[lr * 4 + 0], sk_v[lr * 4 + 0], dltb, hbf, idx_b, cst[lr * 4 + 0], cnt_b, part_buf, Int32(QKV), Int32(H), grid_dim=(ceildiv(QKV, 128), KS), block_dim=128)
                        else:
                            ks.enqueue_function(k_corrksc, hk_v[lr * 4 + 0], sk_v[lr * 4 + 0], dltb, hbf, idx_b, cst[lr * 4 + 0], cnt_b, part_buf, Int32(QKV), Int32(H), grid_dim=(ceildiv(QKV, 128), KS), block_dim=128)
                        ks.enqueue_function(k_comb, part_buf, qkv, Int32(QKV), grid_dim=ceildiv(QKV, 128), block_dim=128)
                    ks.enqueue_function(k_rope, qkv, cv, sv, rp, grid_dim=ceildiv(QKV, 256), block_dim=256)
                    ks.enqueue_function(k_kv, rp, kv_k[lr], kv_v[lr], Int32(q), grid_dim=ceildiv(NKV * HD, 256), block_dim=256)
                    ks.enqueue_function(k_attp, rp, kv_k[lr], kv_v[lr], att_pm, att_pd, att_pa, Int32(q), grid_dim=(NCH, NQ), block_dim=128)
                    ks.enqueue_function(k_attc, att_pm, att_pd, att_pa, attn, attnb, grid_dim=NQ, block_dim=HD)
                    if eph and nh[lr] > 0:
                        ks.record_event(ev_in2[lr * 4 + 1])
                        gs.enqueue_wait_for(ev_in2[lr * 4 + 1])
                        gs.enqueue_function(k_sel, attnb16, idx_b, cnt_b, dummy_log, Int32(0), Int32(H), Int32(nc[lr * 4 + 1]), Int32(tk[lr * 4 + 1]), Int32(nfh), grid_dim=1, block_dim=1024)
                        if eph2 and abl_t != 1:
                            gs.enqueue_function(k_gatx, plugk_w[lr * 4 + 1], idx_b, cnt_b, stg32[(lr % rslots) * 4 + 1], ovl_flags, Int32(nout[1] // 8), Int32(((lr % rslots) * 4 + 1) * INTER), Int32((itn * LAYERS + lr) * 2 + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                        else:
                            gs.enqueue_function(k_corrkse, hk_v[lr * 4 + 1], sk_v[lr * 4 + 1], dltb, attnb, idx_b, cst[lr * 4 + 1], cnt_b, part_buf, Int32(H), Int32(H), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                        gs.record_event(ev_cd[lr * 4 + 1])
                    ks.enqueue_function(k_gkm, hk32[lr * 4 + 1], sk_v[lr * 4 + 1], t0b, attnb, part_g, Int32(H), Int32(H), Int32(24), grid_dim=(ceildiv(H // 8, GTB), 24), block_dim=GTB)
                    ks.enqueue_function(k_gkc, part_g, proj, Int32(H), Int32(24), grid_dim=ceildiv(H, 128), block_dim=128)
                    if nh[lr] > 0:
                        if not eph: ks.enqueue_function(k_sel, attnb16, idx_b, cnt_b, dummy_log, Int32(0), Int32(H), Int32(nc[lr * 4 + 1]), Int32(tk[lr * 4 + 1]), Int32(nfh), grid_dim=1, block_dim=1024)
                        if not eph: ks.enqueue_function(k_gatx, plugk_w[lr * 4 + 1], idx_b, cnt_b, stg32[(lr % rslots) * 4 + 1], ovl_flags, Int32(nout[1] // 8), Int32(((lr % rslots) * 4 + 1) * INTER), Int32(pos * LAYERS + lr + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                        if eph:
                            ks.enqueue_wait_for(ev_cd[lr * 4 + 1])
                            if eph2: ks.enqueue_function(k_corrksc, hk_v[lr * 4 + 1], sk_v[lr * 4 + 1], dltb, attnb, idx_b, cst[lr * 4 + 1], cnt_b, part_buf, Int32(H), Int32(H), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                        else:
                            ks.enqueue_function(k_corrksc, hk_v[lr * 4 + 1], sk_v[lr * 4 + 1], dltb, attnb, idx_b, cst[lr * 4 + 1], cnt_b, part_buf, Int32(H), Int32(H), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                        ks.enqueue_function(k_comb, part_buf, proj, Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
                    ks.enqueue_function(k_res, proj, x, grid_dim=ceildiv(H, 256), block_dim=256)
                    ks.enqueue_function(k_normf, x, postnorm_v[lr], hbf, grid_dim=1, block_dim=256)
                    if eph and nh[lr] > 0:
                        ks.record_event(ev_in2[lr * 4 + 2])
                        gs.enqueue_wait_for(ev_in2[lr * 4 + 2])
                        gs.enqueue_function(k_sel, hbf16, idx_b, cnt_b, dummy_log, Int32(0), Int32(H), Int32(nc[lr * 4 + 2]), Int32(tk[lr * 4 + 2]), Int32(nfh), grid_dim=1, block_dim=1024)
                        if eph2 and abl_t != 1:
                            gs.enqueue_function(k_gatx, plugk_w[lr * 4 + 2], idx_b, cnt_b, stg32[(lr % rslots) * 4 + 2], ovl_flags, Int32(nout[2] // 8), Int32(((lr % rslots) * 4 + 2) * INTER), Int32((itn * LAYERS + lr) * 2 + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                        else:
                            gs.enqueue_function(k_corrkse, hk_v[lr * 4 + 2], sk_v[lr * 4 + 2], dltb, hbf, idx_b, cst[lr * 4 + 2], cnt_b, part_buf, Int32(2 * INTER), Int32(H), grid_dim=(ceildiv(2 * INTER, 128), KS), block_dim=128)
                        gs.record_event(ev_cd[lr * 4 + 2])
                    ks.enqueue_function(k_gkm, hk32[lr * 4 + 2], sk_v[lr * 4 + 2], t0b, hbf, part_g, Int32(2 * INTER), Int32(H), Int32(24), grid_dim=(ceildiv(2 * INTER // 8, GTB), 24), block_dim=GTB)
                    ks.enqueue_function(k_gkc, part_g, gu, Int32(2 * INTER), Int32(24), grid_dim=ceildiv(2 * INTER, 128), block_dim=128)
                    if nh[lr] > 0:
                        if not eph: ks.enqueue_function(k_sel, hbf16, idx_b, cnt_b, dummy_log, Int32(0), Int32(H), Int32(nc[lr * 4 + 2]), Int32(tk[lr * 4 + 2]), Int32(nfh), grid_dim=1, block_dim=1024)
                        if not eph: ks.enqueue_function(k_gatx, plugk_w[lr * 4 + 2], idx_b, cnt_b, stg32[(lr % rslots) * 4 + 2], ovl_flags, Int32(nout[2] // 8), Int32(((lr % rslots) * 4 + 2) * INTER), Int32(pos * LAYERS + lr + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                        if eph:
                            ks.enqueue_wait_for(ev_cd[lr * 4 + 2])
                            if eph2: ks.enqueue_function(k_corrksc, hk_v[lr * 4 + 2], sk_v[lr * 4 + 2], dltb, hbf, idx_b, cst[lr * 4 + 2], cnt_b, part_buf, Int32(2 * INTER), Int32(H), grid_dim=(ceildiv(2 * INTER, 128), KS), block_dim=128)
                        else:
                            ks.enqueue_function(k_corrksc, hk_v[lr * 4 + 2], sk_v[lr * 4 + 2], dltb, hbf, idx_b, cst[lr * 4 + 2], cnt_b, part_buf, Int32(2 * INTER), Int32(H), grid_dim=(ceildiv(2 * INTER, 128), KS), block_dim=128)
                        ks.enqueue_function(k_comb, part_buf, gu, Int32(2 * INTER), grid_dim=ceildiv(2 * INTER, 128), block_dim=128)
                    ks.enqueue_function(k_swi, gu, act, actb, grid_dim=ceildiv(INTER, 256), block_dim=256)
                    if eph and ni[lr] > 0:
                        ks.record_event(ev_in2[lr * 4 + 3])
                        gs.enqueue_wait_for(ev_in2[lr * 4 + 3])
                        gs.enqueue_function(k_sel, actb16, idx_b, cnt_b, dummy_log, Int32(0), Int32(INTER), Int32(nc[lr * 4 + 3]), Int32(tk[lr * 4 + 3]), Int32(nfi), grid_dim=1, block_dim=1024)
                        if eph2 and abl_t != 1:
                            gs.enqueue_function(k_gatx, plugk_w[lr * 4 + 3], idx_b, cnt_b, stg32[(lr % rslots) * 4 + 3], ovl_flags, Int32(nout[3] // 8), Int32(((lr % rslots) * 4 + 3) * INTER), Int32((itn * LAYERS + lr) * 2 + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                        else:
                            gs.enqueue_function(k_corrkse, hk_v[lr * 4 + 3], sk_v[lr * 4 + 3], dltb, actb, idx_b, cst[lr * 4 + 3], cnt_b, part_buf, Int32(H), Int32(INTER), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                        gs.record_event(ev_cd[lr * 4 + 3])
                    ks.enqueue_function(k_gkm, hk32[lr * 4 + 3], sk_v[lr * 4 + 3], t0b, actb, part_g, Int32(H), Int32(INTER), Int32(32), grid_dim=(ceildiv(H // 8, GTB), 32), block_dim=GTB)
                    ks.enqueue_function(k_gkc, part_g, down, Int32(H), Int32(32), grid_dim=ceildiv(H, 128), block_dim=128)
                    if ni[lr] > 0:
                        if not eph: ks.enqueue_function(k_sel, actb16, idx_b, cnt_b, dummy_log, Int32(0), Int32(INTER), Int32(nc[lr * 4 + 3]), Int32(tk[lr * 4 + 3]), Int32(nfi), grid_dim=1, block_dim=1024)
                        if not eph: ks.enqueue_function(k_gatx, plugk_w[lr * 4 + 3], idx_b, cnt_b, stg32[(lr % rslots) * 4 + 3], ovl_flags, Int32(nout[3] // 8), Int32(((lr % rslots) * 4 + 3) * INTER), Int32(pos * LAYERS + lr + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                        if eph:
                            ks.enqueue_wait_for(ev_cd[lr * 4 + 3])
                            if eph2: ks.enqueue_function(k_corrksc, hk_v[lr * 4 + 3], sk_v[lr * 4 + 3], dltb, actb, idx_b, cst[lr * 4 + 3], cnt_b, part_buf, Int32(H), Int32(INTER), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                        else:
                            ks.enqueue_function(k_corrksc, hk_v[lr * 4 + 3], sk_v[lr * 4 + 3], dltb, actb, idx_b, cst[lr * 4 + 3], cnt_b, part_buf, Int32(H), Int32(INTER), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                        ks.enqueue_function(k_comb, part_buf, down, Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
                    ks.enqueue_function(k_res, down, x, grid_dim=ceildiv(H, 256), block_dim=256)
                    ks.record_event(ev_ldone[lr])
        for phase in range(0 if ahead else (2 if skel else 1)):
            if skel and phase == 1:
                ctx.stream().enqueue_wait_for(ev_ph)
            ctx.enqueue_function(k_embed, emb, tokd, x, grid_dim=ceildiv(H, 256), block_dim=256)
            ctx.stream().record_event(emb_done)
            ks.enqueue_wait_for(emb_done)
            var par = pos % 2
            var rd = par if skel else 1 - par
            if stale and not skel:
                if q == 0:
                    for i in range(LAYERS * 4):
                        ctx.enqueue_copy(dst_buf=scnt[(1 - par) * 128 + i], src_buf=zero_h)
                    ctx.synchronize()
                for l in range(LAYERS):
                    for j in range(4):
                        var jj = l * 4 + j
                        gs.enqueue_function(k_gatv, plugk_w[jj], sidx[(1 - par) * 128 + jj], scnt[(1 - par) * 128 + jj], stg32[jj], Int32(nout[j] // 8), grid_dim=kin[j], block_dim=128)
                    gs.record_event(ev_pref[l])
            for l in range(LAYERS):
                if skel and phase == 0 and l > 0:
                    gs.record_event(ev_pref[l - 1])
                if stale and not (skel and phase == 0):
                    ks.enqueue_wait_for(ev_pref[l])
                if fast:
                    if not skn:
                        ks.enqueue_function(k_normf, x, innorm_v[l], hbf, grid_dim=1, block_dim=256)
                else:
                    ks.enqueue_function(k_norm, x, innorm_v[l], hbf, grid_dim=1, block_dim=1)
                # ---- qkv_proj: fold GEMV, then the selected-channel correction ----
                ks.record_event(ev_in)
                if fast:
                    if not skg:
                        if kgemv:
                            ks.enqueue_function(k_gkm, hk32[l * 4 + 0], sk_v[l * 4 + 0], t0b, hbf, part_g, Int32(QKV), Int32(H), Int32(24), grid_dim=(ceildiv(QKV // 8, GTB), 24), block_dim=GTB)
                            ks.enqueue_function(k_gkc, part_g, qkv, Int32(QKV), Int32(24), grid_dim=ceildiv(QKV, 128), block_dim=128)
                        else:
                            ks.enqueue_function(k_gemvf, wq32[l * 4 + 0], wq32[l * 4 + 0], sw_v[l * 4 + 0], t0b, hbf, qkv, Int32(QKV), Int32(H), grid_dim=ceildiv(QKV, 4), block_dim=128)
                else:
                    ks.enqueue_function(k_gemv, wq_v[l * 4 + 0], wq_v[l * 4 + 0], sw_v[l * 4 + 0], t0b, hbf, qkv, Int32(QKV), Int32(H), grid_dim=ceildiv(QKV, 4), block_dim=128)
                if stale and nh[l] > 0:
                    if (not skel) or phase == 0:
                        ks.enqueue_function(k_sel, hbf16, sidx[par * 128 + l * 4 + 0], scnt[par * 128 + l * 4 + 0], cnt_log, Int32((pos * LAYERS + l) * 4 + 0), Int32(H), Int32(nc[l * 4 + 0]), Int32(tk[l * 4 + 0]), Int32(nfh), grid_dim=1, block_dim=1024)
                        if ovl and skel and phase == 0:
                            ks.enqueue_function(k_age, sidx[par * 128 + l * 4 + 0], scnt[par * 128 + l * 4 + 0], ovl_flags, age_hits, Int32((l * 4 + 0) * INTER), Int32(pos + 1000), Int32((pos * LAYERS + l) * 4 + 0), grid_dim=1, block_dim=1)
                            ks.enqueue_function(k_ovl, sidx[par * 128 + l * 4 + 0], scnt[par * 128 + l * 4 + 0], ovl_flags, ovl_hits, Int32((l * 4 + 0) * INTER), Int32(pos + 1000), Int32((pos * LAYERS + l) * 4 + 0), grid_dim=1, block_dim=1)
                    if skel and phase == 0:
                        ks.record_event(ev_sel[l * 4 + 0])
                        gs.enqueue_wait_for(ev_sel[l * 4 + 0])
                        if fix:
                            if pool:
                                gs.enqueue_function(k_plk, sidx[par * 128 + l * 4 + 0], scnt[par * 128 + l * 4 + 0], so_v[l * 4 + 0], st_v[l * 4 + 0], ms_v[l * 4 + 0], mcn_v[l * 4 + 0], pool_log, Int32(pos + 1), Int32((pos * LAYERS + l) * 8 + 0 * 4 + 0), grid_dim=1, block_dim=1024)
                                gs.enqueue_function(k_pfr, st_v[l * 4 + 0], Int32(cap_j[0]), Int32(pos + 1), fr_v[l * 4 + 0], fcn_v[l * 4 + 0], grid_dim=1, block_dim=1024)
                                gs.enqueue_function(k_pas, plugk_w[l * 4 + 0], ms_v[l * 4 + 0], mcn_v[l * 4 + 0], fr_v[l * 4 + 0], fcn_v[l * 4 + 0], so_v[l * 4 + 0], sc_v[l * 4 + 0], st_v[l * 4 + 0], pool32[l * 4 + 0], pool_ovf, Int32(nout[0] // 8), Int32(pos + 1), grid_dim=GB, block_dim=128)
                            else:
                                gs.enqueue_function(k_gatx, plugk_w[l * 4 + 0], sidx[par * 128 + l * 4 + 0], scnt[par * 128 + l * 4 + 0], stg32[l * 4 + 0], ovl_flags, Int32(nout[0] // 8), Int32((l * 4 + 0) * INTER), Int32(pos + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                        else:
                            gs.enqueue_function(k_gatp, plugk_w[l * 4 + 0], sidx[par * 128 + l * 4 + 0], scnt[par * 128 + l * 4 + 0], stg32[l * 4 + 0], Int32(nout[0] // 8), grid_dim=GB, block_dim=128)
                    else:
                        if fix:
                            ks.enqueue_function(k_sel, hbf16, idx_b, cnt_b, dummy_log, Int32(0), Int32(H), Int32(nc[l * 4 + 0]), Int32(tk[l * 4 + 0]), Int32(nfh), grid_dim=1, block_dim=1024)
                            if pool:
                                ks.enqueue_function(k_plk, idx_b, cnt_b, so_v[l * 4 + 0], st_v[l * 4 + 0], ms_v[l * 4 + 0], mcn_v[l * 4 + 0], pool_log, Int32(pos + 1), Int32((pos * LAYERS + l) * 8 + 1 * 4 + 0), grid_dim=1, block_dim=1024)
                                ks.enqueue_function(k_pfr, st_v[l * 4 + 0], Int32(cap_j[0]), Int32(pos + 1), fr_v[l * 4 + 0], fcn_v[l * 4 + 0], grid_dim=1, block_dim=1024)
                                ks.enqueue_function(k_pas, plugk_w[l * 4 + 0], ms_v[l * 4 + 0], mcn_v[l * 4 + 0], fr_v[l * 4 + 0], fcn_v[l * 4 + 0], so_v[l * 4 + 0], sc_v[l * 4 + 0], st_v[l * 4 + 0], pool32[l * 4 + 0], pool_ovf, Int32(nout[0] // 8), Int32(pos + 1), grid_dim=GB, block_dim=128)
                            else:
                                ks.enqueue_function(k_gatx, plugk_w[l * 4 + 0], idx_b, cnt_b, stg32[l * 4 + 0], ovl_flags, Int32(nout[0] // 8), Int32((l * 4 + 0) * INTER), Int32(pos + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                            if pool:
                                ks.enqueue_function(k_cksp, hk_v[l * 4 + 0], sk_v[l * 4 + 0], dltb, hbf, idx_b, pool8[l * 4 + 0], so_v[l * 4 + 0], cnt_b, part_buf, Int32(QKV), Int32(H), grid_dim=(ceildiv(QKV, 128), KS), block_dim=128)
                            else:
                                ks.enqueue_function(k_corrksc, hk_v[l * 4 + 0], sk_v[l * 4 + 0], dltb, hbf, idx_b, stg8[l * 4 + 0], cnt_b, part_buf, Int32(QKV), Int32(H), grid_dim=(ceildiv(QKV, 128), KS), block_dim=128)
                            ks.enqueue_function(k_comb, part_buf, qkv, Int32(QKV), grid_dim=ceildiv(QKV, 128), block_dim=128)
                        else:
                            if ovl and skel:
                                ks.enqueue_function(k_sel, hbf16, idx_b, cnt_b, dummy_log, Int32(0), Int32(H), Int32(nc[l * 4 + 0]), Int32(tk[l * 4 + 0]), Int32(nfh), grid_dim=1, block_dim=1024)
                                ks.enqueue_function(k_prb, idx_b, cnt_b, ovl_flags, prb_hits, Int32((l * 4 + 0) * INTER), Int32(pos + 1000), Int32((pos * LAYERS + l) * 4 + 0), grid_dim=1, block_dim=1)
                            if ksplit:
                                ks.enqueue_function(k_corrks, hk_v[l * 4 + 0], sk_v[l * 4 + 0], dltb, hbf, sidx[rd * 128 + l * 4 + 0], stg8[l * 4 + 0], scnt[rd * 128 + l * 4 + 0], part_buf, Int32(QKV), Int32(H), grid_dim=(ceildiv(QKV, 128), KS), block_dim=128)
                                ks.enqueue_function(k_comb, part_buf, qkv, Int32(QKV), grid_dim=ceildiv(QKV, 128), block_dim=128)
                            else:
                                ks.enqueue_function(k_corrk, hk_v[l * 4 + 0], sk_v[l * 4 + 0], dltb, hbf, sidx[rd * 128 + l * 4 + 0], stg8[l * 4 + 0], scnt[rd * 128 + l * 4 + 0], qkv, Int32(QKV), Int32(H), grid_dim=ceildiv(QKV, 128), block_dim=128)
                elif dsel and nh[l] > 0:
                    ctx.stream().enqueue_wait_for(ev_in)
                    ctx.enqueue_function(k_sel, hbf16, idx_dev, cnt_dev, cnt_log, Int32((pos * LAYERS + l) * 4 + 0), Int32(H), Int32(nc[l * 4 + 0]), Int32(tk[l * 4 + 0]), Int32(nfh), grid_dim=1, block_dim=1024)
                    if prof:
                        ctx.enqueue_function(k_acc, idx_dev, cnt_dev, hist_d, Int32((((1 if pi >= 6 else 0) * LAYERS + l) * 4 + 0) * INTER), grid_dim=1, block_dim=128)
                    if lyr:
                        ctx.enqueue_function(k_ovl, idx_dev, cnt_dev, lyr_flags, lyr_hits, Int32(0 * INTER), Int32(pos * LAYERS + l + 1000), Int32((pos * LAYERS + l) * 4 + 0), grid_dim=1, block_dim=1)
                    if abl == 0 or abl == 2:
                        if vgat:
                            ctx.enqueue_function(k_gatv, plugk_w[l * 4 + 0], idx_dev, cnt_dev, stage_w, Int32(QKV // 8), grid_dim=H, block_dim=128)
                        else:
                            ctx.enqueue_function(k_gatd, plugk_w[l * 4 + 0], idx_dev, cnt_dev, stage_w, Int32(QKV // 8), grid_dim=H, block_dim=128)
                    ctx.stream().record_event(ev_gat)
                    ks.enqueue_wait_for(ev_gat)
                    if abl == 0 or abl == 3:
                        if kmaj:
                            if ksplit:
                                ks.enqueue_function(k_corrks, hk_v[l * 4 + 0], sk_v[l * 4 + 0], dltb, hbf, idx_dev, stage_u8, cnt_dev, part_buf, Int32(QKV), Int32(H), grid_dim=(ceildiv(QKV, 128), KS), block_dim=128)
                                ks.enqueue_function(k_comb, part_buf, qkv, Int32(QKV), grid_dim=ceildiv(QKV, 128), block_dim=128)
                            else:
                                ks.enqueue_function(k_corrk, hk_v[l * 4 + 0], sk_v[l * 4 + 0], dltb, hbf, idx_dev, stage_u8, cnt_dev, qkv, Int32(QKV), Int32(H), grid_dim=ceildiv(QKV, 128), block_dim=128)
                        else:
                            ks.enqueue_function(k_corrd, wq_v[l * 4 + 0], sw_v[l * 4 + 0], dltb, hbf, idx_dev, stage_u8, cnt_dev, qkv, Int32(QKV), Int32(H), grid_dim=ceildiv(QKV, 128), block_dim=128)
                    if tier and nh1[l] > 0:
                        ctx.enqueue_function(k_sel, hbf16, idx_b, cnt_b, dummy_log, Int32(0), Int32(H), Int32(nh1[l]), Int32(tk[l * 4 + 0]), Int32(nfh), grid_dim=1, block_dim=1024)
                        ctx.enqueue_function(k_gatv, plugk_w[l * 4 + 0], idx_b, cnt_b, stage_b32, Int32(QKV // 8), grid_dim=H, block_dim=128)
                        ctx.stream().record_event(ev_gat2)
                        ks.enqueue_wait_for(ev_gat2)
                        ks.enqueue_function(k_corrk, hk_v[l * 4 + 0], sk_v[l * 4 + 0], dltD, hbf, idx_b, stage_b8, cnt_b, qkv, Int32(QKV), Int32(H), grid_dim=ceildiv(QKV, 128), block_dim=128)
                elif nsel_h > 0:
                    ctx.stream().enqueue_wait_for(ev_in)
                    ctx.enqueue_copy(dst_buf=xh_h, src_buf=hbf16)
                    ctx.synchronize()
                    for bk in range(32768):
                        hist[bk] = 0
                    for c in range(H):
                        var key = Int(xh_h[c]) & 32767
                        hist[key] = hist[key] + 1
                    var cum0 = 0
                    var thr0 = 0
                    var t0 = 32767
                    while t0 >= 0:
                        cum0 += Int(hist[t0])
                        if cum0 >= nsel_h:
                            thr0 = t0
                            break
                        t0 -= 1
                    var cnt0 = 0
                    for c in range(H):
                        if (Int(xh_h[c]) & 32767) >= thr0:
                            idxh[cnt0] = Int32(c)
                            cnt0 += 1
                    ctx.enqueue_copy(dst_buf=idx_dev, src_buf=idxh)
                    ctx.enqueue_function(k_gather, plugk_w[l * 4 + 0], idx_dev, stage_w, Int32(cnt0), Int32(QKV // 8), grid_dim=ceildiv(cnt0 * (QKV // 8), 256), block_dim=256)
                    ctx.stream().record_event(ev_gat)
                    ks.enqueue_wait_for(ev_gat)
                    ks.enqueue_function(k_corr, wq_v[l * 4 + 0], sw_v[l * 4 + 0], dltb, hbf, idx_dev, stage_u8, qkv, Int32(cnt0), Int32(QKV), Int32(H), grid_dim=ceildiv(QKV, 128), block_dim=128)
                    bytes_total += cnt0 * (QKV // 2)
                    frac_sum += Float64(cnt0) / Float64(H)
                if not ska:
                    ks.enqueue_function(k_rope, qkv, cv, sv, rp, grid_dim=ceildiv(QKV, 256), block_dim=256)
                if not ska:
                    ks.enqueue_function(k_kv, rp, kv_k[l], kv_v[l], Int32(q), grid_dim=ceildiv(NKV * HD, 256), block_dim=256)
                if attn_split:
                    if not ska:
                        ks.enqueue_function(k_attp, rp, kv_k[l], kv_v[l], att_pm, att_pd, att_pa, Int32(q), grid_dim=(NCH, NQ), block_dim=128)
                    if not ska:
                        ks.enqueue_function(k_attc, att_pm, att_pd, att_pa, attn, attnb, grid_dim=NQ, block_dim=HD)
                else:
                    ks.enqueue_function(k_attn, rp, kv_k[l], kv_v[l], attn, attnb, Int32(q), grid_dim=(1, NQ), block_dim=HD)
                # ---- o_proj: fold GEMV, then the selected-channel correction ----
                ks.record_event(ev_in)
                if fast:
                    if not skg:
                        if kgemv:
                            ks.enqueue_function(k_gkm, hk32[l * 4 + 1], sk_v[l * 4 + 1], t0b, attnb, part_g, Int32(H), Int32(H), Int32(24), grid_dim=(ceildiv(H // 8, GTB), 24), block_dim=GTB)
                            ks.enqueue_function(k_gkc, part_g, proj, Int32(H), Int32(24), grid_dim=ceildiv(H, 128), block_dim=128)
                        else:
                            ks.enqueue_function(k_gemvf, wq32[l * 4 + 1], wq32[l * 4 + 1], sw_v[l * 4 + 1], t0b, attnb, proj, Int32(H), Int32(H), grid_dim=ceildiv(H, 4), block_dim=128)
                else:
                    ks.enqueue_function(k_gemv, wq_v[l * 4 + 1], wq_v[l * 4 + 1], sw_v[l * 4 + 1], t0b, attnb, proj, Int32(H), Int32(H), grid_dim=ceildiv(H, 4), block_dim=128)
                if stale and nh[l] > 0:
                    if (not skel) or phase == 0:
                        ks.enqueue_function(k_sel, attnb16, sidx[par * 128 + l * 4 + 1], scnt[par * 128 + l * 4 + 1], cnt_log, Int32((pos * LAYERS + l) * 4 + 1), Int32(H), Int32(nc[l * 4 + 1]), Int32(tk[l * 4 + 1]), Int32(nfh), grid_dim=1, block_dim=1024)
                        if ovl and skel and phase == 0:
                            ks.enqueue_function(k_age, sidx[par * 128 + l * 4 + 1], scnt[par * 128 + l * 4 + 1], ovl_flags, age_hits, Int32((l * 4 + 1) * INTER), Int32(pos + 1000), Int32((pos * LAYERS + l) * 4 + 1), grid_dim=1, block_dim=1)
                            ks.enqueue_function(k_ovl, sidx[par * 128 + l * 4 + 1], scnt[par * 128 + l * 4 + 1], ovl_flags, ovl_hits, Int32((l * 4 + 1) * INTER), Int32(pos + 1000), Int32((pos * LAYERS + l) * 4 + 1), grid_dim=1, block_dim=1)
                    if skel and phase == 0:
                        ks.record_event(ev_sel[l * 4 + 1])
                        gs.enqueue_wait_for(ev_sel[l * 4 + 1])
                        if fix:
                            if pool:
                                gs.enqueue_function(k_plk, sidx[par * 128 + l * 4 + 1], scnt[par * 128 + l * 4 + 1], so_v[l * 4 + 1], st_v[l * 4 + 1], ms_v[l * 4 + 1], mcn_v[l * 4 + 1], pool_log, Int32(pos + 1), Int32((pos * LAYERS + l) * 8 + 0 * 4 + 1), grid_dim=1, block_dim=1024)
                                gs.enqueue_function(k_pfr, st_v[l * 4 + 1], Int32(cap_j[1]), Int32(pos + 1), fr_v[l * 4 + 1], fcn_v[l * 4 + 1], grid_dim=1, block_dim=1024)
                                gs.enqueue_function(k_pas, plugk_w[l * 4 + 1], ms_v[l * 4 + 1], mcn_v[l * 4 + 1], fr_v[l * 4 + 1], fcn_v[l * 4 + 1], so_v[l * 4 + 1], sc_v[l * 4 + 1], st_v[l * 4 + 1], pool32[l * 4 + 1], pool_ovf, Int32(nout[1] // 8), Int32(pos + 1), grid_dim=GB, block_dim=128)
                            else:
                                gs.enqueue_function(k_gatx, plugk_w[l * 4 + 1], sidx[par * 128 + l * 4 + 1], scnt[par * 128 + l * 4 + 1], stg32[l * 4 + 1], ovl_flags, Int32(nout[1] // 8), Int32((l * 4 + 1) * INTER), Int32(pos + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                        else:
                            gs.enqueue_function(k_gatp, plugk_w[l * 4 + 1], sidx[par * 128 + l * 4 + 1], scnt[par * 128 + l * 4 + 1], stg32[l * 4 + 1], Int32(nout[1] // 8), grid_dim=GB, block_dim=128)
                    else:
                        if fix:
                            ks.enqueue_function(k_sel, attnb16, idx_b, cnt_b, dummy_log, Int32(0), Int32(H), Int32(nc[l * 4 + 1]), Int32(tk[l * 4 + 1]), Int32(nfh), grid_dim=1, block_dim=1024)
                            if pool:
                                ks.enqueue_function(k_plk, idx_b, cnt_b, so_v[l * 4 + 1], st_v[l * 4 + 1], ms_v[l * 4 + 1], mcn_v[l * 4 + 1], pool_log, Int32(pos + 1), Int32((pos * LAYERS + l) * 8 + 1 * 4 + 1), grid_dim=1, block_dim=1024)
                                ks.enqueue_function(k_pfr, st_v[l * 4 + 1], Int32(cap_j[1]), Int32(pos + 1), fr_v[l * 4 + 1], fcn_v[l * 4 + 1], grid_dim=1, block_dim=1024)
                                ks.enqueue_function(k_pas, plugk_w[l * 4 + 1], ms_v[l * 4 + 1], mcn_v[l * 4 + 1], fr_v[l * 4 + 1], fcn_v[l * 4 + 1], so_v[l * 4 + 1], sc_v[l * 4 + 1], st_v[l * 4 + 1], pool32[l * 4 + 1], pool_ovf, Int32(nout[1] // 8), Int32(pos + 1), grid_dim=GB, block_dim=128)
                            else:
                                ks.enqueue_function(k_gatx, plugk_w[l * 4 + 1], idx_b, cnt_b, stg32[l * 4 + 1], ovl_flags, Int32(nout[1] // 8), Int32((l * 4 + 1) * INTER), Int32(pos + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                            if pool:
                                ks.enqueue_function(k_cksp, hk_v[l * 4 + 1], sk_v[l * 4 + 1], dltb, attnb, idx_b, pool8[l * 4 + 1], so_v[l * 4 + 1], cnt_b, part_buf, Int32(H), Int32(H), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                            else:
                                ks.enqueue_function(k_corrksc, hk_v[l * 4 + 1], sk_v[l * 4 + 1], dltb, attnb, idx_b, stg8[l * 4 + 1], cnt_b, part_buf, Int32(H), Int32(H), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                            ks.enqueue_function(k_comb, part_buf, proj, Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
                        else:
                            if ovl and skel:
                                ks.enqueue_function(k_sel, attnb16, idx_b, cnt_b, dummy_log, Int32(0), Int32(H), Int32(nc[l * 4 + 1]), Int32(tk[l * 4 + 1]), Int32(nfh), grid_dim=1, block_dim=1024)
                                ks.enqueue_function(k_prb, idx_b, cnt_b, ovl_flags, prb_hits, Int32((l * 4 + 1) * INTER), Int32(pos + 1000), Int32((pos * LAYERS + l) * 4 + 1), grid_dim=1, block_dim=1)
                            if ksplit:
                                ks.enqueue_function(k_corrks, hk_v[l * 4 + 1], sk_v[l * 4 + 1], dltb, attnb, sidx[rd * 128 + l * 4 + 1], stg8[l * 4 + 1], scnt[rd * 128 + l * 4 + 1], part_buf, Int32(H), Int32(H), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                                ks.enqueue_function(k_comb, part_buf, proj, Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
                            else:
                                ks.enqueue_function(k_corrk, hk_v[l * 4 + 1], sk_v[l * 4 + 1], dltb, attnb, sidx[rd * 128 + l * 4 + 1], stg8[l * 4 + 1], scnt[rd * 128 + l * 4 + 1], proj, Int32(H), Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
                elif dsel and nh[l] > 0:
                    ctx.stream().enqueue_wait_for(ev_in)
                    ctx.enqueue_function(k_sel, attnb16, idx_dev, cnt_dev, cnt_log, Int32((pos * LAYERS + l) * 4 + 1), Int32(H), Int32(nc[l * 4 + 1]), Int32(tk[l * 4 + 1]), Int32(nfh), grid_dim=1, block_dim=1024)
                    if prof:
                        ctx.enqueue_function(k_acc, idx_dev, cnt_dev, hist_d, Int32((((1 if pi >= 6 else 0) * LAYERS + l) * 4 + 1) * INTER), grid_dim=1, block_dim=128)
                    if lyr:
                        ctx.enqueue_function(k_ovl, idx_dev, cnt_dev, lyr_flags, lyr_hits, Int32(1 * INTER), Int32(pos * LAYERS + l + 1000), Int32((pos * LAYERS + l) * 4 + 1), grid_dim=1, block_dim=1)
                    if abl == 0 or abl == 2:
                        if vgat:
                            ctx.enqueue_function(k_gatv, plugk_w[l * 4 + 1], idx_dev, cnt_dev, stage_w, Int32(H // 8), grid_dim=H, block_dim=128)
                        else:
                            ctx.enqueue_function(k_gatd, plugk_w[l * 4 + 1], idx_dev, cnt_dev, stage_w, Int32(H // 8), grid_dim=H, block_dim=128)
                    ctx.stream().record_event(ev_gat)
                    ks.enqueue_wait_for(ev_gat)
                    if abl == 0 or abl == 3:
                        if kmaj:
                            if ksplit:
                                ks.enqueue_function(k_corrks, hk_v[l * 4 + 1], sk_v[l * 4 + 1], dltb, attnb, idx_dev, stage_u8, cnt_dev, part_buf, Int32(H), Int32(H), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                                ks.enqueue_function(k_comb, part_buf, proj, Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
                            else:
                                ks.enqueue_function(k_corrk, hk_v[l * 4 + 1], sk_v[l * 4 + 1], dltb, attnb, idx_dev, stage_u8, cnt_dev, proj, Int32(H), Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
                        else:
                            ks.enqueue_function(k_corrd, wq_v[l * 4 + 1], sw_v[l * 4 + 1], dltb, attnb, idx_dev, stage_u8, cnt_dev, proj, Int32(H), Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
                    if tier and nh1[l] > 0:
                        ctx.enqueue_function(k_sel, attnb16, idx_b, cnt_b, dummy_log, Int32(0), Int32(H), Int32(nh1[l]), Int32(tk[l * 4 + 1]), Int32(nfh), grid_dim=1, block_dim=1024)
                        ctx.enqueue_function(k_gatv, plugk_w[l * 4 + 1], idx_b, cnt_b, stage_b32, Int32(H // 8), grid_dim=H, block_dim=128)
                        ctx.stream().record_event(ev_gat2)
                        ks.enqueue_wait_for(ev_gat2)
                        ks.enqueue_function(k_corrk, hk_v[l * 4 + 1], sk_v[l * 4 + 1], dltD, attnb, idx_b, stage_b8, cnt_b, proj, Int32(H), Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
                elif nsel_h > 0:
                    ctx.stream().enqueue_wait_for(ev_in)
                    ctx.enqueue_copy(dst_buf=xh_h, src_buf=attnb16)
                    ctx.synchronize()
                    for bk in range(32768):
                        hist[bk] = 0
                    for c in range(H):
                        var key = Int(xh_h[c]) & 32767
                        hist[key] = hist[key] + 1
                    var cum1 = 0
                    var thr1 = 0
                    var t1 = 32767
                    while t1 >= 0:
                        cum1 += Int(hist[t1])
                        if cum1 >= nsel_h:
                            thr1 = t1
                            break
                        t1 -= 1
                    var cnt1 = 0
                    for c in range(H):
                        if (Int(xh_h[c]) & 32767) >= thr1:
                            idxh[cnt1] = Int32(c)
                            cnt1 += 1
                    ctx.enqueue_copy(dst_buf=idx_dev, src_buf=idxh)
                    ctx.enqueue_function(k_gather, plugk_w[l * 4 + 1], idx_dev, stage_w, Int32(cnt1), Int32(H // 8), grid_dim=ceildiv(cnt1 * (H // 8), 256), block_dim=256)
                    ctx.stream().record_event(ev_gat)
                    ks.enqueue_wait_for(ev_gat)
                    ks.enqueue_function(k_corr, wq_v[l * 4 + 1], sw_v[l * 4 + 1], dltb, attnb, idx_dev, stage_u8, proj, Int32(cnt1), Int32(H), Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
                    bytes_total += cnt1 * (H // 2)
                    frac_sum += Float64(cnt1) / Float64(H)
                if not skn:
                    ks.enqueue_function(k_res, proj, x, grid_dim=ceildiv(H, 256), block_dim=256)
                if fast:
                    if not skn:
                        ks.enqueue_function(k_normf, x, postnorm_v[l], hbf, grid_dim=1, block_dim=256)
                else:
                    ks.enqueue_function(k_norm, x, postnorm_v[l], hbf, grid_dim=1, block_dim=1)
                # ---- gate_up_proj: fold GEMV, then the selected-channel correction ----
                ks.record_event(ev_in)
                if fast:
                    if not skg:
                        if kgemv:
                            ks.enqueue_function(k_gkm, hk32[l * 4 + 2], sk_v[l * 4 + 2], t0b, hbf, part_g, Int32(2 * INTER), Int32(H), Int32(24), grid_dim=(ceildiv(2 * INTER // 8, GTB), 24), block_dim=GTB)
                            ks.enqueue_function(k_gkc, part_g, gu, Int32(2 * INTER), Int32(24), grid_dim=ceildiv(2 * INTER, 128), block_dim=128)
                        else:
                            ks.enqueue_function(k_gemvf, wq32[l * 4 + 2], wq32[l * 4 + 2], sw_v[l * 4 + 2], t0b, hbf, gu, Int32(2 * INTER), Int32(H), grid_dim=ceildiv(2 * INTER, 4), block_dim=128)
                else:
                    ks.enqueue_function(k_gemv, wq_v[l * 4 + 2], wq_v[l * 4 + 2], sw_v[l * 4 + 2], t0b, hbf, gu, Int32(2 * INTER), Int32(H), grid_dim=ceildiv(2 * INTER, 4), block_dim=128)
                if stale and nh[l] > 0:
                    if (not skel) or phase == 0:
                        ks.enqueue_function(k_sel, hbf16, sidx[par * 128 + l * 4 + 2], scnt[par * 128 + l * 4 + 2], cnt_log, Int32((pos * LAYERS + l) * 4 + 2), Int32(H), Int32(nc[l * 4 + 2]), Int32(tk[l * 4 + 2]), Int32(nfh), grid_dim=1, block_dim=1024)
                        if ovl and skel and phase == 0:
                            ks.enqueue_function(k_age, sidx[par * 128 + l * 4 + 2], scnt[par * 128 + l * 4 + 2], ovl_flags, age_hits, Int32((l * 4 + 2) * INTER), Int32(pos + 1000), Int32((pos * LAYERS + l) * 4 + 2), grid_dim=1, block_dim=1)
                            ks.enqueue_function(k_ovl, sidx[par * 128 + l * 4 + 2], scnt[par * 128 + l * 4 + 2], ovl_flags, ovl_hits, Int32((l * 4 + 2) * INTER), Int32(pos + 1000), Int32((pos * LAYERS + l) * 4 + 2), grid_dim=1, block_dim=1)
                    if skel and phase == 0:
                        ks.record_event(ev_sel[l * 4 + 2])
                        gs.enqueue_wait_for(ev_sel[l * 4 + 2])
                        if fix:
                            if pool:
                                gs.enqueue_function(k_plk, sidx[par * 128 + l * 4 + 2], scnt[par * 128 + l * 4 + 2], so_v[l * 4 + 2], st_v[l * 4 + 2], ms_v[l * 4 + 2], mcn_v[l * 4 + 2], pool_log, Int32(pos + 1), Int32((pos * LAYERS + l) * 8 + 0 * 4 + 2), grid_dim=1, block_dim=1024)
                                gs.enqueue_function(k_pfr, st_v[l * 4 + 2], Int32(cap_j[2]), Int32(pos + 1), fr_v[l * 4 + 2], fcn_v[l * 4 + 2], grid_dim=1, block_dim=1024)
                                gs.enqueue_function(k_pas, plugk_w[l * 4 + 2], ms_v[l * 4 + 2], mcn_v[l * 4 + 2], fr_v[l * 4 + 2], fcn_v[l * 4 + 2], so_v[l * 4 + 2], sc_v[l * 4 + 2], st_v[l * 4 + 2], pool32[l * 4 + 2], pool_ovf, Int32(nout[2] // 8), Int32(pos + 1), grid_dim=GB, block_dim=128)
                            else:
                                gs.enqueue_function(k_gatx, plugk_w[l * 4 + 2], sidx[par * 128 + l * 4 + 2], scnt[par * 128 + l * 4 + 2], stg32[l * 4 + 2], ovl_flags, Int32(nout[2] // 8), Int32((l * 4 + 2) * INTER), Int32(pos + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                        else:
                            gs.enqueue_function(k_gatp, plugk_w[l * 4 + 2], sidx[par * 128 + l * 4 + 2], scnt[par * 128 + l * 4 + 2], stg32[l * 4 + 2], Int32(nout[2] // 8), grid_dim=GB, block_dim=128)
                    else:
                        if fix:
                            ks.enqueue_function(k_sel, hbf16, idx_b, cnt_b, dummy_log, Int32(0), Int32(H), Int32(nc[l * 4 + 2]), Int32(tk[l * 4 + 2]), Int32(nfh), grid_dim=1, block_dim=1024)
                            if pool:
                                ks.enqueue_function(k_plk, idx_b, cnt_b, so_v[l * 4 + 2], st_v[l * 4 + 2], ms_v[l * 4 + 2], mcn_v[l * 4 + 2], pool_log, Int32(pos + 1), Int32((pos * LAYERS + l) * 8 + 1 * 4 + 2), grid_dim=1, block_dim=1024)
                                ks.enqueue_function(k_pfr, st_v[l * 4 + 2], Int32(cap_j[2]), Int32(pos + 1), fr_v[l * 4 + 2], fcn_v[l * 4 + 2], grid_dim=1, block_dim=1024)
                                ks.enqueue_function(k_pas, plugk_w[l * 4 + 2], ms_v[l * 4 + 2], mcn_v[l * 4 + 2], fr_v[l * 4 + 2], fcn_v[l * 4 + 2], so_v[l * 4 + 2], sc_v[l * 4 + 2], st_v[l * 4 + 2], pool32[l * 4 + 2], pool_ovf, Int32(nout[2] // 8), Int32(pos + 1), grid_dim=GB, block_dim=128)
                            else:
                                ks.enqueue_function(k_gatx, plugk_w[l * 4 + 2], idx_b, cnt_b, stg32[l * 4 + 2], ovl_flags, Int32(nout[2] // 8), Int32((l * 4 + 2) * INTER), Int32(pos + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                            if pool:
                                ks.enqueue_function(k_cksp, hk_v[l * 4 + 2], sk_v[l * 4 + 2], dltb, hbf, idx_b, pool8[l * 4 + 2], so_v[l * 4 + 2], cnt_b, part_buf, Int32(2 * INTER), Int32(H), grid_dim=(ceildiv(2 * INTER, 128), KS), block_dim=128)
                            else:
                                ks.enqueue_function(k_corrksc, hk_v[l * 4 + 2], sk_v[l * 4 + 2], dltb, hbf, idx_b, stg8[l * 4 + 2], cnt_b, part_buf, Int32(2 * INTER), Int32(H), grid_dim=(ceildiv(2 * INTER, 128), KS), block_dim=128)
                            ks.enqueue_function(k_comb, part_buf, gu, Int32(2 * INTER), grid_dim=ceildiv(2 * INTER, 128), block_dim=128)
                        else:
                            if ovl and skel:
                                ks.enqueue_function(k_sel, hbf16, idx_b, cnt_b, dummy_log, Int32(0), Int32(H), Int32(nc[l * 4 + 2]), Int32(tk[l * 4 + 2]), Int32(nfh), grid_dim=1, block_dim=1024)
                                ks.enqueue_function(k_prb, idx_b, cnt_b, ovl_flags, prb_hits, Int32((l * 4 + 2) * INTER), Int32(pos + 1000), Int32((pos * LAYERS + l) * 4 + 2), grid_dim=1, block_dim=1)
                            if ksplit:
                                ks.enqueue_function(k_corrks, hk_v[l * 4 + 2], sk_v[l * 4 + 2], dltb, hbf, sidx[rd * 128 + l * 4 + 2], stg8[l * 4 + 2], scnt[rd * 128 + l * 4 + 2], part_buf, Int32(2 * INTER), Int32(H), grid_dim=(ceildiv(2 * INTER, 128), KS), block_dim=128)
                                ks.enqueue_function(k_comb, part_buf, gu, Int32(2 * INTER), grid_dim=ceildiv(2 * INTER, 128), block_dim=128)
                            else:
                                ks.enqueue_function(k_corrk, hk_v[l * 4 + 2], sk_v[l * 4 + 2], dltb, hbf, sidx[rd * 128 + l * 4 + 2], stg8[l * 4 + 2], scnt[rd * 128 + l * 4 + 2], gu, Int32(2 * INTER), Int32(H), grid_dim=ceildiv(2 * INTER, 128), block_dim=128)
                elif dsel and nh[l] > 0:
                    ctx.stream().enqueue_wait_for(ev_in)
                    ctx.enqueue_function(k_sel, hbf16, idx_dev, cnt_dev, cnt_log, Int32((pos * LAYERS + l) * 4 + 2), Int32(H), Int32(nc[l * 4 + 2]), Int32(tk[l * 4 + 2]), Int32(nfh), grid_dim=1, block_dim=1024)
                    if prof:
                        ctx.enqueue_function(k_acc, idx_dev, cnt_dev, hist_d, Int32((((1 if pi >= 6 else 0) * LAYERS + l) * 4 + 2) * INTER), grid_dim=1, block_dim=128)
                    if lyr:
                        ctx.enqueue_function(k_ovl, idx_dev, cnt_dev, lyr_flags, lyr_hits, Int32(2 * INTER), Int32(pos * LAYERS + l + 1000), Int32((pos * LAYERS + l) * 4 + 2), grid_dim=1, block_dim=1)
                    if abl == 0 or abl == 2:
                        if vgat:
                            ctx.enqueue_function(k_gatv, plugk_w[l * 4 + 2], idx_dev, cnt_dev, stage_w, Int32(2 * INTER // 8), grid_dim=H, block_dim=128)
                        else:
                            ctx.enqueue_function(k_gatd, plugk_w[l * 4 + 2], idx_dev, cnt_dev, stage_w, Int32(2 * INTER // 8), grid_dim=H, block_dim=128)
                    ctx.stream().record_event(ev_gat)
                    ks.enqueue_wait_for(ev_gat)
                    if abl == 0 or abl == 3:
                        if kmaj:
                            if ksplit:
                                ks.enqueue_function(k_corrks, hk_v[l * 4 + 2], sk_v[l * 4 + 2], dltb, hbf, idx_dev, stage_u8, cnt_dev, part_buf, Int32(2 * INTER), Int32(H), grid_dim=(ceildiv(2 * INTER, 128), KS), block_dim=128)
                                ks.enqueue_function(k_comb, part_buf, gu, Int32(2 * INTER), grid_dim=ceildiv(2 * INTER, 128), block_dim=128)
                            else:
                                ks.enqueue_function(k_corrk, hk_v[l * 4 + 2], sk_v[l * 4 + 2], dltb, hbf, idx_dev, stage_u8, cnt_dev, gu, Int32(2 * INTER), Int32(H), grid_dim=ceildiv(2 * INTER, 128), block_dim=128)
                        else:
                            ks.enqueue_function(k_corrd, wq_v[l * 4 + 2], sw_v[l * 4 + 2], dltb, hbf, idx_dev, stage_u8, cnt_dev, gu, Int32(2 * INTER), Int32(H), grid_dim=ceildiv(2 * INTER, 128), block_dim=128)
                    if tier and nh1[l] > 0:
                        ctx.enqueue_function(k_sel, hbf16, idx_b, cnt_b, dummy_log, Int32(0), Int32(H), Int32(nh1[l]), Int32(tk[l * 4 + 2]), Int32(nfh), grid_dim=1, block_dim=1024)
                        ctx.enqueue_function(k_gatv, plugk_w[l * 4 + 2], idx_b, cnt_b, stage_b32, Int32(2 * INTER // 8), grid_dim=H, block_dim=128)
                        ctx.stream().record_event(ev_gat2)
                        ks.enqueue_wait_for(ev_gat2)
                        ks.enqueue_function(k_corrk, hk_v[l * 4 + 2], sk_v[l * 4 + 2], dltD, hbf, idx_b, stage_b8, cnt_b, gu, Int32(2 * INTER), Int32(H), grid_dim=ceildiv(2 * INTER, 128), block_dim=128)
                elif nsel_h > 0:
                    ctx.stream().enqueue_wait_for(ev_in)
                    ctx.enqueue_copy(dst_buf=xh_h, src_buf=hbf16)
                    ctx.synchronize()
                    for bk in range(32768):
                        hist[bk] = 0
                    for c in range(H):
                        var key = Int(xh_h[c]) & 32767
                        hist[key] = hist[key] + 1
                    var cum2 = 0
                    var thr2 = 0
                    var t2 = 32767
                    while t2 >= 0:
                        cum2 += Int(hist[t2])
                        if cum2 >= nsel_h:
                            thr2 = t2
                            break
                        t2 -= 1
                    var cnt2 = 0
                    for c in range(H):
                        if (Int(xh_h[c]) & 32767) >= thr2:
                            idxh[cnt2] = Int32(c)
                            cnt2 += 1
                    ctx.enqueue_copy(dst_buf=idx_dev, src_buf=idxh)
                    ctx.enqueue_function(k_gather, plugk_w[l * 4 + 2], idx_dev, stage_w, Int32(cnt2), Int32(2 * INTER // 8), grid_dim=ceildiv(cnt2 * (2 * INTER // 8), 256), block_dim=256)
                    ctx.stream().record_event(ev_gat)
                    ks.enqueue_wait_for(ev_gat)
                    ks.enqueue_function(k_corr, wq_v[l * 4 + 2], sw_v[l * 4 + 2], dltb, hbf, idx_dev, stage_u8, gu, Int32(cnt2), Int32(2 * INTER), Int32(H), grid_dim=ceildiv(2 * INTER, 128), block_dim=128)
                    bytes_total += cnt2 * (2 * INTER // 2)
                    frac_sum += Float64(cnt2) / Float64(H)
                if not skn:
                    ks.enqueue_function(k_swi, gu, act, actb, grid_dim=ceildiv(INTER, 256), block_dim=256)
                # ---- down_proj: fold GEMV, then the selected-channel correction ----
                ks.record_event(ev_in)
                if fast:
                    if not skg:
                        if kgemv:
                            ks.enqueue_function(k_gkm, hk32[l * 4 + 3], sk_v[l * 4 + 3], t0b, actb, part_g, Int32(H), Int32(INTER), Int32(32), grid_dim=(ceildiv(H // 8, GTB), 32), block_dim=GTB)
                            ks.enqueue_function(k_gkc, part_g, down, Int32(H), Int32(32), grid_dim=ceildiv(H, 128), block_dim=128)
                        else:
                            ks.enqueue_function(k_gemvf8, wq32[l * 4 + 3], wq32[l * 4 + 3], sw_v[l * 4 + 3], t0b, actb, down, Int32(H), Int32(INTER), grid_dim=ceildiv(H, 4), block_dim=128)
                else:
                    ks.enqueue_function(k_gemv, wq_v[l * 4 + 3], wq_v[l * 4 + 3], sw_v[l * 4 + 3], t0b, actb, down, Int32(H), Int32(INTER), grid_dim=ceildiv(H, 4), block_dim=128)
                if stale and ni[l] > 0:
                    if (not skel) or phase == 0:
                        ks.enqueue_function(k_sel, actb16, sidx[par * 128 + l * 4 + 3], scnt[par * 128 + l * 4 + 3], cnt_log, Int32((pos * LAYERS + l) * 4 + 3), Int32(INTER), Int32(nc[l * 4 + 3]), Int32(tk[l * 4 + 3]), Int32(nfi), grid_dim=1, block_dim=1024)
                        if ovl and skel and phase == 0:
                            ks.enqueue_function(k_age, sidx[par * 128 + l * 4 + 3], scnt[par * 128 + l * 4 + 3], ovl_flags, age_hits, Int32((l * 4 + 3) * INTER), Int32(pos + 1000), Int32((pos * LAYERS + l) * 4 + 3), grid_dim=1, block_dim=1)
                            ks.enqueue_function(k_ovl, sidx[par * 128 + l * 4 + 3], scnt[par * 128 + l * 4 + 3], ovl_flags, ovl_hits, Int32((l * 4 + 3) * INTER), Int32(pos + 1000), Int32((pos * LAYERS + l) * 4 + 3), grid_dim=1, block_dim=1)
                    if skel and phase == 0:
                        ks.record_event(ev_sel[l * 4 + 3])
                        gs.enqueue_wait_for(ev_sel[l * 4 + 3])
                        if fix:
                            if pool:
                                gs.enqueue_function(k_plk, sidx[par * 128 + l * 4 + 3], scnt[par * 128 + l * 4 + 3], so_v[l * 4 + 3], st_v[l * 4 + 3], ms_v[l * 4 + 3], mcn_v[l * 4 + 3], pool_log, Int32(pos + 1), Int32((pos * LAYERS + l) * 8 + 0 * 4 + 3), grid_dim=1, block_dim=1024)
                                gs.enqueue_function(k_pfr, st_v[l * 4 + 3], Int32(cap_j[3]), Int32(pos + 1), fr_v[l * 4 + 3], fcn_v[l * 4 + 3], grid_dim=1, block_dim=1024)
                                gs.enqueue_function(k_pas, plugk_w[l * 4 + 3], ms_v[l * 4 + 3], mcn_v[l * 4 + 3], fr_v[l * 4 + 3], fcn_v[l * 4 + 3], so_v[l * 4 + 3], sc_v[l * 4 + 3], st_v[l * 4 + 3], pool32[l * 4 + 3], pool_ovf, Int32(nout[3] // 8), Int32(pos + 1), grid_dim=GB, block_dim=128)
                            else:
                                gs.enqueue_function(k_gatx, plugk_w[l * 4 + 3], sidx[par * 128 + l * 4 + 3], scnt[par * 128 + l * 4 + 3], stg32[l * 4 + 3], ovl_flags, Int32(nout[3] // 8), Int32((l * 4 + 3) * INTER), Int32(pos + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                        else:
                            gs.enqueue_function(k_gatp, plugk_w[l * 4 + 3], sidx[par * 128 + l * 4 + 3], scnt[par * 128 + l * 4 + 3], stg32[l * 4 + 3], Int32(nout[3] // 8), grid_dim=GB, block_dim=128)
                    else:
                        if fix:
                            ks.enqueue_function(k_sel, actb16, idx_b, cnt_b, dummy_log, Int32(0), Int32(INTER), Int32(nc[l * 4 + 3]), Int32(tk[l * 4 + 3]), Int32(nfi), grid_dim=1, block_dim=1024)
                            if pool:
                                ks.enqueue_function(k_plk, idx_b, cnt_b, so_v[l * 4 + 3], st_v[l * 4 + 3], ms_v[l * 4 + 3], mcn_v[l * 4 + 3], pool_log, Int32(pos + 1), Int32((pos * LAYERS + l) * 8 + 1 * 4 + 3), grid_dim=1, block_dim=1024)
                                ks.enqueue_function(k_pfr, st_v[l * 4 + 3], Int32(cap_j[3]), Int32(pos + 1), fr_v[l * 4 + 3], fcn_v[l * 4 + 3], grid_dim=1, block_dim=1024)
                                ks.enqueue_function(k_pas, plugk_w[l * 4 + 3], ms_v[l * 4 + 3], mcn_v[l * 4 + 3], fr_v[l * 4 + 3], fcn_v[l * 4 + 3], so_v[l * 4 + 3], sc_v[l * 4 + 3], st_v[l * 4 + 3], pool32[l * 4 + 3], pool_ovf, Int32(nout[3] // 8), Int32(pos + 1), grid_dim=GB, block_dim=128)
                            else:
                                ks.enqueue_function(k_gatx, plugk_w[l * 4 + 3], idx_b, cnt_b, stg32[l * 4 + 3], ovl_flags, Int32(nout[3] // 8), Int32((l * 4 + 3) * INTER), Int32(pos + 1000), Int32(reuse_win), grid_dim=GB, block_dim=128)
                            if pool:
                                ks.enqueue_function(k_cksp, hk_v[l * 4 + 3], sk_v[l * 4 + 3], dltb, actb, idx_b, pool8[l * 4 + 3], so_v[l * 4 + 3], cnt_b, part_buf, Int32(H), Int32(INTER), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                            else:
                                ks.enqueue_function(k_corrksc, hk_v[l * 4 + 3], sk_v[l * 4 + 3], dltb, actb, idx_b, stg8[l * 4 + 3], cnt_b, part_buf, Int32(H), Int32(INTER), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                            ks.enqueue_function(k_comb, part_buf, down, Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
                        else:
                            if ovl and skel:
                                ks.enqueue_function(k_sel, actb16, idx_b, cnt_b, dummy_log, Int32(0), Int32(INTER), Int32(nc[l * 4 + 3]), Int32(tk[l * 4 + 3]), Int32(nfi), grid_dim=1, block_dim=1024)
                                ks.enqueue_function(k_prb, idx_b, cnt_b, ovl_flags, prb_hits, Int32((l * 4 + 3) * INTER), Int32(pos + 1000), Int32((pos * LAYERS + l) * 4 + 3), grid_dim=1, block_dim=1)
                            if ksplit:
                                ks.enqueue_function(k_corrks, hk_v[l * 4 + 3], sk_v[l * 4 + 3], dltb, actb, sidx[rd * 128 + l * 4 + 3], stg8[l * 4 + 3], scnt[rd * 128 + l * 4 + 3], part_buf, Int32(H), Int32(INTER), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                                ks.enqueue_function(k_comb, part_buf, down, Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
                            else:
                                ks.enqueue_function(k_corrk, hk_v[l * 4 + 3], sk_v[l * 4 + 3], dltb, actb, sidx[rd * 128 + l * 4 + 3], stg8[l * 4 + 3], scnt[rd * 128 + l * 4 + 3], down, Int32(H), Int32(INTER), grid_dim=ceildiv(H, 128), block_dim=128)
                elif dsel and ni[l] > 0:
                    ctx.stream().enqueue_wait_for(ev_in)
                    ctx.enqueue_function(k_sel, actb16, idx_dev, cnt_dev, cnt_log, Int32((pos * LAYERS + l) * 4 + 3), Int32(INTER), Int32(nc[l * 4 + 3]), Int32(tk[l * 4 + 3]), Int32(nfi), grid_dim=1, block_dim=1024)
                    if prof:
                        ctx.enqueue_function(k_acc, idx_dev, cnt_dev, hist_d, Int32((((1 if pi >= 6 else 0) * LAYERS + l) * 4 + 3) * INTER), grid_dim=1, block_dim=128)
                    if lyr:
                        ctx.enqueue_function(k_ovl, idx_dev, cnt_dev, lyr_flags, lyr_hits, Int32(3 * INTER), Int32(pos * LAYERS + l + 1000), Int32((pos * LAYERS + l) * 4 + 3), grid_dim=1, block_dim=1)
                    if abl == 0 or abl == 2:
                        if vgat:
                            ctx.enqueue_function(k_gatv, plugk_w[l * 4 + 3], idx_dev, cnt_dev, stage_w, Int32(H // 8), grid_dim=INTER, block_dim=128)
                        else:
                            ctx.enqueue_function(k_gatd, plugk_w[l * 4 + 3], idx_dev, cnt_dev, stage_w, Int32(H // 8), grid_dim=INTER, block_dim=128)
                    ctx.stream().record_event(ev_gat)
                    ks.enqueue_wait_for(ev_gat)
                    if abl == 0 or abl == 3:
                        if kmaj:
                            if ksplit:
                                ks.enqueue_function(k_corrks, hk_v[l * 4 + 3], sk_v[l * 4 + 3], dltb, actb, idx_dev, stage_u8, cnt_dev, part_buf, Int32(H), Int32(INTER), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                                ks.enqueue_function(k_comb, part_buf, down, Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
                            else:
                                ks.enqueue_function(k_corrk, hk_v[l * 4 + 3], sk_v[l * 4 + 3], dltb, actb, idx_dev, stage_u8, cnt_dev, down, Int32(H), Int32(INTER), grid_dim=ceildiv(H, 128), block_dim=128)
                        else:
                            ks.enqueue_function(k_corrd, wq_v[l * 4 + 3], sw_v[l * 4 + 3], dltb, actb, idx_dev, stage_u8, cnt_dev, down, Int32(H), Int32(INTER), grid_dim=ceildiv(H, 128), block_dim=128)
                    if tier and ni1[l] > 0:
                        ctx.enqueue_function(k_sel, actb16, idx_b, cnt_b, dummy_log, Int32(0), Int32(INTER), Int32(ni1[l]), Int32(tk[l * 4 + 3]), Int32(nfi), grid_dim=1, block_dim=1024)
                        ctx.enqueue_function(k_gatv, plugk_w[l * 4 + 3], idx_b, cnt_b, stage_b32, Int32(H // 8), grid_dim=INTER, block_dim=128)
                        ctx.stream().record_event(ev_gat2)
                        ks.enqueue_wait_for(ev_gat2)
                        ks.enqueue_function(k_corrk, hk_v[l * 4 + 3], sk_v[l * 4 + 3], dltD, actb, idx_b, stage_b8, cnt_b, down, Int32(H), Int32(INTER), grid_dim=ceildiv(H, 128), block_dim=128)
                elif nsel_i > 0:
                    ctx.stream().enqueue_wait_for(ev_in)
                    ctx.enqueue_copy(dst_buf=xh_i, src_buf=actb16)
                    ctx.synchronize()
                    for bk in range(32768):
                        hist[bk] = 0
                    for c in range(INTER):
                        var key = Int(xh_i[c]) & 32767
                        hist[key] = hist[key] + 1
                    var cum3 = 0
                    var thr3 = 0
                    var t3 = 32767
                    while t3 >= 0:
                        cum3 += Int(hist[t3])
                        if cum3 >= nsel_i:
                            thr3 = t3
                            break
                        t3 -= 1
                    var cnt3 = 0
                    for c in range(INTER):
                        if (Int(xh_i[c]) & 32767) >= thr3:
                            idxh[cnt3] = Int32(c)
                            cnt3 += 1
                    ctx.enqueue_copy(dst_buf=idx_dev, src_buf=idxh)
                    ctx.enqueue_function(k_gather, plugk_w[l * 4 + 3], idx_dev, stage_w, Int32(cnt3), Int32(H // 8), grid_dim=ceildiv(cnt3 * (H // 8), 256), block_dim=256)
                    ctx.stream().record_event(ev_gat)
                    ks.enqueue_wait_for(ev_gat)
                    ks.enqueue_function(k_corr, wq_v[l * 4 + 3], sw_v[l * 4 + 3], dltb, actb, idx_dev, stage_u8, down, Int32(cnt3), Int32(H), Int32(INTER), grid_dim=ceildiv(H, 128), block_dim=128)
                    bytes_total += cnt3 * (H // 2)
                    frac_sum += Float64(cnt3) / Float64(INTER)
                if not skn:
                    ks.enqueue_function(k_res, down, x, grid_dim=ceildiv(H, 256), block_dim=256)
            if skel and phase == 0:
                gs.record_event(ev_pref[LAYERS - 1])
            ks.record_event(ev_ph)
        if fast:
            if not skn:
                ks.enqueue_function(k_normf, x, fnorm, hbf, grid_dim=1, block_dim=256)
        else:
            ks.enqueue_function(k_norm, x, fnorm, hbf, grid_dim=1, block_dim=1)
        if fast:
            if not skl:
                ks.enqueue_function(k_lgf, hbf, emb, lg, grid_dim=ceildiv(V, 4), block_dim=128)
        else:
            ks.enqueue_function(k_lg, hbf, emb, lg, grid_dim=ceildiv(V, 256), block_dim=256)
        var tv = DeviceBuffer[DType.int32](ctx, tgt.unsafe_ptr().unsafe_offset(pos), 1, owning=False)
        var lv = DeviceBuffer[DType.float32](ctx, loss.unsafe_ptr().unsafe_offset(pos), 1, owning=False)
        var t1v = DeviceBuffer[DType.int32](ctx, top1.unsafe_ptr().unsafe_offset(pos), 1, owning=False)
        if fast:
            if not skl:
                ks.enqueue_function(k_cef, lg, tv, lv, t1v, Int32(1), grid_dim=1, block_dim=256)
        else:
            ks.enqueue_function(k_ce, lg, tv, lv, t1v, Int32(1), grid_dim=1, block_dim=1)
        var mv = DeviceBuffer[DType.float32](ctx, marg.unsafe_ptr().unsafe_offset(pos), 1, owning=False)
        if mrg_on:
            ks.enqueue_function(k_mrg, lg, t1v, mv, grid_dim=1, block_dim=256)
        enq_ns += Int(perf_counter_ns() - t_tok0)
        ks.synchronize()
        ctx.synchronize()
        if mrg_on and (not roll or q >= 63):
            ctx.enqueue_copy(dst_buf=m1h, src_buf=mv)
            ctx.synchronize()
            margs[pos] = Float64(m1h[0])
            if esc_on and not esc_now and Float64(m1h[0]) * 100.0 < Float64(esc_m100):
                redo = True
                esc_count += 1
                continue
        if roll and q >= 63:
            ctx.enqueue_copy(dst_buf=t1h, src_buf=t1v)
            ctx.synchronize()
            gen[pi * 32 + q - 63] = Int(t1h[0])
            for ai in range(len(alt) // 3):
                if alt[ai * 3] == pi and alt[ai * 3 + 1] == q - 63:
                    var av = DeviceBuffer[DType.float32](ctx, lg.unsafe_ptr().unsafe_offset(alt[ai * 3 + 2]), 1, owning=False)
                    var cv = DeviceBuffer[DType.float32](ctx, lg.unsafe_ptr().unsafe_offset(Int(t1h[0])), 1, owning=False)
                    ctx.enqueue_copy(dst_buf=a1h, src_buf=av)
                    ctx.enqueue_copy(dst_buf=a2h, src_buf=cv)
                    ctx.synchronize()
                    print("MOJO_ALT prompt", pi, "step", q - 63, "token", alt[ai * 3 + 2], "chosen", Int(t1h[0]), "gap_x100", Int((bf16r(a2h[0]) - bf16r(a1h[0])) * 100.0))
        if (pos + 1) % 64 == 0:
            print("decode positions", pos + 1, "/", steps)
    var elapsed = Float64(perf_counter_ns() - t0) / 1e9
    var lh = ctx.enqueue_create_host_buffer[DType.float32](steps)
    var th = ctx.enqueue_create_host_buffer[DType.int32](steps)
    ctx.enqueue_copy(dst_buf=lh, src_buf=loss)
    ctx.enqueue_copy(dst_buf=th, src_buf=top1)
    ctx.synchronize()
    if dsel:
        var clh = ctx.enqueue_create_host_buffer[DType.int32](steps * LAYERS * 4)
        ctx.enqueue_copy(dst_buf=clh, src_buf=cnt_log)
        ctx.synchronize()
        for i in range(steps * LAYERS * 4):
            var jj = i % 4
            var cn = Int(clh[i])
            if jj == 0:
                bytes_total += cn * (QKV // 2)
                frac_sum += Float64(cn) / Float64(H)
            elif jj == 1:
                bytes_total += cn * (H // 2)
                frac_sum += Float64(cn) / Float64(H)
            elif jj == 2:
                bytes_total += cn * (2 * INTER // 2)
                frac_sum += Float64(cn) / Float64(H)
            else:
                bytes_total += cn * (H // 2)
                frac_sum += Float64(cn) / Float64(INTER)
    if prof:
        var hh = ctx.enqueue_create_host_buffer[DType.int32](2 * LAYERS * 4 * INTER)
        ctx.enqueue_copy(dst_buf=hh, src_buf=hist_d)
        ctx.synchronize()
        for g in range(2 * LAYERS * 4):
            var line = String("MOJO_PROF ") + String(g)
            for c in range(INTER):
                line += String(" ") + String(Int(hh[g * INTER + c]))
            print(line)
    if ovl:
        var oh = ctx.enqueue_create_host_buffer[DType.int32](steps * LAYERS * 4)
        var oc = ctx.enqueue_create_host_buffer[DType.int32](steps * LAYERS * 4)
        ctx.enqueue_copy(dst_buf=oh, src_buf=ovl_hits)
        ctx.enqueue_copy(dst_buf=oc, src_buf=cnt_log)
        ctx.synchronize()
        for jj in range(4):
            var hs = 0
            var cs = 0
            for pos in range(1, steps):
                if roll and pos % 95 == 0:
                    continue
                for l in range(LAYERS):
                    hs += Int(oh[(pos * LAYERS + l) * 4 + jj])
                    cs += Int(oc[(pos * LAYERS + l) * 4 + jj])
            print("MOJO_OVL linear", jj, "previous_token_overlap", Float64(hs) / Float64(cs))
        var ah = ctx.enqueue_create_host_buffer[DType.int32](steps * LAYERS * 4 * 16)
        ctx.enqueue_copy(dst_buf=ah, src_buf=age_hits)
        ctx.synchronize()
        for jj in range(4):
            var cum = 0
            var tot2 = 0
            for pos in range(1, steps):
                for l in range(LAYERS):
                    tot2 += Int(oc[(pos * LAYERS + l) * 4 + jj])
            var line = String("MOJO_AGE linear ") + String(jj) + String(" cumulative_hit_fraction_by_window")
            for w in range(16):
                for pos in range(1, steps):
                    for l in range(LAYERS):
                        cum += Int(ah[16 * ((pos * LAYERS + l) * 4 + jj) + w])
                line += String(" ") + String(w + 1) + String(":") + String(Float64(cum) / Float64(tot2))
                cum = 0 if False else cum
            print(line)
        var ph = ctx.enqueue_create_host_buffer[DType.int32](steps * LAYERS * 4 * 2)
        ctx.enqueue_copy(dst_buf=ph, src_buf=prb_hits)
        ctx.synchronize()
        for jj in range(4):
            var cov_s = 0
            var cov_b = 0
            var tot = 0
            for pos in range(1, steps):
                if roll and pos % 95 == 0:
                    continue
                for l in range(LAYERS):
                    cov_s += Int(ph[2 * ((pos * LAYERS + l) * 4 + jj)])
                    cov_b += Int(ph[2 * ((pos * LAYERS + l) * 4 + jj) + 1])
                    tot += Int(oc[(pos * LAYERS + l) * 4 + jj])
            print("MOJO_OVL linear", jj, "true_selection_covered_by_skeleton", Float64(cov_s) / Float64(tot), "plus_previous_token_only", Float64(cov_b) / Float64(tot))
    if lyr:
        var lh = ctx.enqueue_create_host_buffer[DType.int32](steps * LAYERS * 4)
        var lc = ctx.enqueue_create_host_buffer[DType.int32](steps * LAYERS * 4)
        ctx.enqueue_copy(dst_buf=lh, src_buf=lyr_hits)
        ctx.enqueue_copy(dst_buf=lc, src_buf=cnt_log)
        ctx.synchronize()
        for jj in range(4):
            var hs = 0
            var cs = 0
            for pos in range(steps):
                for l in range(1, LAYERS):
                    hs += Int(lh[(pos * LAYERS + l) * 4 + jj])
                    cs += Int(lc[(pos * LAYERS + l) * 4 + jj])
            print("MOJO_LYR linear", jj, "covered_by_previous_layer_selection", Float64(hs) / Float64(cs))
    if pool:
        var plh = ctx.enqueue_create_host_buffer[DType.int32](steps * LAYERS * 8 + 16)
        var ovh = ctx.enqueue_create_host_buffer[DType.int32](4)
        ctx.enqueue_copy(dst_buf=plh, src_buf=pool_log)
        ctx.enqueue_copy(dst_buf=ovh, src_buf=pool_ovf)
        ctx.synchronize()
        var fresh = 0
        for pos in range(steps):
            for l in range(LAYERS):
                for ph in range(2):
                    for j in range(4):
                        fresh += Int(plh[(pos * LAYERS + l) * 8 + ph * 4 + j]) * (nout[j] // 2)
        print("MOJO_POOL capfrac", capfrac, "pool_MB", Float64(pl_total) / 1e6, "fresh_plug_MB_per_token", Float64(fresh) / 1e6 / Float64(steps), "overflow", Int(ovh[0]))
    var total_ce = Float64(0)
    var matches = 0
    for pos in range(steps):
        total_ce += Float64(lh[pos])
        var o = Int(ob[4 * pos]) | (Int(ob[4 * pos + 1]) << 8) | (Int(ob[4 * pos + 2]) << 16) | (Int(ob[4 * pos + 3]) << 24)
        if Int(th[pos]) == o:
            matches += 1
        if not roll:
            print("MOJO_GLINT_SELECT_POSITION", pos, Int(th[pos]), Float64(lh[pos]))
    if mrg_on:
        print("MOJO_GLINT_ESC escalated_tokens", esc_count, "of", steps)
        if roll:
            for pi in range(12):
                var ml = String("MOJO_MARGIN prompt ") + String(pi) + String(" m100")
                for si in range(32):
                    ml += String(" ") + String(Int(margs[pi * 95 + 63 + si] * 100.0))
                print(ml)
    if roll:
        for pi in range(12):
            var line = String("MOJO_ROLL prompt ") + String(pi) + String(" tokens")
            for si in range(32):
                line += String(" ") + String(gen[pi * 32 + si])
            print(line)
    print("MOJO_GLINT_SELECT_DECODE steps", steps, "duration_s", elapsed, "ms_per_token", elapsed * 1000 / Float64(steps), "tokens_per_s", Float64(steps) / elapsed)
    print("MOJO_GLINT_SELECT_DECODE ppl", exp(total_ce / Float64(steps)), "oracle_top1", matches, "/", steps)
    print("MOJO_GLINT_SELECT_STATS percent", pct, "gathered_bytes_per_token_mean", Float64(bytes_total) / Float64(steps), "mean_selected_channel_fraction", frac_sum / Float64(steps * LAYERS * 4))
    print("MOJO_ENQ host_enqueue_ms_per_token", Float64(enq_ns) / 1e6 / Float64(steps))
    print("MOJO_GLINT_SELECT_DECODE_DONE")
    # Keep all owning buffers alive while their typed borrowed views are used.
    _ = emb_u8.unsafe_ptr()
    _ = fnorm_u8.unsafe_ptr()
    _ = kvk.unsafe_ptr()
    _ = kvv.unsafe_ptr()
    _ = table_u8.unsafe_ptr()
    _ = part_g.unsafe_ptr()
    _ = xs.unsafe_ptr()
    _ = hbf_s.unsafe_ptr()
    _ = qkv_s.unsafe_ptr()
    _ = rp_s.unsafe_ptr()
    _ = attn_s.unsafe_ptr()
    _ = attnb_s.unsafe_ptr()
    _ = att_pm_s.unsafe_ptr()
    _ = att_pd_s.unsafe_ptr()
    _ = att_pa_s.unsafe_ptr()
    _ = proj_s.unsafe_ptr()
    _ = gu_s.unsafe_ptr()
    _ = actb_s.unsafe_ptr()
    _ = act_s.unsafe_ptr()
    _ = down_s.unsafe_ptr()
    _ = part_gs.unsafe_ptr()
    _ = len(hk32)
    _ = pool_all.unsafe_ptr()
    _ = so_all.unsafe_ptr()
    _ = sc_all.unsafe_ptr()
    _ = st_all.unsafe_ptr()
    _ = ms_all.unsafe_ptr()
    _ = fr_all.unsafe_ptr()
    _ = cn_all.unsafe_ptr()
    _ = pool_log.unsafe_ptr()
    _ = pool_ovf.unsafe_ptr()
    _ = len(pool8) + len(pool32) + len(so_v) + len(sc_v) + len(st_v) + len(ms_v) + len(fr_v) + len(mcn_v) + len(fcn_v)
    _ = len(in_norm) + len(post_norm) + len(lq) + len(plugk_host) + len(plugk_w) + len(ls) + len(hk_u8) + len(sk_f) + len(stg8) + len(stg32) + len(sidx) + len(scnt) + len(ev_pref)
    _ = stg_all.unsafe_ptr()
    _ = sel_idx_all.unsafe_ptr()
    _ = sel_cnt_all.unsafe_ptr()
