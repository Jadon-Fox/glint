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
from p7_decode_fast_kernels import rms_norm_block, logits_warp, ce_argmax_block
from p7_attention_split import attention_split_partial, attention_split_combine, NCH
from p7_select_device import select_topk, gather_rows_d, apply_corr_d, apply_corr_k, transpose_scales, gather_rows_d4, apply_corr_ks, combine_corr, KS

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
    var ksplit = False
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
        ksplit = "ksplit" in mode
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
    # Optional K-major copies of the hole nibbles and block scales (extra VRAM ~ the size of the resident hole + scales) so the
    # selected-channel correction reads are coalesced across rows.
    var hk_u8 = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var sk_f = List[type_of(ctx.enqueue_create_buffer[DType.float32](1))]()
    var hk_v = List[DeviceBuffer[DType.uint8]]()
    var sk_v = List[DeviceBuffer[DType.float32]]()
    if kmaj:
        var k_trs = ctx.compile_function[transpose_scales[ImmUntrackedOrigin, MutUntrackedOrigin]]()
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
    var cnt_log = ctx.enqueue_create_buffer[DType.int32](steps * LAYERS * 4)
    # Prefetch-timing control "stale": the channels fetched for token t are the ones selected on token t-1 (kept per linear, double
    # buffered), so the plug rows for a whole token can be gathered on a second stream while the compute stream runs. The values
    # multiplied are still the current token's activations. Needs dsel + kmaj.
    if stale and not (dsel and kmaj):
        raise Error("stale needs dsel and kmaj")
    var stg_total = 0
    var stg_off = List[Int]()
    if stale:
        for i in range(LAYERS * 4):
            var jj = i % 4
            stg_off.append(stg_total)
            stg_total += kin[jj] * nout[jj] // 2
    var stg_all = ctx.enqueue_create_buffer[DType.uint8](stg_total + 16)
    var sel_idx_all = ctx.enqueue_create_buffer[DType.int32]((2 * LAYERS * 4 * 8192 if stale else 1) + 16)
    var sel_cnt_all = ctx.enqueue_create_buffer[DType.int32]((2 * LAYERS * 4 if stale else 1) + 16)
    var stg8 = List[DeviceBuffer[DType.uint8]]()
    var stg32 = List[DeviceBuffer[DType.uint32]]()
    var sidx = List[DeviceBuffer[DType.int32]]()
    var scnt = List[DeviceBuffer[DType.int32]]()
    var gs = ctx.create_stream()
    var ev_pref = List[type_of(ev_gat)]()
    var zero_h = ctx.enqueue_create_host_buffer[DType.int32](1)
    zero_h[0] = 0
    if stale:
        for i in range(LAYERS * 4):
            var jj = i % 4
            stg8.append(DeviceBuffer[DType.uint8](ctx, stg_all.unsafe_ptr().unsafe_offset(stg_off[i]), kin[jj] * nout[jj] // 2, owning=False))
            stg32.append(DeviceBuffer[DType.uint32](ctx, stg_all.unsafe_ptr().unsafe_offset(stg_off[i]).unsafe_bitcast[UInt32](), kin[jj] * nout[jj] // 8, owning=False))
        for i in range(2 * LAYERS * 4):
            sidx.append(DeviceBuffer[DType.int32](ctx, sel_idx_all.unsafe_ptr().unsafe_offset(i * 8192), 8192, owning=False))
            scnt.append(DeviceBuffer[DType.int32](ctx, sel_cnt_all.unsafe_ptr().unsafe_offset(i), 1, owning=False))
        for _ in range(LAYERS):
            ev_pref.append(ctx.create_event())
        for i in range(2 * LAYERS * 4):
            ctx.enqueue_copy(dst_buf=scnt[i], src_buf=zero_h)
        ctx.synchronize()
    var k_normf = ctx.compile_function[rms_norm_block[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_lgf = ctx.compile_function[logits_warp[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_cef = ctx.compile_function[ce_argmax_block[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_gemvf = ctx.compile_function[glint_gemv_fast[False, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var wq32 = List[DeviceBuffer[DType.uint32]]()
    for i in range(LAYERS * 4):
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
    for l in range(LAYERS):
        var pl = pct
        if ends > 0 and (l < ends or l >= LAYERS - ends):
            pl = endpct
        nh.append((pl * H + 99) // 100)
        ni.append((pl * INTER + 99) // 100)
    var bytes_total = 0
    var frac_sum = Float64(0)
    print("MOJO_GLINT_SELECT_CONFIG fast", fast, "attn_split", attn_split, "dsel", dsel, "kmaj", kmaj, "vgat", vgat, "roll", roll, "stale", stale, "ksplit", ksplit, "abl", abl, "ends", ends, "endpct", endpct, "percent", pct, "fixed_n", fixed_n, "nsel_hidden", nsel_h, "nsel_inter", nsel_i)
    print("MOJO_GLINT_SELECT_DECODE_BEGIN", steps)
    var t0 = perf_counter_ns()
    var gen = List[Int]()
    for _ in range(12 * 32):
        gen.append(0)
    var t1h = ctx.enqueue_create_host_buffer[DType.int32](1)
    for pos in range(steps):
        # Free-running rollout (Amendment 3 protocol): 12 prompts of 64 evaluation-text tokens at offsets 0,128,...,1408, then 32 greedy
        # steps; per prompt 95 forward passes (positions 0..94), generated token s is the argmax at position 63+s and is fed at 64+s.
        var q = pos
        var pi = 0
        if roll:
            q = pos % 95
            pi = pos // 95
            if q < 64:
                tokh[0] = toks[pi * 128 + q]
            else:
                tokh[0] = Int32(gen[pi * 32 + q - 64])
        else:
            tokh[0] = toks[pos]
        ctx.enqueue_copy(dst_buf=tokd, src_buf=tokh)
        var cv = DeviceBuffer[DType.float32](ctx, ct.unsafe_ptr().unsafe_offset(q * (ROT // 2)), ROT // 2, owning=False)
        var sv = DeviceBuffer[DType.float32](ctx, st.unsafe_ptr().unsafe_offset(q * (ROT // 2)), ROT // 2, owning=False)
        ctx.enqueue_function(k_embed, emb, tokd, x, grid_dim=ceildiv(H, 256), block_dim=256)
        ctx.stream().record_event(emb_done)
        ks.enqueue_wait_for(emb_done)
        var par = pos % 2
        if stale:
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
            if stale:
                ks.enqueue_wait_for(ev_pref[l])
            if fast:
                ks.enqueue_function(k_normf, x, innorm_v[l], hbf, grid_dim=1, block_dim=256)
            else:
                ks.enqueue_function(k_norm, x, innorm_v[l], hbf, grid_dim=1, block_dim=1)
            # ---- qkv_proj: fold GEMV, then the selected-channel correction ----
            ks.record_event(ev_in)
            if fast:
                ks.enqueue_function(k_gemvf, wq32[l * 4 + 0], wq32[l * 4 + 0], sw_v[l * 4 + 0], t0b, hbf, qkv, Int32(QKV), Int32(H), grid_dim=ceildiv(QKV, 4), block_dim=128)
            else:
                ks.enqueue_function(k_gemv, wq_v[l * 4 + 0], wq_v[l * 4 + 0], sw_v[l * 4 + 0], t0b, hbf, qkv, Int32(QKV), Int32(H), grid_dim=ceildiv(QKV, 4), block_dim=128)
            if stale and nh[l] > 0:
                ks.enqueue_function(k_sel, hbf16, sidx[par * 128 + l * 4 + 0], scnt[par * 128 + l * 4 + 0], cnt_log, Int32((pos * LAYERS + l) * 4 + 0), Int32(H), Int32(nh[l]), grid_dim=1, block_dim=1024)
                if ksplit:
                    ks.enqueue_function(k_corrks, hk_v[l * 4 + 0], sk_v[l * 4 + 0], dltb, hbf, sidx[(1 - par) * 128 + l * 4 + 0], stg8[l * 4 + 0], scnt[(1 - par) * 128 + l * 4 + 0], part_buf, Int32(QKV), Int32(H), grid_dim=(ceildiv(QKV, 128), KS), block_dim=128)
                    ks.enqueue_function(k_comb, part_buf, qkv, Int32(QKV), grid_dim=ceildiv(QKV, 128), block_dim=128)
                else:
                    ks.enqueue_function(k_corrk, hk_v[l * 4 + 0], sk_v[l * 4 + 0], dltb, hbf, sidx[(1 - par) * 128 + l * 4 + 0], stg8[l * 4 + 0], scnt[(1 - par) * 128 + l * 4 + 0], qkv, Int32(QKV), Int32(H), grid_dim=ceildiv(QKV, 128), block_dim=128)
            elif dsel and nh[l] > 0:
                ctx.stream().enqueue_wait_for(ev_in)
                ctx.enqueue_function(k_sel, hbf16, idx_dev, cnt_dev, cnt_log, Int32((pos * LAYERS + l) * 4 + 0), Int32(H), Int32(nh[l]), grid_dim=1, block_dim=1024)
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
            ks.enqueue_function(k_rope, qkv, cv, sv, rp, grid_dim=ceildiv(QKV, 256), block_dim=256)
            ks.enqueue_function(k_kv, rp, kv_k[l], kv_v[l], Int32(q), grid_dim=ceildiv(NKV * HD, 256), block_dim=256)
            if attn_split:
                ks.enqueue_function(k_attp, rp, kv_k[l], kv_v[l], att_pm, att_pd, att_pa, Int32(q), grid_dim=(NCH, NQ), block_dim=128)
                ks.enqueue_function(k_attc, att_pm, att_pd, att_pa, attn, attnb, grid_dim=NQ, block_dim=HD)
            else:
                ks.enqueue_function(k_attn, rp, kv_k[l], kv_v[l], attn, attnb, Int32(q), grid_dim=(1, NQ), block_dim=HD)
            # ---- o_proj: fold GEMV, then the selected-channel correction ----
            ks.record_event(ev_in)
            if fast:
                ks.enqueue_function(k_gemvf, wq32[l * 4 + 1], wq32[l * 4 + 1], sw_v[l * 4 + 1], t0b, attnb, proj, Int32(H), Int32(H), grid_dim=ceildiv(H, 4), block_dim=128)
            else:
                ks.enqueue_function(k_gemv, wq_v[l * 4 + 1], wq_v[l * 4 + 1], sw_v[l * 4 + 1], t0b, attnb, proj, Int32(H), Int32(H), grid_dim=ceildiv(H, 4), block_dim=128)
            if stale and nh[l] > 0:
                ks.enqueue_function(k_sel, attnb16, sidx[par * 128 + l * 4 + 1], scnt[par * 128 + l * 4 + 1], cnt_log, Int32((pos * LAYERS + l) * 4 + 1), Int32(H), Int32(nh[l]), grid_dim=1, block_dim=1024)
                if ksplit:
                    ks.enqueue_function(k_corrks, hk_v[l * 4 + 1], sk_v[l * 4 + 1], dltb, attnb, sidx[(1 - par) * 128 + l * 4 + 1], stg8[l * 4 + 1], scnt[(1 - par) * 128 + l * 4 + 1], part_buf, Int32(H), Int32(H), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                    ks.enqueue_function(k_comb, part_buf, proj, Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
                else:
                    ks.enqueue_function(k_corrk, hk_v[l * 4 + 1], sk_v[l * 4 + 1], dltb, attnb, sidx[(1 - par) * 128 + l * 4 + 1], stg8[l * 4 + 1], scnt[(1 - par) * 128 + l * 4 + 1], proj, Int32(H), Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
            elif dsel and nh[l] > 0:
                ctx.stream().enqueue_wait_for(ev_in)
                ctx.enqueue_function(k_sel, attnb16, idx_dev, cnt_dev, cnt_log, Int32((pos * LAYERS + l) * 4 + 1), Int32(H), Int32(nh[l]), grid_dim=1, block_dim=1024)
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
            ks.enqueue_function(k_res, proj, x, grid_dim=ceildiv(H, 256), block_dim=256)
            if fast:
                ks.enqueue_function(k_normf, x, postnorm_v[l], hbf, grid_dim=1, block_dim=256)
            else:
                ks.enqueue_function(k_norm, x, postnorm_v[l], hbf, grid_dim=1, block_dim=1)
            # ---- gate_up_proj: fold GEMV, then the selected-channel correction ----
            ks.record_event(ev_in)
            if fast:
                ks.enqueue_function(k_gemvf, wq32[l * 4 + 2], wq32[l * 4 + 2], sw_v[l * 4 + 2], t0b, hbf, gu, Int32(2 * INTER), Int32(H), grid_dim=ceildiv(2 * INTER, 4), block_dim=128)
            else:
                ks.enqueue_function(k_gemv, wq_v[l * 4 + 2], wq_v[l * 4 + 2], sw_v[l * 4 + 2], t0b, hbf, gu, Int32(2 * INTER), Int32(H), grid_dim=ceildiv(2 * INTER, 4), block_dim=128)
            if stale and nh[l] > 0:
                ks.enqueue_function(k_sel, hbf16, sidx[par * 128 + l * 4 + 2], scnt[par * 128 + l * 4 + 2], cnt_log, Int32((pos * LAYERS + l) * 4 + 2), Int32(H), Int32(nh[l]), grid_dim=1, block_dim=1024)
                if ksplit:
                    ks.enqueue_function(k_corrks, hk_v[l * 4 + 2], sk_v[l * 4 + 2], dltb, hbf, sidx[(1 - par) * 128 + l * 4 + 2], stg8[l * 4 + 2], scnt[(1 - par) * 128 + l * 4 + 2], part_buf, Int32(2 * INTER), Int32(H), grid_dim=(ceildiv(2 * INTER, 128), KS), block_dim=128)
                    ks.enqueue_function(k_comb, part_buf, gu, Int32(2 * INTER), grid_dim=ceildiv(2 * INTER, 128), block_dim=128)
                else:
                    ks.enqueue_function(k_corrk, hk_v[l * 4 + 2], sk_v[l * 4 + 2], dltb, hbf, sidx[(1 - par) * 128 + l * 4 + 2], stg8[l * 4 + 2], scnt[(1 - par) * 128 + l * 4 + 2], gu, Int32(2 * INTER), Int32(H), grid_dim=ceildiv(2 * INTER, 128), block_dim=128)
            elif dsel and nh[l] > 0:
                ctx.stream().enqueue_wait_for(ev_in)
                ctx.enqueue_function(k_sel, hbf16, idx_dev, cnt_dev, cnt_log, Int32((pos * LAYERS + l) * 4 + 2), Int32(H), Int32(nh[l]), grid_dim=1, block_dim=1024)
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
            ks.enqueue_function(k_swi, gu, act, actb, grid_dim=ceildiv(INTER, 256), block_dim=256)
            # ---- down_proj: fold GEMV, then the selected-channel correction ----
            ks.record_event(ev_in)
            if fast:
                ks.enqueue_function(k_gemvf, wq32[l * 4 + 3], wq32[l * 4 + 3], sw_v[l * 4 + 3], t0b, actb, down, Int32(H), Int32(INTER), grid_dim=ceildiv(H, 4), block_dim=128)
            else:
                ks.enqueue_function(k_gemv, wq_v[l * 4 + 3], wq_v[l * 4 + 3], sw_v[l * 4 + 3], t0b, actb, down, Int32(H), Int32(INTER), grid_dim=ceildiv(H, 4), block_dim=128)
            if stale and ni[l] > 0:
                ks.enqueue_function(k_sel, actb16, sidx[par * 128 + l * 4 + 3], scnt[par * 128 + l * 4 + 3], cnt_log, Int32((pos * LAYERS + l) * 4 + 3), Int32(INTER), Int32(ni[l]), grid_dim=1, block_dim=1024)
                if ksplit:
                    ks.enqueue_function(k_corrks, hk_v[l * 4 + 3], sk_v[l * 4 + 3], dltb, actb, sidx[(1 - par) * 128 + l * 4 + 3], stg8[l * 4 + 3], scnt[(1 - par) * 128 + l * 4 + 3], part_buf, Int32(H), Int32(INTER), grid_dim=(ceildiv(H, 128), KS), block_dim=128)
                    ks.enqueue_function(k_comb, part_buf, down, Int32(H), grid_dim=ceildiv(H, 128), block_dim=128)
                else:
                    ks.enqueue_function(k_corrk, hk_v[l * 4 + 3], sk_v[l * 4 + 3], dltb, actb, sidx[(1 - par) * 128 + l * 4 + 3], stg8[l * 4 + 3], scnt[(1 - par) * 128 + l * 4 + 3], down, Int32(H), Int32(INTER), grid_dim=ceildiv(H, 128), block_dim=128)
            elif dsel and ni[l] > 0:
                ctx.stream().enqueue_wait_for(ev_in)
                ctx.enqueue_function(k_sel, actb16, idx_dev, cnt_dev, cnt_log, Int32((pos * LAYERS + l) * 4 + 3), Int32(INTER), Int32(ni[l]), grid_dim=1, block_dim=1024)
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
            ks.enqueue_function(k_res, down, x, grid_dim=ceildiv(H, 256), block_dim=256)
        if fast:
            ks.enqueue_function(k_normf, x, fnorm, hbf, grid_dim=1, block_dim=256)
        else:
            ks.enqueue_function(k_norm, x, fnorm, hbf, grid_dim=1, block_dim=1)
        if fast:
            ks.enqueue_function(k_lgf, hbf, emb, lg, grid_dim=ceildiv(V, 4), block_dim=128)
        else:
            ks.enqueue_function(k_lg, hbf, emb, lg, grid_dim=ceildiv(V, 256), block_dim=256)
        var tv = DeviceBuffer[DType.int32](ctx, tgt.unsafe_ptr().unsafe_offset(pos), 1, owning=False)
        var lv = DeviceBuffer[DType.float32](ctx, loss.unsafe_ptr().unsafe_offset(pos), 1, owning=False)
        var t1v = DeviceBuffer[DType.int32](ctx, top1.unsafe_ptr().unsafe_offset(pos), 1, owning=False)
        if fast:
            ks.enqueue_function(k_cef, lg, tv, lv, t1v, Int32(1), grid_dim=1, block_dim=256)
        else:
            ks.enqueue_function(k_ce, lg, tv, lv, t1v, Int32(1), grid_dim=1, block_dim=1)
        ks.synchronize()
        ctx.synchronize()
        if roll and q >= 63:
            ctx.enqueue_copy(dst_buf=t1h, src_buf=t1v)
            ctx.synchronize()
            gen[pi * 32 + q - 63] = Int(t1h[0])
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
    var total_ce = Float64(0)
    var matches = 0
    for pos in range(steps):
        total_ce += Float64(lh[pos])
        var o = Int(ob[4 * pos]) | (Int(ob[4 * pos + 1]) << 8) | (Int(ob[4 * pos + 2]) << 16) | (Int(ob[4 * pos + 3]) << 24)
        if Int(th[pos]) == o:
            matches += 1
        if not roll:
            print("MOJO_GLINT_SELECT_POSITION", pos, Int(th[pos]), Float64(lh[pos]))
    if roll:
        for pi in range(12):
            var line = String("MOJO_ROLL prompt ") + String(pi) + String(" tokens")
            for si in range(32):
                line += String(" ") + String(gen[pi * 32 + si])
            print(line)
    print("MOJO_GLINT_SELECT_DECODE steps", steps, "duration_s", elapsed, "ms_per_token", elapsed * 1000 / Float64(steps), "tokens_per_s", Float64(steps) / elapsed)
    print("MOJO_GLINT_SELECT_DECODE ppl", exp(total_ce / Float64(steps)), "oracle_top1", matches, "/", steps)
    print("MOJO_GLINT_SELECT_STATS percent", pct, "gathered_bytes_per_token_mean", Float64(bytes_total) / Float64(steps), "mean_selected_channel_fraction", frac_sum / Float64(steps * LAYERS * 4))
    print("MOJO_GLINT_SELECT_DECODE_DONE")
    # Keep all owning buffers alive while their typed borrowed views are used.
    _ = emb_u8.unsafe_ptr()
    _ = fnorm_u8.unsafe_ptr()
    _ = kvk.unsafe_ptr()
    _ = kvv.unsafe_ptr()
    _ = table_u8.unsafe_ptr()
    _ = len(in_norm) + len(post_norm) + len(lq) + len(plugk_host) + len(plugk_w) + len(ls) + len(hk_u8) + len(sk_f) + len(stg8) + len(stg32) + len(sidx) + len(scnt) + len(ev_pref)
    _ = stg_all.unsafe_ptr()
    _ = sel_idx_all.unsafe_ptr()
    _ = sel_cnt_all.unsafe_ptr()
