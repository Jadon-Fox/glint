# Project 7 Mojo/MAX single-token full GLINT KV-cache decode — Phase B prototype, GLINT #116.
# Branch feat/p7-mojo-kv-decode, separate from existing p7_forward.mojo.
# Identical source model/tokenizer/oracle to the existing M1 INT8 prefill.
# Usage: mojo build -I mojo -o p7_glint_decode mojo/p7_glint_decode.mojo
#        ./p7_glint_decode GLINT_PIN_DIR MODEL_DIR ARTIFACTS_DIR [steps]   (steps default: all 2047 positions)
# PIN is a glint_pin_v1 directory (hole + plug + absmax + _glint_cells); see scripts/p7_mojo_glint_decode.sh.
# Note: a successful build is not evidence of a fidelity/throughput pass.
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

    var load_t0 = perf_counter_ns()
    var emb_u8 = bf16_from(ctx, "model.embed_tokens.weight", s1_h, s2_h, s1_path, s2_path)
    var fnorm_u8 = bf16_from(ctx, "model.norm.weight", s1_h, s2_h, s1_path, s2_path)
    var in_norm = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var post_norm = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var lq = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var lp = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
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
            lp.append(load_glint(ctx,nm+".glint_plug",nbytes,"U8",pin_h,pin_h2,pin_path,pin_path2))
            ls.append(load_glint(ctx,nm+".absmax",nout[j]*kin[j]//64*4,"F32",pin_h,pin_h2,pin_path,pin_path2))
    var table_ent=pin_h[0][find_entry(pin_h,"_glint_cells")].copy()
    if table_ent.dtype!="F32" or len(table_ent.shape)!=2 or table_ent.shape[0]!=16 or table_ent.shape[1]!=16:
        raise Error("invalid pin _glint_cells shape")
    var table_u8=load_glint(ctx,"_glint_cells",1024,"F32",pin_h,pin_h2,pin_path,pin_path2)
    print("GLINT weights loaded in", Float64(perf_counter_ns() - load_t0) / 1e9, "s")

    var emb = DeviceBuffer[DType.bfloat16](ctx, emb_u8.unsafe_ptr().unsafe_bitcast[BFloat16](), V * H, owning=False)
    var fnorm = DeviceBuffer[DType.bfloat16](ctx, fnorm_u8.unsafe_ptr().unsafe_bitcast[BFloat16](), H, owning=False)
    var wq_v = List[DeviceBuffer[DType.uint8]]()
    var plug_v = List[DeviceBuffer[DType.uint8]]()
    var sw_v = List[DeviceBuffer[DType.float32]]()
    var table = DeviceBuffer[DType.float32](ctx,table_u8.unsafe_ptr().unsafe_bitcast[Float32](),256,owning=False)
    var innorm_v = List[DeviceBuffer[DType.bfloat16]]()
    var postnorm_v = List[DeviceBuffer[DType.bfloat16]]()
    for l in range(LAYERS):
        innorm_v.append(DeviceBuffer[DType.bfloat16](ctx, in_norm[l].unsafe_ptr().unsafe_bitcast[BFloat16](), H, owning=False))
        postnorm_v.append(DeviceBuffer[DType.bfloat16](ctx, post_norm[l].unsafe_ptr().unsafe_bitcast[BFloat16](), H, owning=False))
        for j in range(4):
            wq_v.append(DeviceBuffer[DType.uint8](ctx, lq[l * 4 + j].unsafe_ptr(), nout[j] * kin[j] // 2, owning=False))
            plug_v.append(DeviceBuffer[DType.uint8](ctx, lp[l * 4 + j].unsafe_ptr(), nout[j] * kin[j] // 2, owning=False))
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
    var k_lg = ctx.compile_function[logits_one[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_ce = ctx.compile_function[ce_argmax[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()

    print("MOJO_GLINT_DECODE_BEGIN", steps)
    var t0 = perf_counter_ns()
    for pos in range(steps):
        tokh[0] = toks[pos]
        ctx.enqueue_copy(dst_buf=tokd, src_buf=tokh)
        var cv = DeviceBuffer[DType.float32](ctx, ct.unsafe_ptr().unsafe_offset(pos * (ROT // 2)), ROT // 2, owning=False)
        var sv = DeviceBuffer[DType.float32](ctx, st.unsafe_ptr().unsafe_offset(pos * (ROT // 2)), ROT // 2, owning=False)
        ctx.enqueue_function(k_embed, emb, tokd, x, grid_dim=ceildiv(H, 256), block_dim=256)
        for l in range(LAYERS):
            # Imported original math, launched with one thread because its T=2048
            # bound cannot safely be overridden at module import time.
            ctx.enqueue_function(k_norm, x, innorm_v[l], hbf, grid_dim=1, block_dim=1)
            ctx.enqueue_function(k_gemv, wq_v[l * 4 + 0], plug_v[l * 4 + 0], sw_v[l * 4 + 0], table, hbf, qkv, Int32(QKV), Int32(H), grid_dim=ceildiv(QKV, 4), block_dim=128)
            ctx.enqueue_function(k_rope, qkv, cv, sv, rp, grid_dim=ceildiv(QKV, 256), block_dim=256)
            ctx.enqueue_function(k_kv, rp, kv_k[l], kv_v[l], Int32(pos), grid_dim=ceildiv(NKV * HD, 256), block_dim=256)
            ctx.enqueue_function(k_attn, rp, kv_k[l], kv_v[l], attn, attnb, Int32(pos), grid_dim=(1, NQ), block_dim=HD)
            ctx.enqueue_function(k_gemv, wq_v[l * 4 + 1], plug_v[l * 4 + 1], sw_v[l * 4 + 1], table, attnb, proj, Int32(H), Int32(H), grid_dim=ceildiv(H, 4), block_dim=128)
            ctx.enqueue_function(k_res, proj, x, grid_dim=ceildiv(H, 256), block_dim=256)
            ctx.enqueue_function(k_norm, x, postnorm_v[l], hbf, grid_dim=1, block_dim=1)
            ctx.enqueue_function(k_gemv, wq_v[l * 4 + 2], plug_v[l * 4 + 2], sw_v[l * 4 + 2], table, hbf, gu, Int32(2 * INTER), Int32(H), grid_dim=ceildiv(2 * INTER, 4), block_dim=128)
            ctx.enqueue_function(k_swi, gu, act, actb, grid_dim=ceildiv(INTER, 256), block_dim=256)
            ctx.enqueue_function(k_gemv, wq_v[l * 4 + 3], plug_v[l * 4 + 3], sw_v[l * 4 + 3], table, actb, down, Int32(H), Int32(INTER), grid_dim=ceildiv(H, 4), block_dim=128)
            ctx.enqueue_function(k_res, down, x, grid_dim=ceildiv(H, 256), block_dim=256)
        ctx.enqueue_function(k_norm, x, fnorm, hbf, grid_dim=1, block_dim=1)
        ctx.enqueue_function(k_lg, hbf, emb, lg, grid_dim=ceildiv(V, 256), block_dim=256)
        var tv = DeviceBuffer[DType.int32](ctx, tgt.unsafe_ptr().unsafe_offset(pos), 1, owning=False)
        var lv = DeviceBuffer[DType.float32](ctx, loss.unsafe_ptr().unsafe_offset(pos), 1, owning=False)
        var t1v = DeviceBuffer[DType.int32](ctx, top1.unsafe_ptr().unsafe_offset(pos), 1, owning=False)
        ctx.enqueue_function(k_ce, lg, tv, lv, t1v, Int32(1), grid_dim=1, block_dim=1)
        ctx.synchronize()
        if (pos + 1) % 64 == 0:
            print("decode positions", pos + 1, "/", steps)
    var elapsed = Float64(perf_counter_ns() - t0) / 1e9
    var lh = ctx.enqueue_create_host_buffer[DType.float32](steps)
    var th = ctx.enqueue_create_host_buffer[DType.int32](steps)
    ctx.enqueue_copy(dst_buf=lh, src_buf=loss)
    ctx.enqueue_copy(dst_buf=th, src_buf=top1)
    ctx.synchronize()
    var total_ce = Float64(0)
    var matches = 0
    for pos in range(steps):
        total_ce += Float64(lh[pos])
        var o = Int(ob[4 * pos]) | (Int(ob[4 * pos + 1]) << 8) | (Int(ob[4 * pos + 2]) << 16) | (Int(ob[4 * pos + 3]) << 24)
        if Int(th[pos]) == o:
            matches += 1
        print("MOJO_GLINT_DECODE_POSITION", pos, Int(th[pos]), Float64(lh[pos]))
    print("MOJO_GLINT_DECODE steps", steps, "duration_s", elapsed, "ms_per_token", elapsed * 1000 / Float64(steps), "tokens_per_s", Float64(steps) / elapsed)
    print("MOJO_GLINT_DECODE ppl", exp(total_ce / Float64(steps)), "oracle_top1", matches, "/", steps)
    print("MOJO_GLINT_DECODE_DONE")
    # Keep all owning buffers alive while their typed borrowed views are used.
    _ = emb_u8.unsafe_ptr()
    _ = fnorm_u8.unsafe_ptr()
    _ = kvk.unsafe_ptr()
    _ = kvv.unsafe_ptr()
    _ = table_u8.unsafe_ptr()
    _ = len(in_norm) + len(post_norm) + len(lq) + len(lp) + len(ls)
