# Project 7 lean GLINT decoder: only the transient ephemeral path of p7_glint_select_decode.mojo
# (mode fast+attn+dsel+kmaj+vgat+skel+ksplit+fix+kgemv+ahead with GLINT_EPHEMERAL=2), same loader, pin, kernels and launch order.
# Per linear: the compute stream runs the K-major fold GEMV while the gather stream selects the current-|x| top channels and copies
# their plug rows from pinned host memory into one device slot; the compute stream then applies the split correction and combines.
# Usage: mojo build -I mojo -o p7_glint_lean mojo/p7_glint_lean.mojo
#        GLINT_EPHEMERAL=2 ./p7_glint_lean PIN MODEL_DIR ARTIFACTS_DIR STEPS PERCENT 0 MODE 0 0 0
#        MODE containing "roll" = free-running rollout (12 prompts x 32 greedy steps); otherwise teacher-forced.
#        Positional args 6, 8, 9, 10 (fixed_n, ends, endpct, pct2) are kept for CLI compatibility and must be 0.
from std.math import ceildiv, exp, cos, sin, pow
from std.sys import argv
from std.os import getenv
from std.time import perf_counter_ns
from std.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from p7_safetensors import parse_header, STEntry
from p7_forward import bf16r, embed_gather, rope, residual_add, swiglu, read_range
from p7_glint_gemv_km import glint_gemv_km, gemv_km_combine, GTB
from p7_decode_fast_kernels import rms_norm_block, logits_warp, ce_argmax_block, margin_block
from p7_attention_split import attention_split_partial, attention_split_combine, NCH
from p7_select_device import select_topk, transpose_scales, gather_rows_fix, apply_corr_ksc, combine_corr, KS, GB
from p7_plug_planes import split_planes, apply_corr_ksc_planes, select_topk2, apply_corr_ksc_tier, gather_planes2, apply_corr_ksc_tier_x, combine_corr2

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

def parse_uint(s: String, what: String) raises -> Int:
    var v = 0
    for b in s.as_bytes():
        if b < 48 or b > 57:
            raise Error(what + " must be a decimal integer")
        v = v * 10 + Int(b) - 48
    return v

def parse_ints(text: String) -> List[Int]:
    var vals = List[Int]()
    var cur = 0
    var have = False
    for b in text.as_bytes():
        if b >= 48 and b <= 57:
            cur = cur * 10 + Int(b) - 48
            have = True
        elif have:
            vals.append(cur)
            cur = 0
            have = False
    if have:
        vals.append(cur)
    return vals^

# "a,b,c,d" -> four per-class values; unset entries keep `dflt`.
def parse_class4(s: String, dflt: Int) -> List[Int]:
    var out = List[Int]()
    for _ in range(4):
        out.append(dflt)
    var ci = 0
    var cv = 0
    for b in s.as_bytes():
        if b == 44:
            out[ci] = cv
            ci += 1
            cv = 0
        else:
            cv = cv * 10 + Int(b) - 48
    if s != "":
        out[ci] = cv
    return out^

def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error("Usage: p7_glint_lean PIN MODEL_DIR ARTIFACTS_DIR [steps (1..2047)] [percent] 0 [mode] 0 0 0")
    if getenv("GLINT_EPHEMERAL", "0") != "2":
        raise Error("p7_glint_lean implements only the transient ephemeral path; set GLINT_EPHEMERAL=2")
    var pin_dir = String(args[1])
    var pin_path = pin_dir + "/model-00001-of-00002.safetensors"
    var pin_path2 = pin_dir + "/model-00002-of-00002.safetensors"
    var model_dir = String(args[2])
    var art_dir = String(args[3])
    var steps = 1
    if len(args) > 4:
        steps = parse_uint(String(args[4]), "steps")
    if steps < 1 or steps > NT:
        raise Error("steps outside 1..2047")
    var pct = 10
    if len(args) > 5:
        pct = parse_uint(String(args[5]), "percent")
    if pct < 0 or pct > 100:
        raise Error("percent must be 0..100")
    for ai in range(6, len(args)):
        if ai != 7 and ai <= 10 and parse_uint(String(args[ai]), "positional arg " + String(ai)) != 0:
            raise Error("positional arg " + String(ai) + " (fixed_n/ends/endpct/pct2) is not supported by the lean decoder; pass 0")
    var roll = len(args) > 7 and "roll" in String(args[7])
    if roll:
        steps = 12 * 95
    var ctx = DeviceContext()
    var toks_i = parse_ints(open(art_dir + "/eval_tokens_2048.txt", "r").read())
    if len(toks_i) != CTX:
        raise Error("expected 2048 pinned eval tokens")
    var toks = List[Int32]()
    for t in toks_i:
        toks.append(Int32(t))
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

    # Plug bytes stay in pinned host memory; the gather kernel reads them zero-copy.
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

    # GLINT_TIER2_REAL=1: the two-tier plug moves real 2-bit rows. Each linear's K-major plug is also kept as two pinned 2-bit planes
    # (p7_plug_planes.mojo); wide-set channels fetch the HI row, top-set channels the HI and LO rows. Outputs equal the emulation.
    var tier_real = getenv("GLINT_TIER2_REAL", "0") == "1"
    # GLINT_TIER2_FUSED (on top of GLINT_TIER2_REAL): "sel" = one selection launch for both sets (select_topk2), gathers and
    # corrections unfused; "x" = also one gather launch (gather_planes2) and one correction + combine with the unfused arithmetic
    # (apply_corr_ksc_tier_x, combine_corr2); "1" = like x but the correction walks only the wide list (apply_corr_ksc_tier), which
    # changes the summation order. sel and x are bit-identical to GLINT_TIER2_REAL.
    var fused_env = getenv("GLINT_TIER2_FUSED", "0")
    if fused_env != "0" and fused_env != "1" and fused_env != "x" and fused_env != "sel":
        raise Error("GLINT_TIER2_FUSED must be 0, sel, x or 1")
    var fused_sel = fused_env != "0"
    var fused_gat = fused_env == "1" or fused_env == "x"
    var fused_corr = fused_env == "1"
    var fused_x = fused_env == "x"
    if fused_sel and not tier_real:
        raise Error("GLINT_TIER2_FUSED needs GLINT_TIER2_REAL=1")
    var k_spl = ctx.compile_function[split_planes[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var plane_hi_host = List[type_of(ctx.enqueue_create_host_buffer[DType.uint8](1))]()
    var plane_lo_host = List[type_of(ctx.enqueue_create_host_buffer[DType.uint8](1))]()
    var load_t0 = perf_counter_ns()
    var k_tr = ctx.compile_function[transpose_plug[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_trs = ctx.compile_function[transpose_scales[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var hk_u8 = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var sk_f = List[type_of(ctx.enqueue_create_buffer[DType.float32](1))]()
    var emb_u8 = bf16_from(ctx, "model.embed_tokens.weight", s1_h, s2_h, s1_path, s2_path)
    var fnorm_u8 = bf16_from(ctx, "model.norm.weight", s1_h, s2_h, s1_path, s2_path)
    var in_norm = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var post_norm = List[type_of(ctx.enqueue_create_buffer[DType.uint8](1))]()
    var plugk_host = List[type_of(ctx.enqueue_create_host_buffer[DType.uint8](1))]()
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
            var rm_h = load_glint(ctx,nm,nbytes,"U8",pin_h,pin_h2,pin_path,pin_path2)
            var rm_s = load_glint(ctx,nm+".absmax",nout[j]*kin[j]//64*4,"F32",pin_h,pin_h2,pin_path,pin_path2)
            var rm_sv = DeviceBuffer[DType.float32](ctx, rm_s.unsafe_ptr().unsafe_bitcast[Float32](), nout[j] * kin[j] // 64, owning=False)
            var hb0 = ctx.enqueue_create_buffer[DType.uint8](nbytes)
            var sb0 = ctx.enqueue_create_buffer[DType.float32](nout[j] * kin[j] // 64)
            ctx.enqueue_function(k_tr, rm_h, hb0, Int32(nout[j]), Int32(kin[j]), grid_dim=ceildiv(kin[j] * (nout[j] // 2), 256), block_dim=256)
            ctx.enqueue_function(k_trs, rm_sv, sb0, Int32(nout[j]), Int32(kin[j]), grid_dim=ceildiv(nout[j] * (kin[j] // 64), 256), block_dim=256)
            ctx.synchronize()
            _ = rm_s.unsafe_ptr()
            _ = rm_h.unsafe_ptr()
            hk_u8.append(hb0^)
            sk_f.append(sb0^)
            var rowmaj = load_glint_host(ctx,nm+".glint_plug",nbytes,"U8",pin_h,pin_h2,pin_path,pin_path2)
            var src_dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
            var dst_dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
            ctx.enqueue_copy(dst_buf=src_dev, src_buf=rowmaj)
            ctx.enqueue_function(k_tr, src_dev, dst_dev, Int32(nout[j]), Int32(kin[j]), grid_dim=ceildiv(kin[j] * (nout[j] // 2), 256), block_dim=256)
            var pk = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
            ctx.enqueue_copy(dst_buf=pk, src_buf=dst_dev)
            if tier_real:
                var hi_dev = ctx.enqueue_create_buffer[DType.uint8](nbytes // 2)
                var lo_dev = ctx.enqueue_create_buffer[DType.uint8](nbytes // 2)
                ctx.enqueue_function(k_spl, dst_dev, hi_dev, lo_dev, Int32(nout[j]), Int32(kin[j]), grid_dim=ceildiv(kin[j] * (nout[j] // 4), 256), block_dim=256)
                var hh = ctx.enqueue_create_host_buffer[DType.uint8](nbytes // 2)
                var lh = ctx.enqueue_create_host_buffer[DType.uint8](nbytes // 2)
                ctx.enqueue_copy(dst_buf=hh, src_buf=hi_dev)
                ctx.enqueue_copy(dst_buf=lh, src_buf=lo_dev)
                ctx.synchronize()
                plane_hi_host.append(hh^)
                plane_lo_host.append(lh^)
            ctx.synchronize()
            plugk_host.append(pk^)
    var table_ent=pin_h[0][find_entry(pin_h,"_glint_cells")].copy()
    if table_ent.dtype!="F32" or len(table_ent.shape)!=2 or table_ent.shape[0]!=16 or table_ent.shape[1]!=16:
        raise Error("invalid pin _glint_cells shape")
    var table_u8=load_glint(ctx,"_glint_cells",1024,"F32",pin_h,pin_h2,pin_path,pin_path2)
    print("GLINT weights loaded in", Float64(perf_counter_ns() - load_t0) / 1e9, "s")

    var emb = DeviceBuffer[DType.bfloat16](ctx, emb_u8.unsafe_ptr().unsafe_bitcast[BFloat16](), V * H, owning=False)
    var fnorm = DeviceBuffer[DType.bfloat16](ctx, fnorm_u8.unsafe_ptr().unsafe_bitcast[BFloat16](), H, owning=False)
    var table = DeviceBuffer[DType.float32](ctx,table_u8.unsafe_ptr().unsafe_bitcast[Float32](),256,owning=False)
    var innorm_v = List[DeviceBuffer[DType.bfloat16]]()
    var postnorm_v = List[DeviceBuffer[DType.bfloat16]]()
    for l in range(LAYERS):
        innorm_v.append(DeviceBuffer[DType.bfloat16](ctx, in_norm[l].unsafe_ptr().unsafe_bitcast[BFloat16](), H, owning=False))
        postnorm_v.append(DeviceBuffer[DType.bfloat16](ctx, post_norm[l].unsafe_ptr().unsafe_bitcast[BFloat16](), H, owning=False))

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
    var k_rope = ctx.compile_function[rope[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_res = ctx.compile_function[residual_add[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_swi = ctx.compile_function[swiglu[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_kv = ctx.compile_function[kv_write[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_attp = ctx.compile_function[attention_split_partial[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_attc = ctx.compile_function[attention_split_combine[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_normf = ctx.compile_function[rms_norm_block[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_lgf = ctx.compile_function[logits_warp[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_cef = ctx.compile_function[ce_argmax_block[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_mrg = ctx.compile_function[margin_block[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_gkm = ctx.compile_function[glint_gemv_km[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_gkc = ctx.compile_function[gemv_km_combine[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_sel = ctx.compile_function[select_topk[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_gatx = ctx.compile_function[gather_rows_fix[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_corrksc = ctx.compile_function[apply_corr_ksc[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_comb = ctx.compile_function[combine_corr[ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_corrpl = ctx.compile_function[apply_corr_ksc_planes[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_sel2 = ctx.compile_function[select_topk2[ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_corrt = ctx.compile_function[apply_corr_ksc_tier[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_gat2 = ctx.compile_function[gather_planes2[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_corrx = ctx.compile_function[apply_corr_ksc_tier_x[ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
    var k_comb2 = ctx.compile_function[combine_corr2[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin]]()
    var k_mk = ctx.compile_function[make_tables[ImmUntrackedOrigin, ImmUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()

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
    # GLINT_PLUG_BITS = b < 4 (landing emulation of a partial plug): each cell of a hole row is replaced by the mean of the
    # cells sharing the plug's top b bits, so the low 4-b plug bits change nothing. Same bytes move (speed unchanged);
    # outputs are those of a b-bit prefix plug on the current (ordered) table.
    var pbits = parse_uint(getenv("GLINT_PLUG_BITS", "4"), "GLINT_PLUG_BITS")
    # GLINT_TIER2_PCT = p2 (> percent): two plug levels at once. The top `percent` channels get the full plug and the next
    # channels up to p2 get the GLINT_PLUG_BITS-bit prefix. Applied as corr(prefix table, top p2) + corr(full - prefix, top
    # percent), which is exact because the correction is linear in channels and the top set is inside the wider set.
    var tier2 = parse_uint(getenv("GLINT_TIER2_PCT", "0"), "GLINT_TIER2_PCT")
    var t0x = ctx.enqueue_create_buffer[DType.float32](256)
    var dltb2 = ctx.enqueue_create_buffer[DType.float32](256)
    var dltbd = ctx.enqueue_create_buffer[DType.float32](256)
    if tier_real and (tier2 == 0 or pbits != 2):
        raise Error("GLINT_TIER2_REAL needs GLINT_TIER2_PCT and GLINT_PLUG_BITS=2")
    if tier2 > 0:
        if pbits >= 4:
            raise Error("GLINT_TIER2_PCT needs GLINT_PLUG_BITS < 4")
        var c4 = ctx.enqueue_create_host_buffer[DType.float32](256)
        var c2 = ctx.enqueue_create_host_buffer[DType.float32](256)
        var cd = ctx.enqueue_create_host_buffer[DType.float32](256)
        ctx.enqueue_copy(dst_buf=c4, src_buf=table)
        ctx.synchronize()
        var g2 = 1 << (4 - pbits)
        for hh in range(16):
            for q0 in range(0, 16, g2):
                var acc2 = Float32(0)
                for r in range(g2):
                    acc2 += c4[hh * 16 + q0 + r]
                for r in range(g2):
                    c2[hh * 16 + q0 + r] = acc2 / Float32(g2)
        for i2 in range(256):
            cd[i2] = c4[i2] - c2[i2] + parent_h[i2 // 16]
        var c2d = ctx.enqueue_create_buffer[DType.float32](256)
        var cdd = ctx.enqueue_create_buffer[DType.float32](256)
        ctx.enqueue_copy(dst_buf=c2d, src_buf=c2)
        ctx.enqueue_copy(dst_buf=cdd, src_buf=cd)
        ctx.enqueue_function(k_mk, c2d, parent_b, t0x, dltb2, grid_dim=1, block_dim=256)
        ctx.enqueue_function(k_mk, cdd, parent_b, t0x, dltbd, grid_dim=1, block_dim=256)
        ctx.synchronize()
        print("MOJO_GLINT_TIER2 pct", tier2, "tier2_bits", pbits)
    if pbits < 4 and tier2 == 0:
        var thh = ctx.enqueue_create_host_buffer[DType.float32](256)
        ctx.enqueue_copy(dst_buf=thh, src_buf=table)
        ctx.synchronize()
        var g = 1 << (4 - pbits)
        for hh in range(16):
            for q0 in range(0, 16, g):
                var acc = Float32(0)
                for r in range(g):
                    acc += thh[hh * 16 + q0 + r]
                for r in range(g):
                    thh[hh * 16 + q0 + r] = acc / Float32(g)
        ctx.enqueue_copy(dst_buf=table, src_buf=thh)
        ctx.synchronize()
        print("MOJO_GLINT_PLUG_BITS", pbits)
    ctx.enqueue_function(k_mk, table, parent_b, t0b, dltb, grid_dim=1, block_dim=256)
    ctx.synchronize()

    var hk_v = List[DeviceBuffer[DType.uint8]]()
    var sk_v = List[DeviceBuffer[DType.float32]]()
    var hk32 = List[DeviceBuffer[DType.uint32]]()
    var plugk_w = List[DeviceBuffer[DType.uint32]]()
    for i in range(LAYERS * 4):
        var jj = i % 4
        hk_v.append(DeviceBuffer[DType.uint8](ctx, hk_u8[i].unsafe_ptr(), nout[jj] * kin[jj] // 2, owning=False))
        sk_v.append(DeviceBuffer[DType.float32](ctx, sk_f[i].unsafe_ptr(), nout[jj] * kin[jj] // 64, owning=False))
        hk32.append(DeviceBuffer[DType.uint32](ctx, hk_u8[i].unsafe_ptr().unsafe_bitcast[UInt32](), nout[jj] * kin[jj] // 8, owning=False))
        plugk_w.append(DeviceBuffer[DType.uint32](ctx, plugk_host[i].unsafe_ptr().unsafe_bitcast[UInt32](), kin[jj] * nout[jj] // 8, owning=False))
    var phi_w = List[DeviceBuffer[DType.uint32]]()
    var plo_w = List[DeviceBuffer[DType.uint32]]()
    for i in range(len(plane_hi_host)):
        var jj = i % 4
        phi_w.append(DeviceBuffer[DType.uint32](ctx, plane_hi_host[i].unsafe_ptr().unsafe_bitcast[UInt32](), kin[jj] * nout[jj] // 16, owning=False))
        plo_w.append(DeviceBuffer[DType.uint32](ctx, plane_lo_host[i].unsafe_ptr().unsafe_bitcast[UInt32](), kin[jj] * nout[jj] // 16, owning=False))
    # One transient staging slot per linear class, reused by every layer; channel-addressed (row c of the K-major plug).
    var stg_off = List[Int]()
    var stg_total = 0
    for j in range(4):
        stg_off.append(stg_total)
        stg_total += kin[j] * nout[j] // 2
    var stg_all = ctx.enqueue_create_buffer[DType.uint8](stg_total + 16)
    var stg8 = List[DeviceBuffer[DType.uint8]]()
    var stg32 = List[DeviceBuffer[DType.uint32]]()
    for j in range(4):
        stg8.append(DeviceBuffer[DType.uint8](ctx, stg_all.unsafe_ptr().unsafe_offset(stg_off[j]), kin[j] * nout[j] // 2, owning=False))
        stg32.append(DeviceBuffer[DType.uint32](ctx, stg_all.unsafe_ptr().unsafe_offset(stg_off[j]).unsafe_bitcast[UInt32](), kin[j] * nout[j] // 8, owning=False))
    # Plane staging (GLINT_TIER2_REAL): per class one HI and one LO channel-addressed slot of kin * nout / 4 bytes each.
    var pst_all = ctx.enqueue_create_buffer[DType.uint8](stg_total + 16 if tier_real else 16)
    var hst8 = List[DeviceBuffer[DType.uint8]]()
    var lst8 = List[DeviceBuffer[DType.uint8]]()
    var hst32 = List[DeviceBuffer[DType.uint32]]()
    var lst32 = List[DeviceBuffer[DType.uint32]]()
    if tier_real:
        for j in range(4):
            var q4 = kin[j] * nout[j] // 4
            var hp = pst_all.unsafe_ptr().unsafe_offset(stg_off[j])
            var lp = pst_all.unsafe_ptr().unsafe_offset(stg_off[j] + q4)
            hst8.append(DeviceBuffer[DType.uint8](ctx, hp, q4, owning=False))
            lst8.append(DeviceBuffer[DType.uint8](ctx, lp, q4, owning=False))
            hst32.append(DeviceBuffer[DType.uint32](ctx, hp.unsafe_bitcast[UInt32](), q4 // 4, owning=False))
            lst32.append(DeviceBuffer[DType.uint32](ctx, lp.unsafe_bitcast[UInt32](), q4 // 4, owning=False))
    # Last-fetch stamp per (class, channel). Stamps grow across layers, so with no reuse window every selected row is fetched.
    # GLINT_ORACLE=1 (timing study): every token runs twice. Pass A is the normal path and records each linear's true selection;
    # pass B recomputes the token with all 128 plug copies queued at its start (perfect prediction, full overlap). Only pass B is
    # timed as MOJO_ORACLE; outputs come from pass B and must equal the normal path. Needs a per-linear staging slot (whole plug).
    var oracle = getenv("GLINT_ORACLE", "0") == "1"
    # GLINT_ORACLE_REUSE = W tokens: in pass B a row already fetched into its linear's slot within the last W tokens is not
    # fetched again (plug rows never change, so outputs cannot change; only link bytes drop).
    var oreuse = parse_uint(getenv("GLINT_ORACLE_REUSE", "0"), "GLINT_ORACLE_REUSE")
    # GLINT_REUSE = W tokens (real decode, no oracle): each linear keeps its own channel-addressed slot and a selected row
    # fetched within the last W tokens is not fetched again. GLINT_REUSE_STAT=1 samples, every 16 tokens, the plug bytes
    # whose rows were fetched within the last W tokens (the VRAM a bounded pool would need for that window).
    var reuse = parse_uint(getenv("GLINT_REUSE", "0"), "GLINT_REUSE")
    var rstat = getenv("GLINT_REUSE_STAT", "0") == "1"
    var perlin = oracle or reuse > 0
    if tier_real and perlin:
        raise Error("GLINT_TIER2_REAL does not combine with GLINT_ORACLE or GLINT_REUSE")
    # The last 4 * INTER flags are the LO-plane fetch stamps of the real two-tier path (class c at (LAYERS * 4 + 4 + c) * INTER).
    var nflag = (LAYERS * 4 + 8) * INTER
    var ovl_flags = ctx.enqueue_create_buffer[DType.int32](nflag)
    var ovl_zero = ctx.enqueue_create_host_buffer[DType.int32](nflag)
    for i in range(nflag):
        ovl_zero[i] = 0
    ctx.enqueue_copy(dst_buf=ovl_flags, src_buf=ovl_zero)
    var idx_b = ctx.enqueue_create_buffer[DType.int32](8192)
    var cnt_b = ctx.enqueue_create_buffer[DType.int32](1)
    var idx_all = ctx.enqueue_create_buffer[DType.int32](LAYERS * 4 * 8192 if perlin else 8)
    var cnt_all = ctx.enqueue_create_buffer[DType.int32](LAYERS * 4 if perlin else 8)
    var ostg_total = 0
    var ostg_off = List[Int]()
    for i in range(LAYERS * 4):
        ostg_off.append(ostg_total)
        if perlin:
            ostg_total += kin[i % 4] * nout[i % 4] // 2
    var ostg_all = ctx.enqueue_create_buffer[DType.uint8](ostg_total + 16)
    var idxL = List[DeviceBuffer[DType.int32]]()
    var cntL = List[DeviceBuffer[DType.int32]]()
    var ostg8 = List[DeviceBuffer[DType.uint8]]()
    var ostg32 = List[DeviceBuffer[DType.uint32]]()
    for i in range(LAYERS * 4):
        if perlin:
            idxL.append(DeviceBuffer[DType.int32](ctx, idx_all.unsafe_ptr().unsafe_offset(i * 8192), 8192, owning=False))
            cntL.append(DeviceBuffer[DType.int32](ctx, cnt_all.unsafe_ptr().unsafe_offset(i), 1, owning=False))
            var sz = kin[i % 4] * nout[i % 4] // 2
            ostg8.append(DeviceBuffer[DType.uint8](ctx, ostg_all.unsafe_ptr().unsafe_offset(ostg_off[i]), sz, owning=False))
            ostg32.append(DeviceBuffer[DType.uint32](ctx, ostg_all.unsafe_ptr().unsafe_offset(ostg_off[i]).unsafe_bitcast[UInt32](), sz // 4, owning=False))
        else:
            idxL.append(DeviceBuffer[DType.int32](ctx, idx_b.unsafe_ptr(), 8192, owning=False))
            cntL.append(DeviceBuffer[DType.int32](ctx, cnt_b.unsafe_ptr(), 1, owning=False))
    var dummy_log = ctx.enqueue_create_buffer[DType.int32](4)
    # Selected-channel counts of the current token: slot i = top set of linear i, LAYERS * 4 + i = wide (tier-2) set.
    var blog = ctx.enqueue_create_buffer[DType.int32](LAYERS * 8)
    var blog_h = ctx.enqueue_create_host_buffer[DType.int32](LAYERS * 8)
    var byte_log = getenv("GLINT_BYTE_LOG", "0") == "1"
    var link_bytes = 0
    var part_buf = ctx.enqueue_create_buffer[DType.float32](KS * 16384)
    var part_buf2 = ctx.enqueue_create_buffer[DType.float32](KS * 16384 if fused_x else 16)
    var part_g = ctx.enqueue_create_buffer[DType.float32](32 * 16384)
    var hbf16 = DeviceBuffer[DType.uint16](ctx, hbf.unsafe_ptr().unsafe_bitcast[UInt16](), H, owning=False)
    var attnb16 = DeviceBuffer[DType.uint16](ctx, attnb.unsafe_ptr().unsafe_bitcast[UInt16](), H, owning=False)
    var actb16 = DeviceBuffer[DType.uint16](ctx, actb.unsafe_ptr().unsafe_bitcast[UInt16](), INTER, owning=False)
    # Per linear class (qkv, o, gate+up, down): bf16 input, its raw-bits view for selection, fp32 output, K-split count.
    var xin_c = List[DeviceBuffer[DType.bfloat16]]()
    var xin16_c = List[DeviceBuffer[DType.uint16]]()
    var yout_c = List[DeviceBuffer[DType.float32]]()
    var splits_c = List[Int]()
    xin_c.append(hbf)
    xin_c.append(attnb)
    xin_c.append(hbf)
    xin_c.append(actb)
    xin16_c.append(hbf16)
    xin16_c.append(attnb16)
    xin16_c.append(hbf16)
    xin16_c.append(actb16)
    yout_c.append(qkv)
    yout_c.append(proj)
    yout_c.append(gu)
    yout_c.append(down)
    splits_c.append(24)
    splits_c.append(24)
    splits_c.append(24)
    splits_c.append(32)

    var ks = ctx.create_stream()
    var gs = ctx.create_stream()
    # GLINT_ONE_STREAM=1: the gather stream is the compute stream, so every cross-stream event wait becomes a same-stream no-op.
    if getenv("GLINT_ONE_STREAM", "0") == "1":
        gs = ks
        print("MOJO_GLINT_ONE_STREAM 1")
    var emb_done = ctx.create_event()
    var ev_in2 = List[type_of(emb_done)]()
    var ev_cd = List[type_of(emb_done)]()
    for _ in range(LAYERS * 4):
        ev_in2.append(ctx.create_event())
        ev_cd.append(ctx.create_event())

    # GLINT_ROLL_OFFSET: tokens added to each rollout prompt's start (64 gives 12 prompts disjoint from the default set).
    var roll_off = parse_uint(getenv("GLINT_ROLL_OFFSET", "0"), "GLINT_ROLL_OFFSET")
    # GLINT_MARGIN=1: record the bf16 logit gap top1 - top2 per token (printed per rollout prompt).
    var mrg_on = getenv("GLINT_MARGIN", "0") == "1"
    var marg = ctx.enqueue_create_buffer[DType.float32](steps)
    var m1h = ctx.enqueue_create_host_buffer[DType.float32](1)
    var margs = List[Float64]()
    for _ in range(steps):
        margs.append(0.0)
    # GLINT_ALT_FILE: integer triplets (prompt, generated step, token). At that step the run prints its own bf16 logit gap
    # between its chosen token and the listed token.
    var alt = List[Int]()
    var altp = getenv("GLINT_ALT_FILE", "")
    if altp != "":
        alt = parse_ints(open(altp, "r").read())
    var a1h = ctx.enqueue_create_host_buffer[DType.float32](1)
    var a2h = ctx.enqueue_create_host_buffer[DType.float32](1)
    # GLINT_CLASS_PCT = "q,o,gu,d": plug share per linear class (percent), replacing the uniform share in every layer.
    # GLINT_GB = "q,o,gu,d": blocks of the transient gather kernel per class (default GB each).
    var cls_pct = parse_class4(getenv("GLINT_CLASS_PCT", ""), -1)
    var gbk = parse_class4(getenv("GLINT_GB", ""), GB)
    print("MOJO_GLINT_CLASS pct", cls_pct[0], cls_pct[1], cls_pct[2], cls_pct[3], "gather_blocks", gbk[0], gbk[1], gbk[2], gbk[3])
    # GLINT_LAYER_PCT = 32 comma-separated percents: plug share per layer (every class), replacing the uniform share.
    # A class share from GLINT_CLASS_PCT, when set, still wins for that class.
    var lay_pct = List[Int]()
    var lps = getenv("GLINT_LAYER_PCT", "")
    if lps != "":
        lay_pct = parse_ints(lps)
        if len(lay_pct) != LAYERS:
            raise Error("GLINT_LAYER_PCT must hold 32 integers")
    var nc = List[Int]()
    for l in range(LAYERS):
        for c in range(4):
            var share = pct
            if len(lay_pct) > 0:
                share = lay_pct[l]
            if cls_pct[c] >= 0:
                share = cls_pct[c]
            nc.append((share * kin[c] + 99) // 100)
    var nc2 = List[Int]()
    for l in range(LAYERS):
        for c in range(4):
            nc2.append((tier2 * kin[c] + 99) // 100 if tier2 > 0 else 0)
    var idx_t2 = ctx.enqueue_create_buffer[DType.int32](8192)
    var cnt_t2 = ctx.enqueue_create_buffer[DType.int32](2)
    if tier2 > 0:
        var tb = 0
        for l in range(LAYERS):
            for c in range(4):
                tb += nc[l * 4 + c] * nout[c] // 2 + (nc2[l * 4 + c] - nc[l * 4 + c]) * nout[c] // 2 * pbits // 4
        print("MOJO_GLINT_TIER2 real_layout_plug_bytes_per_token", tb)
    var nb_top = 0
    var nb_wide = 0
    for i in range(LAYERS * 4):
        nb_top += nc[i] * nout[i % 4] // 2
        nb_wide += nc2[i] * nout[i % 4] // 2
    var nb_arm = nb_top
    if tier2 > 0:
        nb_arm = nb_top // 2 + nb_wide // 2 if tier_real else nb_wide
    print("MOJO_LINK_BYTES nominal_per_token", nb_arm, "tier_real", tier_real, "fused", fused_env)
    if len(lay_pct) > 0:
        var lb = 0
        for l in range(LAYERS):
            for c in range(4):
                lb += nc[l * 4 + c] * nout[c] // 2
        print("MOJO_GLINT_LAYER_PCT plug_bytes_per_token", lb)
    # GLINT_FORCE_FILE = 12*32 integers: in rollout mode the generated positions are fed these tokens instead of the run's own
    # argmax, so every position is scored in the same context.
    var frc = List[Int]()
    var frcp = getenv("GLINT_FORCE_FILE", "")
    if frcp != "":
        frc = parse_ints(open(frcp, "r").read())
        if len(frc) != 12 * 32:
            raise Error("GLINT_FORCE_FILE must hold 12*32 integers")
    # Overlap baseline (timing only, outputs invalid when set): GLINT_TIME_ABL=1 compute only (the host-to-slot copy
    # is skipped and nothing reads host memory; corrections use the stale slot), 2 copy only (per token, only the selection and the
    # host-to-slot copy for every linear), 3 selection only (2 without the copy), 4 no correction (the corrections and their
    # combine are skipped), 5 = 1 and 4 together.
    var abl_t = parse_uint(getenv("GLINT_TIME_ABL", "0"), "GLINT_TIME_ABL")
    if abl_t > 0:
        print("MOJO_GLINT_TIME_ABL", abl_t, "outputs invalid")
    print("MOJO_GLINT_SELECT_CONFIG lean_ephemeral2 roll", roll, "percent", pct, "roll_offset", roll_off, "margin", mrg_on, "forced", len(frc) > 0)
    print("MOJO_GLINT_SELECT_DECODE_BEGIN", steps)
    var t0 = perf_counter_ns()
    var gen = List[Int]()
    for _ in range(12 * 32):
        gen.append(0)
    var t1h = ctx.enqueue_create_host_buffer[DType.int32](1)
    var enq_ns = 0
    var orc_ns = 0
    var rs_sum = 0
    var rs_n = 0
    var flh = ctx.enqueue_create_host_buffer[DType.int32](nflag if rstat else 8)
    for pos in range(steps):
        var stamp_base = (pos + 1) * LAYERS
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
        if abl_t == 2 or abl_t == 3:
            for lr in range(LAYERS):
                for c in range(4):
                    var i = lr * 4 + c
                    if nc[i] == 0:
                        continue
                    gs.enqueue_function(k_sel, xin16_c[c], idx_b, cnt_b, dummy_log, Int32(0), Int32(kin[c]), Int32(nc[i]), Int32(0), Int32(0), grid_dim=1, block_dim=1024)
                    if abl_t == 2:
                        gs.enqueue_function(k_gatx, plugk_w[i], idx_b, cnt_b, stg32[c], ovl_flags, Int32(nout[c] // 8), Int32(c * INTER), Int32((stamp_base + lr) * 2 + 1000), Int32(0), grid_dim=gbk[c], block_dim=128)
            gs.synchronize()
            continue
        ctx.enqueue_function(k_embed, emb, tokd, x, grid_dim=ceildiv(H, 256), block_dim=256)
        ctx.stream().record_event(emb_done)
        ks.enqueue_wait_for(emb_done)
        for lr in range(LAYERS):
            for c in range(4):
                var i = lr * 4 + c
                var n = nout[c]
                var kd = kin[c]
                if c == 0:
                    ks.enqueue_function(k_normf, x, innorm_v[lr], hbf, grid_dim=1, block_dim=256)
                elif c == 1:
                    ks.enqueue_function(k_rope, qkv, cv, sv, rp, grid_dim=ceildiv(QKV, 256), block_dim=256)
                    ks.enqueue_function(k_kv, rp, kv_k[lr], kv_v[lr], Int32(q), grid_dim=ceildiv(NKV * HD, 256), block_dim=256)
                    ks.enqueue_function(k_attp, rp, kv_k[lr], kv_v[lr], att_pm, att_pd, att_pa, Int32(q), grid_dim=(NCH, NQ), block_dim=128)
                    ks.enqueue_function(k_attc, att_pm, att_pd, att_pa, attn, attnb, grid_dim=NQ, block_dim=HD)
                elif c == 2:
                    ks.enqueue_function(k_res, proj, x, grid_dim=ceildiv(H, 256), block_dim=256)
                    ks.enqueue_function(k_normf, x, postnorm_v[lr], hbf, grid_dim=1, block_dim=256)
                else:
                    ks.enqueue_function(k_swi, gu, act, actb, grid_dim=ceildiv(INTER, 256), block_dim=256)
                if nc[i] > 0:
                    ks.record_event(ev_in2[i])
                    gs.enqueue_wait_for(ev_in2[i])
                    if fused_sel:
                        gs.enqueue_function(k_sel2, xin16_c[c], idx_t2, cnt_t2, idxL[i], cntL[i], blog, Int32(LAYERS * 4 + i), Int32(i), Int32(kd), Int32(nc2[i]), Int32(nc[i]), grid_dim=1, block_dim=1024)
                        if fused_gat:
                            gs.enqueue_function(k_gat2, phi_w[i], plo_w[i], xin16_c[c], idx_t2, cnt_t2, hst32[c], lst32[c], Int32(n // 16), grid_dim=gbk[c], block_dim=128)
                        else:
                            gs.enqueue_function(k_gatx, phi_w[i], idx_t2, cnt_t2, hst32[c], ovl_flags, Int32(n // 16), Int32(c * INTER), Int32((stamp_base + lr) * 2 + 1000), Int32(0), grid_dim=gbk[c], block_dim=128)
                    elif tier2 > 0:
                        gs.enqueue_function(k_sel, xin16_c[c], idx_t2, cnt_t2, blog, Int32(LAYERS * 4 + i), Int32(kd), Int32(nc2[i]), Int32(0), Int32(0), grid_dim=1, block_dim=1024)
                        if tier_real:
                            gs.enqueue_function(k_gatx, phi_w[i], idx_t2, cnt_t2, hst32[c], ovl_flags, Int32(n // 16), Int32(c * INTER), Int32((stamp_base + lr) * 2 + 1000), Int32(0), grid_dim=gbk[c], block_dim=128)
                        else:
                            gs.enqueue_function(k_gatx, plugk_w[i], idx_t2, cnt_t2, stg32[c], ovl_flags, Int32(n // 8), Int32(c * INTER), Int32((stamp_base + lr) * 2 + 1000), Int32(0), grid_dim=gbk[c], block_dim=128)
                    if not fused_sel:
                        gs.enqueue_function(k_sel, xin16_c[c], idxL[i], cntL[i], blog, Int32(i), Int32(kd), Int32(nc[i]), Int32(0), Int32(0), grid_dim=1, block_dim=1024)
                    if tier2 > 0:
                        if tier_real and not fused_gat:
                            gs.enqueue_function(k_gatx, plo_w[i], idxL[i], cntL[i], lst32[c], ovl_flags, Int32(n // 16), Int32((LAYERS * 4 + 4 + c) * INTER), Int32((stamp_base + lr) * 2 + 1000), Int32(0), grid_dim=gbk[c], block_dim=128)
                    elif abl_t != 1 and abl_t != 5:
                        if reuse > 0:
                            gs.enqueue_function(k_gatx, plugk_w[i], idxL[i], cntL[i], ostg32[i], ovl_flags, Int32(n // 8), Int32((i + 4) * INTER), Int32(2000 + pos * 2), Int32(2 * reuse), grid_dim=gbk[c], block_dim=128)
                        else:
                            gs.enqueue_function(k_gatx, plugk_w[i], idxL[i], cntL[i], stg32[c], ovl_flags, Int32(n // 8), Int32(c * INTER), Int32((stamp_base + lr) * 2 + 1000), Int32(0), grid_dim=gbk[c], block_dim=128)
                    gs.record_event(ev_cd[i])
                ks.enqueue_function(k_gkm, hk32[i], sk_v[i], t0b, xin_c[c], part_g, Int32(n), Int32(kd), Int32(splits_c[c]), grid_dim=(ceildiv(n // 8, GTB), splits_c[c]), block_dim=GTB)
                ks.enqueue_function(k_gkc, part_g, yout_c[c], Int32(n), Int32(splits_c[c]), grid_dim=ceildiv(n, 128), block_dim=128)
                if nc[i] > 0:
                    ks.enqueue_wait_for(ev_cd[i])
                    if abl_t != 4 and abl_t != 5:
                        if fused_x:
                            ks.enqueue_function(k_corrx, hk_v[i], sk_v[i], dltb2, dltbd, xin_c[c], idx_t2, cnt_t2, idxL[i], cntL[i], hst8[c], lst8[c], part_buf, part_buf2, Int32(n), Int32(kd), grid_dim=(ceildiv(n, 128), KS), block_dim=128)
                            ks.enqueue_function(k_comb2, part_buf, part_buf2, yout_c[c], Int32(n), grid_dim=ceildiv(n, 128), block_dim=128)
                        elif fused_corr:
                            ks.enqueue_function(k_corrt, hk_v[i], sk_v[i], dltb2, dltbd, xin_c[c], xin16_c[c], idx_t2, hst8[c], lst8[c], cnt_t2, part_buf, Int32(n), Int32(kd), grid_dim=(ceildiv(n, 128), KS), block_dim=128)
                            ks.enqueue_function(k_comb, part_buf, yout_c[c], Int32(n), grid_dim=ceildiv(n, 128), block_dim=128)
                        elif tier_real:
                            ks.enqueue_function(k_corrpl, hk_v[i], sk_v[i], dltb2, xin_c[c], idx_t2, hst8[c], hst8[c], cnt_t2, part_buf, Int32(n), Int32(kd), Int32(0), grid_dim=(ceildiv(n, 128), KS), block_dim=128)
                            ks.enqueue_function(k_comb, part_buf, yout_c[c], Int32(n), grid_dim=ceildiv(n, 128), block_dim=128)
                            ks.enqueue_function(k_corrpl, hk_v[i], sk_v[i], dltbd, xin_c[c], idxL[i], hst8[c], lst8[c], cntL[i], part_buf, Int32(n), Int32(kd), Int32(1), grid_dim=(ceildiv(n, 128), KS), block_dim=128)
                            ks.enqueue_function(k_comb, part_buf, yout_c[c], Int32(n), grid_dim=ceildiv(n, 128), block_dim=128)
                        elif tier2 > 0:
                            ks.enqueue_function(k_corrksc, hk_v[i], sk_v[i], dltb2, xin_c[c], idx_t2, stg8[c], cnt_t2, part_buf, Int32(n), Int32(kd), grid_dim=(ceildiv(n, 128), KS), block_dim=128)
                            ks.enqueue_function(k_comb, part_buf, yout_c[c], Int32(n), grid_dim=ceildiv(n, 128), block_dim=128)
                            ks.enqueue_function(k_corrksc, hk_v[i], sk_v[i], dltbd, xin_c[c], idxL[i], stg8[c], cntL[i], part_buf, Int32(n), Int32(kd), grid_dim=(ceildiv(n, 128), KS), block_dim=128)
                            ks.enqueue_function(k_comb, part_buf, yout_c[c], Int32(n), grid_dim=ceildiv(n, 128), block_dim=128)
                        else:
                            ks.enqueue_function(k_corrksc, hk_v[i], sk_v[i], dltb, xin_c[c], idxL[i], ostg8[i] if reuse > 0 else stg8[c], cntL[i], part_buf, Int32(n), Int32(kd), grid_dim=(ceildiv(n, 128), KS), block_dim=128)
                            ks.enqueue_function(k_comb, part_buf, yout_c[c], Int32(n), grid_dim=ceildiv(n, 128), block_dim=128)
            ks.enqueue_function(k_res, down, x, grid_dim=ceildiv(H, 256), block_dim=256)
        ks.enqueue_function(k_normf, x, fnorm, hbf, grid_dim=1, block_dim=256)
        ks.enqueue_function(k_lgf, hbf, emb, lg, grid_dim=ceildiv(V, 4), block_dim=128)
        var tv = DeviceBuffer[DType.int32](ctx, tgt.unsafe_ptr().unsafe_offset(pos), 1, owning=False)
        var lv = DeviceBuffer[DType.float32](ctx, loss.unsafe_ptr().unsafe_offset(pos), 1, owning=False)
        var t1v = DeviceBuffer[DType.int32](ctx, top1.unsafe_ptr().unsafe_offset(pos), 1, owning=False)
        ks.enqueue_function(k_cef, lg, tv, lv, t1v, Int32(1), grid_dim=1, block_dim=256)
        var mv = DeviceBuffer[DType.float32](ctx, marg.unsafe_ptr().unsafe_offset(pos), 1, owning=False)
        if mrg_on:
            ks.enqueue_function(k_mrg, lg, t1v, mv, grid_dim=1, block_dim=256)
        enq_ns += Int(perf_counter_ns() - t_tok0)
        ks.synchronize()
        ctx.synchronize()
        if oracle:
            gs.synchronize()
            var tb0 = perf_counter_ns()
            var ost = Int32(4000000 + pos * 2)
            ctx.enqueue_function(k_embed, emb, tokd, x, grid_dim=ceildiv(H, 256), block_dim=256)
            ctx.stream().record_event(emb_done)
            ks.enqueue_wait_for(emb_done)
            gs.enqueue_wait_for(emb_done)
            for i in range(LAYERS * 4):
                if nc[i] > 0:
                    gs.enqueue_function(k_gatx, plugk_w[i], idxL[i], cntL[i], ostg32[i], ovl_flags, Int32(nout[i % 4] // 8), Int32((i + 4) * INTER), ost, Int32(2 * oreuse), grid_dim=gbk[i % 4], block_dim=128)
                    gs.record_event(ev_cd[i])
            for lr in range(LAYERS):
                for c in range(4):
                    var i = lr * 4 + c
                    var n = nout[c]
                    var kd = kin[c]
                    if c == 0:
                        ks.enqueue_function(k_normf, x, innorm_v[lr], hbf, grid_dim=1, block_dim=256)
                    elif c == 1:
                        ks.enqueue_function(k_rope, qkv, cv, sv, rp, grid_dim=ceildiv(QKV, 256), block_dim=256)
                        ks.enqueue_function(k_kv, rp, kv_k[lr], kv_v[lr], Int32(q), grid_dim=ceildiv(NKV * HD, 256), block_dim=256)
                        ks.enqueue_function(k_attp, rp, kv_k[lr], kv_v[lr], att_pm, att_pd, att_pa, Int32(q), grid_dim=(NCH, NQ), block_dim=128)
                        ks.enqueue_function(k_attc, att_pm, att_pd, att_pa, attn, attnb, grid_dim=NQ, block_dim=HD)
                    elif c == 2:
                        ks.enqueue_function(k_res, proj, x, grid_dim=ceildiv(H, 256), block_dim=256)
                        ks.enqueue_function(k_normf, x, postnorm_v[lr], hbf, grid_dim=1, block_dim=256)
                    else:
                        ks.enqueue_function(k_swi, gu, act, actb, grid_dim=ceildiv(INTER, 256), block_dim=256)
                    ks.enqueue_function(k_gkm, hk32[i], sk_v[i], t0b, xin_c[c], part_g, Int32(n), Int32(kd), Int32(splits_c[c]), grid_dim=(ceildiv(n // 8, GTB), splits_c[c]), block_dim=GTB)
                    ks.enqueue_function(k_gkc, part_g, yout_c[c], Int32(n), Int32(splits_c[c]), grid_dim=ceildiv(n, 128), block_dim=128)
                    if nc[i] > 0:
                        ks.enqueue_wait_for(ev_cd[i])
                        ks.enqueue_function(k_corrksc, hk_v[i], sk_v[i], dltb, xin_c[c], idxL[i], ostg8[i], cntL[i], part_buf, Int32(n), Int32(kd), grid_dim=(ceildiv(n, 128), KS), block_dim=128)
                        ks.enqueue_function(k_comb, part_buf, yout_c[c], Int32(n), grid_dim=ceildiv(n, 128), block_dim=128)
                ks.enqueue_function(k_res, down, x, grid_dim=ceildiv(H, 256), block_dim=256)
            ks.enqueue_function(k_normf, x, fnorm, hbf, grid_dim=1, block_dim=256)
            ks.enqueue_function(k_lgf, hbf, emb, lg, grid_dim=ceildiv(V, 4), block_dim=128)
            ks.enqueue_function(k_cef, lg, tv, lv, t1v, Int32(1), grid_dim=1, block_dim=256)
            if mrg_on:
                ks.enqueue_function(k_mrg, lg, t1v, mv, grid_dim=1, block_dim=256)
            ks.synchronize()
            gs.synchronize()
            ctx.synchronize()
            orc_ns += Int(perf_counter_ns() - tb0)
        if byte_log:
            ctx.enqueue_copy(dst_buf=blog_h, src_buf=blog)
            ctx.synchronize()
            for i in range(LAYERS * 4):
                if nc[i] == 0:
                    continue
                var top = Int(blog_h[i])
                var wide = Int(blog_h[LAYERS * 4 + i])
                var rb = nout[i % 4] // 2
                if tier_real:
                    link_bytes += top * rb // 2 + wide * rb // 2
                elif tier2 > 0:
                    link_bytes += wide * rb
                else:
                    link_bytes += top * rb
        if rstat and reuse > 0 and pos % 16 == 15:
            ctx.enqueue_copy(dst_buf=flh, src_buf=ovl_flags)
            ctx.synchronize()
            var rb = 0
            var lim = 2000 + pos * 2 + 1 - 2 * reuse
            for i in range(LAYERS * 4):
                var rw = nout[i % 4] // 2
                for ch in range(kin[i % 4]):
                    if Int(flh[(i + 4) * INTER + ch]) >= lim:
                        rb += rw
            rs_sum += rb
            rs_n += 1
        if mrg_on and (not roll or q >= 63):
            ctx.enqueue_copy(dst_buf=m1h, src_buf=mv)
            ctx.synchronize()
            margs[pos] = Float64(m1h[0])
        if roll and q >= 63:
            ctx.enqueue_copy(dst_buf=t1h, src_buf=t1v)
            ctx.synchronize()
            gen[pi * 32 + q - 63] = Int(t1h[0])
            for ai in range(len(alt) // 3):
                if alt[ai * 3] == pi and alt[ai * 3 + 1] == q - 63:
                    var av = DeviceBuffer[DType.float32](ctx, lg.unsafe_ptr().unsafe_offset(alt[ai * 3 + 2]), 1, owning=False)
                    var chv = DeviceBuffer[DType.float32](ctx, lg.unsafe_ptr().unsafe_offset(Int(t1h[0])), 1, owning=False)
                    ctx.enqueue_copy(dst_buf=a1h, src_buf=av)
                    ctx.enqueue_copy(dst_buf=a2h, src_buf=chv)
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
    var total_ce = Float64(0)
    var matches = 0
    for pos in range(steps):
        total_ce += Float64(lh[pos])
        var o = Int(ob[4 * pos]) | (Int(ob[4 * pos + 1]) << 8) | (Int(ob[4 * pos + 2]) << 16) | (Int(ob[4 * pos + 3]) << 24)
        if Int(th[pos]) == o:
            matches += 1
        if not roll:
            print("MOJO_GLINT_SELECT_POSITION", pos, Int(th[pos]), Float64(lh[pos]))
    if mrg_on and roll:
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
    print("MOJO_ENQ host_enqueue_ms_per_token", Float64(enq_ns) / 1e6 / Float64(steps))
    if byte_log:
        print("MOJO_LINK_BYTES measured_per_token", Float64(link_bytes) / Float64(steps), "tier_real", tier_real)
    if reuse > 0:
        print("MOJO_REUSE tokens", reuse, "slot_MB", Float64(ostg_total) / 1e6, "resident_window_MB_mean", Float64(rs_sum) / 1e6 / Float64(rs_n) if rs_n > 0 else 0.0, "samples", rs_n)
    if oracle:
        print("MOJO_ORACLE pass_b_ms_per_token", Float64(orc_ns) / 1e6 / Float64(steps), "plug_slots_MB", Float64(ostg_total) / 1e6, "reuse_tokens", oreuse)
    print("MOJO_GLINT_SELECT_DECODE_DONE")
    # Keep all owning buffers alive while their typed borrowed views are used.
    _ = emb_u8.unsafe_ptr()
    _ = fnorm_u8.unsafe_ptr()
    _ = kvk.unsafe_ptr()
    _ = kvv.unsafe_ptr()
    _ = table_u8.unsafe_ptr()
    _ = stg_all.unsafe_ptr()
    _ = pst_all.unsafe_ptr()
    _ = len(plane_hi_host) + len(plane_lo_host) + len(phi_w) + len(plo_w) + len(hst8) + len(lst8) + len(hst32) + len(lst32)
    _ = ostg_all.unsafe_ptr()
    _ = idx_all.unsafe_ptr()
    _ = cnt_all.unsafe_ptr()
    _ = part_g.unsafe_ptr()
    _ = hbf.unsafe_ptr()
    _ = attnb.unsafe_ptr()
    _ = actb.unsafe_ptr()
    _ = len(in_norm) + len(post_norm) + len(plugk_host) + len(hk_u8) + len(sk_f) + len(hk32) + len(plugk_w) + len(stg8) + len(stg32)
