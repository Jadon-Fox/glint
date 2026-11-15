# Project 7 / GLINT: recorded step-through tracer for SmolLM2-135M (Llama architecture) on a glint_pin_v1 pin. Pure Mojo, CPU, f32.
# Runs the prompt token by token (KV cache) and records EVERY intermediate of the LAST prompt token as glint_trace_v2:
#   OUT_DIR/manifest.json   model dims, tokens, op list (embedding decode, RMSNorm, q/k/v/o/gate/up/down/logits matmuls, RoPE,
#                           attention, SiLU gate, residual adds, final logits), tensor index, scalar bit patterns
#   OUT_DIR/tensors.bin     every recorded activation as little-endian binary32, indexed by manifest.tensors [offset, length] in floats
# Weights are NOT copied. The viewer reads hole nibbles, plug nibbles, block absmax and the 16x16 cell table from the pin itself
# (model.safetensors), so every rendered weight is the pin's own bytes and any matmul output can be recomputed in full:
#   w[r,k] = cells[h][p] * absmax[(r*K + k) / 64],  y[r] = sum_k round(w[r,k] * x[k]), strictly left to right in f32.
# Arithmetic is f32 with that fixed order and no activation rounding (the production Mojo/CUDA paths round some activations to bf16).
# Usage: mojo build -I mojo -o p7_trace_smol mojo/p7_trace_smol.mojo
#        ./p7_trace_smol PIN_DIR OUT_DIR ID,ID,...      (comma-separated SmolLM2 token ids; the last one is traced)
from std.math import exp, sqrt, cos, sin, pow
from std.memory import bitcast
from std.sys import argv
from p7_safetensors import parse_header, STEntry

comptime H = 576
comptime NH = 9
comptime NKV = 3
comptime HD = 64
comptime KVD = 192
comptime INTER = 1536
comptime NL = 30
comptime V = 49152
comptime EPS = Float32(1e-5)
comptime THETA = 100000.0

def u32_to_f32(v: UInt32) -> Float32:
    return bitcast[DType.float32](SIMD[DType.uint32, 1](v))

def f32_at(data: List[UInt8], off: Int) -> Float32:
    var v = UInt32(data[off]) | (UInt32(data[off + 1]) << 8) | (UInt32(data[off + 2]) << 16) | (UInt32(data[off + 3]) << 24)
    return u32_to_f32(v)

def bf16_at(data: List[UInt8], off: Int) -> Float32:
    var v = (UInt32(data[off]) << 16) | (UInt32(data[off + 1]) << 24)
    return u32_to_f32(v)

def entry_off(ents: List[STEntry], base: Int, name: String) raises -> Int:
    for i in range(len(ents)):
        if ents[i].name == name:
            return base + ents[i].begin
    raise Error("missing tensor " + name)

def entry_len(ents: List[STEntry], name: String) raises -> Int:
    for i in range(len(ents)):
        if ents[i].name == name:
            return ents[i].end - ents[i].begin
    raise Error("missing tensor " + name)

def load_f32s(data: List[UInt8], off: Int, n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(f32_at(data, off + 4 * i))
    return out^

def load_bf16s(data: List[UInt8], off: Int, n: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(n):
        out.append(bf16_at(data, off + 2 * i))
    return out^

# nibble j of a packed tensor (low nibble first)
def nib(data: List[UInt8], off: Int, j: Int) -> Int:
    return Int((data[off + (j >> 1)] >> UInt8(4 * (j & 1))) & UInt8(15))

# One linear projection of the pin: hole nibbles, plug nibbles, block-64 absmax over the flattened row-major index.
struct Lin(Copyable, Movable):
    var name: String
    var ho: Int
    var po: Int
    var scales: List[Float32]
    var n: Int
    var k: Int

    def __init__(out self, name: String):
        self.name = name
        self.ho = 0
        self.po = 0
        self.scales = List[Float32]()
        self.n = 0
        self.k = 0

def make_lin(data: List[UInt8], ents: List[STEntry], base: Int, name: String, n: Int, k: Int) raises -> Lin:
    var l = Lin(name)
    l.ho = entry_off(ents, base, name)
    l.po = entry_off(ents, base, name + ".glint_plug")
    l.scales = load_f32s(data, entry_off(ents, base, name + ".absmax"), n * k // 64)
    l.n = n
    l.k = k
    return l^

def weight_at(data: List[UInt8], l: Lin, cells: List[Float32], j: Int) -> Float32:
    var h = nib(data, l.ho, j)
    var p = nib(data, l.po, j)
    return cells[h * 16 + p] * l.scales[j // 64]

# Product rounded to f32 as its own step: a non-inlined call stops the compiler fusing acc + w*x into one FMA, so the recorded
# arithmetic is exactly "round(w*x), then round(acc + product)", which the viewer reproduces bit for bit.
@no_inline
def mul32(a: Float32, b: Float32) -> Float32:
    return a * b

# Same idea for the reduction itself: a non-inlined add keeps the sum strictly left to right (the compiler otherwise reassociates
# simple reductions such as the sum of squares).
@no_inline
def add32(a: Float32, b: Float32) -> Float32:
    return a + b

def matvec(data: List[UInt8], l: Lin, cells: List[Float32], x: List[Float32]) -> List[Float32]:
    var y = List[Float32]()
    for r in range(l.n):
        var acc = Float32(0)
        var b = r * l.k
        for kk in range(l.k):
            var w = weight_at(data, l, cells, b + kk)
            var prod = mul32(w, x[kk])
            acc = acc + prod
        y.append(acc)
    return y^

# Floats in the manifest (scalars) are written as their raw IEEE-754 binary32 bit pattern (uint32); tensors go to tensors.bin.
def f2u(v: Float32) -> UInt32:
    return bitcast[DType.uint32](SIMD[DType.float32, 1](v))[0]

def jf(v: Float32) -> String:
    return String(f2u(v))

def arr_i(v: List[Int]) -> String:
    var s = String("[")
    for i in range(len(v)):
        if i > 0:
            s += ","
        s += String(v[i])
    s += "]"
    return s^

struct Store(Movable):
    var names: List[String]
    var offs: List[Int]
    var lens: List[Int]
    var data: List[Float32]

    def __init__(out self):
        self.names = List[String]()
        self.offs = List[Int]()
        self.lens = List[Int]()
        self.data = List[Float32]()

    def add(mut self, name: String, v: List[Float32]):
        self.names.append(name)
        self.offs.append(len(self.data))
        self.lens.append(len(v))
        for i in range(len(v)):
            self.data.append(v[i])

    def index_json(self) -> String:
        var s = String("{")
        for i in range(len(self.names)):
            if i > 0:
                s += ","
            s += String('"') + self.names[i] + '":[' + String(self.offs[i]) + "," + String(self.lens[i]) + "]"
        s += "}"
        return s^

    def to_bytes(self) -> List[UInt8]:
        var b = List[UInt8]()
        for i in range(len(self.data)):
            var u = f2u(self.data[i])
            b.append(UInt8(u & UInt32(255)))
            b.append(UInt8((u >> 8) & UInt32(255)))
            b.append(UInt8((u >> 16) & UInt32(255)))
            b.append(UInt8((u >> 24) & UInt32(255)))
        return b^

def rms_norm(x: List[Float32], w: List[Float32]) -> List[Float32]:
    var ss = Float32(0)
    for i in range(H):
        ss = add32(ss, mul32(x[i], x[i]))
    var mean = ss / Float32(H)
    var rs = Float32(1) / sqrt(mean + EPS)
    var out = List[Float32]()
    for i in range(H):
        var n = x[i] * rs
        out.append(w[i] * n)
    return out^

def op_norm(id: Int, layer: Int, name: String, xn: String, wn: String, yn: String, x: List[Float32]) -> String:
    var ss = Float32(0)
    for i in range(H):
        ss = add32(ss, mul32(x[i], x[i]))
    var mean = ss / Float32(H)
    var rs = Float32(1) / sqrt(mean + EPS)
    return String('{"id":') + String(id) + ',"layer":' + String(layer) + ',"kind":"rmsnorm","name":"' + name + '","x":"' + xn + '","w":"' + wn + '","y":"' + yn + '","sumsq":' + jf(ss) + ',"mean":' + jf(mean) + ',"rsqrt":' + jf(rs) + ',"eps":' + jf(EPS) + "},"

def op_matmul(id: Int, layer: Int, name: String, l: Lin, xn: String, yn: String) -> String:
    return String('{"id":') + String(id) + ',"layer":' + String(layer) + ',"kind":"matmul","name":"' + name + '","weight":"' + l.name + '","N":' + String(l.n) + ',"K":' + String(l.k) + ',"x":"' + xn + '","y":"' + yn + '"},'

def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error("usage: p7_trace_smol PIN_DIR OUT_DIR ID,ID,...")
    var pin_dir = String(args[1])
    var out_dir = String(args[2])
    var ids = List[Int]()
    var cur = 0
    var have = False
    for b in String(args[3]).as_bytes():
        if b == 44:
            ids.append(cur)
            cur = 0
            have = False
        else:
            cur = cur * 10 + Int(b) - 48
            have = True
    if have:
        ids.append(cur)
    var T = len(ids)
    var path = pin_dir + "/model.safetensors"
    var f = open(path, "r")
    var data = f.read_bytes()
    var hl = 0
    for j in range(8):
        hl |= Int(data[j]) << (8 * j)
    var hdr = List[UInt8]()
    for j in range(hl):
        hdr.append(data[8 + j])
    var ents = parse_header(Span(hdr))
    var base = 8 + hl
    var cells = load_f32s(data, entry_off(ents, base, "_glint_cells"), 256)
    var emb = make_lin(data, ents, base, "model.embed_tokens.weight", V, H)
    var fnorm = load_bf16s(data, entry_off(ents, base, "model.norm.weight"), H)

    var lq = List[Lin]()
    var lk = List[Lin]()
    var lv = List[Lin]()
    var lo = List[Lin]()
    var lg = List[Lin]()
    var lu = List[Lin]()
    var ld = List[Lin]()
    var in_w = List[List[Float32]]()
    var post_w = List[List[Float32]]()
    for l in range(NL):
        var p = String("model.layers.") + String(l) + "."
        lq.append(make_lin(data, ents, base, p + "self_attn.q_proj.weight", H, H))
        lk.append(make_lin(data, ents, base, p + "self_attn.k_proj.weight", KVD, H))
        lv.append(make_lin(data, ents, base, p + "self_attn.v_proj.weight", KVD, H))
        lo.append(make_lin(data, ents, base, p + "self_attn.o_proj.weight", H, H))
        lg.append(make_lin(data, ents, base, p + "mlp.gate_proj.weight", INTER, H))
        lu.append(make_lin(data, ents, base, p + "mlp.up_proj.weight", INTER, H))
        ld.append(make_lin(data, ents, base, p + "mlp.down_proj.weight", H, INTER))
        in_w.append(load_bf16s(data, entry_off(ents, base, p + "input_layernorm.weight"), H))
        post_w.append(load_bf16s(data, entry_off(ents, base, p + "post_attention_layernorm.weight"), H))
    print("SMOL_TRACE loaded pin, tokens", T)

    var kc = List[List[Float32]]()
    var vc = List[List[Float32]]()
    for _ in range(NL):
        kc.append(List[Float32]())
        vc.append(List[Float32]())

    var store = Store()
    var oj = String("")
    var opn = 0
    var tpos = T - 1

    for pos in range(T):
        var trace = pos == tpos
        var x = List[Float32]()
        for kk in range(H):
            x.append(weight_at(data, emb, cells, ids[pos] * H + kk))
        if trace:
            store.add("embed", x)
            oj += String('{"id":') + String(opn) + ',"layer":-1,"kind":"embed","name":"token embedding (GLINT decode)","token":' + String(ids[pos]) + ',"weight":"' + emb.name + '","K":' + String(H) + ',"y":"embed"},'
            opn += 1
        for l in range(NL):
            var ln = String("L") + String(l) + "."
            if trace:
                store.add(ln + "x_in", x)
            var h1 = rms_norm(x, in_w[l])
            if trace:
                store.add(ln + "in_ln_w", in_w[l])
                store.add(ln + "norm1", h1)
                oj += op_norm(opn, l, "input_layernorm", ln + "x_in", ln + "in_ln_w", ln + "norm1", x)
                opn += 1
            var q = matvec(data, lq[l], cells, h1)
            var k = matvec(data, lk[l], cells, h1)
            var v = matvec(data, lv[l], cells, h1)
            if trace:
                store.add(ln + "q", q)
                store.add(ln + "k", k)
                store.add(ln + "v", v)
                oj += op_matmul(opn, l, "q_proj", lq[l], ln + "norm1", ln + "q")
                opn += 1
                oj += op_matmul(opn, l, "k_proj", lk[l], ln + "norm1", ln + "k")
                opn += 1
                oj += op_matmul(opn, l, "v_proj", lv[l], ln + "norm1", ln + "v")
                opn += 1
            var cs = List[Float32]()
            var sn = List[Float32]()
            for i in range(HD // 2):
                var inv = Float64(1) / pow(Float64(THETA), (Float64(2) * Float64(i)) / Float64(HD))
                var a = Float64(pos) * inv
                cs.append(Float32(cos(a)))
                sn.append(Float32(sin(a)))
            var qr = q.copy()
            var kr = k.copy()
            for hh in range(NH):
                for i in range(HD // 2):
                    var a1 = q[hh * HD + i]
                    var a2 = q[hh * HD + i + HD // 2]
                    qr[hh * HD + i] = mul32(a1, cs[i]) - mul32(a2, sn[i])
                    qr[hh * HD + i + HD // 2] = mul32(a2, cs[i]) + mul32(a1, sn[i])
            for hh in range(NKV):
                for i in range(HD // 2):
                    var a1 = k[hh * HD + i]
                    var a2 = k[hh * HD + i + HD // 2]
                    kr[hh * HD + i] = mul32(a1, cs[i]) - mul32(a2, sn[i])
                    kr[hh * HD + i + HD // 2] = mul32(a2, cs[i]) + mul32(a1, sn[i])
            if trace:
                store.add(ln + "cos", cs)
                store.add(ln + "sin", sn)
                store.add(ln + "q_rope", qr)
                store.add(ln + "k_rope", kr)
                oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"rope","name":"rotary position embedding","position":' + String(pos) + ',"theta":' + jf(Float32(THETA)) + ',"q":"' + ln + 'q","k":"' + ln + 'k","cos":"' + ln + 'cos","sin":"' + ln + 'sin","q_out":"' + ln + 'q_rope","k_out":"' + ln + 'k_rope","heads_q":' + String(NH) + ',"heads_kv":' + String(NKV) + ',"head_dim":' + String(HD) + "},"
                opn += 1
            for i in range(KVD):
                kc[l].append(kr[i])
                vc[l].append(v[i])
            var T1 = pos + 1
            var attn = List[Float32]()
            for _ in range(H):
                attn.append(Float32(0))
            var scores = List[Float32]()
            var probs = List[Float32]()
            for hq in range(NH):
                var kvh = hq // (NH // NKV)
                var sc = List[Float32]()
                var mx = Float32(-1e30)
                for t in range(T1):
                    var dot = Float32(0)
                    for d in range(HD):
                        dot = dot + mul32(qr[hq * HD + d], kc[l][t * KVD + kvh * HD + d])
                    var s = dot * Float32(0.125)
                    sc.append(s)
                    if s > mx:
                        mx = s
                var den = Float32(0)
                var ex = List[Float32]()
                for t in range(T1):
                    var e = exp(sc[t] - mx)
                    ex.append(e)
                    den = den + e
                for t in range(T1):
                    scores.append(sc[t])
                    probs.append(ex[t] / den)
                for d in range(HD):
                    var acc = Float32(0)
                    for t in range(T1):
                        acc = acc + mul32(ex[t] / den, vc[l][t * KVD + kvh * HD + d])
                    attn[hq * HD + d] = acc
            if trace:
                store.add(ln + "kc", kc[l])
                store.add(ln + "vc", vc[l])
                store.add(ln + "scores", scores)
                store.add(ln + "probs", probs)
                store.add(ln + "attn", attn)
                oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"attention","name":"attention (softmax over cached keys)","T":' + String(T1) + ',"q":"' + ln + 'q_rope","kc":"' + ln + 'kc","vc":"' + ln + 'vc","scores":"' + ln + 'scores","probs":"' + ln + 'probs","y":"' + ln + 'attn","heads":' + String(NH) + ',"kv_heads":' + String(NKV) + ',"head_dim":' + String(HD) + ',"scale":0.125},'
                opn += 1
            var o = matvec(data, lo[l], cells, attn)
            var xm = List[Float32]()
            for i in range(H):
                xm.append(x[i] + o[i])
            if trace:
                store.add(ln + "o", o)
                store.add(ln + "x_mid", xm)
                oj += op_matmul(opn, l, "o_proj", lo[l], ln + "attn", ln + "o")
                opn += 1
                oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"residual","name":"residual add (attention)","a":"' + ln + 'x_in","b":"' + ln + 'o","y":"' + ln + 'x_mid"},'
                opn += 1
            var h2 = rms_norm(xm, post_w[l])
            if trace:
                store.add(ln + "post_ln_w", post_w[l])
                store.add(ln + "norm2", h2)
                oj += op_norm(opn, l, "post_attention_layernorm", ln + "x_mid", ln + "post_ln_w", ln + "norm2", xm)
                opn += 1
            var gt = matvec(data, lg[l], cells, h2)
            var up = matvec(data, lu[l], cells, h2)
            var sg = List[Float32]()
            var act = List[Float32]()
            for i in range(INTER):
                var s = Float32(1) / (Float32(1) + exp(-gt[i]))
                sg.append(s)
                act.append((gt[i] * s) * up[i])
            var dn = matvec(data, ld[l], cells, act)
            var xo = List[Float32]()
            for i in range(H):
                xo.append(xm[i] + dn[i])
            if trace:
                store.add(ln + "gate", gt)
                store.add(ln + "up", up)
                store.add(ln + "sigmoid", sg)
                store.add(ln + "act", act)
                store.add(ln + "down", dn)
                store.add(ln + "x_out", xo)
                oj += op_matmul(opn, l, "gate_proj", lg[l], ln + "norm2", ln + "gate")
                opn += 1
                oj += op_matmul(opn, l, "up_proj", lu[l], ln + "norm2", ln + "up")
                opn += 1
                oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"silu_gate","name":"SiLU(gate) * up","gate":"' + ln + 'gate","up":"' + ln + 'up","sigmoid":"' + ln + 'sigmoid","y":"' + ln + 'act"},'
                opn += 1
                oj += op_matmul(opn, l, "down_proj", ld[l], ln + "act", ln + "down")
                opn += 1
                oj += String('{"id":') + String(opn) + ',"layer":' + String(l) + ',"kind":"residual","name":"residual add (MLP)","a":"' + ln + 'x_mid","b":"' + ln + 'down","y":"' + ln + 'x_out"},'
                opn += 1
            x = xo.copy()
        if pos == T - 1:
            var hf = rms_norm(x, fnorm)
            store.add("final_x", x)
            store.add("final_ln_w", fnorm)
            store.add("final_norm", hf)
            oj += op_norm(opn, -1, "final norm", "final_x", "final_ln_w", "final_norm", x)
            opn += 1
            var logits = matvec(data, emb, cells, hf)
            store.add("logits", logits)
            oj += op_matmul(opn, -1, "logits (tied embedding head)", emb, "final_norm", "logits")
            opn += 1
            var best = 0
            for i in range(len(logits)):
                if logits[i] > logits[best]:
                    best = i
            oj += String('{"id":') + String(opn) + ',"layer":-1,"kind":"output","name":"next-token prediction (greedy)","top_id":' + String(best) + ',"logits":"logits"}'
            opn += 1
            print("SMOL_TRACE top1 token id", best, "logit", logits[best])
    var man = String('{"format":"glint_trace_v2","model":{"name":"SmolLM2-135M","hidden":') + String(H) + ',"layers":' + String(NL) + ',"heads":' + String(NH) + ',"kv_heads":' + String(NKV) + ',"head_dim":' + String(HD) + ',"inter":' + String(INTER) + ',"vocab":' + String(V) + ',"rope_theta":100000,"pin_note":"theta fixed by the model config"},"pin":{"schema":"glint_pin_v1","file":"model.safetensors"},"float_encoding":"scalar fields are binary32 bit patterns as uint32; tensors.bin is little-endian binary32","arithmetic":"f32, sequential k=0..K-1, product rounded before the add (no FMA), no activation rounding","tokens":' + arr_i(ids) + ',"trace_pos":' + String(tpos) + ',"tensors":' + store.index_json() + ',"ops":[' + oj + "]}"
    var mf = open(out_dir + "/manifest.json", "w")
    mf.write(man)
    var bf = open(out_dir + "/tensors.bin", "w")
    var bytes = store.to_bytes()
    bf.write_bytes(Span(bytes))
    print("SMOL_TRACE wrote", out_dir, "ops", opn, "tensors", len(store.names), "floats", len(store.data))
