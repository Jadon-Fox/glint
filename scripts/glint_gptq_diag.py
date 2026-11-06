import sys, glob, numpy as np
sys.path.insert(0,'/home/workspace/fractal-deployment/worktrees/glint/qwen3-forward')
import glint_np
from glint_codec import CELLS, PARENT
from safetensors import safe_open
P=np.asarray(PARENT,dtype=np.float32); C=np.asarray(CELLS,dtype=np.float32)
name='model.layers.5.mlp.down_proj.weight'
def get(d,k):
    for f in glob.glob(d+'/*.safetensors'):
        with safe_open(f,framework='np') as s:
            if k in s.keys(): return s.get_tensor(k)
import torch
w0=np.asarray(torch.from_numpy(get('/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083',name).astype(np.float32)) if get('/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083',name).dtype!=np.uint16 else 0) if False else None
from safetensors import safe_open as so
for f in glob.glob('/opt/models/.hf_cache/hub/models--microsoft--Phi-4-mini-instruct/snapshots/cfbefacb99257ffa30c83adab238a50856ac3083/*.safetensors'):
    with so(f,framework='pt') as s:
        if name in s.keys(): w0=s.get_tensor(name).float().numpy().reshape(-1)
N=w0.size
for tag,pin in (('nearest(old)','/home/workspace/pins/Phi-4-mini-glint-from-bf16'),('gptq','/home/workspace/lakes/p7_glint/gptq_hole_20261106/pin')):
    for f in glob.glob(pin+'/*.safetensors'):
        with so(f,framework='pt') as s:
            if name in s.keys():
                h=glint_np.unpack_nibbles(s.get_tensor(name).numpy().reshape(-1),N).astype(np.int64); am=s.get_tensor(name+'.absmax').numpy(); pl=glint_np.unpack_nibbles(s.get_tensor(name+'.glint_plug').numpy().reshape(-1),N).astype(np.int64)
    sc=np.repeat(am,64)[:N]; v=np.where(sc==0,0,w0/sc)
    near=np.abs(P[None,:]-v[:,None]).argmin(1)
    rms=lambda a: float(np.sqrt(np.mean((a-v)**2)))
    print(tag,'hole!=nearest: %.1f%%'%(100*np.mean(h!=near)),'rmse hole-only %.5f'%rms(P[h]),'rmse stored plug(narrow) %.5f'%rms(C[h,pl]))
    for F in (1.5,2.0):
        Wd=np.clip(P[:,None]+F*(C-P[:,None]),-1,1); pw=np.abs(Wd[h]-v[:,None]).argmin(1); print('   F=%.1f rmse wide plug %.5f'%(F,rms(Wd[h,pw])))
