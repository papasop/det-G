"""
K1 V4.5 read-only GPT-2 failure-mechanism diagnostic.
Requires completed k1_throttle_v44_gpt2_screen_results/.
Loads checkpoints only; performs no training and changes no V4.4 conclusion.
"""

import sys, subprocess, importlib.util, json, math, random, warnings, types
from dataclasses import dataclass, asdict
from pathlib import Path

def install():
    req={"transformers":"transformers>=4.44,<5","datasets":"datasets>=2.20,<4",
         "pandas":"pandas>=2","matplotlib":"matplotlib>=3.7","seaborn":"seaborn>=0.13"}
    miss=[v for k,v in req.items() if importlib.util.find_spec(k) is None]
    if miss: subprocess.check_call([sys.executable,"-m","pip","install","-q"]+miss)
install()

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM

warnings.filterwarnings("ignore",category=FutureWarning)

@dataclass
class CFG:
    seed:int=20260719
    seeds:tuple=(10103,10301,10501)
    model_name:str="gpt2"  # frozen V4.4 cross-model screening target
    seq_len:int=96
    train_blocks:int=500
    val_blocks:int=160
    test_blocks:int=500
    ood_blocks:int=500
    batch:int=4
    pretrain_epochs:int=3
    epochs:int=3
    lr:float=8e-4
    weight_decay:float=1e-4
    planes:int=8                 # adapter rank = 16
    max_dt:float=0.08
    grad_clip:float=1.0
    patience:int=2
    ood_tolerance_nats:float=0.02
    outdir:str="k1_throttle_v44_gpt2_screen_results"
cfg=CFG(); Path(cfg.outdir).mkdir(exist_ok=True,parents=True)
random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =",device); print(json.dumps(asdict(cfg),indent=2))


# -----------------------------------------------------------------------------
# Data — explicit namespaces, chunked tokenisation (no >1024 warning)
# -----------------------------------------------------------------------------
tok=AutoTokenizer.from_pretrained(cfg.model_name); tok.pad_token=tok.eos_token

def texts(repo,config,split):
    d=load_dataset(repo,config,split=split,trust_remote_code=False)
    key="text" if "text" in d.column_names else "sentence"
    return [str(x) for x in d[key] if str(x).strip()]

def blocks_from_text(xs,n):
    blocks=[]; carry=[]
    for chunk in xs:
        carry += tok(chunk+"\n",add_special_tokens=False,truncation=False)["input_ids"]
        while len(carry)>=cfg.seq_len+1 and len(blocks)<n:
            blocks.append(torch.tensor(carry[:cfg.seq_len+1],dtype=torch.long))
            carry=carry[cfg.seq_len:]
        if len(blocks)>=n: break
    if len(blocks)<n: print("warning blocks",len(blocks),"requested",n)
    return blocks

id_train=texts("Salesforce/wikitext","wikitext-2-raw-v1","train")
id_val=texts("Salesforce/wikitext","wikitext-2-raw-v1","validation")
id_test=texts("Salesforce/wikitext","wikitext-2-raw-v1","test")
ood=texts("fancyzhx/ag_news",None,"test")
B={"train":blocks_from_text(id_train,cfg.train_blocks),"val":blocks_from_text(id_val,cfg.val_blocks),
   "test":blocks_from_text(id_test,cfg.test_blocks),"ood":blocks_from_text(ood,cfg.ood_blocks)}
print({k:len(v) for k,v in B.items()})

class BlockDS(Dataset):
    def __init__(self,x): self.x=x
    def __len__(self): return len(self.x)
    def __getitem__(self,i): return self.x[i][:-1],self.x[i][1:]

loaders={k:DataLoader(BlockDS(v),batch_size=cfg.batch,shuffle=(k=="train")) for k,v in B.items()}


# -----------------------------------------------------------------------------
# Frozen GPT-2 and intervention adapters
# -----------------------------------------------------------------------------
base=AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
for p in base.parameters(): p.requires_grad_(False)
base.eval(); D=base.config.n_embd; L=base.config.n_layer; R=2*cfg.planes

class DynamicsAdapter(nn.Module):
    def __init__(self,kind,chart_trainable=False,rms_target=0.0):
        super().__init__(); self.kind=kind
        self.chart_trainable=chart_trainable
        self.register_buffer("rms_target",torch.tensor(float(rms_target)))
        self.down=nn.Linear(D,R,bias=False); self.up=nn.Linear(R,D,bias=False)
        # Stage A learns the chart; Stage B loads and freezes the same chart.
        nn.init.normal_(self.down.weight,std=1/math.sqrt(D))
        nn.init.normal_(self.up.weight,std=1e-4)
        self.down.weight.requires_grad_(chart_trainable)
        self.up.weight.requires_grad_(chart_trainable)
        self.log_a=nn.Parameter(torch.zeros(cfg.planes)); self.log_c=nn.Parameter(torch.zeros(cfg.planes))
        self.log_alpha=nn.Parameter(torch.full((cfg.planes,),-1.0))
        self.log_d=nn.Parameter(torch.full((cfg.planes,),-1.0))
        self.dt_raw=nn.Parameter(torch.tensor(-2.0)); self.gate=nn.Parameter(torch.tensor(-2.0))
        self.ratio_raw=nn.Parameter(torch.tensor(-3.0))
        # z(2) + layer position + entropy + margin + inter-layer TVD
        self.token_gate=nn.Linear(6,1)
        nn.init.zeros_(self.token_gate.weight)
        nn.init.constant_(self.token_gate.bias,-3.0)
        # matched extra parameters used by residual/free controls
        self.mix=nn.Parameter(torch.empty(cfg.planes,2,2))
        self.control_mix=nn.Parameter(torch.empty(cfg.planes,2,2))
        self.tau_raw=nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.mix,std=0.02)
        nn.init.normal_(self.control_mix,std=0.02)

    def forward(self,h,commit_features=None,return_audit=False):
        z=self.down(h).view(*h.shape[:-1],cfg.planes,2); x,y=z[...,0],z[...,1]
        a=F.softplus(self.log_a)+0.1; c=F.softplus(self.log_c)+0.1
        alpha=F.softplus(self.log_alpha)+1e-4; dc=alpha/torch.sqrt(a*c)
        dt=cfg.max_dt*torch.sigmoid(self.dt_raw)
        # Neutral chart pretraining.
        if self.kind=="chart":
            dz=z; lam=torch.zeros_like(x); K=-a*x*x+c*y*y
            dz_res=geo=res_hat=geo_hat=torch.zeros_like(z)
            detg=torch.full_like(a,float("nan")); deta=torch.full_like(a,float("nan"))
            rank_ratio=torch.full_like(a,float("nan")); null_res=torch.full_like(a,float("nan"))
        else:
            # Free residual transport is present in every Stage-B variant.
            dz_res=torch.einsum("...pi,pij->...pj",z,self.mix)
            K=-a*x*x+c*y*y; eps=K-1
            gx=-2*a*x*eps; gy=2*c*y*eps
            jx=-alpha*gy/a; jy=-alpha*gx/c
            dz_lor=torch.stack([jx-dc*gx,jy-dc*gy],-1)
            # Alternative equal-throttle controls.
            gxE=2*a*x*(a*x*x+c*y*y-1); gyE=2*c*y*(a*x*x+c*y*y-1)
            dz_euc=torch.stack([alpha*gyE/a-dc*gxE,-alpha*gxE/c-dc*gyE],-1)
            # V4.2: match branch scale BEFORE mixing. Per-token RMS over
            # planes/components prevents a huge geometric generator from
            # dominating despite a small lambda.
            def branch_unit(v):
                scale=v.float().square().mean(dim=(-2,-1),keepdim=True).sqrt().to(v.dtype)
                return v/(scale+1e-6)
            res_hat=branch_unit(dz_res)
            # V4.3 preregistration: the negative sign is fixed for every
            # active branch. No sign or gate is selected after seeing results.
            if self.kind=="residual":
                lam=torch.zeros_like(x); geo=dz_lor; sign=-1.0
            elif self.kind=="lorentz_neg":
                lam=torch.ones_like(x); geo=dz_lor; sign=-1.0
            elif self.kind=="euclid_neg":
                lam=torch.ones_like(x); geo=dz_euc; sign=-1.0
            elif self.kind=="random_neg":
                lam=torch.ones_like(x)
                geo=torch.einsum("...pi,pij->...pj",z,self.control_mix); sign=-1.0
            else: raise ValueError(self.kind)
            geo_hat=branch_unit(geo)
            dz=res_hat+sign*lam[...,None]*geo_hat
            detg=-a*c if self.kind!="euclid_neg" else a*c
            deta=torch.zeros_like(a) if self.kind!="euclid_neg" else 2*dc*dc
            # The internal Lorentz generator remains exactly critical; lambda only activates it.
            A=torch.zeros(cfg.planes,2,2,device=h.device,dtype=h.dtype)
            A[:,0,0]=-dc; A[:,0,1]=-alpha/a; A[:,1,0]=-alpha/c; A[:,1,1]=-dc
            sv=torch.linalg.svdvals(A); rank_ratio=sv[:,1]/sv[:,0].clamp_min(1e-12)
            u=A[:,:,0]; Gu=torch.stack([-a*u[:,0],c*u[:,1]],-1)
            null_res=(u*Gu).sum(-1).abs()/(u.square().sum(-1)*(a+c)).clamp_min(1e-12)
        dz=dz/torch.sqrt(1+dz.square().sum(-1,keepdim=True)); znew=z+dt*dz
        eps0=K-1; K1=-a*znew[...,0].square()+c*znew[...,1].square(); eps1=K1-1
        correction=self.up((znew-z).reshape(*h.shape[:-1],R))*torch.sigmoid(self.gate)
        if (not self.chart_trainable) and self.rms_target.item()>0:
            rms=correction.float().square().mean().sqrt().clamp_min(1e-12)
            correction=correction*(self.rms_target/rms.detach()).clamp(0.05,20.0)
        out=h+correction
        if return_audit:
            return out,{"detG":detg.detach(),"detA":deta.detach(),"dt":dt.detach(),
              "corr_rms":correction.detach().float().square().mean().sqrt(),
              "rank_ratio":rank_ratio.detach(),"null_res":null_res.detach(),
              "ratio":lam.detach(),"eps0":eps0.detach(),"eps1":eps1.detach(),
              "res_raw_rms":dz_res.detach().float().square().mean().sqrt(),
              "geo_raw_rms":geo.detach().float().square().mean().sqrt(),
              "res_hat_rms":res_hat.detach().float().square().mean().sqrt(),
              "geo_hat_rms":geo_hat.detach().float().square().mean().sqrt()}
        return out

class IntervenedGPT2(nn.Module):
    def __init__(self,kind,chart_trainable=False,rms_targets=None):
        super().__init__(); self.kind=kind
        if rms_targets is None: rms_targets=[0.0]*L
        self.adapters=nn.ModuleList([DynamicsAdapter(kind,chart_trainable,rms_targets[i]) for i in range(L)])
    def forward(self,input_ids,labels=None,audit=False):
        tr=base.transformer; Bn,T=input_ids.shape
        pos=torch.arange(T,device=input_ids.device)[None].expand(Bn,T)
        h=tr.drop(tr.wte(input_ids)+tr.wpe(pos)); audits=[]; prev_prob=None
        for li,(block,ad) in enumerate(zip(tr.h,self.adapters)):
            # transformers releases differ: GPT2Block may return the hidden
            # tensor directly or a tuple whose first item is that tensor.
            block_out=block(h,use_cache=False)
            h=block_out[0] if isinstance(block_out,(tuple,list)) else block_out
            if False:  # V4.3 uses fixed gates; no commit-feature computation
                # Detached commit diagnostics: gate learns how to use them but
                # does not backpropagate through the frozen vocabulary head.
                with torch.no_grad():
                    dl=base.lm_head(tr.ln_f(h)); lp=dl.log_softmax(-1); prob=lp.exp()
                    entropy=(-(prob*lp).sum(-1)/math.log(prob.size(-1))).clamp(0,1)
                    top2=prob.topk(2,-1).values; margin=top2[...,0]-top2[...,1]
                    tvd=torch.zeros_like(entropy) if prev_prob is None else 0.5*(prob-prev_prob).abs().sum(-1)
                    layerfrac=torch.full_like(entropy,li/max(1,L-1))
                    commit_features=torch.stack([layerfrac,entropy,margin,tvd],-1)
                    prev_prob=prob
            else:
                commit_features=torch.zeros(Bn,T,4,device=h.device,dtype=h.dtype)
            if audit:
                h,q=ad(h,commit_features,True)
                with torch.no_grad(): q["layer_top1"]=base.lm_head(tr.ln_f(h)).argmax(-1)
                audits.append(q)
            else: h=ad(h,commit_features)
        logits=base.lm_head(tr.ln_f(h)); loss=None
        if labels is not None: loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),labels.reshape(-1))
        if audit:
            final_top1=logits.argmax(-1)
            for q in audits: q["stable_final"]=q["layer_top1"].eq(final_top1)
        return loss,logits,audits

def count_trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


RESULTS=Path(cfg.outdir)
OUT=Path("k1_v45_gpt2_mechanism_results"); OUT.mkdir(exist_ok=True,parents=True)
if not RESULTS.exists(): raise FileNotFoundError("Missing "+str(RESULTS))
DIAG_BLOCKS=96
EPS=1e-12

def load_model(kind,seed):
    ck=torch.load(RESULTS/f"{kind}_seed{seed}.pt",map_location="cpu")
    ch=torch.load(RESULTS/f"shared_chart_seed{seed}.pt",map_location="cpu")
    m=IntervenedGPT2(kind,False,ch["rms_targets"]).to(device)
    m.load_state_dict(ck["state"],strict=True); m.eval()
    return m,ch

def unit_branch(v):
    sc=v.float().square().mean(dim=(-2,-1),keepdim=True).sqrt().to(v.dtype)
    return v/(sc+1e-6)

def adapter_terms(ad,h):
    z=ad.down(h).view(*h.shape[:-1],cfg.planes,2); x,y=z[...,0],z[...,1]
    a=F.softplus(ad.log_a)+.1; c=F.softplus(ad.log_c)+.1
    alpha=F.softplus(ad.log_alpha)+1e-4; dc=alpha/torch.sqrt(a*c)
    eps=-a*x*x+c*y*y-1
    gx,gy=-2*a*x*eps,2*c*y*eps
    lor=torch.stack([-alpha*gy/a-dc*gx,-alpha*gx/c-dc*gy],-1)
    res=torch.einsum("...pi,pij->...pj",z,ad.mix)
    rh,gh=unit_branch(res),unit_branch(lor)
    def squash(v): return v/torch.sqrt(1+v.square().sum(-1,keepdim=True))
    full=squash(rh-gh); off=squash(rh)
    dt=cfg.max_dt*torch.sigmoid(ad.dt_raw); gate=torch.sigmoid(ad.gate)
    pre_full=ad.up((dt*full).reshape(*h.shape[:-1],-1))*gate
    pre_off=ad.up((dt*off).reshape(*h.shape[:-1],-1))*gate
    target=float(ad.rms_target.item())
    def match(v):
        if target<=0:return v
        scale=(target/v.detach().float().square().mean().sqrt().clamp_min(EPS)).clamp(.05,20.)
        return v*scale
    return {"z":z,"res_hat":rh,"geo_hat":gh,"full":match(pre_full),"off":match(pre_off),
            "eps":eps}

def cos(a,b):
    a=a.detach().float().reshape(-1); b=b.detach().float().reshape(-1)
    return float((a@b)/(a.norm()*b.norm()+EPS))
def rms(x): return float(x.detach().float().square().mean().sqrt())

@torch.no_grad()
def direction_audit(m,split,max_batches=8):
    tr=base.transformer; rows=[]
    for bi,(ids,_) in enumerate(loaders[split]):
        if bi>=max_batches:break
        ids=ids.to(device); pos=torch.arange(ids.size(1),device=device)[None].expand_as(ids)
        h=tr.drop(tr.wte(ids)+tr.wpe(pos))
        for li,(block,ad) in enumerate(zip(tr.h,m.adapters)):
            bo=block(h,use_cache=False); h=bo[0] if isinstance(bo,(tuple,list)) else bo
            q=adapter_terms(ad,h)
            rows.append({"batch":bi,"layer":li,"cos_geo_res":cos(q["geo_hat"],q["res_hat"]),
              "full_off_cos":cos(q["full"],q["off"]),"full_off_delta_rms":rms(q["full"]-q["off"]),
              "full_rms":rms(q["full"]),"off_rms":rms(q["off"]),
              "eps2":float(q["eps"].float().square().mean())})
            h=h+q["full"]
    return rows

def patch_ablation(m,off_layers):
    off_layers=set(off_layers)
    for li,ad in enumerate(m.adapters):
        if li not in off_layers: continue
        def wrapped(self,h,commit_features=None,return_audit=False):
            q=adapter_terms(self,h); out=h+q["off"]
            if return_audit:
                nan=torch.full((cfg.planes,),float("nan"),device=h.device)
                return out,{"detG":nan,"detA":nan,"dt":torch.tensor(0.,device=h.device),
                 "corr_rms":q["off"].float().square().mean().sqrt(),"rank_ratio":nan,
                 "null_res":nan,"ratio":torch.zeros_like(q["eps"]),"eps0":q["eps"],
                 "eps1":q["eps"]}
            return out
        ad.forward=types.MethodType(wrapped,ad)

@torch.no_grad()
def subset_loss(m,split,nblocks=DIAG_BLOCKS):
    ds=BlockDS(B[split][:nblocks]); ld=DataLoader(ds,batch_size=cfg.batch,shuffle=False)
    total=n=0
    for x,y in ld:
        x,y=x.to(device),y.to(device); loss,_,_=m(x,y)
        total+=float(loss)*y.numel(); n+=y.numel()
    return total/n

def rowspace_similarity(W1,W2):
    # Mean squared canonical correlation between chart row spaces.
    q1=torch.linalg.qr(W1.float().T,mode="reduced").Q
    q2=torch.linalg.qr(W2.float().T,mode="reduced").Q
    sv=torch.linalg.svdvals(q1.T@q2)
    return float(sv.square().mean())

layer_rows=[]; direction_rows=[]; overall=[]; charts={}
for seed in cfg.seeds:
    print("#"*92,"\nSEED",seed)
    m,ch=load_model("lorentz_neg",seed); charts[seed]=ch
    for split in ("test","ood"):
        enabled=subset_loss(m,split)
        for r in direction_audit(m,split): direction_rows.append(dict(r,seed=seed,split=split))
        # all-off counterfactual
        ma,_=load_model("lorentz_neg",seed); patch_ablation(ma,range(L))
        alloff=subset_loss(ma,split)
        overall.append({"seed":seed,"split":split,"enabled":enabled,"all_off":alloff,
                        "all_off_minus_enabled":alloff-enabled})
        del ma
        # Per-layer ablation is the main test diagnostic; OOD keeps all-off only.
        if split=="test":
            for li in range(L):
                ml,_=load_model("lorentz_neg",seed); patch_ablation(ml,[li])
                lo=subset_loss(ml,split)
                layer_rows.append({"seed":seed,"layer":li,"enabled":enabled,
                  "layer_off":lo,"off_minus_enabled":lo-enabled})
                del ml
                if torch.cuda.is_available():torch.cuda.empty_cache()
    del m
    if torch.cuda.is_available():torch.cuda.empty_cache()

# Frozen chart comparison: seed 10301 versus the two successful screen seeds.
chart_rows=[]
bad=10301
for other in (10103,10501):
    for li in range(L):
        wbad=charts[bad]["charts"][li]["down"]; wother=charts[other]["charts"][li]["down"]
        chart_rows.append({"seed_a":bad,"seed_b":other,"layer":li,
                           "down_rowspace_similarity":rowspace_similarity(wbad,wother)})

layers=pd.DataFrame(layer_rows); dirs=pd.DataFrame(direction_rows)
overall=pd.DataFrame(overall); chartdf=pd.DataFrame(chart_rows)
layers.to_csv(OUT/"layer_ablation_test.csv",index=False)
dirs.to_csv(OUT/"direction_audit.csv",index=False)
overall.to_csv(OUT/"all_off_counterfactual.csv",index=False)
chartdf.to_csv(OUT/"chart_similarity.csv",index=False)

ls=layers.groupby(["seed","layer"]).off_minus_enabled.mean().reset_index()
ds=dirs.groupby(["seed","split","layer"]).agg(cos_geo_res=("cos_geo_res","mean"),
 full_off_cos=("full_off_cos","mean"),full_off_delta_rms=("full_off_delta_rms","mean"),
 eps2=("eps2","mean")).reset_index()
cs=chartdf.groupby("layer").down_rowspace_similarity.mean().reset_index()
print("\nALL-OFF COUNTERFACTUAL (positive means Lorentz helps)\n",overall.to_string(index=False))
print("\nPER-LAYER OFF TEST DELTA (positive means that Lorentz layer helps)\n",ls.to_string(index=False))
print("\nDIRECTION SUMMARY\n",ds.to_string(index=False))
print("\nCHART SIMILARITY TO SEED 10301\n",cs.to_string(index=False))

# A layer is a reproducible helper/hurter only if its sign agrees in all seeds.
pivot=ls.pivot(index="layer",columns="seed",values="off_minus_enabled")
helper=[int(i) for i in pivot.index[(pivot>0).all(axis=1)]]
hurter=[int(i) for i in pivot.index[(pivot<0).all(axis=1)]]
bad_only=[]
for li in pivot.index:
    if pivot.loc[li,bad]<0 and pivot.loc[li,[10103,10501]].mean()>0: bad_only.append(int(li))
audit={"protocol":"read-only V4.4 checkpoint diagnosis; 96-block fixed subsets; no training",
 "all_off_minus_enabled":{f"{r.seed}:{r.split}":float(r.all_off_minus_enabled) for _,r in overall.iterrows()},
 "layers_help_all_3_seeds":helper,"layers_hurt_all_3_seeds":hurter,
 "layers_hurt_10301_but_help_other_two":bad_only,
 "seed10301_chart_similarity_mean":float(chartdf.down_rowspace_similarity.mean()),
 "seed10301_chart_similarity_min":float(chartdf.down_rowspace_similarity.min()),
 "diagnosis":("LAYER_LOCAL_SIGN_REVERSAL" if bad_only else
              "GLOBAL_SEED_SENSITIVITY_WITHOUT_LOCAL_REVERSAL")}
print("\n"+"="*96+"\nK1 V4.5 GPT-2 READ-ONLY FAILURE-MECHANISM AUDIT\n"+"="*96)
print(json.dumps(audit,indent=2)); (OUT/"audit_v45.json").write_text(json.dumps(audit,indent=2))
print("Saved to",OUT)

