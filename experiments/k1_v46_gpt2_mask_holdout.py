"""
K1 V4.6 — held-out GPT-2 fixed layer-mask validation

This is the first intervention experiment: the metric changes hidden-state
evolution. Frozen DistilGPT-2 is compared with parameter-matched adapters:
  residual : ordinary low-rank nonlinear residual
  euclid   : positive-definite metric + learned damping
  free     : unconstrained 2x2 generator + learned damping
  lorentz  : det(G)<0 and exact critical damping det(A_c)=0

Primary claim gate: Lorentz must beat ALL matched adapters on held-out ID loss,
be non-inferior OOD, and pass exact signature/criticality numerical audits.
This script is standalone; use a T4 GPU. Default run is a screening experiment.
"""

import sys, subprocess, importlib.util, json, math, random, warnings
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
    seeds:tuple=(10709,10903,11113,11311,11503,11701,11909)
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
    outdir:str="k1_v46_gpt2_mask_holdout_results"
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

MASK_LAYERS=(0,2,4,5,7,8,9,11)  # frozen from V4.5 before these seeds are evaluated

class DynamicsAdapter(nn.Module):
    def __init__(self,kind,chart_trainable=False,rms_target=0.0,layer_idx=-1):
        super().__init__(); self.kind=kind; self.layer_idx=int(layer_idx)
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
            elif self.kind=="lorentz_mask":
                lam=torch.ones_like(x) if self.layer_idx in MASK_LAYERS else torch.zeros_like(x)
                geo=dz_lor; sign=-1.0
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
        self.adapters=nn.ModuleList([DynamicsAdapter(kind,chart_trainable,rms_targets[i],layer_idx=i) for i in range(L)])
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


# -----------------------------------------------------------------------------
# Train matched variants from the exact same base and seed
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model,loader,with_audit=False):
    model.eval(); loss_sum=correct=n=0; confs=[]; oks=[]; audit_rows=[]
    for x,y in loader:
        x,y=x.to(device),y.to(device); loss,logits,aa=model(x,y,with_audit)
        nt=y.numel(); loss_sum+=loss.item()*nt; n+=nt
        prob=logits.softmax(-1); cf,pr=prob.max(-1); ok=pr.eq(y)
        correct+=ok.sum().item(); confs.append(cf.cpu()); oks.append(ok.cpu())
        if with_audit:
            for li,q in enumerate(aa):
                ratio=q["ratio"].float(); rf=torch.isfinite(ratio)
                eps0=q["eps0"].float(); eps1=q["eps1"].float(); ef=torch.isfinite(eps0)&torch.isfinite(eps1)
                rvals=ratio[rf]
                e0=eps0[ef]; e1=eps1[ef]
                stable=q["stable_final"].float()
                rr=q["rank_ratio"].float(); nr=q["null_res"].float()
                if rr.ndim==1 and ratio.ndim==3:
                    rr=rr[None,None,:].expand_as(ratio); nr=nr[None,None,:].expand_as(ratio)
                joint=((ratio-1).abs()<0.1)&(rr<0.1)&(nr<1e-3)&stable[...,None].bool()
                audit_rows.append(dict(layer=li,detG_max=float(torch.nan_to_num(q["detG"],nan=-999).max()),
                    detG_min=float(torch.nan_to_num(q["detG"],nan=999).min()),
                    detA_absmax=float(torch.nan_to_num(q["detA"],nan=999).abs().max()),
                    dt=float(q["dt"]),corr_rms=float(q["corr_rms"]),
                    rank_ratio_max=float(torch.nan_to_num(q["rank_ratio"],nan=999).max()),
                    null_res_max=float(torch.nan_to_num(q["null_res"],nan=999).max()),
                    ratio_mean=float(rvals.mean()) if rvals.numel() else np.nan,
                    ratio_std=float(rvals.std()) if rvals.numel()>1 else 0.0,
                    near_zero=float((rvals<0.1).float().mean()) if rvals.numel() else np.nan,
                    near_critical=float(((rvals-1).abs()<0.1).float().mean()) if rvals.numel() else np.nan,
                    contraction=float((e1.abs()<e0.abs()).float().mean()) if e0.numel() else np.nan,
                    eps2=float((e0*e0).sum()) if e0.numel() else 0.0,
                    eps_delta=float((e0*(e1-e0)).sum()) if e0.numel() else 0.0,
                    token_freeze=float(stable.mean()),
                    rank_small=float((rr<0.1).float().mean()),
                    null_small=float((nr<1e-3).float().mean()),
                    joint_collapse=float(joint.float().mean()),
                    res_raw_rms=float(q.get("res_raw_rms",torch.tensor(float("nan")))),
                    geo_raw_rms=float(q.get("geo_raw_rms",torch.tensor(float("nan")))),
                    res_hat_rms=float(q.get("res_hat_rms",torch.tensor(float("nan")))),
                    geo_hat_rms=float(q.get("geo_hat_rms",torch.tensor(float("nan"))))))
    conf=torch.cat(confs).numpy().ravel(); ok=torch.cat(oks).numpy().ravel().astype(float)
    bins=np.linspace(0,1,16); ece=0
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(conf>=lo)&(conf<(hi if hi<1 else hi+1e-8))
        if m.any(): ece+=m.mean()*abs(ok[m].mean()-conf[m].mean())
    return dict(loss=loss_sum/n,ppl=float(math.exp(min(20,loss_sum/n))),accuracy=correct/n,ece=ece),audit_rows

def train_variant(kind,seed,chart_state,rms_targets):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    loaders["train"]=DataLoader(BlockDS(B["train"]),batch_size=cfg.batch,shuffle=True,generator=torch.Generator().manual_seed(seed))
    m=IntervenedGPT2(kind,False,rms_targets).to(device)
    with torch.no_grad():
        for i,ad in enumerate(m.adapters):
            ad.down.weight.copy_(chart_state[i]["down"])
            ad.up.weight.copy_(chart_state[i]["up"])
    print("\n",kind,"trainable=",count_trainable(m))
    opt=torch.optim.AdamW(m.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    best=1e9; state=None; bad=0
    for ep in range(cfg.epochs):
        m.train(); ls=[]
        for x,y in loaders["train"]:
            x,y=x.to(device),y.to(device); loss,_,_=m(x,y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),cfg.grad_clip); opt.step(); ls.append(loss.item())
        va,_=evaluate(m,loaders["val"])
        print(f"{kind} epoch {ep+1}: train={np.mean(ls):.4f} val={va['loss']:.4f}")
        if va["loss"]<best-1e-4:
            best=va["loss"]; state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=cfg.patience: break
    m.load_state_dict(state); torch.save({"kind":kind,"seed":seed,"cfg":asdict(cfg),"state":state},Path(cfg.outdir)/f"{kind}_seed{seed}.pt")
    return m

def learn_shared_chart(seed):
    """Stage A: learn one task-responsive residual chart, then freeze it."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    loaders["train"]=DataLoader(BlockDS(B["train"]),batch_size=cfg.batch,shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    m=IntervenedGPT2("chart",True).to(device)
    opt=torch.optim.AdamW(m.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    best=1e9; state=None
    for ep in range(cfg.pretrain_epochs):
        m.train(); ls=[]
        for x,y in loaders["train"]:
            x,y=x.to(device),y.to(device); loss,_,_=m(x,y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),cfg.grad_clip); opt.step(); ls.append(loss.item())
        va,_=evaluate(m,loaders["val"])
        print(f"chart seed={seed} epoch {ep+1}: train={np.mean(ls):.4f} val={va['loss']:.4f}")
        if va["loss"]<best:
            best=va["loss"]; state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    m.load_state_dict(state)
    # Calibration RMS is measured on validation data, averaged per layer.
    _,rows=evaluate(m,loaders["val"],with_audit=True)
    rr=pd.DataFrame(rows); targets=rr.groupby("layer").corr_rms.mean().reindex(range(L)).values
    targets=np.maximum(targets,1e-7).tolist()
    charts=[{"down":ad.down.weight.detach().cpu().clone(),
             "up":ad.up.weight.detach().cpu().clone()} for ad in m.adapters]
    torch.save({"seed":seed,"charts":charts,"rms_targets":targets},Path(cfg.outdir)/f"shared_chart_seed{seed}.pt")
    print("shared RMS targets =",[f"{x:.3e}" for x in targets])
    del m; torch.cuda.empty_cache()
    return charts,targets

# Frozen base metric before interventions.
class BaseWrap(nn.Module):
    def forward(self,x,labels=None,audit=False):
        # labels are already the one-token-ahead targets, so compute CE directly;
        # passing them to HF would apply a second internal causal shift.
        o=base(x,use_cache=False)
        loss=None if labels is None else F.cross_entropy(o.logits.reshape(-1,o.logits.size(-1)),labels.reshape(-1))
        return loss,o.logits,[]

results=[]; audits={}
variants=["residual","lorentz_neg","lorentz_mask","euclid_neg","random_neg"]
for seed in cfg.seeds:
    print("\n"+"#"*92+"\nPAIRED SEED",seed)
    chart_state,rms_targets=learn_shared_chart(seed)
    for kind in variants:
        m=train_variant(kind,seed,chart_state,rms_targets)
        for split in ["val","test","ood"]:
            q,aud=evaluate(m,loaders[split],with_audit=(split in ("test","ood")))
            results.append({"seed":seed,"variant":kind,"split":split,**q})
            if split in ("test","ood"): audits[(seed,split,kind)]=aud
        del m; torch.cuda.empty_cache()
df=pd.DataFrame(results); df.to_csv(Path(cfg.outdir)/"metrics.csv",index=False)
summary=df.groupby(["variant","split"])[["loss","accuracy","ece"]].agg(["mean","std"])
print("\nPERFORMANCE MEAN ± SD\n",summary.to_string(float_format=lambda x:f"{x:.6f}"))
summary.to_csv(Path(cfg.outdir)/"summary.csv")

# Re-run compact dynamic audit directly from saved row-level audit summaries.
# evaluate() below stores tensor-derived regression sufficient statistics.
dyn=pd.concat([pd.DataFrame(v).assign(seed=k[0],split=k[1],variant=k[2])
               for k,v in audits.items()],ignore_index=True)
dyn.to_csv(Path(cfg.outdir)/"dynamics_by_batch_layer.csv",index=False)
agg=dyn.groupby(["variant","split","layer"]).agg(
    ratio_mean=("ratio_mean","mean"),ratio_std=("ratio_std","mean"),
    near_zero=("near_zero","mean"),near_critical=("near_critical","mean"),
    contraction=("contraction","mean"),eps2=("eps2","sum"),eps_delta=("eps_delta","sum"),
    dt=("dt","mean"),rank_ratio=("rank_ratio_max","max"),null_res=("null_res_max","max"),
    token_freeze=("token_freeze","mean"),rank_small=("rank_small","mean"),
    null_small=("null_small","mean"),joint_collapse=("joint_collapse","mean")).reset_index()
agg["kappa_hat"]=-agg.eps_delta/(agg.dt*agg.eps2).replace(0,np.nan)
agg.to_csv(Path(cfg.outdir)/"adaptive_dynamics_summary.csv",index=False)
print("\nADAPTIVE DAMPING / K=1 REGRESSION\n",agg.to_string(index=False,float_format=lambda x:f"{x:.6g}"))

def meanloss(v,s): return float(df[(df.variant==v)&(df.split==s)].loss.mean())

TCRIT95={2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262}
def paired_delta(a_name,b_name,split):
    a=df[(df.variant==a_name)&(df.split==split)].set_index("seed").loss
    b=df[(df.variant==b_name)&(df.split==split)].set_index("seed").loss
    d=(a-b).dropna().values; n=len(d); m=float(d.mean())
    sd=float(d.std(ddof=1)) if n>1 else 0.; tc=TCRIT95.get(n-1,1.96)
    half=tc*sd/math.sqrt(max(1,n))
    return {"values":d.tolist(),"n":n,"mean":m,"sd":sd,
      "paired_dz":m/(sd+1e-12),"ci95_t":[m-half,m+half],"wins":int((d<0).sum())}

comparisons={}
for split in ("test","ood"):
    for control in ("residual","lorentz_neg","euclid_neg","random_neg"):
        comparisons[f"{split}:lorentz_mask-{control}"]=paired_delta("lorentz_mask",control,split)
    comparisons[f"{split}:lorentz_neg-residual"]=paired_delta("lorentz_neg","residual",split)

normcheck=dyn.groupby(["variant","split"])[["res_raw_rms","geo_raw_rms","res_hat_rms","geo_hat_rms"]].mean().reset_index()
normcheck.to_csv(Path(cfg.outdir)/"branch_normalization.csv",index=False)
print("\nBRANCH NORMALIZATION CHECK\n",normcheck.to_string(index=False,float_format=lambda x:f"{x:.6g}"))
ratio_check=agg[(agg.variant=="lorentz_mask")&(agg.split=="test")][["layer","ratio_mean"]].sort_values("layer")
print("\nFIXED MASK CHECK\n",ratio_check.to_string(index=False))
mask_exact=all((float(r.ratio_mean)>.999)==(int(r.layer) in MASK_LAYERS) for _,r in ratio_check.iterrows())
matched=bool(np.nanmax(np.abs(normcheck.res_hat_rms-1))<.05 and
             np.nanmax(np.abs(normcheck.geo_hat_rms-1))<.05)
def upper_below(k,x=0.):return comparisons[k]["ci95_t"][1]<x
id_res=upper_below("test:lorentz_mask-residual")
id_full=upper_below("test:lorentz_mask-lorentz_neg")
id_euc=upper_below("test:lorentz_mask-euclid_neg")
id_rnd=upper_below("test:lorentz_mask-random_neg")
ood_noninferior=all(upper_below(f"ood:lorentz_mask-{c}",cfg.ood_tolerance_nats)
 for c in ("residual","lorentz_neg","euclid_neg","random_neg"))
minimum_effect=comparisons["test:lorentz_mask-residual"]["mean"]<=-1e-4
stable_wins=all(comparisons[f"test:lorentz_mask-{c}"]["wins"]>=5
 for c in ("residual","lorentz_neg","euclid_neg","random_neg"))
audit={"protocol":"V4.6 held-out validation; mask fixed to [0,2,4,5,7,8,9,11]; seven previously unseen GPT-2 seeds",
 "mask_layers":list(MASK_LAYERS),"test_loss":{v:meanloss(v,"test") for v in variants},
 "ood_loss":{v:meanloss(v,"ood") for v in variants},"paired_comparisons":comparisons,
 "G1_mask_exact_and_pre_mix_matched":bool(mask_exact and matched),
 "G2_mask_beats_residual_95CI":id_res,"G3_mask_beats_full_lorentz_95CI":id_full,
 "G4_mask_beats_euclid_95CI":id_euc,"G5_mask_beats_random_95CI":id_rnd,
 "G6_OOD_noninferior_to_all":ood_noninferior,"G7_minimum_effect_1e-4_nats":minimum_effect,
 "G8_wins_at_least_5_of_7_vs_all":stable_wins,
 "PASS_LAYER_LOCAL_SIGN_REVERSAL_GENERALIZES":bool(mask_exact and matched and id_res and id_full and
   id_euc and id_rnd and ood_noninferior and minimum_effect and stable_wins)}
print("\n"+"="*100+"\nK1 V4.6 GPT-2 HELD-OUT FIXED LAYER-MASK VALIDATION\n"+"="*100)
print(json.dumps(audit,indent=2)); json.dump(audit,open(Path(cfg.outdir)/"audit_v46.json","w"),indent=2)
sns.set_theme(style="whitegrid"); fig,ax=plt.subplots(1,2,figsize=(12,4.5))
sns.barplot(data=df[df.seed>=0],x="split",y="loss",hue="variant",ax=ax[0]); ax[0].set_title("Cross-entropy (lower is better)")
delta_rows=[]
for key,q in comparisons.items():
    if key.startswith("test:lorentz_mask-"):
        control=key.split("-")[-1]
        for i,v in enumerate(q["values"]): delta_rows.append({"control":control,"seed":cfg.seeds[i],"delta":v})
dd=pd.DataFrame(delta_rows)
sns.pointplot(data=dd,x="control",y="delta",errorbar=("ci",95),ax=ax[1])
ax[1].axhline(0,color="black",lw=1); ax[1].set_title("Held-out fixed-mask paired test loss delta")
ax[1].set_ylabel("Mask − control (lower is better)")
plt.tight_layout(); plt.savefig(Path(cfg.outdir)/"v46_gpt2_mask_holdout.png",dpi=180,bbox_inches="tight"); plt.show()
print("Saved to",cfg.outdir)

