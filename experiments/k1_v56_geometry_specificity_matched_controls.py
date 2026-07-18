"""K1 V5.6 — matched-capacity geometric-specificity competition on OPT-125M.

Seven untouched paired seeds compare the token-conditioned Lorentz generator
with a conventional scalar gate and a learned non-geometric direction gate.
Every variant uses the same chart, nominal parameter budget, RMS calibration,
negative orientation, optimiser, batches and stopping rule.

This model's effective metric can change with
token and layer: G[t,l,p]=diag(-a[p], sigma[t,l,p]*c[p]).  It trains a dynamic
crossing model and matched fixed-Lorentz/no-cross controls.  On the frozen best
checkpoint it then applies inference-only counterfactuals: preserve |sigma|
but force Lorentz signature, exclude the zero neighbourhood, or shuffle sigma
between token positions.  The primary causal estimand is loss(counterfactual)
- loss(enabled); positive means the learned token-level crossing helps.

This tests an engineered LLM mechanism.  It is not evidence of physical
wavefunction collapse unless the preregistered behavioural and causal gates
both pass on untouched seeds/data.
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
    seeds:tuple=(42101,42307,42509,42703,42901,43103,43307)
    model_name:str="facebook/opt-125m"  # frozen cross-family target
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
    sigma_max:float=2.0
    zero_band:float=0.10
    equivalence_margin:float=1e-4
    outdir:str="k1_v56_geometry_specificity_matched_results"
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
# Frozen OPT causal LM and hook-based intervention adapters
# -----------------------------------------------------------------------------
base=AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
for p in base.parameters(): p.requires_grad_(False)
base.eval()
if base.config.model_type!="opt":
    raise ValueError(f"V5.6 is frozen to OPT, got {base.config.model_type}")
decoder=base.model.decoder
layers=decoder.layers
D=int(base.config.hidden_size); L=int(base.config.num_hidden_layers); R=2*cfg.planes
if len(layers)!=L: raise RuntimeError("OPT layer count mismatch")


class DynamicsAdapter(nn.Module):
    def __init__(self,kind,chart_trainable=False,rms_target=0.0):
        super().__init__(); self.kind=kind
        self.chart_trainable=chart_trainable
        self.register_buffer("rms_target",torch.tensor(float(rms_target)))
        self.register_buffer("cf_abs_target",torch.ones(cfg.planes))
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
        nn.init.constant_(self.token_gate.bias,0.55)
        # matched extra parameters used by residual/free controls
        self.mix=nn.Parameter(torch.empty(cfg.planes,2,2))
        self.control_mix=nn.Parameter(torch.empty(cfg.planes,2,2))
        self.tau_raw=nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.mix,std=0.02)
        nn.init.normal_(self.control_mix,std=0.02)

    def forward(self,h,commit_features=None,return_audit=False,cf_mode="enabled"):
        # OPT checkpoints may emit FP16 hidden states while the small adapter
        # intentionally trains in FP32. Compute the adapter in its parameter
        # dtype, then cast only the final correction back to the backbone dtype.
        backbone_dtype=h.dtype
        ah=h.to(dtype=self.down.weight.dtype)
        z=self.down(ah).view(*h.shape[:-1],cfg.planes,2); x,y=z[...,0],z[...,1]
        a=F.softplus(self.log_a)+0.1; c=F.softplus(self.log_c)+0.1
        alpha=F.softplus(self.log_alpha)+1e-4; dc=alpha/torch.sqrt(a*c)
        dt=cfg.max_dt*torch.sigmoid(self.dt_raw)
        # Neutral chart pretraining.
        if self.kind=="chart":
            dz=z; lam=torch.zeros_like(x); K=-a*x*x+c*y*y
            # Stage-A learns only the shared chart and has no metric-signature
            # gate.  Keep an explicit token-shaped audit placeholder so the
            # common audit schema remains well-defined.
            sigma=torch.full_like(x,float("nan"))
            dz_res=geo=res_hat=geo_hat=torch.zeros_like(z)
            detg=torch.full_like(a,float("nan")); deta=torch.full_like(a,float("nan"))
            rank_ratio=torch.full_like(a,float("nan")); null_res=torch.full_like(a,float("nan"))
        else:
            # Free residual transport is present in every Stage-B variant.
            dz_res=torch.einsum("...pi,pij->...pj",z,self.mix)
            K=-a*x*x+c*y*y; eps=K-1
            # V4.2: match branch scale BEFORE mixing. Per-token RMS over
            # planes/components prevents a huge geometric generator from
            # dominating despite a small lambda.
            def branch_unit(v):
                scale=v.float().square().mean(dim=(-2,-1),keepdim=True).sqrt().to(v.dtype)
                return v/(scale+1e-6)
            res_hat=branch_unit(dz_res)
            # V4.3 preregistration: the negative sign is fixed for every
            # active branch. No sign or gate is selected after seeing results.
            feats=torch.stack([x,y,x*x,y*y,x*y,torch.ones_like(x)],-1)
            raw=self.token_gate(feats).squeeze(-1)
            if self.kind=="residual":
                sigma=torch.ones_like(x); lam=torch.zeros_like(x); geo=z; sign=-1.0
            elif self.kind=="scalar_gate":
                sigma=cfg.sigma_max*torch.sigmoid(raw)
                lam=sigma; geo=z; sign=-1.0
            elif self.kind=="learned_direction_gate":
                sigma=cfg.sigma_max*torch.sigmoid(raw); w=sigma/cfg.sigma_max
                alt=torch.tanh(torch.einsum("...pi,pij->...pj",z,self.control_mix))
                geo=(1-w[...,None])*z+w[...,None]*alt
                lam=torch.ones_like(x); sign=-1.0
            else:
                if self.kind=="fixed_lorentz": sigma=torch.ones_like(x)
                elif self.kind=="dynamic_cross": sigma=cfg.sigma_max*torch.tanh(raw)
                elif self.kind=="dynamic_nocross": sigma=cfg.sigma_max*torch.sigmoid(raw)
                elif self.kind=="lagged_control":
                    local=cfg.sigma_max*torch.tanh(raw)
                    sigma=torch.cat([local[:,:1],local[:,:-1]],1)
                else: raise ValueError(self.kind)
                if cf_mode in ("force_lorentz","abs_sigma"): sigma=sigma.abs()
                elif cf_mode=="exclude_zero":
                    sg=torch.where(sigma>=0,torch.ones_like(sigma),-torch.ones_like(sigma))
                    sigma=sg*sigma.abs().clamp_min(cfg.zero_band)
                elif cf_mode=="shuffle": sigma=torch.cat([sigma[:,:1],sigma[:,:-1]],1)
                elif cf_mode=="layer_mean_abs": sigma=self.cf_abs_target[None,None,:].expand_as(sigma)
                elif cf_mode=="token_shuffle_abs":
                    local=sigma.abs(); sigma=torch.cat([local[:,:1],local[:,:-1]],1)
                elif cf_mode=="sign_only":
                    sg=torch.where(sigma>=0,torch.ones_like(sigma),-torch.ones_like(sigma))
                    sigma=sg*self.cf_abs_target[None,None,:]
                elif cf_mode=="fixed_one": sigma=torch.ones_like(sigma)
                elif cf_mode!="enabled": raise ValueError(cf_mode)
                d_sig=alpha*torch.sqrt(sigma.abs().clamp_min(1e-8)/(a*c))
                g0=-d_sig*z[...,0]-(alpha*sigma/a)*z[...,1]
                g1=-(alpha/c)*z[...,0]-d_sig*z[...,1]
                geo=torch.stack([g0,g1],-1)
                lam=torch.ones_like(x); sign=-1.0
            geo_hat=branch_unit(geo)
            dz=res_hat+sign*lam[...,None]*geo_hat
            if self.kind in ("fixed_lorentz","dynamic_cross","dynamic_nocross","lagged_control"):
                detg=-a*c*sigma; deta=(alpha*alpha/(a*c))*(sigma.abs()-sigma)
            else:
                detg=torch.full_like(sigma,float("nan")); deta=torch.full_like(sigma,float("nan"))
            # The internal Lorentz generator remains exactly critical; lambda only activates it.
            # Structural audit only: keep the tiny batched matrices in FP32.
            # CUDA does not implement torch.linalg.svdvals for Half here, and
            # casting this diagnostic does not alter the intervention path.
            A=torch.zeros(cfg.planes,2,2,device=h.device,dtype=torch.float32)
            if self.kind=="residual":
                rank_ratio=torch.full_like(a,float("nan"))
                null_res=torch.full_like(a,float("nan"))
            else:
                rank_ratio=torch.full_like(a,float("nan")); null_res=torch.full_like(a,float("nan"))
        dz=dz/torch.sqrt(1+dz.square().sum(-1,keepdim=True)); znew=z+dt*dz
        eps0=K-1; K1=-a*znew[...,0].square()+c*znew[...,1].square(); eps1=K1-1
        correction=self.up((znew-z).reshape(*h.shape[:-1],R))*torch.sigmoid(self.gate)
        if (not self.chart_trainable) and self.rms_target.item()>0:
            rms=correction.float().square().mean().sqrt().clamp_min(1e-12)
            correction=correction*(self.rms_target/rms.detach()).clamp(0.05,20.0)
        correction_backbone=correction.to(dtype=backbone_dtype)
        out=h+correction_backbone
        if return_audit:
            return out,{"detG":detg.detach(),"detA":deta.detach(),"dt":dt.detach(),
              "corr_rms":correction_backbone.detach().float().square().mean().sqrt(),
              "rank_ratio":rank_ratio.detach(),"null_res":null_res.detach(),
              "ratio":lam.detach(),"sigma":sigma.detach(),"eps0":eps0.detach(),"eps1":eps1.detach(),
              "res_raw_rms":dz_res.detach().float().square().mean().sqrt(),
              "geo_raw_rms":geo.detach().float().square().mean().sqrt(),
              "res_hat_rms":res_hat.detach().float().square().mean().sqrt(),
              "geo_hat_rms":geo_hat.detach().float().square().mean().sqrt()}
        return out

class IntervenedCausalLM(nn.Module):
    """Architecture-independent intervention via decoder-layer forward hooks."""
    def __init__(self,kind,chart_trainable=False,rms_targets=None):
        super().__init__(); self.kind=kind
        if rms_targets is None: rms_targets=[0.0]*L
        self.adapters=nn.ModuleList([DynamicsAdapter(kind,chart_trainable,rms_targets[i]) for i in range(L)])

    def _layer_logits(self,h):
        q=h
        if decoder.final_layer_norm is not None: q=decoder.final_layer_norm(q)
        if getattr(decoder,"project_out",None) is not None: q=decoder.project_out(q)
        return base.lm_head(q)

    def forward(self,input_ids,labels=None,audit=False,cf_mode="enabled"):
        audits=[]; handles=[]
        def make_hook(li,ad):
            def hook(_module,_inputs,output):
                h=output[0] if isinstance(output,(tuple,list)) else output
                if audit:
                    hn,q=ad(h,None,True,cf_mode)
                    with torch.no_grad(): q["layer_top1"]=self._layer_logits(hn).argmax(-1)
                    audits.append((li,q))
                else: hn=ad(h,None,False,cf_mode)
                if isinstance(output,tuple): return (hn,)+output[1:]
                if isinstance(output,list): return [hn]+output[1:]
                return hn
            return hook
        try:
            for li,(layer,ad) in enumerate(zip(layers,self.adapters)):
                handles.append(layer.register_forward_hook(make_hook(li,ad)))
            out=base(input_ids=input_ids,use_cache=False,output_attentions=False,
                     output_hidden_states=False,return_dict=True)
        finally:
            for h in handles: h.remove()
        logits=out.logits
        loss=None if labels is None else F.cross_entropy(
            logits.float().reshape(-1,logits.size(-1)),labels.reshape(-1))
        ordered=[]
        if audit:
            audits.sort(key=lambda x:x[0]); ordered=[q for _,q in audits]
            if len(ordered)!=L: raise RuntimeError(f"Expected {L} hook audits, got {len(ordered)}")
            final_top1=logits.argmax(-1)
            for q in ordered: q["stable_final"]=q["layer_top1"].eq(final_top1)
        return loss,logits,ordered

def count_trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Train matched variants from the exact same base and seed
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model,loader,with_audit=False,cf_mode="enabled"):
    model.eval(); loss_sum=correct=n=0; confs=[]; oks=[]; audit_rows=[]
    for x,y in loader:
        x,y=x.to(device),y.to(device); loss,logits,aa=model(x,y,with_audit,cf_mode)
        nt=y.numel(); loss_sum+=loss.item()*nt; n+=nt
        prob=logits.float().softmax(-1); cf,pr=prob.max(-1); ok=pr.eq(y)
        correct+=ok.sum().item(); confs.append(cf.cpu()); oks.append(ok.cpu())
        if with_audit:
            for li,q in enumerate(aa):
                ratio=q["ratio"].float(); rf=torch.isfinite(ratio)
                eps0=q["eps0"].float(); eps1=q["eps1"].float(); ef=torch.isfinite(eps0)&torch.isfinite(eps1)
                rvals=ratio[rf]
                e0=eps0[ef]; e1=eps1[ef]
                stable=q["stable_final"].float()
                dg_raw=q["detG"].float()
                if dg_raw.ndim==1:
                    dg_raw=dg_raw[None,None,:].expand(y.shape[0],y.shape[1],-1)
                dg=dg_raw.mean(-1)
                sg_raw=q["sigma"].float()
                if sg_raw.ndim==1:
                    sg_raw=sg_raw[None,None,:].expand_as(dg_raw)
                good3=ok[...,None].expand_as(dg_raw)
                dg_good=dg_raw[good3]; dg_bad=dg_raw[~good3]
                sg_good=sg_raw[good3]; sg_bad=sg_raw[~good3]
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
                audit_rows[-1].update(
                    detG_mean=float(dg.mean()),abs_detG_mean=float(dg.abs().mean()),
                    sigma_mean=float(torch.nanmean(sg_raw)),
                    abs_sigma_mean=float(torch.nanmean(sg_raw.abs())),
                    # Boundary is defined in the dimensionless signature
                    # coordinate, separately for every plane.
                    near_boundary=float((sg_raw.abs()<cfg.zero_band).float().mean()),
                    euclid_fraction=float((sg_raw<0).float().mean()),
                    detG_correct=float(dg_good.mean()) if dg_good.numel() else np.nan,
                    detG_error=float(dg_bad.mean()) if dg_bad.numel() else np.nan,
                    near_boundary_correct=float((sg_good.abs()<cfg.zero_band).float().mean()) if sg_good.numel() else np.nan,
                    near_boundary_error=float((sg_bad.abs()<cfg.zero_band).float().mean()) if sg_bad.numel() else np.nan)
    conf=torch.cat(confs).numpy().ravel(); ok=torch.cat(oks).numpy().ravel().astype(float)
    bins=np.linspace(0,1,16); ece=0
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(conf>=lo)&(conf<(hi if hi<1 else hi+1e-8))
        if m.any(): ece+=m.mean()*abs(ok[m].mean()-conf[m].mean())
    return dict(loss=loss_sum/n,ppl=float(math.exp(min(20,loss_sum/n))),accuracy=correct/n,ece=ece),audit_rows

def train_variant(kind,seed,chart_state,rms_targets):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    loaders["train"]=DataLoader(BlockDS(B["train"]),batch_size=cfg.batch,shuffle=True,generator=torch.Generator().manual_seed(seed))
    m=IntervenedCausalLM(kind,False,rms_targets).to(device)
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
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss: kind={kind}, seed={seed}, epoch={ep+1}")
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),cfg.grad_clip); opt.step(); ls.append(loss.item())
        va,_=evaluate(m,loaders["val"])
        print(f"{kind} epoch {ep+1}: train={np.mean(ls):.4f} val={va['loss']:.4f}")
        if va["loss"]<best-1e-4:
            best=va["loss"]; state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=cfg.patience: break
    if state is None: raise RuntimeError(f"No finite validation checkpoint: kind={kind}, seed={seed}")
    m.load_state_dict(state); torch.save({"kind":kind,"seed":seed,"cfg":asdict(cfg),"state":state},Path(cfg.outdir)/f"{kind}_seed{seed}.pt")
    return m

def learn_shared_chart(seed):
    """Stage A: learn one task-responsive residual chart, then freeze it."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    loaders["train"]=DataLoader(BlockDS(B["train"]),batch_size=cfg.batch,shuffle=True,
        generator=torch.Generator().manual_seed(seed))
    m=IntervenedCausalLM("chart",True).to(device)
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
        loss=None if labels is None else F.cross_entropy(o.logits.float().reshape(-1,o.logits.size(-1)),labels.reshape(-1))
        return loss,o.logits,[]

def preflight():
    """Fail fast before the three-seed mechanism screen."""
    x,y=next(iter(loaders["val"])); x,y=x.to(device),y.to(device)
    with torch.no_grad(): ref=base(x,use_cache=False).logits.detach().clone()
    counts={}; checks={}
    kinds=["residual","fixed_lorentz","scalar_gate","learned_direction_gate","dynamic_cross"]
    for kind in kinds:
        m=IntervenedCausalLM(kind,False,[0.0]*L).to(device); counts[kind]=count_trainable(m)
        loss,logits,aa=m(x,y,audit=True)
        if len(aa)!=L or not torch.isfinite(loss): raise RuntimeError(f"{kind} hook preflight failed")
        if any(len(layer._forward_hooks) for layer in layers): raise RuntimeError("Hook leak detected")
        if kind!="residual":
            if kind in ("fixed_lorentz","dynamic_nocross"):
                checks[kind+"_detG_negative"]=all(float(q["detG"].max())<0 for q in aa)
            m.zero_grad(set_to_none=True); loss.backward()
            ps=[m.adapters[0].log_c] if kind=="fixed_lorentz" else [m.adapters[0].token_gate.weight]
            if kind=="learned_direction_gate": ps.append(m.adapters[0].control_mix)
            checks[kind+"_gradient_nonzero"]=all(p.grad is not None and float(p.grad.norm())>0 for p in ps)
        del m
    if len(set(counts.values()))!=1: raise RuntimeError(f"Parameter budgets differ: {counts}")
    if not all(checks.values()): raise RuntimeError(f"Structural preflight failed: {checks}")
    with torch.no_grad(): after=base(x,use_cache=False).logits
    if not torch.allclose(ref,after,atol=0,rtol=0): raise RuntimeError("Base changed or hook was not removed")
    print("PREFLIGHT PASS",{"trainable":counts,"checks":checks,"layers":L,"hidden":D})
    if torch.cuda.is_available(): torch.cuda.empty_cache()

preflight()

@torch.no_grad()
def calibrate_abs_targets(model):
    sums=[torch.zeros(cfg.planes,device=device) for _ in range(L)]; counts=[0]*L
    model.eval()
    for x,y in loaders["val"]:
        x,y=x.to(device),y.to(device); _,_,aa=model(x,y,True,"enabled")
        for li,q in enumerate(aa):
            s=q["sigma"].float().abs(); sums[li]+=s.sum((0,1)); counts[li]+=s.shape[0]*s.shape[1]
    rows=[]
    for li,ad in enumerate(model.adapters):
        target=(sums[li]/max(1,counts[li])).clamp_min(1e-4); ad.cf_abs_target.copy_(target)
        rows.extend({"layer":li,"plane":p,"val_mean_abs_sigma":float(v)} for p,v in enumerate(target.cpu()))
    return rows

results=[]; audits={}; counterfactual=[]; calibration=[]
variants=["residual","fixed_lorentz","scalar_gate","learned_direction_gate","dynamic_cross"]
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
cf=pd.DataFrame(counterfactual); cf.to_csv(Path(cfg.outdir)/"counterfactual_metrics.csv",index=False)
pd.DataFrame(calibration).to_csv(Path(cfg.outdir)/"validation_abs_sigma_targets.csv",index=False)

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

def v56_stat(d):
    d=np.asarray(d,float); n=len(d); mean=float(d.mean()); sd=float(d.std(ddof=1)) if n>1 else 0.0
    tc={2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262}.get(n-1,1.96)
    half=tc*sd/math.sqrt(max(1,n))
    return {"values":d.tolist(),"mean":mean,"sd":sd,"ci95_t":[mean-half,mean+half],
            "wins":int((d<0).sum()),"n":n}

controls=("residual","fixed_lorentz","scalar_gate","learned_direction_gate")
comp56={}
for split in ("test","ood"):
    tab=df[df.split==split].pivot(index="seed",columns="variant",values="loss")
    for c in controls:
        comp56[f"{split}:dynamic_geometry-{c}"]=v56_stat((tab.dynamic_cross-tab[c]).values)
norm56=dyn.groupby(["variant","split"])[["res_hat_rms","geo_hat_rms"]].mean().reset_index()
matched=bool(np.nanmax(np.abs(norm56.res_hat_rms-1))<0.05 and
             np.nanmax(np.abs(norm56.geo_hat_rms-1))<0.05)
id_all=all(comp56[f"test:dynamic_geometry-{c}"]["ci95_t"][1]<0 and
           comp56[f"test:dynamic_geometry-{c}"]["wins"]>=6 for c in controls)
ood_noninferior=all(comp56[f"ood:dynamic_geometry-{c}"]["ci95_t"][1]<cfg.ood_tolerance_nats for c in controls)
beats_general=comp56["test:dynamic_geometry-learned_direction_gate"]["ci95_t"][1]<0
minimum_effect=comp56["test:dynamic_geometry-learned_direction_gate"]["mean"]<=-1e-4
audit56={"protocol":"OPT-125M; seven untouched paired seeds; shared chart/RMS/batches; nominal parameter-matched adaptive controls; FP32 CE",
         "comparisons":comp56,
         "G1_pre_mix_branch_RMS_matched":matched,
         "G2_dynamic_geometry_beats_all_ID_95CI_and_6of7":id_all,
         "G3_beats_learned_nongeometric_direction_gate_95CI":beats_general,
         "G4_OOD_noninferior_to_all":ood_noninferior,
         "G5_effect_vs_general_gate_at_least_1e4_nats":minimum_effect,
         "PASS_V56_GEOMETRY_SPECIFICITY_MATCHED_CONTROLS":bool(matched and id_all and beats_general and ood_noninferior and minimum_effect),
         "interpretation":"Pass supports this engineered Lorentz-generator parameterisation over these matched gates. It does not prove quantum collapse, decoherence, or uniqueness among all possible controllers."}
print("\n"+"="*104+"\nK1 V5.6 MATCHED-CAPACITY GEOMETRY-SPECIFICITY COMPETITION\n"+"="*104)
print(json.dumps(audit56,indent=2)); json.dump(audit56,open(Path(cfg.outdir)/"audit_v56.json","w"),indent=2)
sns.set_theme(style="whitegrid"); fig,ax=plt.subplots(figsize=(11,4.5))
sns.barplot(data=df[df.split=="test"],x="variant",y="loss",ax=ax); ax.tick_params(axis="x",rotation=20)
ax.set_title("Seven-seed matched-control held-out loss")
plt.tight_layout(); plt.savefig(Path(cfg.outdir)/"v56_geometry_specificity.png",dpi=180,bbox_inches="tight"); plt.show()
print("Saved to",cfg.outdir)
sys.exit(0)

def confirm_stat(d,positive_win=True):
    d=np.asarray(d,float); n=len(d); mean=float(d.mean()); sd=float(d.std(ddof=1)) if n>1 else 0.0
    tc={2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262}.get(n-1,1.96)
    half=tc*sd/math.sqrt(max(1,n))
    return {"values":d.tolist(),"mean":mean,"sd":sd,"ci95_t":[mean-half,mean+half],
            "wins":int(((d>0) if positive_win else (d<0)).sum()),"n":n}

confirm={}
for split in ("test","ood"):
    pt=df[df.split==split].pivot(index="seed",columns="variant",values="loss")
    confirm[f"{split}:dynamic-fixed"]=confirm_stat((pt.dynamic_cross-pt.fixed_lorentz).values,False)
    confirm[f"{split}:dynamic-residual"]=confirm_stat((pt.dynamic_cross-pt.residual).values,False)
    ct=cf[cf.split==split].pivot(index="seed",columns="mode",values="loss")
    for mode in ("abs_sigma","layer_mean_abs","token_shuffle_abs","sign_only","fixed_one"):
        confirm[f"{split}:{mode}-enabled"]=confirm_stat((ct[mode]-ct.enabled).values,True)
    confirm[f"{split}:layer_mean_abs-abs_sigma"]=confirm_stat((ct.layer_mean_abs-ct.abs_sigma).values,True)
    confirm[f"{split}:token_shuffle_abs-abs_sigma"]=confirm_stat((ct.token_shuffle_abs-ct.abs_sigma).values,True)

absq=confirm["test:abs_sigma-enabled"]
meanq=confirm["test:layer_mean_abs-abs_sigma"]
shufq=confirm["test:token_shuffle_abs-abs_sigma"]
signq=confirm["test:sign_only-enabled"]
dynfix=confirm["test:dynamic-fixed"]; dynres=confirm["test:dynamic-residual"]
equiv=(absq["ci95_t"][0]>-cfg.equivalence_margin and absq["ci95_t"][1]<cfg.equivalence_margin)
amplitude=(meanq["ci95_t"][0]>0 and shufq["ci95_t"][0]>0 and signq["ci95_t"][0]>0)
performance=(dynfix["ci95_t"][1]<0 and dynres["ci95_t"][1]<0)
ood_noninferior=(confirm["ood:dynamic-fixed"]["ci95_t"][1]<cfg.ood_tolerance_nats and
                 confirm["ood:dynamic-residual"]["ci95_t"][1]<cfg.ood_tolerance_nats)
audit53={"protocol":"frozen V5.2 hypotheses; seven untouched seeds; FP32 CE; validation-only |sigma| calibration",
         "comparisons":confirm,
         "G1_dynamic_beats_fixed_and_residual_ID_95CI":performance,
         "G2_sign_removal_equivalent_within_1e4_nats":equiv,
         "G3_token_amplitude_three_tests_95CI":amplitude,
         "G4_OOD_noninferior":ood_noninferior,
         "PASS_V53_TOKEN_CONDITIONED_THROTTLE_CONFIRMATION":bool(performance and equiv and amplitude and ood_noninferior),
         "interpretation":"Pass confirms a small engineered token-conditioned throttle effect; it does not confirm signature crossing or wavefunction collapse."}
print("\n"+"="*104+"\nK1 V5.3 SEVEN-SEED TOKEN-THROTTLE CONFIRMATION\n"+"="*104)
print(json.dumps(audit53,indent=2)); json.dump(audit53,open(Path(cfg.outdir)/"audit_v53.json","w"),indent=2)
sns.set_theme(style="whitegrid"); fig,ax=plt.subplots(1,2,figsize=(13,4.5))
sns.barplot(data=df[df.split=="test"],x="variant",y="loss",ax=ax[0]); ax[0].set_title("Seven-seed held-out loss")
sns.barplot(data=cf[cf.split=="test"],x="mode",y="loss",ax=ax[1]); ax[1].tick_params(axis="x",rotation=25)
ax[1].set_title("Same-checkpoint amplitude/sign counterfactuals")
plt.tight_layout(); plt.savefig(Path(cfg.outdir)/"v53_confirmation.png",dpi=180,bbox_inches="tight"); plt.show()
print("Saved to",cfg.outdir)
sys.exit(0)

def meanloss(v,s): return float(df[(df.variant==v)&(df.split==s)].loss.mean())
TCRIT95={2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262}
def stat(d,positive_win=True):
    d=np.asarray(d,float); n=len(d); m=float(d.mean()); sd=float(d.std(ddof=1)) if n>1 else 0.0
    half=TCRIT95.get(n-1,1.96)*sd/math.sqrt(max(n,1))
    return {"values":d.tolist(),"mean":m,"sd":sd,"ci95_t":[m-half,m+half],
            "wins":int(((d>0) if positive_win else (d<0)).sum()),"n":n}

comparisons={}
for split in ("test","ood"):
    tab=df[df.split==split].pivot(index="seed",columns="variant",values="loss")
    for control in ("residual","fixed_lorentz","dynamic_nocross","lagged_control"):
        comparisons[f"{split}:dynamic_cross-{control}"]=stat((tab.dynamic_cross-tab[control]).values,False)
    ctab=cf[cf.split==split].pivot(index="seed",columns="mode",values="loss")
    for mode in ("force_lorentz","exclude_zero","shuffle"):
        # positive means disabling/alignment-breaking increases loss
        comparisons[f"{split}:{mode}-enabled"]=stat((ctab[mode]-ctab.enabled).values,True)

# Behavioural collapse markers are computed seed-wise to avoid treating tokens
# or batches as independent replicates.
behaviour={}
dc=dyn[(dyn.variant=="dynamic_cross")&(dyn.split=="test")]
final=[]; correct_gap=[]; crossing=[]
for seed,g in dc.groupby("seed"):
    by=g.groupby("layer").mean(numeric_only=True)
    final.append(by.loc[L-1,"near_boundary"]-by.loc[:L-2,"near_boundary"].mean())
    correct_gap.append(by.loc[L-1,"near_boundary_correct"]-by.loc[L-1,"near_boundary_error"])
    crossing.append(g.euclid_fraction.mean())
behaviour["final_layer_boundary_dip_minus_earlier"]=stat(final,True)
behaviour["final_layer_correct_minus_error_boundary_rate"]=stat(correct_gap,True)
behaviour["euclidean_excursion_fraction_by_seed"]={"values":crossing,"mean":float(np.mean(crossing))}

normcheck=dyn.groupby(["variant","split"])[["res_hat_rms","geo_hat_rms"]].mean().reset_index()
matched=bool(np.nanmax(np.abs(normcheck.res_hat_rms-1))<0.05 and
             np.nanmax(np.abs(normcheck.geo_hat_rms-1))<0.05)
causal=comparisons["test:force_lorentz-enabled"]
zero_cf=comparisons["test:exclude_zero-enabled"]
alignment=comparisons["test:shuffle-enabled"]
dynamic_vs_fixed=comparisons["test:dynamic_cross-fixed_lorentz"]
g_causal=causal["ci95_t"][0]>0 and causal["wins"]==len(cfg.seeds)
g_zero=zero_cf["ci95_t"][0]>0
g_align=alignment["ci95_t"][0]>0
g_beh=(behaviour["final_layer_boundary_dip_minus_earlier"]["ci95_t"][0]>0 and
       behaviour["final_layer_correct_minus_error_boundary_rate"]["ci95_t"][0]>0)
g_perf=dynamic_vs_fixed["ci95_t"][1]<0
audit={"protocol":"OPT-125M token-conditioned G_eff; frozen V5.0 hypotheses/seeds; FP32 CE; token-by-plane signature audit; same-checkpoint counterfactuals",
       "test_loss":{v:meanloss(v,"test") for v in variants},
       "ood_loss":{v:meanloss(v,"ood") for v in variants},
       "paired_comparisons":comparisons,"behaviour":behaviour,
       "G1_branch_RMS_matched":matched,
       "G2_forcing_Lorentz_hurts_95CI":g_causal,
       "G3_excluding_zero_band_hurts_95CI":g_zero,
       "G4_token_metric_shuffle_hurts_95CI":g_align,
       "G5_final_boundary_dip_is_correct_token_specific":g_beh,
       "G6_dynamic_beats_fixed_Lorentz_95CI":g_perf,
       "PASS_V51_TOKEN_SIGNATURE_COLLAPSE_REPLICATION":bool(matched and g_causal and g_zero and g_align and g_beh and g_perf),
       "note":"A screen pass requires a frozen seven-new-seed confirmation; it remains an engineered analogue, not physical wavefunction collapse."}
print("\n"+"="*100+"\nK1 V5.1 FP32 TOKEN-LEVEL METRIC-SIGNATURE REPLICATION\n"+"="*100)
print(json.dumps(audit,indent=2)); json.dump(audit,open(Path(cfg.outdir)/"audit_v51.json","w"),indent=2)
sns.set_theme(style="whitegrid"); fig,ax=plt.subplots(1,2,figsize=(13,4.5))
traj=dc.groupby(["seed","layer"])[["detG_mean","near_boundary"]].mean().reset_index()
sns.lineplot(data=traj,x="layer",y="detG_mean",hue="seed",marker="o",ax=ax[0]); ax[0].axhline(0,color="black",lw=1)
ax[0].set_title("Token-conditioned det(G) by layer")
cfplot=cf[cf.split=="test"]; sns.barplot(data=cfplot,x="mode",y="loss",ax=ax[1]); ax[1].set_title("Same-checkpoint inference counterfactuals")
plt.tight_layout(); plt.savefig(Path(cfg.outdir)/"v51_token_signature_collapse.png",dpi=180,bbox_inches="tight"); plt.show()
print("Saved to",cfg.outdir)
