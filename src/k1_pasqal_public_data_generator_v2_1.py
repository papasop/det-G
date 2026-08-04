#!/usr/bin/env python3
"""No-account public two-level data generator for the K=1 bridge v2.1.

The signed objective is frozen as

    F_signed = endpoint_infidelity - centred_dimensionless_Rabi_exposure.

No Hessian sign is inspected when defining it.  Two information-time laws are
audited separately:

  full_gradient  : both calibration coordinates follow -grad(infidelity)
  detuning_only  : amplitude is neutral; detuning follows its gradient

This separation diagnoses whether rank one is native to the response or is
introduced by a controller policy.  Output CSV is compatible with v2.0.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math,platform,sys,time
from pathlib import Path
import numpy as np

TITLE="PASQAL PUBLIC TWO-LEVEL SIGNED-COST / INFORMATION-TIME DATA GENERATOR"
VERSION="2.1"
SX=np.array([[0,1],[1,0]],complex); SY=np.array([[0,-1j],[1j,0]],complex); SZ=np.diag([1,-1]).astype(complex)
I2=np.eye(2,dtype=complex); KET0=np.array([1,0],complex)

def jd(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 raise TypeError(type(x).__name__)
def can(x):return json.dumps(x,sort_keys=True,separators=(",",":"),default=jd,allow_nan=False)
def sha(x):return hashlib.sha256(can(x).encode()).hexdigest()

def ustep(om,de,ph,dur):
 v=np.array([om*math.cos(ph),om*math.sin(ph),de]); r=float(np.linalg.norm(v)); th=.5*r*dur
 if r<1e-14:return I2.copy()
 return math.cos(th)*I2-1j*math.sin(th)*(v[0]*SX+v[1]*SY+v[2]*SZ)/r

def endpoint(z,segments,omref,dref):
 U=I2.copy(); a,d=z
 for om,de,ph,dur in segments:U=ustep(om*(1+a),de+d*dref,ph,dur)@U
 return U@KET0

def infidelity(z,target,segments,omref,dref):
 p=endpoint(z,segments,omref,dref);return float(max(0.,1-abs(np.vdot(target,p))**2))

def exposure(z,segments,omref):
 a=float(z[0]); num=sum((om*(1+a)/omref)**2*dur for om,de,ph,dur in segments)
 den=sum(dur for *_,dur in segments);return num/den

def deriv(f,z,h):
 z=np.array(z,float);g=np.zeros(2)
 for i in range(2):
  e=np.zeros(2);e[i]=h;g[i]=(f(z+e)-f(z-e))/(2*h)
 return g

def hessian(f,h):
 G=np.zeros((2,2));o=np.zeros(2);f0=f(o)
 for i in range(2):
  ei=np.zeros(2);ei[i]=h;G[i,i]=(f(ei)-2*f0+f(-ei))/h**2
 e0=np.array([h,0.]);e1=np.array([0.,h])
 G[0,1]=G[1,0]=(f(e0+e1)-f(e0-e1)-f(-e0+e1)+f(-e0-e1))/(4*h*h)
 return G

def fit_A(Z,D):return np.linalg.lstsq(Z,D,rcond=None)[0].T
def assess(rows):
 tr=[r for r in rows if r[0]=="train"];te=[r for r in rows if r[0]=="heldout"]
 def cv(a):return np.array([[x[1],x[2]] for x in a]),np.array([x[3] for x in a]),np.array([[x[4],x[5]] for x in a])
 z,c,d=cv(tr);zt,ct,dt=cv(te)
 X=np.c_[np.ones(len(z)),z[:,0],z[:,1],.5*z[:,0]**2,z[:,0]*z[:,1],.5*z[:,1]**2]
 q=np.linalg.lstsq(X,c,rcond=None)[0];G=np.array([[q[3],q[4]],[q[4],q[5]]]);A=fit_A(z,d)
 Xt=np.c_[np.ones(len(zt)),zt[:,0],zt[:,1],.5*zt[:,0]**2,zt[:,0]*zt[:,1],.5*zt[:,1]**2]
 ev=np.linalg.eigvalsh(G);sv=np.linalg.svd(A,compute_uv=False)
 return {"G_estimated":G,"G_eigenvalues":ev,"detG":float(np.linalg.det(G)),"A_estimated":A,
  "A_singular_values":sv,"rank_ratio":float(sv[1]/max(sv[0],1e-15)),
  "heldout_cost_mse":float(np.mean((Xt@q-ct)**2)),"heldout_transport_mse":float(np.mean((zt@A.T-dt)**2)),
  "Lorentzian_cost":bool(ev[0]<0<ev[1]),"rank_one_transport":bool(sv[0]>1e-8 and sv[1]/sv[0]<1e-2)}

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--grid",type=int,default=21);ap.add_argument("--span",type=float,default=.12)
 ap.add_argument("--fd",type=float,default=2e-4);ap.add_argument("--mobility",type=float,default=1.)
 ap.add_argument("--outdir",default="k1_pasqal_public_data_v2_1_results")
 args,unk=ap.parse_known_args()
 if unk:print("[notice] ignored notebook/kernel arguments:",unk)
 t=time.time();out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
 # rad/us and us; fixed before any Hessian/transport calculation.
 segments=[(2.0,.25,0.0,.32),(1.7,-.35,1.1,.28),(2.2,.15,2.2,.36),(1.8,-.20,-.7,.30)]
 omref=2.;dref=1.;target=endpoint([0,0],segments,omref,dref)
 task=lambda z:infidelity(z,target,segments,omref,dref)
 # Remove constant and tangent of exposure, leaving an independently scaled
 # quadratic implementation-exposure penalty.  Coefficient is fixed to one.
 e0=exposure([0,0],segments,omref);eg=deriv(lambda z:exposure(z,segments,omref),[0,0],args.fd)
 signed=lambda z:task(z)-(exposure(z,segments,omref)-e0-np.dot(eg,z))
 G_direct=hessian(signed,args.fd)
 vals=np.linspace(-args.span,args.span,args.grid);base=[]
 for i,a in enumerate(vals):
  for j,d in enumerate(vals):
   z=np.array([a,d]);c=signed(z);gt=deriv(task,z,args.fd)
   split="train" if (i+j)%2==0 else "heldout"
   base.append((split,a,d,c,gt))
 rows_by={}
 for kind in ("full_gradient","detuning_only"):
  rows=[]
  for split,a,d,c,gt in base:
   vel=-args.mobility*gt
   if kind=="detuning_only":vel[0]=0.
   rows.append((split,a,d,c,float(vel[0]),float(vel[1])))
  rows_by[kind]=rows
  p=out/f"{kind}_bridge_data.csv"
  with p.open("w",newline="") as f:
   w=csv.writer(f);w.writerow(["split","z0","z1","signed_cost","dz0_dt","dz1_dt"]);w.writerows(rows)
 audits={k:assess(v) for k,v in rows_by.items()}
 full_native=audits["full_gradient"]["rank_one_transport"]
 policy_rank1=audits["detuning_only"]["rank_one_transport"]
 report={"title":TITLE,"version":VERSION,"scientific_status":
  ("SIGNED_COST_LORENTZIAN_NATIVE_FULL_TRANSPORT_RANK_ONE" if audits["full_gradient"]["Lorentzian_cost"] and full_native
   else "SIGNED_COST_AND_TRANSPORT_GENERATED_NATIVE_K1_BRIDGE_NOT_ESTABLISHED"),
  "protocol_sha256":sha({"segments":segments,"grid":args.grid,"span":args.span,"fd":args.fd,"mobility":args.mobility,
   "signed_cost":"endpoint infidelity minus centred dimensionless Rabi exposure"}),
  "frozen_model":{"segments":segments,"omega_ref":omref,"detuning_ref":dref,
   "signed_cost":"endpoint_infidelity-(Rabi_exposure-value-tangent_at_reference)","information_time":"calibration iteration"},
  "direct_signed_cost_Hessian":G_direct,"direct_signed_cost_eigenvalues":np.linalg.eigvalsh(G_direct),
  "audits":audits,"gates":{"signed_cost_Lorentzian":audits["full_gradient"]["Lorentzian_cost"],
   "full_gradient_transport_rank_one":full_native,"detuning_only_policy_rank_one":policy_rank1,
   "rank_one_not_attributed_to_physics_when_only_policy_passes":bool(policy_rank1 and not full_native)},
  "native_K1_PASQAL_bridge_supported":bool(audits["full_gradient"]["Lorentzian_cost"] and full_native),
  "interpretation":"The cost signature is extracted from a frozen task-minus-exposure definition. Full-gradient transport is the non-policy diagnostic. A rank-one result appearing only after suppressing amplitude updates is controller design, not native K=1 dynamics.",
  "next_required_step":"Feed full_gradient_bridge_data.csv to v2.0. If full-gradient rank one fails, do not use detuning-only success as native evidence; redesign the physical realization map or retain it as a control-policy implementation.",
  "claim_boundary":"No-account deterministic two-level model and information-time calibration law; no Cloud/QPU, microscopic PASQAL action, Born rule, collapse or physical-wavefunction claim.",
  "artifacts":{"full_gradient_csv":str((out/"full_gradient_bridge_data.csv").resolve()),"detuning_only_csv":str((out/"detuning_only_bridge_data.csv").resolve())},
  "elapsed_seconds":time.time()-t,"environment":{"python":platform.python_version(),"numpy":np.__version__}}
 (out/"run_summary.json").write_text(json.dumps(report,indent=2,default=jd)+"\n")
 print("="*112);print(f"{TITLE} v{VERSION}");print("="*112);print(json.dumps(report,indent=2,default=jd));return 0
if __name__=="__main__":
 rc=main()
 if not any(x in sys.modules for x in ("ipykernel","IPython","google.colab")):raise SystemExit(rc)
