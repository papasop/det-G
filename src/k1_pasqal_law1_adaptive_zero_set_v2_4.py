#!/usr/bin/env python3
"""Adaptive-boundary candidate-3 Law-I(e) zero-set audit v2.4."""
from __future__ import annotations
import argparse,json,hashlib,math,platform,sys,time
from pathlib import Path
import numpy as np
TITLE="K=1 / PASQAL CANDIDATE-3 ADAPTIVE FINITE ZERO-SET COMPLETENESS AUDIT";VERSION="2.4"
SX=np.array([[0,1],[1,0]],complex);SY=np.array([[0,-1j],[1j,0]],complex);SZ=np.diag([1,-1]).astype(complex);I=np.eye(2,dtype=complex);Q=np.array([1,0],complex)
def jd(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 raise TypeError(type(x).__name__)
def can(x):return json.dumps(x,sort_keys=True,separators=(",",":"),default=jd,allow_nan=False)
def us(o,d,p,t):
 v=np.array([o*math.cos(p),o*math.sin(p),d]);r=np.linalg.norm(v);a=r*t/2
 return I if r<1e-15 else math.cos(a)*I-1j*math.sin(a)*(v[0]*SX+v[1]*SY+v[2]*SZ)/r
def ep(z,segs):
 u=I.copy()
 for o,d,p,t in segs:u=us(o*(1+z[0]),d+z[1],p,t)@u
 return u@Q
def bis(f,a,b,tol=1e-13):
 fa=f(a)
 for _ in range(100):
  m=(a+b)/2;fm=f(m)
  if abs(fm)<tol or b-a<tol:return m
  if fa*fm<=0:b=m
  else:a=m;fa=fm
 return (a+b)/2
def roots(F,x,Y,step):
 n=max(1001,int(math.ceil(2*Y/step))+1);ys=np.linspace(-Y,Y,n);fs=np.array([F([x,y]) for y in ys]);rr=[]
 for i in range(n-1):
  if abs(fs[i])<1e-11:rr.append(float(ys[i]))
  if fs[i]*fs[i+1]<0:rr.append(float(bis(lambda y:F([x,y]),ys[i],ys[i+1])))
 rr.sort();out=[]
 for r in rr:
  if not out or abs(r-out[-1])>2*step:out.append(r)
 return out
def H2(F,h):
 z=np.zeros(2);f=F(z);G=np.zeros((2,2))
 for i in range(2):
  e=np.zeros(2);e[i]=h;G[i,i]=(F(e)-2*f+F(-e))/h**2
 a=np.array([h,0]);b=np.array([0,h]);G[0,1]=G[1,0]=(F(a+b)-F(a-b)-F(-a+b)+F(-a-b))/(4*h*h);return G
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--xmax",type=float,default=.08);ap.add_argument("--initial-ymax",type=float,default=.20)
 ap.add_argument("--maximum-ymax",type=float,default=.40);ap.add_argument("--growth",type=float,default=1.6)
 ap.add_argument("--sections",type=int,default=65);ap.add_argument("--scan-step",type=float,default=1e-3)
 ap.add_argument("--fd",type=float,default=2e-4);ap.add_argument("--outdir",default="k1_pasqal_law1_adaptive_v2_4_results")
 a,u=ap.parse_known_args()
 if u:print("[notice] ignored notebook/kernel arguments:",u)
 t=time.time();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 seg=[(2.,.25,0.,.32),(1.7,-.35,1.1,.28),(2.2,.15,2.2,.36),(1.8,-.2,-.7,.30)];tar=ep([0,0],seg)
 task=lambda z:float(max(0,1-abs(np.vdot(tar,ep(z,seg)))**2));expo=lambda z:sum((o*(1+z[0])/2)**2*dt for o,d,p,dt in seg)/sum(dt for o,d,p,dt in seg)
 e0=expo([0,0]);eg=np.array([(expo([a.fd,0])-expo([-a.fd,0]))/(2*a.fd),(expo([0,a.fd])-expo([0,-a.fd]))/(2*a.fd)])
 F=lambda z:task(z)-(expo(z)-e0-eg@np.asarray(z));G=H2(F,a.fd)
 disc=G[0,1]**2-G[0,0]*G[1,1];sl=sorted([(-G[0,1]-math.sqrt(disc))/G[1,1],(-G[0,1]+math.sqrt(disc))/G[1,1]])
 rec=[]
 for x in np.linspace(-a.xmax,a.xmax,a.sections):
  if abs(x)<1e-14:continue
  Y=a.initial_ymax;history=[]
  while True:
   rs=roots(F,float(x),Y,a.scan_step);history.append({"ymax":Y,"root_count":len(rs),"roots":rs})
   if len(rs)>=2 or Y>=a.maximum_ymax-1e-15:break
   Y=min(a.maximum_ymax,Y*a.growth)
  # one final scan at global maximum diagnoses extra distant branches.
  global_rs=rs if abs(Y-a.maximum_ymax)<1e-14 else roots(F,float(x),a.maximum_ymax,a.scan_step)
  rec.append({"x":float(x),"adaptive_ymax":Y,"adaptive_roots":rs,"global_roots":global_rs,"history":history,
   "max_residual":max([abs(F([x,y])) for y in global_rs],default=None)})
 counts=[len(r["global_roots"]) for r in rec];recovered=[r for r in rec if len(r["history"][0]["roots"])<2 and len(r["global_roots"])==2]
 missing=[r for r in rec if len(r["global_roots"])<2];extra=[r for r in rec if len(r["global_roots"])>2]
 maxres=max((r["max_residual"] for r in rec if r["max_residual"] is not None),default=float("inf"))
 # Tangent tracking from closest nonzero sections.
 near=sorted([r for r in rec if len(r["global_roots"])==2],key=lambda r:abs(r["x"]))[:8]
 terr=max((min(abs(y/r["x"]-m) for m in sl) for r in near for y in r["global_roots"]),default=float("inf"))
 gates={"Lorentzian_local_Hessian":bool(np.linalg.det(G)<0),"all_sections_have_two_roots":not missing and all(c>=2 for c in counts),
  "no_extra_zero_branches_to_maximum_boundary":not extra and all(c<=2 for c in counts),"all_root_residuals_small":maxres<1e-9,
  "branches_tangent_to_Hessian_null_rays":terr<.15,"adaptive_expansion_recovers_initial_missing_roots":len(recovered)>0}
 passed=all(gates.values());report={"title":TITLE,"version":VERSION,"scientific_status":
  ("ADAPTIVE_BOUNDED_TWO_BRANCH_ZERO_SET_SUPPORTED" if passed else "ADAPTIVE_ZERO_SET_COMPLETENESS_NOT_SUPPORTED_FAIL_CLOSED"),
  "protocol_sha256":hashlib.sha256(can({"seg":seg,"xmax":a.xmax,"initial":a.initial_ymax,"maximum":a.maximum_ymax,"growth":a.growth,"step":a.scan_step}).encode()).hexdigest(),
  "G_local":G,"G_eigenvalues":np.linalg.eigvalsh(G),"null_tangent_slopes":sl,
  "metrics":{"sections":len(rec),"initial_missing_sections":sum(len(r["history"][0]["roots"])<2 for r in rec),"recovered_sections":len(recovered),
   "still_missing_sections":len(missing),"extra_branch_sections":len(extra),"root_count_min":min(counts),"root_count_max":max(counts),
   "maximum_y_used":max(r["adaptive_ymax"] for r in rec),"maximum_root_residual":maxres,"tangent_slope_error":terr},
  "gates":gates,"Law_I_d_finite_extension_supported":all(len(r["global_roots"])>=2 for r in rec),
  "Law_I_e_complete_in_frozen_adaptive_domain":passed,"Law_I_f_Principle_R_origin_supported":False,
  "complete_Law_I_candidate_supported":False,
  "interpretation":"Adaptive expansion distinguishes roots leaving the original rectangle from genuine branch termination. Exhaustive numerical completeness is asserted only inside the frozen maximum rectangle.",
  "next_required_step":"If I-e passes, derive/certify I-f from Principle R and seek an analytic implicit-function/Morse continuation theorem. If missing roots remain, inspect the listed sections rather than widening post hoc again.",
  "claim_boundary":"Bounded numerical zero-set enumeration only; no global theorem, native Law-II/III, hardware or wavefunction claim.",
  "artifacts":{"adaptive_sections":str((out/"adaptive_sections.json").resolve())},"elapsed_seconds":time.time()-t,"environment":{"python":platform.python_version(),"numpy":np.__version__}}
 (out/"adaptive_sections.json").write_text(json.dumps(rec,indent=2,default=jd)+"\n");(out/"run_summary.json").write_text(json.dumps(report,indent=2,default=jd)+"\n")
 print("="*112);print(f"{TITLE} v{VERSION}");print("="*112);print(json.dumps(report,indent=2,default=jd));return 0
if __name__=="__main__":
 rc=main()
 if not any(x in sys.modules for x in ("ipykernel","IPython","google.colab")):raise SystemExit(rc)
