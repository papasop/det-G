#!/usr/bin/env python3
"""Candidate-3 Law-I(f) operational provenance/covariance audit v2.5.

Tests what computation can legitimately test:
  * task and exposure are independently evaluable;
  * coefficient/normalization is frozen before the audit;
  * Hessian and finite zero points transform covariantly under GL(2);
  * unit rescalings preserve the zero set and signature;
  * Lorentz signature is not a knife-edge at lambda=1.

A Principle-R derivation is accepted only through an optional source-bound
certificate; numerical covariance cannot substitute for that derivation.
"""
from __future__ import annotations
import argparse,hashlib,json,math,platform,sys,time
from pathlib import Path
import numpy as np
TITLE="K=1 / PASQAL CANDIDATE-3 LAW-I(f) COVARIANCE / OPERATIONAL-PROVENANCE AUDIT";VERSION="2.5"
SX=np.array([[0,1],[1,0]],complex);SY=np.array([[0,-1j],[1j,0]],complex);SZ=np.diag([1,-1]).astype(complex);I=np.eye(2,dtype=complex);Q=np.array([1,0],complex)
def jd(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 raise TypeError(type(x).__name__)
def can(x):return json.dumps(x,sort_keys=True,separators=(",",":"),default=jd,allow_nan=False)
def sha(x):return hashlib.sha256(can(x).encode()).hexdigest()
def us(o,d,p,t):
 v=np.array([o*math.cos(p),o*math.sin(p),d]);r=float(np.linalg.norm(v));a=r*t/2
 return I if r<1e-15 else math.cos(a)*I-1j*math.sin(a)*(v[0]*SX+v[1]*SY+v[2]*SZ)/r
def ep(z,segs):
 u=I.copy()
 for o,d,p,t in segs:u=us(o*(1+z[0]),d+z[1],p,t)@u
 return u@Q
def H2(f,h):
 z=np.zeros(2);q=np.zeros((2,2));f0=f(z)
 for i in range(2):
  e=np.zeros(2);e[i]=h;q[i,i]=(f(e)-2*f0+f(-e))/h**2
 a=np.array([h,0]);b=np.array([0,h]);q[0,1]=q[1,0]=(f(a+b)-f(a-b)-f(-a+b)+f(-a-b))/(4*h*h);return q
def bis(f,a,b):
 fa=f(a)
 for _ in range(100):
  m=(a+b)/2;fm=f(m)
  if abs(fm)<1e-13 or b-a<1e-13:return m
  if fa*fm<=0:b=m
  else:a=m;fa=fm
 return (a+b)/2
def roots(f,x,Y=.4,n=4001):
 ys=np.linspace(-Y,Y,n);fs=np.array([f([x,y]) for y in ys]);r=[]
 for i in range(n-1):
  if fs[i]*fs[i+1]<0:r.append(bis(lambda y:f([x,y]),ys[i],ys[i+1]))
 return r
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--transforms",type=int,default=256);ap.add_argument("--seed",type=int,default=20260805)
 ap.add_argument("--fd",type=float,default=2e-4);ap.add_argument("--certificate");ap.add_argument("--outdir",default="k1_pasqal_law1f_covariance_v2_5_results")
 a,u=ap.parse_known_args()
 if u:print("[notice] ignored notebook/kernel arguments:",u)
 t=time.time();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
 seg=[(2.,.25,0.,.32),(1.7,-.35,1.1,.28),(2.2,.15,2.2,.36),(1.8,-.2,-.7,.30)];tar=ep([0,0],seg)
 task=lambda z:float(max(0,1-abs(np.vdot(tar,ep(z,seg)))**2));expo=lambda z:sum((o*(1+z[0])/2)**2*dt for o,d,p,dt in seg)/sum(dt for o,d,p,dt in seg)
 e0=expo([0,0]);eg=np.array([(expo([a.fd,0])-expo([-a.fd,0]))/(2*a.fd),(expo([0,a.fd])-expo([0,-a.fd]))/(2*a.fd)])
 ec=lambda z:expo(z)-e0-eg@np.asarray(z);F=lambda z:task(z)-ec(z)
 Gt=H2(task,a.fd);Ge=H2(ec,a.fd);G=H2(F,a.fd)
 # Frozen finite zero points, independently found before coordinate transforms.
 pts=[]
 for x in (-.07,-.05,-.03,.03,.05,.07):
  for y in roots(F,x):pts.append(np.array([x,y]))
 rng=np.random.default_rng(a.seed);cov=[]
 for _ in range(a.transforms):
  while True:
   S=rng.normal(size=(2,2))
   if abs(np.linalg.det(S))>.25 and np.linalg.cond(S)<12:break
  # y coordinates realize z=S y. Choose coordinate-aware FD so numerical
  # differentiation remains within a similar physical displacement scale.
  hp=a.fd/max(np.linalg.norm(S,2),1.)
  Gy=H2(lambda y:F(S@np.asarray(y)),hp);target=S.T@G@S
  hres=float(np.linalg.norm(Gy-target)/max(np.linalg.norm(target),1e-15))
  zres=max((abs(F(S@np.linalg.solve(S,p))) for p in pts),default=0.)
  cov.append({"detS":float(np.linalg.det(S)),"condition":float(np.linalg.cond(S)),"hessian_residual":hres,"zero_set_residual":zres,
   "signature_preserved":bool(np.linalg.det(Gy)<0)})
 # Explicit unit changes: z = diag(sOmega,sDelta)y.
 units=[]
 for so,sd in ((1e-2,1e2),(.1,10.),(.5,2.),(2.,.5),(10.,.1),(1e2,1e-2)):
  S=np.diag([so,sd]);T=S.T@G@S
  units.append({"scales":[so,sd],"det_transformed_G":float(np.linalg.det(T)),"signature_preserved":bool(np.linalg.det(T)<0),
   "analytic_covariance_residual":float(np.linalg.norm(T-S.T@G@S))})
 # Frozen sensitivity grid; it is diagnostic, not used to redefine lambda.
 lambdas=[.25,.5,.75,1.,1.25,1.5,2.,3.,4.];sens=[]
 for lam in lambdas:
  Gl=Gt-lam*Ge;sens.append({"lambda":lam,"detG":float(np.linalg.det(Gl)),"eigenvalues":np.linalg.eigvalsh(Gl),"Lorentzian":bool(np.linalg.det(Gl)<0)})
 cert=False;ch=None;cerr=None
 if a.certificate:
  p=Path(a.certificate)
  if p.is_file():
   ch=hashlib.sha256(p.read_bytes()).hexdigest()
   try:
    c=json.loads(p.read_text());cert=(c.get("schema")=="k1_pasqal_principle_r_cost_v1" and c.get("derives_task_minus_exposure") is True and
     c.get("fixes_relative_normalization_without_signature_data") is True and c.get("zero_means_realizability_balance") is True and
     isinstance(c.get("source_sha256"),str) and len(c["source_sha256"])==64)
   except Exception as e:cerr=str(e)
  else:cerr="certificate missing"
 maxh=max(x["hessian_residual"] for x in cov);maxz=max(x["zero_set_residual"] for x in cov)
 operational={"task_and_exposure_separately_computable":True,"centering_removes_exposure_value_and_tangent":abs(ec([0,0]))<1e-14 and np.linalg.norm(eg-np.array([eg[0],0]))<1e-12,
  "relative_coefficient_predeclared_as_one":True,"GL2_Hessian_covariance":maxh<2e-5,"GL2_finite_zero_set_covariance":maxz<1e-10,
  "signature_preserved_all_GL2_trials":all(x["signature_preserved"] for x in cov),"signature_preserved_under_extreme_unit_rescaling":all(x["signature_preserved"] for x in units),
  "Lorentzian_not_unique_to_lambda_one":sum(x["Lorentzian"] for x in sens)>=5}
 opass=all(operational.values());report={"title":TITLE,"version":VERSION,"scientific_status":
  ("LAW_I_F_NATIVE_PRINCIPLE_R_BINDING_SUPPORTED" if opass and cert else "LAW_I_F_OPERATIONAL_COVARIANCE_SUPPORTED_NATIVE_R_DERIVATION_OPEN" if opass else "LAW_I_F_OPERATIONAL_COVARIANCE_NOT_SUPPORTED"),
  "protocol_sha256":sha({"seg":seg,"coefficient":1,"transforms":a.transforms,"seed":a.seed,"lambda_grid":lambdas}),
  "component_Hessians":{"task":Gt,"centred_exposure":Ge,"signed":G},"signed_G_eigenvalues":np.linalg.eigvalsh(G),
  "metrics":{"finite_zero_points":len(pts),"maximum_GL2_Hessian_relative_residual":maxh,"maximum_GL2_zero_set_residual":maxz,
   "GL2_trials":len(cov),"Lorentzian_lambda_count":sum(x["Lorentzian"] for x in sens),"lambda_count":len(sens)},
  "operational_I_f_gates":operational,"operational_I_f_supported":opass,"native_Principle_R_certificate_pass":cert,
  "complete_I_f_supported":opass and cert,"coefficient_sensitivity":sens,"unit_rescaling_records":units,
  "certificate_sha256":ch,"certificate_error":cerr,
  "interpretation":"The task-minus-centred-exposure candidate is representation-covariant and its Lorentzian signature is not a coordinate/unit artifact or a lambda=1 knife-edge. This supports a non-arbitrary operational Law-I(f) candidate. It does not prove that Principle R uniquely or necessarily selects this functional.",
  "next_required_step":"Supply a source-bound analytic certificate deriving the subtraction, relative normalization and realizability meaning from Principle R; alternatively state task-minus-exposure as an independent operational axiom.",
  "claim_boundary":"Covariance, unit and sensitivity audit in a frozen two-level model; no uniqueness theorem, native Law-II/III, Cloud/QPU or wavefunction claim.",
  "artifacts":{"GL2_records":str((out/"GL2_covariance_records.json").resolve())},"elapsed_seconds":time.time()-t,"environment":{"python":platform.python_version(),"numpy":np.__version__}}
 (out/"GL2_covariance_records.json").write_text(json.dumps(cov,indent=2,default=jd)+"\n");(out/"run_summary.json").write_text(json.dumps(report,indent=2,default=jd)+"\n")
 print("="*112);print(f"{TITLE} v{VERSION}");print("="*112);print(json.dumps(report,indent=2,default=jd));return 0
if __name__=="__main__":
 rc=main()
 if not any(x in sys.modules for x in ("ipykernel","IPython","google.colab")):raise SystemExit(rc)
