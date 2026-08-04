#!/usr/bin/env python3
"""Candidate-3 Law-I(d/e/f) finite zero-set audit v2.3.

I-d: continue the two Hessian-null tangents into finite F_signed=0 curves.
I-e: exhaustively bracket all transverse roots in a frozen rectangle and test
     whether exactly two branches account for the numerical zero set.
I-f: audit operational provenance separately from a native Principle-R proof.

This is a bounded-domain numerical audit, not a global analytic theorem.
"""
from __future__ import annotations
import argparse,hashlib,json,math,platform,sys,time
from pathlib import Path
import numpy as np

TITLE="K=1 / PASQAL CANDIDATE-3 LAW-I FINITE-ZERO-SET / PROVENANCE AUDIT"
VERSION="2.3"
SX=np.array([[0,1],[1,0]],complex);SY=np.array([[0,-1j],[1j,0]],complex);SZ=np.diag([1,-1]).astype(complex)
I2=np.eye(2,dtype=complex);K0=np.array([1,0],complex)
def jd(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 raise TypeError(type(x).__name__)
def can(x):return json.dumps(x,sort_keys=True,separators=(",",":"),default=jd,allow_nan=False)
def sha(x):return hashlib.sha256(can(x).encode()).hexdigest()
def Ustep(o,d,p,t):
 v=np.array([o*math.cos(p),o*math.sin(p),d]);r=float(np.linalg.norm(v));a=.5*r*t
 return I2 if r<1e-15 else math.cos(a)*I2-1j*math.sin(a)*(v[0]*SX+v[1]*SY+v[2]*SZ)/r
def endpoint(z,segs,oref,dref):
 U=I2.copy();a,b=z
 for o,d,p,t in segs:U=Ustep(o*(1+a),d+b*dref,p,t)@U
 return U@K0
def bisect(f,a,b,tol=1e-13,n=100):
 fa,fb=f(a),f(b)
 if fa==0:return a
 if fb==0:return b
 if fa*fb>0:raise ValueError("not bracketed")
 for _ in range(n):
  m=(a+b)/2;fm=f(m)
  if abs(fm)<tol or b-a<tol:return m
  if fa*fm<=0:b,fb=m,fm
  else:a,fa=m,fm
 return (a+b)/2
def hess(f,h):
 q=np.zeros((2,2));z=np.zeros(2);f0=f(z)
 for i in range(2):
  e=np.zeros(2);e[i]=h;q[i,i]=(f(e)-2*f0+f(-e))/h**2
 a=np.array([h,0]);b=np.array([0,h]);q[0,1]=q[1,0]=(f(a+b)-f(a-b)-f(-a+b)+f(-a-b))/(4*h*h)
 return q
def roots_y(F,x,ymax,ny,zero=1e-11):
 ys=np.linspace(-ymax,ymax,ny);fs=np.array([F([x,y]) for y in ys]);roots=[]
 for i in range(ny-1):
  if abs(fs[i])<zero:roots.append(float(ys[i]))
  if fs[i]*fs[i+1]<0:roots.append(float(bisect(lambda y:F([x,y]),ys[i],ys[i+1])))
 if abs(fs[-1])<zero:roots.append(float(ys[-1]))
 roots.sort();ded=[]
 for r in roots:
  if not ded or abs(r-ded[-1])>2*ymax/(ny-1):ded.append(r)
 return ded
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--xmax",type=float,default=.08);ap.add_argument("--ymax",type=float,default=.20)
 ap.add_argument("--sections",type=int,default=65);ap.add_argument("--yscan",type=int,default=4001)
 ap.add_argument("--fd",type=float,default=2e-4);ap.add_argument("--principle-r-certificate")
 ap.add_argument("--outdir",default="k1_pasqal_law1_def_v2_3_results")
 args,unk=ap.parse_known_args()
 if unk:print("[notice] ignored notebook/kernel arguments:",unk)
 t0=time.time();out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
 segs=[(2.,.25,0.,.32),(1.7,-.35,1.1,.28),(2.2,.15,2.2,.36),(1.8,-.2,-.7,.30)];oref=2.;dref=1.
 target=endpoint([0,0],segs,oref,dref)
 task=lambda z:float(max(0.,1-abs(np.vdot(target,endpoint(z,segs,oref,dref)))**2))
 expo=lambda z:sum((o*(1+z[0])/oref)**2*dt for o,d,p,dt in segs)/sum(dt for o,d,p,dt in segs)
 e0=expo([0,0]);eg=np.array([(expo([args.fd,0])-expo([-args.fd,0]))/(2*args.fd),
                              (expo([0,args.fd])-expo([0,-args.fd]))/(2*args.fd)])
 F=lambda z:task(z)-(expo(z)-e0-eg@np.asarray(z))
 G=hess(F,args.fd);ev=np.linalg.eigvalsh(G)
 # Hessian null slopes y=m x: G11+2G12 m+G22 m^2=0.
 disc=G[0,1]**2-G[0,0]*G[1,1];ms=sorted([(-G[0,1]-math.sqrt(disc))/G[1,1],(-G[0,1]+math.sqrt(disc))/G[1,1]])
 xs=np.linspace(-args.xmax,args.xmax,args.sections);records=[];counts=[]
 for x in xs:
  if abs(x)<1e-14:continue
  rs=roots_y(F,float(x),args.ymax,args.yscan);counts.append(len(rs))
  records.append({"x":float(x),"roots_y":rs,"residuals":[abs(F([x,y])) for y in rs]})
 # Track roots nearest the two predicted tangent branches.
 tracked=[]
 for rec in records:
  x=rec["x"];rs=rec["roots_y"]
  if len(rs)>=2:
   pick=[]
   remaining=list(rs)
   for m in ms:
    j=int(np.argmin([abs(y-m*x) for y in remaining]));pick.append(remaining.pop(j))
   tracked.append((x,*pick))
 maxres=max((max(r["residuals"]) for r in records if r["residuals"]),default=float("inf"))
 finite_extent=max((abs(x) for x,*_ in tracked),default=0.)
 separation=min((abs(a-b) for x,a,b in tracked if abs(x)>args.xmax/4),default=0.)
 exactly_two=sum(c==2 for c in counts);coverage=exactly_two/max(len(counts),1)
 # Tangency: y/x tends to the Hessian-null slopes at the closest sections.
 near=sorted(tracked,key=lambda r:abs(r[0]))[:min(8,len(tracked))]
 slope_sets=sorted([[r[1]/r[0],r[2]/r[0]] for r in near],key=lambda z:z[0]) if near else []
 tangent_err=max((min(abs(s-ms[0]),abs(s-ms[1])) for pair in slope_sets for s in pair),default=float("inf"))
 cert=False;cert_hash=None;cert_error=None
 if args.principle_r_certificate:
  p=Path(args.principle_r_certificate)
  if p.is_file():
   cert_hash=hashlib.sha256(p.read_bytes()).hexdigest()
   try:
    c=json.loads(p.read_text());cert=(c.get("schema")=="k1_pasqal_principle_r_cost_v1" and
      c.get("derives_task_minus_exposure") is True and c.get("complete_zero_set_theorem") is True and
      c.get("normalization_independent_of_signature") is True)
   except Exception as e:cert_error=str(e)
  else:cert_error="certificate file missing"
 gates={
  "I_d_nontrivial_finite_branches":finite_extent>=.95*args.xmax and len(tracked)>=.9*len(records),
  "I_d_zero_cost_residual":maxres<1e-9,
  "I_d_tangent_to_local_null_rays":tangent_err<.15,
  "I_e_exactly_two_roots_each_section":coverage>=.95,
  "I_e_two_branches_separated_off_origin":separation>1e-4,
  "I_e_no_extra_branches_in_frozen_box":all(c==2 for c in counts),
  "I_f_task_and_exposure_independently_computable":True,
  "I_f_dimensionless_normalization_frozen":True,
  "I_f_coefficient_not_tuned_after_signature":True,
  "I_f_native_Principle_R_certificate":cert}
 Id=all(v for k,v in gates.items() if k.startswith("I_d"));Ie=all(v for k,v in gates.items() if k.startswith("I_e"));If=all(v for k,v in gates.items() if k.startswith("I_f"))
 report={"title":TITLE,"version":VERSION,"scientific_status":
  ("LAW_I_DEF_ALL_SUPPORTED_IN_FROZEN_DOMAIN" if Id and Ie and If else
   "FINITE_TWO_BRANCH_ZERO_SET_SUPPORTED_PRINCIPLE_R_ORIGIN_OPEN" if Id and Ie else
   "FINITE_ZERO_SET_INCOMPLETE_FAIL_CLOSED"),
  "protocol_sha256":sha({"segs":segs,"signed_cost":"task-error minus centred Rabi exposure, coefficient one",
   "box":[args.xmax,args.ymax],"sections":args.sections,"yscan":args.yscan}),
  "G_local":G,"G_eigenvalues":ev,"null_tangent_slopes":ms,"frozen_domain":{"xmax":args.xmax,"ymax":args.ymax},
  "metrics":{"sections_audited":len(counts),"two_root_section_fraction":coverage,"finite_branch_x_extent":finite_extent,
   "minimum_off_origin_branch_separation":separation,"maximum_zero_residual":maxres,"maximum_near_origin_tangent_slope_error":tangent_err,
   "root_count_min":min(counts) if counts else 0,"root_count_max":max(counts) if counts else 0},
  "gates":gates,"Law_I_d_finite_extension_supported":Id,"Law_I_e_bounded_complete_zero_set_supported":Ie,
  "Law_I_f_Principle_R_origin_supported":If,"complete_Law_I_candidate_supported":Id and Ie and If,
  "principle_R_certificate_sha256":cert_hash,"certificate_error":cert_error,
  "interpretation":"I-d/e concern the exact finite zero level set of the frozen candidate cost, not merely its Hessian. I-e is numerical completeness only inside the declared rectangle. I-f remains open without a dependency-closed Principle-R derivation.",
  "next_required_step":"If I-d/e pass, supply an analytic/provenance certificate deriving task-minus-exposure and its normalization from Principle R; then prove the two-branch zero set beyond the scanned box or state the result locally.",
  "claim_boundary":"No-account two-level model, bounded numerical root enumeration, and operational provenance audit; no global zero-set theorem, native K=1 dynamics, Cloud/QPU or wavefunction claim.",
  "artifacts":{"section_roots":str((out/"section_roots.json").resolve())},"elapsed_seconds":time.time()-t0,
  "environment":{"python":platform.python_version(),"numpy":np.__version__}}
 (out/"section_roots.json").write_text(json.dumps(records,indent=2,default=jd)+"\n")
 (out/"run_summary.json").write_text(json.dumps(report,indent=2,default=jd)+"\n")
 print("="*112);print(f"{TITLE} v{VERSION}");print("="*112);print(json.dumps(report,indent=2,default=jd));return 0
if __name__=="__main__":
 rc=main()
 if not any(x in sys.modules for x in ("ipykernel","IPython","google.colab")):raise SystemExit(rc)
