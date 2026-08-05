#!/usr/bin/env python3
"""Prospective cross-protocol zero-cone naturality audit for R -> Law-I.

Manifest JSON schema is emitted on first run.  Every protocol has its own CSV
with columns x,y,F,split.  F must be independently defined and nonnegative.
The frozen map z=T x sends protocol coordinates to canonical coordinates.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, sys, time
from pathlib import Path
import numpy as np

try:
    from audits.common import source_hash_matches
except ModuleNotFoundError:
    from common import source_hash_matches

TITLE="PRINCIPLE R -> LAW-I CROSS-PROTOCOL ZERO-CONE NATURALITY AUDIT"
VERSION="3.2"

def J(x):
    if isinstance(x,dict): return {str(k):J(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [J(v) for v in x]
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    return x
def sha(b): return hashlib.sha256(b).hexdigest()
def hfile(p): return sha(Path(p).read_bytes())
def hobj(o): return sha(json.dumps(J(o),sort_keys=True,separators=(",",":")).encode())
def load_csv(p):
    a=np.genfromtxt(p,delimiter=",",names=True,dtype=None,encoding="utf-8")
    if not a.dtype.names or not {"x","y","F","split"}.issubset(a.dtype.names): raise ValueError(f"{p}: need x,y,F,split")
    X=np.c_[a["x"].astype(float),a["y"].astype(float)]; F=a["F"].astype(float); S=np.asarray(a["split"],str)
    if np.any(~np.isfinite(X)) or np.any(~np.isfinite(F)) or np.any(F < -1e-12): raise ValueError(f"{p}: finite nonnegative F required")
    if not {"train","heldout"}.issubset(set(S)): raise ValueError(f"{p}: train and heldout required")
    return X,np.maximum(F,0),S

def nq(X,G):
    q=np.einsum("ni,ij,nj->n",X,G,X); r2=np.sum(X*X,axis=1)
    return np.abs(q)/np.maximum(np.linalg.norm(G)*r2,1e-300)

def fit_G(X,zero):
    Z=X[zero]
    if len(Z)<6:return None
    M=np.c_[Z[:,0]**2,2*Z[:,0]*Z[:,1],Z[:,1]**2]
    _,_,vh=np.linalg.svd(M,full_matrices=False);g=vh[-1];G=np.array([[g[0],g[1]],[g[1],g[2]]]);n=np.linalg.norm(G)
    return None if n<1e-15 else G/n

def canon(G,T):
    Ti=np.linalg.inv(T); C=Ti.T@G@Ti;return C/np.linalg.norm(C)

def projective_res(A,B):
    # Null sets determine G only up to any nonzero scalar: compare both signs.
    rp=np.linalg.norm(A-B);rm=np.linalg.norm(A+B)
    return float(min(rp,rm)), (1 if rp<=rm else -1), float(rp)

def cone_angles(G):
    th=np.linspace(0,math.pi,300000,endpoint=False);U=np.c_[np.cos(th),np.sin(th)];v=nq(U,G);chosen=[]
    for i in np.argsort(v):
        t=float(th[i])
        if all(abs(((t-s+math.pi/2)%math.pi)-math.pi/2)>1e-3 for s in chosen):chosen.append(t)
        if len(chosen)==2:break
    return np.sort(chosen)

def branch_distance(a,b):
    d=lambda x,y:abs(((x-y+math.pi/2)%math.pi)-math.pi/2)
    return float(min(max(d(a[0],b[0]),d(a[1],b[1])),max(d(a[0],b[1]),d(a[1],b[0]))))

def one_protocol(item,base,cfg):
    p=(base/item["data_csv"]).resolve();X,F,S=load_csv(p);tr=S=="train";te=S=="heldout";fz=F<=cfg["F_zero_tol"]
    G=fit_G(X,tr&fz); T=np.asarray(item["to_canonical_T"],float)
    if G is None: return {"name":item["name"],"gate":False,"error":"insufficient training F-zero geometry"}
    qz=nq(X,G)<=cfg["q_zero_tol"];nf=int(np.sum(te&fz));nqz=int(np.sum(te&qz))
    fbad=int(np.sum(te&fz&~qz));qbad=int(np.sum(te&~fz&qz))
    ev=np.linalg.eigvalsh(G);det=float(np.linalg.det(G));Cg=canon(G,T)
    gates={"data_hash_matches":hfile(p)==item["data_sha256"],
      "cost_definition_source_bound":source_hash_matches(item,path_key="cost_definition_source_path",hash_key="cost_definition_source_sha256",base_dir=base),
      "protocol_predeclared":bool(item["predeclared_before_cross_comparison"]),"mapping_predeclared":bool(item["mapping_predeclared_before_outcomes"]),
      "mapping_invertible":abs(float(np.linalg.det(T)))>1e-12,"heldout_Fzero_coverage":nf>=cfg["minimum_zero_points"],
      "heldout_qzero_coverage":nqz>=cfg["minimum_zero_points"],"Fzero_implies_qzero":fbad/max(1,nf)<=cfg["maximum_violation_rate"],
      "qzero_implies_Fzero":qbad/max(1,nqz)<=cfg["maximum_violation_rate"],"fitted_G_Lorentzian":ev[0]<0<ev[1]}
    return {"name":item["name"],"data_sha256":hfile(p),"G_protocol":G,"G_canonical":Cg,"eigenvalues":ev,"detG":det,
      "canonical_branch_angles":cone_angles(Cg),"heldout":{"Fzero":nf,"qzero":nqz,"Fzero_qnonzero":fbad,"qzero_Fpositive":qbad},
      "gates":gates,"gate":all(gates.values())}

def audit_manifest(m,path,cfg):
    base=path.parent; rec=[one_protocol(x,base,cfg) for x in m["protocols"]];valid=[r for r in rec if "G_canonical" in r]
    pairs=[]
    for i in range(len(valid)):
      for j in range(i+1,len(valid)):
        r,s=valid[i],valid[j];pr,sgn,pos=projective_res(r["G_canonical"],s["G_canonical"]);bd=branch_distance(r["canonical_branch_angles"],s["canonical_branch_angles"])
        pairs.append({"pair":[r["name"],s["name"]],"unoriented_projective_residual":pr,"relative_sign":sgn,
          "positive_conformal_residual":pos,"branch_pair_distance_radians":bd,
          "unoriented_cone_match":pr<=cfg["cone_residual_tol"] and bd<=cfg["branch_angle_tol"],
          "positive_coorientation_match":pos<=cfg["cone_residual_tol"]})
    provenance={"at_least_two_protocols":len(rec)>=2,"all_protocol_costs_independently_defined":bool(m["provenance"]["costs_independent_of_each_other_and_TESC"]),
      "comparison_rule_frozen":bool(m["provenance"]["comparison_rule_frozen_before_outcomes"]),
      "no_outcome_based_protocol_selection":bool(m["provenance"]["no_protocol_selected_after_outcomes"])}
    gates={"provenance":all(provenance.values()),"all_protocols_pass_local_binding":len(rec)>=2 and all(r["gate"] for r in rec),
      "all_pairs_same_unoriented_cone":bool(pairs) and all(p["unoriented_cone_match"] for p in pairs)}
    oriented=gates["all_pairs_same_unoriented_cone"] and all(p["positive_coorientation_match"] for p in pairs)
    return {"protocol_records":rec,"pairwise_naturality":pairs,"provenance_gates":provenance,"gates":gates,
      "cross_protocol_unoriented_cone_class_supported":all(gates.values()),"positive_coorientation_selected":bool(oriented),
      "gate":all(gates.values())}

def synth(out,cfg):
    rng=np.random.default_rng(20260805);G=np.array([[1.,0.],[0.,-1.]])
    source=out/"selftest_cost_definition.txt";source.write_text("synthetic cost definition for v3.2 self-test\n")
    source_hash=hfile(source)
    items=[]
    for k,T in enumerate((np.eye(2),np.array([[1.4,.3],[-.2,.9]]))):
      rows=["x,y,F,split"];Ti=np.linalg.inv(T)
      for i in range(800):
        z=rng.normal(size=2);z/=np.linalg.norm(z);x=Ti@z;q=float(z@G@z);F=abs(q);sp="train" if i%2==0 else "heldout"
        rows.append(f"{x[0]:.17g},{x[1]:.17g},{F:.17g},{sp}")
      p=out/f"selftest_protocol_{k}.csv";p.write_text("\n".join(rows)+"\n")
      items.append({"name":f"p{k}","data_csv":p.name,"data_sha256":hfile(p),
        "cost_definition_source_path":source.name,"cost_definition_source_sha256":source_hash,
        "predeclared_before_cross_comparison":True,"mapping_predeclared_before_outcomes":True,"to_canonical_T":T.tolist()})
    m={"protocols":items,"provenance":{"costs_independent_of_each_other_and_TESC":True,"comparison_rule_frozen_before_outcomes":True,"no_protocol_selected_after_outcomes":True}}
    # Self-test needs tolerance-compatible near-zero samples; F tolerance selects q near zero statistically poorly.
    # Add exact ray points to both files.
    for k,item in enumerate(items):
      p=out/item["data_csv"];T=np.asarray(item["to_canonical_T"]);Ti=np.linalg.inv(T);lines=p.read_text().splitlines()
      for i in range(80):
        z=np.array([1.,1. if i%2==0 else -1.])*rng.uniform(.1,1);x=Ti@z;lines.append(f"{x[0]:.17g},{x[1]:.17g},0,{'train' if i%4<2 else 'heldout'}")
      p.write_text("\n".join(lines)+"\n");item["data_sha256"]=hfile(p)
    good=audit_manifest(m,out/"selftest_manifest.json",cfg)
    mb=json.loads(json.dumps(m));mb["protocols"][1]["to_canonical_T"]=[[1,0],[0,1]]
    bad=audit_manifest(mb,out/"selftest_bad_manifest.json",cfg)
    return {"matched_cross_protocol_positive_control_pass":good["gate"],"wrong_mapping_negative_control_rejected":not bad["gate"]}

def template(out):
    for name in ("protocol_A.csv","protocol_B.csv"):(out/name).write_text("x,y,F,split\n0.0,0.0,nan,train\n")
    m={"schema":"r-law1-cross-protocol-v3.2","protocols":[
      {"name":"protocol_A","data_csv":"protocol_A.csv","data_sha256":"","cost_definition_source_path":"","cost_definition_source_sha256":"","predeclared_before_cross_comparison":False,"mapping_predeclared_before_outcomes":False,"to_canonical_T":[[1,0],[0,1]]},
      {"name":"protocol_B","data_csv":"protocol_B.csv","data_sha256":"","cost_definition_source_path":"","cost_definition_source_sha256":"","predeclared_before_cross_comparison":False,"mapping_predeclared_before_outcomes":False,"to_canonical_T":[[1,0],[0,1]]}],
      "provenance":{"costs_independent_of_each_other_and_TESC":False,"comparison_rule_frozen_before_outcomes":False,"no_protocol_selected_after_outcomes":False}}
    (out/"cross_protocol_manifest_template.json").write_text(json.dumps(m,indent=2)+"\n")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--manifest",default="");ap.add_argument("--outdir",default="r_law1_cross_protocol_v3_2_results")
    a,u=ap.parse_known_args();
    if u:print("[notice] ignored notebook/kernel arguments:",u)
    t=time.time();out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    cfg={"F_zero_tol":1e-8,"q_zero_tol":2e-4,"minimum_zero_points":12,"maximum_violation_rate":.05,"cone_residual_tol":.02,"branch_angle_tol":.02}
    controls=synth(out,cfg);mp=Path(a.manifest) if a.manifest else None
    if mp and mp.is_file():
      m=json.loads(mp.read_text());emp=audit_manifest(m,mp,cfg);msha=hfile(mp)
      status="CROSS_PROTOCOL_UNORIENTED_LAWI_CONE_CLASS_SUPPORTED" if emp["gate"] else "CROSS_PROTOCOL_CONE_NATURALITY_NOT_SUPPORTED_FAIL_CLOSED"
    else:template(out);emp=None;msha=None;status="PIPELINE_CALIBRATED_INDEPENDENT_PROTOCOL_DATA_REQUIRED"
    protocol={"title":TITLE,"version":VERSION,"criteria":cfg};protocol["protocol_sha256"]=hobj(protocol)
    report={"title":TITLE,"version":VERSION,"scientific_status":status,"protocol_sha256":protocol["protocol_sha256"],"manifest_supplied":emp is not None,"manifest_sha256":msha,
      "self_tests":controls,"empirical_audit":emp,"all_scientific_gates_pass":bool(emp and emp["gate"] and all(controls.values())),
      "interpretation":"Each protocol must independently recover a held-out two-branch zero cone. Frozen realization maps must carry the unordered branch pair to one common projective conformal class. Zero sets alone determine G only up to nonzero scale; positive coorientation is reported separately.",
      "next_required_step":"Populate two independently sourced protocol CSVs and freeze their coordinate maps before comparing outcomes; then rerun with --manifest.",
      "claim_boundary":"A pass supports a local cross-protocol unoriented cone class, not a physical time orientation, metric scale, spacetime, Law-II/III or wavefunction.","elapsed_seconds":time.time()-t,"environment":{"python":platform.python_version(),"numpy":np.__version__}}
    (out/"protocol.json").write_text(json.dumps(J(protocol),indent=2)+"\n");(out/"run_summary.json").write_text(json.dumps(J(report),indent=2)+"\n")
    print("="*112);print(f"{TITLE} v{VERSION}");print("="*112);print(json.dumps(J(report),indent=2));return 0
if __name__=="__main__":
    rc=main()
    if not any(x in sys.modules for x in ("ipykernel","IPython","google.colab")):raise SystemExit(rc)
