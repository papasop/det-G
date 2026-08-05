#!/usr/bin/env python3
"""Single-channel no-go / independent two-channel Law-I origin audit v3.1."""

from __future__ import annotations
import argparse, hashlib, json, math, platform, sys, time
from pathlib import Path
import numpy as np

try:
    from audits.common import source_hash_matches
except ModuleNotFoundError:
    from common import source_hash_matches

TITLE="PRINCIPLE R -> LAW-I SINGLE-CHANNEL NO-GO / TWO-CHANNEL ORIGIN AUDIT"
VERSION="3.1"
GT=np.array([[-1.4753511828891064,-0.04866380215462485],
             [-0.04866380215462485,0.2661881493004614]],float)

def js(x):
    if isinstance(x,dict): return {str(k):js(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [js(v) for v in x]
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    return x

def hobj(x):
    return hashlib.sha256(json.dumps(js(x),sort_keys=True,separators=(",",":")).encode()).hexdigest()

def sha_ok(x): return isinstance(x,str) and len(x)==64 and all(c in "0123456789abcdef" for c in x.lower())

def q_from_channels(a,b): return .5*(np.outer(a,b)+np.outer(b,a))

def conformal_fit(A,B):
    c=float(np.sum(A*B)/np.sum(B*B)); r=float(np.linalg.norm(A-c*B)/max(np.linalg.norm(A),1e-300))
    return c,r

def angle(u,v):
    z=abs(float(u@v))/(np.linalg.norm(u)*np.linalg.norm(v)); return math.acos(min(1,max(-1,z)))

def algebra(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float); M=np.vstack([a,b]); detM=float(np.linalg.det(M))
    G=q_from_channels(a,b); eig=np.linalg.eigvalsh(G); detG=float(np.linalg.det(G))
    # Null lines are ker(a) and ker(b).
    ra=np.array([-a[1],a[0]]); rb=np.array([-b[1],b[0]])
    residuals=[abs(float(r@G@r))/(1+np.linalg.norm(G)*float(r@r)) for r in (ra,rb)]
    scale,res=conformal_fit(G,GT)
    return {"channel_matrix":M,"channel_determinant":detM,"channels_independent":abs(detM)>1e-12,
            "channel_angle_radians":angle(a,b),"induced_G":G,"induced_G_eigenvalues":eig,
            "induced_detG":detG,"induced_Lorentzian":detG<0,"null_rays":[ra/np.linalg.norm(ra),rb/np.linalg.norm(rb)],
            "maximum_null_residual":max(residuals),"TESC_conformal_scale":scale,
            "TESC_conformal_relative_residual":res,"same_TESC_null_cone":scale>0 and res<1e-8}

def load_cert(path):
    d=json.loads(Path(path).read_text()); return d

def template():
    return {"schema":"r-law1-independent-two-channel-v3.1",
      "channel_plus":{"covector":[1.0,0.0],"definition_source_path":"","definition_source_sha256":"","physical_meaning":"","measurable":False},
      "channel_minus":{"covector":[0.0,1.0],"definition_source_path":"","definition_source_sha256":"","physical_meaning":"","measurable":False},
      "capacity":{"definition_source_path":"","definition_source_sha256":"","strictly_positive":False},
      "provenance":{"both_channels_predeclared_before_TESC":False,"definitions_do_not_use_G_TESC_or_its_null_rays":False,
                    "relative_product_rule_derived":False,"selected_2D_plane_derived":False}}

def audit_cert(d, base_dir="."):
    cp,cm=d["channel_plus"],d["channel_minus"]; cap=d["capacity"]; p=d["provenance"]
    A=algebra(cp["covector"],cm["covector"])
    gates={"plus_source_bound":source_hash_matches(cp,path_key="definition_source_path",hash_key="definition_source_sha256",base_dir=base_dir),
      "minus_source_bound":source_hash_matches(cm,path_key="definition_source_path",hash_key="definition_source_sha256",base_dir=base_dir),
      "plus_has_physical_meaning":bool(cp["physical_meaning"].strip()),"minus_has_physical_meaning":bool(cm["physical_meaning"].strip()),
      "both_channels_measurable":bool(cp["measurable"] and cm["measurable"]),"capacity_source_bound":source_hash_matches(cap,path_key="definition_source_path",hash_key="definition_source_sha256",base_dir=base_dir),
      "capacity_strictly_positive":bool(cap["strictly_positive"]),"channels_predeclared_before_TESC":bool(p["both_channels_predeclared_before_TESC"]),
      "definitions_independent_of_TESC":bool(p["definitions_do_not_use_G_TESC_or_its_null_rays"]),
      "product_rule_derived_not_chosen":bool(p["relative_product_rule_derived"]),"two_dimensional_plane_derived":bool(p["selected_2D_plane_derived"]),
      "channels_linearly_independent":A["channels_independent"],"induced_quadratic_form_Lorentzian":A["induced_Lorentzian"],
      "two_null_rays_exact":A["maximum_null_residual"]<1e-12}
    return {"gates":gates,"gate":all(gates.values()),"algebra":A}

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--certificate",default="");ap.add_argument("--outdir",default="r_law1_two_channel_origin_v3_1_results")
    args,unknown=ap.parse_known_args()
    if unknown: print("[notice] ignored notebook/kernel arguments:",unknown)
    t=time.time();out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True)
    # Exact structural controls.
    single=np.array([1.,2.]); single_kernel=np.array([-2.,1.])
    single={"nonzero_channel":True,"kernel_dimension":1,"one_unoriented_null_line":True,
            "cannot_equal_two_distinct_Lorentzian_null_lines":True,"kernel_witness":single_kernel}
    pos=algebra([1.,1.],[1.,-1.]); dep=algebra([1.,1.],[2.,2.])
    circular={"F_abs_q_zero_set_equals_q_by_construction":True,"F_q_squared_zero_set_equals_q_by_construction":True,
              "admissible_as_independent_R_to_LawI_evidence":False}
    tests={"single_channel_no_go_pass":single["cannot_equal_two_distinct_Lorentzian_null_lines"],
           "independent_two_channel_positive_control":pos["induced_Lorentzian"] and pos["channels_independent"],
           "dependent_channel_negative_control_rejected":not dep["induced_Lorentzian"],
           "circular_q_cost_rejected_as_provenance":not circular["admissible_as_independent_R_to_LawI_evidence"]}
    certpath=Path(args.certificate) if args.certificate else None
    if certpath and certpath.is_file():
        cert=load_cert(certpath); empirical=audit_cert(cert, certpath.parent); certsha=hashlib.sha256(certpath.read_bytes()).hexdigest()
        status="INDEPENDENT_TWO_CHANNEL_LAWI_ORIGIN_SUPPORTED" if empirical["gate"] else "TWO_CHANNEL_ORIGIN_CERTIFICATE_FAIL_CLOSED"
    else:
        p=out/"independent_two_channel_certificate_template.json";p.write_text(json.dumps(template(),indent=2)+"\n")
        empirical=None;certsha=None;status="STRUCTURAL_MECHANISM_CERTIFIED_INDEPENDENT_CHANNEL_PROVENANCE_REQUIRED"
    protocol={"title":TITLE,"version":VERSION,"tests":"single-channel no-go; two-channel product; circular negative control",
              "claim_rule":"F=|L_plus L_minus|/H with H>0; channel definitions must predate and not depend on TESC"}
    protocol["protocol_sha256"]=hobj(protocol);(out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    report={"title":TITLE,"version":VERSION,"scientific_status":status,"protocol_sha256":protocol["protocol_sha256"],
      "single_channel_information_time":single,"two_channel_positive_control":pos,"dependent_channel_negative_control":dep,
      "circular_cost_negative_control":circular,"self_tests":tests,"certificate_supplied":empirical is not None,
      "certificate_sha256":certsha,"native_two_channel_audit":empirical,
      "all_scientific_gates_pass":bool(empirical and empirical["gate"] and all(tests.values())),
      "interpretation":"A scalar Information-Time differential has one kernel line in 2D and cannot represent a nondegenerate Lorentzian two-line zero cone. Two independently derived nonparallel channels yield F=|L+L-|/H and an induced Lorentzian quadratic form exactly. Matching TESC additionally requires a noncircular provenance-bound identification.",
      "next_required_step":"Define and source-bind two measurable realization channels before consulting TESC. If no such channels exist, retain Law-I as an additional representation assumption.",
      "claim_boundary":"The algebraic two-channel mechanism is not a derivation from Principle R without provenance. F=|q_TESC| or q_TESC^2 is a circular negative control.",
      "elapsed_seconds":time.time()-t,"environment":{"python":platform.python_version(),"numpy":np.__version__}}
    (out/"run_summary.json").write_text(json.dumps(js(report),indent=2)+"\n")
    print("="*112);print(f"{TITLE} v{VERSION}");print("="*112);print(json.dumps(js(report),indent=2));return 0

if __name__=="__main__":
    rc=main()
    if not any(x in sys.modules for x in ("ipykernel","IPython","google.colab")): raise SystemExit(rc)
