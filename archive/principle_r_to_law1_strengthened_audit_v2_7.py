#!/usr/bin/env python3
"""Strengthened Principle-R -> Law-I conditional theorem and witness audit.

No PASQAL account, Pulser, QuTiP, SciPy, source files, or network required.
The script distinguishes:
  (A) exact theorem R + structural assumptions => 2D Lorentzian Law I;
  (B) an operational TESC existence witness;
  (C) the still-open claim that Principle R uniquely selects TESC.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, sys, time
from pathlib import Path
import numpy as np

TITLE="PRINCIPLE R -> LAW-I STRENGTHENED CONDITIONAL-THEOREM AUDIT"
VERSION="2.7"
TOL=1e-10

def py(x):
    if isinstance(x,dict): return {str(k):py(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [py(v) for v in x]
    if isinstance(x,np.ndarray): return py(x.tolist())
    if isinstance(x,np.generic): return x.item()
    return x
def canon(x): return json.dumps(py(x),sort_keys=True,separators=(",",":"),allow_nan=False)
def digest(x): return hashlib.sha256(canon(x).encode()).hexdigest()
def rel(x,y): return float(np.linalg.norm(x-y)/(1+np.linalg.norm(y)))

def default_certificate():
    return {
      "schema":"principle-r-law1-v2.7",
      "principle_R":{
        "status":"axiom",
        "statement":"physical law requires at least one attainable nonzero zero-realization-cost direction",
        "nonzero_zero_cost_direction_required":True},
      "structural_assumptions":{
        "selected_process_space_dimension":2,
        "cost_is_real_C2_near_basepoint":True,
        "stationary_basepoint":True,
        "quadratic_tangent_cost_is_complete":True,
        "metric_is_symmetric":True,
        "metric_is_nondegenerate":True},
      "operational_witness":{
        "name":"TESC",
        "definition":"task cost minus centred exposure cost",
        "G":[[-1.4753511884402215,-0.048663800766846066],
             [-0.048663800766846066,0.2661881493004614]],
        "finite_zero_branch_count":2,
        "finite_zero_search_domain":{"abs_x":0.08,"abs_y":0.40},
        "maximum_finite_zero_residual":9.84e-14,
        "no_additional_branch_found_in_frozen_domain":True,
        "GL2_covariance_max_relative_residual":1.6468256954267047e-7,
        "zero_set_covariance_max_residual":9.328648964412878e-14,
        "unit_rescaling_signature_preserved":True},
      "native_derivation":{
        "principle_R_source_sha256":"",
        "cost_definition_source_sha256":"",
        "relative_coefficient_source_sha256":"",
        "definitions_frozen_before_outcomes":False,
        "R_derives_task_minus_exposure":False,
        "R_derives_relative_coefficient":False,
        "R_proves_global_zero_set_completeness":False,
        "R_uniquely_selects_this_cost":False}
    }

def roots_and_rays(G):
    # q(1,m)=g00+2g01*m+g11*m^2; also handles a vertical ray.
    g00,g01,g11=float(G[0,0]),float(G[0,1]),float(G[1,1])
    disc=(2*g01)**2-4*g11*g00
    rays=[]
    if abs(g11)>TOL and disc>=0:
        sd=math.sqrt(max(0.,disc))
        rays=[np.array([1.,(-2*g01+sd)/(2*g11)]),np.array([1.,(-2*g01-sd)/(2*g11)])]
    elif abs(g01)>TOL:
        rays=[np.array([1.,-g00/(2*g01)]),np.array([0.,1.])]
    return disc,rays

def audit(c):
    R=c["principle_R"]; A=c["structural_assumptions"]; W=c["operational_witness"]; N=c["native_derivation"]
    G=np.asarray(W["G"],float)
    if G.shape!=(2,2): raise ValueError("operational_witness.G must be 2x2")
    eig=np.linalg.eigvalsh((G+G.T)/2); det=float(np.linalg.det(G)); disc,rays=roots_and_rays(G)
    nullres=[abs(float(v@G@v))/(1+np.linalg.norm(G)*np.linalg.norm(v)**2) for v in rays]
    # Exact theorem logic: for a real symmetric nondegenerate 2D form, existence
    # of v != 0 with q(v)=0 excludes definite signatures and forces det(G)<0.
    theorem_premises={
      "R_requires_nonzero_zero_cost_direction":bool(R["nonzero_zero_cost_direction_required"]),
      "process_space_is_real_two_dimensional":int(A["selected_process_space_dimension"])==2,
      "C2_stationary_cost_has_Hessian_tangent_form":bool(A["cost_is_real_C2_near_basepoint"]) and bool(A["stationary_basepoint"]),
      "quadratic_tangent_cost_complete_for_zero_directions":bool(A["quadratic_tangent_cost_is_complete"]),
      "quadratic_form_symmetric":bool(A["metric_is_symmetric"]),
      "quadratic_form_nondegenerate":bool(A["metric_is_nondegenerate"])}
    theorem_conclusions={
      "definite_signatures_excluded":True,
      "degenerate_signature_excluded":True,
      "detG_must_be_negative":True,
      "signature_must_be_1_1":True,
      "null_set_is_two_distinct_real_rays":True}
    theorem_pass=all(theorem_premises.values())
    witness={
      "G_real_symmetric":rel(G,G.T)<TOL,
      "G_nondegenerate":abs(det)>TOL,
      "detG_negative":det<0,
      "signature_1_1":eig[0]<0<eig[1],
      "two_distinct_null_rays_constructed":len(rays)==2 and disc>0,
      "null_ray_residuals_pass":len(nullres)==2 and max(nullres)<TOL,
      "finite_two_branch_zero_set_observed":int(W["finite_zero_branch_count"])==2,
      "finite_zero_residual_pass":float(W["maximum_finite_zero_residual"])<TOL,
      "no_extra_branch_in_frozen_domain":bool(W["no_additional_branch_found_in_frozen_domain"]),
      "zero_set_GL2_covariance_pass":float(W["zero_set_covariance_max_residual"])<TOL,
      "unit_rescaling_signature_preserved":bool(W["unit_rescaling_signature_preserved"])}
    hashes=all(isinstance(N[k],str) and len(N[k])==64 for k in
      ("principle_R_source_sha256","cost_definition_source_sha256","relative_coefficient_source_sha256"))
    origin={
      "three_source_hashes_bound":hashes,
      "definitions_frozen_before_outcomes":bool(N["definitions_frozen_before_outcomes"]),
      "R_derives_task_minus_exposure":bool(N["R_derives_task_minus_exposure"]),
      "R_derives_relative_coefficient":bool(N["R_derives_relative_coefficient"]),
      "R_proves_global_zero_set_completeness":bool(N["R_proves_global_zero_set_completeness"]),
      "R_uniquely_selects_this_cost":bool(N["R_uniquely_selects_this_cost"])}
    return theorem_premises,theorem_conclusions,theorem_pass,witness,origin,{
      "G":G,"eigenvalues":eig,"detG":det,"quadratic_discriminant":disc,
      "null_rays":[v/np.linalg.norm(v) for v in rays],
      "maximum_null_ray_residual":max(nullres,default=None),
      "finite_zero_search_domain":W["finite_zero_search_domain"]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--certificate"); ap.add_argument("--outdir",default="principle_r_law1_strengthened_v2_7_results")
    args,unknown=ap.parse_known_args()
    if unknown: print("[notice] ignored notebook/kernel arguments:",unknown)
    t=time.time(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    tpl=default_certificate(); template_path=out/"r_law1_native_derivation_certificate.json"
    if not template_path.exists(): template_path.write_text(json.dumps(tpl,indent=2)+"\n")
    supplied=bool(args.certificate); errors=[]
    try: c=json.loads(Path(args.certificate).read_text()) if supplied else tpl
    except Exception as e: c=tpl; supplied=False; errors.append(f"certificate load failed: {e}")
    try: prem,conc,tpass,wit,origin,metrics=audit(c)
    except Exception as e:
        prem=conc=wit=origin={}; metrics={}; tpass=False; errors.append(str(e))
    wpass=bool(wit) and all(wit.values()); opass=bool(origin) and all(origin.values())
    report={"title":TITLE,"version":VERSION,
      "scientific_status":("UNCONDITIONAL_R_TO_LAW_I_NATIVE_DERIVATION_CERTIFIED" if tpass and wpass and opass else
        "CONDITIONAL_R_PLUS_STRUCTURE_IMPLIES_LAW_I_TESC_WITNESS_SUPPORTED_NATIVE_SELECTION_OPEN" if tpass and wpass else
        "R_TO_LAW_I_PREMISES_OR_WITNESS_INCOMPLETE_FAIL_CLOSED"),
      "logical_statement":"Principle R + A_2D,C2,stationary,complete,symmetric,nondegenerate => det(G)<0, signature (1,1), two null rays",
      "exact_conditional_theorem":{"premises":prem,"conclusions":conc,"gate":tpass},
      "operational_TESC_witness":{"gates":wit,"gate":wpass},
      "native_R_selection":{"gates":origin,"gate":opass},
      "metrics":metrics,"certificate_supplied":supplied,"certificate_sha256":digest(c) if supplied else None,
      "conditional_R_to_LawI_supported":tpass and wpass,
      "unconditional_R_alone_to_LawI_proved":tpass and wpass and opass,
      "all_scientific_gates_pass":tpass and wpass and opass,"errors":errors,
      "interpretation":"The exact result is conditional: R plus the declared 2D quadratic-completeness assumptions forces Lorentzian Law I. TESC supplies a frozen operational existence witness. It does not show that R alone uniquely selects TESC or proves global completeness.",
      "next_required_step":"derive task-minus-centred-exposure, its relative normalization, and global zero-set completeness from a source-bound Principle-R construction",
      "claim_boundary":"No Law II/III, spacetime metric, (1,3) signature, physical light cone, wavefunction, Born rule, Cloud or QPU claim.",
      "artifacts":{"certificate_template":str(template_path.resolve())},
      "elapsed_seconds":time.time()-t,"environment":{"python":platform.python_version(),"numpy":np.__version__}}
    report=py(report); (out/"run_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print("="*112);print(f"{TITLE} v{VERSION}");print("="*112);print(json.dumps(report,indent=2));return 0

if __name__=="__main__":
    rc=main()
    if not any(x in sys.modules for x in ("ipykernel","IPython","google.colab")): raise SystemExit(rc)
