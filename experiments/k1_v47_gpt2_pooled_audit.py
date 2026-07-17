"""
K1 V4.7 — read-only pooled GPT-2 audit

This is a secondary pooled analysis of two consecutive, non-overlapping runs:
  V4.4 screen: seeds 10103, 10301, 10501
  V4.6 holdout: seeds 10709, 10903, 11113, 11311, 11503, 11701, 11909

It performs no training. It preserves these historical facts:
  * V4.4's preregistered 3-seed GO decision failed.
  * V4.6's fixed-mask primary hypothesis failed.
The pooled analysis concerns only the common, prespecified full-layer controls.
"""

import json, math
from pathlib import Path
import numpy as np
import pandas as pd

V44 = Path("k1_throttle_v44_gpt2_screen_results")
V46 = Path("k1_v46_gpt2_mask_holdout_results")
OUT = Path("k1_v47_gpt2_pooled_audit_results")
OUT.mkdir(exist_ok=True, parents=True)

SEEDS44 = (10103, 10301, 10501)
SEEDS46 = (10709, 10903, 11113, 11311, 11503, 11701, 11909)
EXPECTED = SEEDS44 + SEEDS46
COMMON = ("residual", "lorentz_neg", "euclid_neg", "random_neg")
SPLITS = ("val", "test", "ood")
TCRIT95 = {2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447,
           7:2.365, 8:2.306, 9:2.262}


def require_file(p):
    if not p.is_file():
        raise FileNotFoundError(f"Missing required file: {p}")
    return p


def load_metrics(root, cohort, expected_seeds):
    p = require_file(root / "metrics.csv")
    d = pd.read_csv(p)
    needed = {"seed", "variant", "split", "loss", "accuracy", "ece"}
    if not needed.issubset(d.columns):
        raise ValueError(f"{p} lacks columns {sorted(needed-set(d.columns))}")
    d = d[d.variant.isin(COMMON) & d.split.isin(SPLITS)].copy()
    d["seed"] = d.seed.astype(int)
    got = tuple(sorted(d.seed.unique()))
    if got != tuple(sorted(expected_seeds)):
        raise ValueError(f"{cohort} seed mismatch: expected {expected_seeds}, got {got}")
    dup = d.duplicated(["seed", "variant", "split"], keep=False)
    if dup.any():
        raise ValueError(f"{cohort} contains duplicate seed/variant/split cells")
    expected_n = len(expected_seeds)*len(COMMON)*len(SPLITS)
    if len(d) != expected_n:
        raise ValueError(f"{cohort}: expected {expected_n} common cells, got {len(d)}")
    if not np.isfinite(d[["loss", "accuracy", "ece"]].to_numpy()).all():
        raise ValueError(f"{cohort} contains non-finite metrics")
    d["cohort"] = cohort
    return d


def load_cfg(root, seed):
    # torch is only used to read the small checkpoint metadata.
    import torch
    p = require_file(root / f"residual_seed{seed}.pt")
    q = torch.load(p, map_location="cpu")
    if "cfg" not in q:
        raise ValueError(f"Checkpoint has no cfg metadata: {p}")
    return q["cfg"]


def validate_protocol():
    keys = ("model_name", "seq_len", "train_blocks", "val_blocks",
            "test_blocks", "ood_blocks", "batch", "pretrain_epochs", "epochs",
            "lr", "weight_decay", "planes", "max_dt", "grad_clip", "patience",
            "ood_tolerance_nats")
    cfgs = []
    for root, seeds, cohort in ((V44, SEEDS44, "V4.4"), (V46, SEEDS46, "V4.6")):
        for seed in seeds:
            c = load_cfg(root, seed)
            cfgs.append((cohort, seed, c))
    ref = {k:cfgs[0][2].get(k) for k in keys}
    mismatches = []
    for cohort, seed, c in cfgs:
        for k in keys:
            if c.get(k) != ref[k]:
                mismatches.append({"cohort":cohort,"seed":seed,"field":k,
                                   "expected":ref[k],"actual":c.get(k)})
    if mismatches:
        raise ValueError("Protocol metadata mismatch:\n"+json.dumps(mismatches,indent=2))
    if ref["model_name"] != "gpt2":
        raise ValueError(f"Expected gpt2, got {ref['model_name']}")
    return ref


def paired_delta(d, a, b, split, cohort=None):
    q = d[d.split.eq(split)]
    if cohort is not None: q = q[q.cohort.eq(cohort)]
    pa = q[q.variant.eq(a)].set_index("seed").loss
    pb = q[q.variant.eq(b)].set_index("seed").loss
    x = (pa-pb).dropna().sort_index()
    n = len(x); vals=x.to_numpy(); mean=float(vals.mean())
    sd=float(vals.std(ddof=1)) if n>1 else 0.0
    tc=TCRIT95.get(n-1,1.96); half=tc*sd/math.sqrt(max(n,1))
    return {"a":a,"b":b,"split":split,"cohort":cohort or "pooled",
            "seeds":[int(i) for i in x.index],"values":vals.tolist(),"n":n,
            "mean":mean,"sd":sd,"paired_dz":mean/(sd+1e-12),
            "ci95_t":[mean-half,mean+half],"wins":int((vals<0).sum())}


def cohort_heterogeneity(q44, q46):
    # Fixed-effect two-cohort Q/I² diagnostic. Descriptive with only 2 cohorts.
    effects=np.array([q44["mean"],q46["mean"]],float)
    variances=np.array([q44["sd"]**2/q44["n"],q46["sd"]**2/q46["n"]],float)
    w=1/np.maximum(variances,1e-30); mu=float((w*effects).sum()/w.sum())
    Q=float((w*(effects-mu)**2).sum()); df=1
    I2=float(max(0.0,(Q-df)/Q)*100) if Q>0 else 0.0
    diff=float(effects[0]-effects[1]); se=float(np.sqrt(variances.sum()))
    return {"V44_mean":effects[0],"V46_mean":effects[1],
            "cohort_difference_V44_minus_V46":diff,
            "difference_normal95":[diff-1.96*se,diff+1.96*se],
            "Q":Q,"df":df,"I2_percent":I2}


if set(SEEDS44) & set(SEEDS46):
    raise ValueError("V4.4 and V4.6 seed sets overlap")
if tuple(sorted(EXPECTED)) != tuple(sorted(set(EXPECTED))) or len(EXPECTED)!=10:
    raise ValueError("Expected exactly ten unique pooled seeds")

protocol = validate_protocol()
d44 = load_metrics(V44,"V4.4-screen",SEEDS44)
d46 = load_metrics(V46,"V4.6-holdout",SEEDS46)
df = pd.concat([d44,d46],ignore_index=True)
df.to_csv(OUT/"pooled_metrics.csv",index=False)

comparisons={}; heterogeneity={}
for split in ("test","ood"):
    for control in ("residual","euclid_neg","random_neg"):
        key=f"{split}:lorentz_neg-{control}"
        pooled=paired_delta(df,"lorentz_neg",control,split)
        q44=paired_delta(df,"lorentz_neg",control,split,"V4.4-screen")
        q46=paired_delta(df,"lorentz_neg",control,split,"V4.6-holdout")
        comparisons[key]={"pooled":pooled,"V4.4":q44,"V4.6":q46}
        heterogeneity[key]=cohort_heterogeneity(q44,q46)

summary=df.groupby(["variant","split"])[["loss","accuracy","ece"]].agg(["mean","std"])
summary.to_csv(OUT/"pooled_summary.csv")
print("\nPOOLED PERFORMANCE MEAN ± SD\n",summary.to_string(float_format=lambda x:f"{x:.7f}"))
print("\nPAIRED COMPARISONS\n",json.dumps(comparisons,indent=2))
print("\nCOHORT HETEROGENEITY (descriptive; only two cohorts)\n",json.dumps(heterogeneity,indent=2))

def pooled_q(split,control):
    return comparisons[f"{split}:lorentz_neg-{control}"]["pooled"]
def superiority(control): return pooled_q("test",control)["ci95_t"][1] < 0
ood_tol=float(protocol["ood_tolerance_nats"])
ood_noninferior=all(pooled_q("ood",c)["ci95_t"][1] < ood_tol
                    for c in ("residual","euclid_neg","random_neg"))
wins_all=all(pooled_q("test",c)["wins"]>=8
             for c in ("residual","euclid_neg","random_neg"))
minimum_effect=pooled_q("test","residual")["mean"]<=-1e-4

audit={
 "analysis_type":"secondary pooled audit of consecutive experiments; no retraining",
 "historical_facts":{
   "V4.4_preregistered_screen_passed":False,
   "V4.6_fixed_mask_primary_hypothesis_passed":False
 },
 "protocol_metadata":protocol,
 "seed_sets":{"V4.4":list(SEEDS44),"V4.6":list(SEEDS46),"pooled":list(EXPECTED)},
 "G1_protocol_consistent_and_10_unique_seeds":True,
 "G2_lorentz_beats_residual_95CI":superiority("residual"),
 "G3_lorentz_beats_euclid_95CI":superiority("euclid_neg"),
 "G4_lorentz_beats_random_95CI":superiority("random_neg"),
 "G5_test_wins_at_least_8_of_10_vs_all":wins_all,
 "G6_OOD_noninferior_to_all":ood_noninferior,
 "G7_minimum_test_effect_1e-4_nats":minimum_effect,
 "paired_comparisons":comparisons,
 "cohort_heterogeneity":heterogeneity,
 "PASS_GPT2_10SEED_SECONDARY_POOLED_REPLICATION":bool(
     superiority("residual") and superiority("euclid_neg") and
     superiority("random_neg") and wins_all and ood_noninferior and minimum_effect),
 "claim_boundary":"Supports a small full-layer negative-Lorentz adapter effect in a secondary pooled GPT-2 analysis; does not validate OU K=1 attraction or physical token collapse."
}

print("\n"+"="*104+"\nK1 V4.7 GPT-2 TEN-SEED SECONDARY POOLED AUDIT\n"+"="*104)
print(json.dumps(audit,indent=2))
(OUT/"audit_v47.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
print("Saved to",OUT)
