#!/usr/bin/env python3
"""Principle-R -> Law-I bidirectional zero-set binding / falsification audit.

Input CSV columns: x,y,F,split.  F must be an independently defined,
nonnegative realization cost. split must be train or heldout.  The signed
quadratic representative q=x^T G x is evaluated by this script; it must not
be used to construct F.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from r_to_law1.protocol import protocol_sha256
from r_to_law1.tesc import derive_tesc_hessian, load_frozen_protocol

TITLE = "PRINCIPLE R -> LAW-I BIDIRECTIONAL ZERO-SET BINDING / FALSIFICATION AUDIT"
VERSION = "3.0.1"


def jsonable(v):
    if isinstance(v, dict): return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    return v


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def canonical_hash(obj) -> str:
    b = json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b).hexdigest()


def qvals(xy: np.ndarray, G: np.ndarray) -> np.ndarray:
    return np.einsum("ni,ij,nj->n", xy, G, xy)


def normalized_q(xy: np.ndarray, G: np.ndarray) -> np.ndarray:
    r2 = np.sum(xy * xy, axis=1)
    return np.abs(qvals(xy, G)) / np.maximum(np.linalg.norm(G) * r2, 1e-300)


def load_csv(path: Path):
    a = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    required = {"x", "y", "F", "split"}
    if not a.dtype.names or not required.issubset(a.dtype.names):
        raise ValueError("CSV requires columns: x,y,F,split")
    xy = np.c_[a["x"].astype(float), a["y"].astype(float)]
    F = a["F"].astype(float)
    split = np.asarray(a["split"], dtype=str)
    if len(xy) < 20: raise ValueError("at least 20 observations required")
    if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(F)):
        raise ValueError("x,y,F must be finite")
    if np.any(F < -1e-12): raise ValueError("F must be independently nonnegative")
    if not {"train", "heldout"}.issubset(set(split)):
        raise ValueError("split must contain both train and heldout")
    return xy, np.maximum(F, 0.0), split


def ray_angles(G: np.ndarray) -> np.ndarray:
    a, b, c = G[0, 0], G[0, 1], G[1, 1]
    # Solve a cos^2+t 2b cos sin+c sin^2=0 densely, then polish by clustering.
    th = np.linspace(0, math.pi, 200001, endpoint=False)
    u = np.c_[np.cos(th), np.sin(th)]
    z = np.abs(qvals(u, G))
    ids = np.argsort(z)
    chosen = []
    for i in ids:
        t = float(th[i])
        if all(abs(((t-s+math.pi/2) % math.pi)-math.pi/2) > 1e-3 for s in chosen):
            chosen.append(t)
        if len(chosen) == 2: break
    return np.sort(chosen)


def angle_to_rays(xy: np.ndarray, rays: np.ndarray) -> np.ndarray:
    th = np.mod(np.arctan2(xy[:, 1], xy[:, 0]), math.pi)
    ds = [np.abs(((th-r+math.pi/2) % math.pi)-math.pi/2) for r in rays]
    return np.min(np.vstack(ds), axis=0)


def confusion(Fzero, qzero):
    n = len(Fzero)
    both = int(np.sum(Fzero & qzero))
    f_only = int(np.sum(Fzero & ~qzero))
    q_only = int(np.sum(~Fzero & qzero))
    neither = int(np.sum(~Fzero & ~qzero))
    return {
        "n": n, "both_zero": both,
        "F_zero_q_nonzero": f_only,
        "q_zero_F_positive": q_only,
        "neither_zero": neither,
        "F_to_q_violation_rate": f_only / max(1, int(np.sum(Fzero))),
        "q_to_F_violation_rate": q_only / max(1, int(np.sum(qzero))),
    }


def fit_alternative_q(xy, Fzero):
    # Find the symmetric quadratic form whose normalized evaluations are
    # smallest on training F-zero points; smallest right singular vector.
    z = xy[Fzero]
    if len(z) < 3: return None
    M = np.c_[z[:, 0]**2, 2*z[:, 0]*z[:, 1], z[:, 1]**2]
    _, _, vh = np.linalg.svd(M, full_matrices=False)
    g = vh[-1]
    G = np.array([[g[0], g[1]], [g[1], g[2]]])
    n = np.linalg.norm(G)
    return None if n == 0 else G/n


def audit(xy, F, split, G, cfg):
    train = split == "train"
    held = split == "heldout"
    radius = np.linalg.norm(xy, axis=1)
    nq = normalized_q(xy, G)
    rays = ray_angles(G)

    # Threshold is frozen by CLI/protocol, never optimized on heldout outcomes.
    Fzero = F <= cfg["F_zero_tol"]
    qzero = nq <= cfg["q_zero_tol"]
    held_conf = confusion(Fzero[held], qzero[held])

    # Neighbourhood contraction: violations must not persist toward the origin.
    positive_r = radius[radius > 0]
    rmax = float(np.max(positive_r))
    cuts = np.array(cfg["radius_fractions"]) * rmax
    contraction = []
    for cut in cuts:
        m = held & (radius > 0) & (radius <= cut)
        if np.sum(m) < cfg["minimum_points_per_radius"]: continue
        c = confusion(Fzero[m], qzero[m])
        fz = m & Fzero
        angular = angle_to_rays(xy[fz], rays) if np.any(fz) else np.array([math.inf])
        contraction.append({
            "radius_cut": float(cut), "points": int(np.sum(m)),
            **c, "maximum_Fzero_angle_to_q_rays": float(np.max(angular)),
        })

    # Equal-class competitor: fit a generic symmetric q only on training F-zero points.
    Galt = fit_alternative_q(xy[train], Fzero[train])
    target_score = float(np.mean(nq[held & Fzero])) if np.any(held & Fzero) else math.inf
    if Galt is None:
        alt_score, advantage = None, None
    else:
        alt_score = float(np.mean(normalized_q(xy[held & Fzero], Galt))) if np.any(held & Fzero) else math.inf
        advantage = (alt_score-target_score)/max(alt_score, 1e-300)

    lorentz = float(np.linalg.det(G)) < 0
    coverage = int(np.sum(held & Fzero)) >= cfg["minimum_zero_points"] and int(np.sum(held & qzero)) >= cfg["minimum_zero_points"]
    f_to_q = held_conf["F_to_q_violation_rate"] <= cfg["maximum_violation_rate"]
    q_to_f = held_conf["q_to_F_violation_rate"] <= cfg["maximum_violation_rate"]
    enough_scales = len(contraction) >= 3
    contraction_gate = enough_scales and all(
        c["F_to_q_violation_rate"] <= cfg["maximum_violation_rate"]
        and c["q_to_F_violation_rate"] <= cfg["maximum_violation_rate"]
        for c in contraction[-3:]
    )
    competitor_gate = advantage is not None and advantage >= -cfg["alternative_advantage_tolerance"]
    gates = {
        "G_is_Lorentzian": lorentz,
        "heldout_zero_set_coverage": coverage,
        "heldout_F_zero_implies_q_zero": f_to_q,
        "heldout_q_zero_implies_F_zero": q_to_f,
        "neighbourhood_contraction": contraction_gate,
        "TESC_not_beaten_by_equal_class_alternative": competitor_gate,
    }
    falsified = coverage and (not f_to_q or not q_to_f or (enough_scales and not contraction_gate))
    return {
        "gates": gates, "gate": all(gates.values()),
        "current_TESC_binding_falsified": bool(falsified),
        "heldout_confusion": held_conf,
        "neighbourhood_contraction": contraction,
        "target_G": G, "target_null_ray_angles_radians": rays,
        "alternative_G": Galt,
        "target_heldout_Fzero_score": target_score,
        "alternative_heldout_Fzero_score": alt_score,
        "target_advantage_over_alternative": advantage,
    }


def self_tests(cfg, G: np.ndarray):
    rng = np.random.default_rng(20260805)
    rays = ray_angles(G)
    n = 800
    r = rng.uniform(.01, 1, n)
    choose = rng.integers(0, 2, n)
    theta = rays[choose] + rng.normal(0, 2e-5, n)
    xy0 = np.c_[r*np.cos(theta), r*np.sin(theta)]
    off = rng.uniform(0, math.pi, n)
    xy1 = np.c_[r*np.cos(off), r*np.sin(off)]
    xy = np.r_[xy0, xy1]
    Fgood = np.r_[np.zeros(n), np.ones(n)]
    Fbad = np.r_[np.ones(n), np.zeros(n)]
    split = np.array(["train" if i % 2 == 0 else "heldout" for i in range(2*n)])
    good = audit(xy, Fgood, split, G, cfg)
    bad = audit(xy, Fbad, split, G, cfg)
    return {
        "synthetic_positive_control_pass": bool(good["gate"]),
        "synthetic_wrong_zero_set_rejected": bool(not bad["gate"] and bad["current_TESC_binding_falsified"]),
    }


def write_template(path: Path, G: np.ndarray):
    rng = np.random.default_rng(20260805)
    rows = ["x,y,F,split"]
    for i in range(256):
        x, y = rng.uniform(-.1, .1, 2)
        rows.append(f"{x:.17g},{y:.17g},nan,{'train' if i%2==0 else 'heldout'}")
    path.write_text("\n".join(rows)+"\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="")
    p.add_argument("--outdir", default="r_law1_bidirectional_audit_v3_0_results")
    p.add_argument("--protocol", default="protocols/frozen_tesc_protocol.json")
    args, unknown = p.parse_known_args()
    if unknown: print("[notice] ignored notebook/kernel arguments:", unknown)
    t0 = time.time()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    frozen_protocol = load_frozen_protocol(args.protocol)
    frozen_protocol_sha = protocol_sha256(frozen_protocol)
    G = derive_tesc_hessian(frozen_protocol)
    cfg = frozen_protocol["prospective_audit_thresholds"]["v3_0_zero_set_binding"]
    audit_protocol = {"title": TITLE, "version": VERSION, "G_TESC": G,
                "criteria": cfg, "frozen_tesc_protocol_sha256": frozen_protocol_sha,
                "derivation_version": frozen_protocol["derivation_version"],
                "data_contract": "F is independent, nonnegative, prospective; q/TESC not used to construct F"}
    audit_protocol["protocol_sha256"] = canonical_hash(audit_protocol)
    (out/"frozen_protocol.json").write_text(json.dumps(jsonable(audit_protocol), indent=2)+"\n")
    controls = self_tests(cfg, G)
    data = Path(args.data) if args.data else None
    if data and data.is_file():
        xy, F, split = load_csv(data)
        empirical = audit(xy, F, split, G, cfg)
        data_hash = sha256_file(data)
        status = ("PHYSICAL_ZERO_SET_BINDING_SUPPORTED_IN_FROZEN_MODEL" if empirical["gate"]
                  else "CURRENT_TESC_BINDING_FALSIFIED" if empirical["current_TESC_binding_falsified"]
                  else "ZERO_SET_BINDING_INCONCLUSIVE_FAIL_CLOSED")
    else:
        empirical = None; data_hash = None
        template = out/"prospective_F_data_template.csv"; write_template(template, G)
        status = "PIPELINE_CALIBRATED_INDEPENDENT_PHYSICAL_F_DATA_REQUIRED"
    report = {
        "title": TITLE, "version": VERSION, "scientific_status": status,
        "protocol_sha256": audit_protocol["protocol_sha256"],
        "frozen_tesc_protocol_sha256": frozen_protocol_sha,
        "derivation_version": frozen_protocol["derivation_version"],
        "data_supplied": empirical is not None, "data_sha256": data_hash,
        "self_tests": controls, "empirical_audit": empirical,
        "all_scientific_gates_pass": bool(empirical and empirical["gate"] and all(controls.values())),
        "interpretation": "This audit tests Z(F)∩V=Z(q_TESC)∩V in both directions. Failure falsifies the current F/V/TESC binding, not Principle R, the analytic 2D theorem, or every possible Law-I representative.",
        "claim_boundary": "No independent F data means calibration only. A pass is local model-bound zero-set evidence, not spacetime, Law-II/III, wavefunction, Born-rule or hardware evidence.",
        "next_required_step": "Populate the template using an independently defined nonnegative realization cost F, freeze it before inspecting q/TESC agreement, then rerun with --data FILE.",
        "elapsed_seconds": time.time()-t0,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    (out/"run_summary.json").write_text(json.dumps(jsonable(report), indent=2)+"\n")
    print("="*112); print(f"{TITLE} v{VERSION}"); print("="*112)
    print(json.dumps(jsonable(report), indent=2))
    return 0


if __name__ == "__main__":
    rc = main()
    if not any(x in sys.modules for x in ("ipykernel", "IPython", "google.colab")):
        raise SystemExit(rc)
