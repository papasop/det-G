"""K1 V5.8 — GeoFlow controller-specificity screen.

Standalone Colab script.  GeoFlow is fixed while four parameter-matched
token-local controllers compete: det-G amplitude, generic learned scalar,
entropy-only, and deterministic random-feature.  A residual (lambda=1) anchor
is also trained.  The low-rank intervention is

    h' = h + gate * lambda(h) * (B @ A @ h),

so the trainable factor map depends on A and B only through M=B@A.  This is
required before applying the quotient direction from Geometric-Flow.  The
controller parameters are updated by the same auxiliary AdamW in every arm and
A/B always use the same GeoFlow direction.  Every minibatch is normalized to
the same first-order product-displacement budget.  After checkpoint selection,
each adaptive throttle is monotonically quantile-calibrated per layer on the
validation set to one common schedule.  Test/OOD never determine calibration.

This is a three-seed screening experiment, not confirmatory evidence.  A GO
means run a preregistered ten-new-seed replication.  It does not establish
physical wavefunction collapse, Lorentzian spacetime, or universal optimizer
superiority.
"""

import sys, subprocess, importlib.util, json, math, random, warnings
from dataclasses import dataclass, asdict
from pathlib import Path

GEOFLOW_COMMIT = "17f87315c5e442e548721e82034111071f9f4f8f"

def install():
    req = {
        "transformers": "transformers>=4.44,<5",
        "datasets": "datasets>=2.20,<4",
        "pandas": "pandas>=2",
        "matplotlib": "matplotlib>=3.7",
        "seaborn": "seaborn>=0.13",
    }
    missing = [spec for mod, spec in req.items() if importlib.util.find_spec(mod) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    # Pin the exact audited upstream implementation.  Do not silently use an
    # arbitrary preinstalled GeoFlow version.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        f"git+https://github.com/papasop/Geometric-Flow.git@{GEOFLOW_COMMIT}",
    ])

install()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from geometric_flow import inverse_gram_direction

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class CFG:
    seed: int = 20260719
    seeds: tuple = (50707, 50909, 51113)  # untouched screening seeds
    model_name: str = "facebook/opt-125m"
    seq_len: int = 96
    train_blocks: int = 500
    val_blocks: int = 160
    test_blocks: int = 500
    ood_blocks: int = 500
    batch: int = 4
    epochs: int = 3
    lr_aux: float = 8e-4
    weight_decay: float = 1e-4
    rank: int = 16
    controller_hidden: int = 8
    product_budget: float = 2.0e-4
    product_budget_eps: float = 1e-12
    max_factor_scale: float = 1e4
    grad_clip_aux: float = 1.0
    patience: int = 2
    sigma_max: float = 2.0
    correction_cap: float = 0.10
    ood_tolerance_nats: float = 0.02
    budget_match_rtol: float = 0.05
    calibration_quantiles: tuple = (0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
    calibration_targets: tuple = (0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.40)
    calibration_tolerance: float = 0.05
    outdir: str = "k1_v58_geoflow_controller_specificity_screen_results"


cfg = CFG()
Path(cfg.outdir).mkdir(parents=True, exist_ok=True)
random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)
print(json.dumps({**asdict(cfg), "geoflow_commit": GEOFLOW_COMMIT}, indent=2))


# ---------------------------------------------------------------------------
# Data: explicit HF namespaces and chunked tokenization.
# ---------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(cfg.model_name)
tok.pad_token = tok.eos_token

def dataset_texts(repo, config, split):
    ds = load_dataset(repo, config, split=split, trust_remote_code=False)
    key = "text" if "text" in ds.column_names else "sentence"
    return [str(x) for x in ds[key] if str(x).strip()]

def make_blocks(texts, count):
    out, carry = [], []
    for text in texts:
        carry += tok(text + "\n", add_special_tokens=False, truncation=False)["input_ids"]
        while len(carry) >= cfg.seq_len + 1 and len(out) < count:
            out.append(torch.tensor(carry[:cfg.seq_len + 1], dtype=torch.long))
            carry = carry[cfg.seq_len:]
        if len(out) >= count: break
    if len(out) < count: print("warning: blocks", len(out), "requested", count)
    return out

blocks = {
    "train": make_blocks(dataset_texts("Salesforce/wikitext", "wikitext-2-raw-v1", "train"), cfg.train_blocks),
    "val": make_blocks(dataset_texts("Salesforce/wikitext", "wikitext-2-raw-v1", "validation"), cfg.val_blocks),
    "test": make_blocks(dataset_texts("Salesforce/wikitext", "wikitext-2-raw-v1", "test"), cfg.test_blocks),
    "ood": make_blocks(dataset_texts("fancyzhx/ag_news", None, "test"), cfg.ood_blocks),
}
print({k: len(v) for k, v in blocks.items()})

class BlockDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i][:-1], self.rows[i][1:]

def loader(split, shuffle=False, seed=0):
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(BlockDataset(blocks[split]), batch_size=cfg.batch,
                      shuffle=shuffle, generator=gen)


# ---------------------------------------------------------------------------
# Frozen OPT backbone.  Every adapter exposes A=(r,d), B=(d,r).
# Controller features depend only on h, never on the A/B gauge coordinates.
# ---------------------------------------------------------------------------
base = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
for p in base.parameters(): p.requires_grad_(False)
base.eval()
if base.config.model_type != "opt":
    raise ValueError(f"V5.8 is frozen to OPT, got {base.config.model_type}")
decoder = base.model.decoder
layers = decoder.layers
D = int(base.config.hidden_size)
L = int(base.config.num_hidden_layers)

class GaugeCompatibleAdapter(nn.Module):
    def __init__(self, controller_kind, layer_index):
        super().__init__()
        self.controller_kind = controller_kind
        self.layer_index = layer_index
        self.A = nn.Parameter(torch.empty(cfg.rank, D))
        self.B = nn.Parameter(torch.empty(D, cfg.rank))
        nn.init.normal_(self.A, std=1 / math.sqrt(D))
        nn.init.normal_(self.B, std=1e-4)

        # Same nominal auxiliary budget in every controller arm.
        self.controller = nn.Sequential(
            nn.Linear(6, cfg.controller_hidden), nn.Tanh(),
            nn.Linear(cfg.controller_hidden, 1),
        )
        nn.init.zeros_(self.controller[-1].weight)
        nn.init.zeros_(self.controller[-1].bias)
        self.log_a = nn.Parameter(torch.zeros(1))
        self.log_c = nn.Parameter(torch.zeros(1))
        self.log_alpha = nn.Parameter(torch.full((1,), -1.0))
        self.output_gate = nn.Parameter(torch.tensor(-3.0))
        # Fixed token chart: it is a buffer, so geometry does not gain trainable
        # directions unavailable to the generic scalar controller.
        g = torch.Generator().manual_seed(8000 + layer_index)
        q = torch.randn(2, D, generator=g)
        q = F.normalize(q, dim=-1)
        self.register_buffer("q", q)
        self.register_buffer("source_quantiles", torch.tensor(cfg.calibration_targets, dtype=torch.float32))
        self.register_buffer("target_quantiles", torch.tensor(cfg.calibration_targets, dtype=torch.float32))
        self.calibration_enabled = False
        self.last_lambda = None

    def token_features(self, h, context):
        hf = h.float()
        q0 = F.linear(hf, self.q[0:1]).squeeze(-1)
        q1 = F.linear(hf, self.q[1:2]).squeeze(-1)
        rms = hf.square().mean(-1).sqrt()
        depth = torch.full_like(rms, self.layer_index / max(1, L - 1))
        entropy = context["entropy"]
        margin = context["margin"]
        if self.controller_kind == "entropy":
            feats = torch.stack([entropy, margin, entropy.square(), margin.square(),
                                 entropy*margin, depth], -1)
            x, y = entropy, margin
        elif self.controller_kind == "random_feature":
            ids = context["input_ids"].float()
            feats = torch.stack([torch.sin(ids*0.013), torch.cos(ids*0.017),
                                 torch.sin(ids*0.031), torch.cos(ids*0.047),
                                 torch.sin(ids*0.071), depth], -1)
            x, y = feats[...,0], feats[...,1]
        else:
            feats = torch.stack([q0, q1, entropy, margin, rms, depth], -1)
            x, y = q0, q1
        return feats, x, y

    def quantile_map(self, value):
        src = self.source_quantiles.to(value)
        tgt = self.target_quantiles.to(value)
        idx = torch.bucketize(value.contiguous(), src).clamp(1, src.numel()-1)
        x0, x1 = src[idx-1], src[idx]
        y0, y1 = tgt[idx-1], tgt[idx]
        w = ((value-x0)/(x1-x0).clamp_min(1e-6)).clamp(0,1)
        return y0+w*(y1-y0)

    def throttle(self, h, context):
        feats, x, y = self.token_features(h, context)
        raw = self.controller(feats).squeeze(-1)
        if self.controller_kind == "residual":
            lam = torch.ones_like(raw)
            sigma = torch.full_like(raw, float("nan"))
        elif self.controller_kind in ("generic", "entropy", "random_feature"):
            # Activate the same three scalar degrees of freedom used by the
            # geometric arm, but without imposing its Lorentz generator.
            temp = F.softplus(self.log_a) + 0.1
            u = (temp * raw + self.log_c * torch.tanh(x)
                 + (self.log_alpha + 1.0) * torch.tanh(y))
            lam = 2.0 * torch.sigmoid(u)
            sigma = torch.full_like(raw, float("nan"))
        elif self.controller_kind == "geometry":
            # Lorentz-side amplitude proxy.  Sign crossing is not claimed:
            # V5.1--V5.6 support token amplitude, not physical collapse.
            sigma = cfg.sigma_max * torch.tanh(raw)
            a = F.softplus(self.log_a) + 0.1
            c = F.softplus(self.log_c) + 0.1
            alpha = F.softplus(self.log_alpha) + 1e-4
            d = alpha * torch.sqrt(sigma.abs().clamp_min(1e-8) / (a * c))
            g0 = -d * x - (alpha * sigma / a) * y
            g1 = -(alpha / c) * x - d * y
            magnitude = torch.sqrt(0.5 * (g0.square() + g1.square()) + 1e-8)
            # Bounded, token-local throttle; mean scale near one.
            centered = torch.log(magnitude + 1e-6)
            centered = centered - centered.detach().mean()
            lam = 2.0 * torch.sigmoid(centered)
        else:
            raise ValueError(self.controller_kind)
        self.last_lambda = lam.detach()
        if self.calibration_enabled and self.controller_kind != "residual":
            lam = self.quantile_map(lam)
        return lam, sigma

    def forward(self, h, context, audit=False):
        backbone_dtype = h.dtype
        hf = h.float()
        lam, sigma = self.throttle(hf, context)
        low = F.linear(hf, self.A)
        product = F.linear(low, self.B)
        corr = torch.sigmoid(self.output_gate) * lam[..., None] * product
        # Smooth cap prevents a single factor proposal from destabilizing the
        # frozen backbone while preserving direction.
        rms = corr.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
        corr = corr * (cfg.correction_cap / rms).clamp(max=1.0)
        out = h + corr.to(backbone_dtype)
        if not audit: return out
        return out, {
            "lambda_mean": float(lam.detach().mean()),
            "lambda_std": float(lam.detach().std()),
            "sigma_abs_mean": float(torch.nan_to_num(sigma.detach().abs()).mean()),
            "corr_rms": float(corr.detach().square().mean().sqrt()),
        }

class IntervenedLM(nn.Module):
    def __init__(self, controller_kind):
        super().__init__()
        self.controller_kind = controller_kind
        self.adapters = nn.ModuleList([
            GaugeCompatibleAdapter(controller_kind, i) for i in range(L)
        ])

    def forward(self, input_ids, labels=None, audit=False):
        rows, handles = [], []
        # Frozen, intervention-free predictive uncertainty.  It is computed
        # identically for every arm and detached before controller use.
        with torch.no_grad():
            pre_logits = base(input_ids=input_ids, use_cache=False, return_dict=True).logits.float()
            logp = F.log_softmax(pre_logits, -1); prob = logp.exp()
            entropy = -(prob*logp).sum(-1) / math.log(pre_logits.size(-1))
            top2 = prob.topk(2, dim=-1).values
            margin = top2[...,0] - top2[...,1]
        context = {"entropy": entropy.detach(), "margin": margin.detach(),
                   "input_ids": input_ids.detach()}
        del pre_logits, logp, prob, top2, entropy, margin
        def make_hook(li, adapter):
            def hook(_module, _inputs, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                if audit:
                    hn, q = adapter(h, context, True); q["layer"] = li; rows.append(q)
                else:
                    hn = adapter(h, context, False)
                if isinstance(output, tuple): return (hn,) + output[1:]
                if isinstance(output, list): return [hn] + output[1:]
                return hn
            return hook
        try:
            for li, (layer, adapter) in enumerate(zip(layers, self.adapters)):
                handles.append(layer.register_forward_hook(make_hook(li, adapter)))
            out = base(input_ids=input_ids, use_cache=False, return_dict=True)
        finally:
            for handle in handles: handle.remove()
        logits = out.logits.float()
        loss = None if labels is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return loss, logits, rows

def factor_parameters(model):
    return [p for ad in model.adapters for p in (ad.A, ad.B)]

def auxiliary_parameters(model):
    factor_ids = {id(p) for p in factor_parameters(model)}
    return [p for p in model.parameters() if p.requires_grad and id(p) not in factor_ids]

def trainable_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factor optimizers with a shared first-order ||d(BA)||_F budget.
# ---------------------------------------------------------------------------
def product_motion_sq(adapters, deltas):
    total = None
    for ad, (dA, dB) in zip(adapters, deltas):
        dM = dB @ ad.A.detach() + ad.B.detach() @ dA
        term = dM.float().square().sum()
        total = term if total is None else total + term
    return total if total is not None else torch.zeros((), device=device)

@torch.no_grad()
def apply_budgeted(adapters, deltas):
    predicted = torch.sqrt(product_motion_sq(adapters, deltas)).clamp_min(cfg.product_budget_eps)
    scale = min(cfg.max_factor_scale, cfg.product_budget / float(predicted))
    for ad, (dA, dB) in zip(adapters, deltas):
        ad.A.add_(dA, alpha=scale)
        ad.B.add_(dB, alpha=scale)
    realized_first_order = float(predicted) * scale
    return realized_first_order, scale

class BudgetedGeoFlow:
    def __init__(self, adapters): self.adapters = list(adapters); self.fallbacks = 0
    @torch.no_grad()
    def step(self):
        deltas = []
        for ad in self.adapters:
            if ad.A.grad is None or ad.B.grad is None:
                raise RuntimeError("missing factor gradient")
            direction = inverse_gram_direction(
                ad.A, ad.B, ad.A.grad, ad.B.grad, condition_limit=1e10)
            self.fallbacks += int(direction.diagnostics.fallback_count)
            deltas.append((direction.velocity_A, direction.velocity_B))
        return apply_budgeted(self.adapters, deltas)


# ---------------------------------------------------------------------------
# Evaluation and paired training.
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, split, audit=False):
    model.eval(); loss_sum = correct = n = 0; confs = []; oks = []; audit_rows = []
    for x, y in loader(split):
        x, y = x.to(device), y.to(device)
        loss, logits, rows = model(x, y, audit)
        nt = y.numel(); loss_sum += float(loss) * nt; n += nt
        prob = logits.softmax(-1); conf, pred = prob.max(-1); ok = pred.eq(y)
        correct += int(ok.sum()); confs.append(conf.cpu()); oks.append(ok.cpu())
        audit_rows.extend(rows)
    conf = torch.cat(confs).numpy().ravel(); ok = torch.cat(oks).numpy().ravel().astype(float)
    ece = 0.0
    edges = np.linspace(0, 1, 16)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < (hi if hi < 1 else hi + 1e-8))
        if mask.any(): ece += mask.mean() * abs(ok[mask].mean() - conf[mask].mean())
    mean_loss = loss_sum / n
    return {"loss": mean_loss, "ppl": math.exp(min(20, mean_loss)),
            "accuracy": correct/n, "ece": ece}, audit_rows

def initial_state(controller, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = IntervenedLM(controller).to(device)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del model
    return state

def train_arm(controller, seed, shared_state):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = IntervenedLM(controller).to(device); model.load_state_dict(shared_state)
    aux = auxiliary_parameters(model)
    aux_opt = torch.optim.AdamW(aux, lr=cfg.lr_aux, weight_decay=cfg.weight_decay)
    factor_opt = BudgetedGeoFlow(model.adapters)
    best = float("inf"); best_state = None; bad = 0; budget_rows = []
    train_loader = loader("train", True, seed)
    print(f"\ngeoflow x {controller:14s} trainable={trainable_count(model)}")
    for epoch in range(cfg.epochs):
        model.train(); epoch_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            model.zero_grad(set_to_none=True)
            loss, _, _ = model(x, y, False)
            if not torch.isfinite(loss): raise RuntimeError("non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(aux, cfg.grad_clip_aux)
            # Factor and auxiliary proposals are based on the same gradient.
            realized, scale = factor_opt.step(); aux_opt.step()
            budget_rows.append({"budget": realized, "scale": scale})
            epoch_losses.append(float(loss.detach()))
        val, _ = evaluate(model, "val")
        print(f"epoch {epoch+1}: train={np.mean(epoch_losses):.4f} val={val['loss']:.4f} "
              f"budget={np.mean([r['budget'] for r in budget_rows[-len(train_loader):]]):.3e}")
        if val["loss"] < best - 1e-4:
            best = val["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience: break
    if best_state is None: raise RuntimeError("no finite checkpoint")
    model.load_state_dict(best_state)
    cell = f"geoflow_{controller}"
    torch.save({"arm": cell, "seed": seed, "cfg": asdict(cfg),
                "geoflow_commit": GEOFLOW_COMMIT, "state": best_state},
               Path(cfg.outdir) / f"{cell}_seed{seed}.pt")
    return model, budget_rows, getattr(factor_opt, "fallbacks", 0)

def preflight():
    counts = {}
    for controller in ("residual", "geometry", "generic", "entropy", "random_feature"):
        m = IntervenedLM(controller).to(device); counts[controller] = trainable_count(m)
        x, y = next(iter(loader("val"))); x, y = x.to(device), y.to(device)
        loss, _, rows = m(x, y, True); loss.backward()
        if not torch.isfinite(loss) or len(rows) != L: raise RuntimeError("hook preflight failed")
        if m.adapters[0].A.grad is None or m.adapters[0].B.grad is None:
            raise RuntimeError("factor gradient missing")
        if controller != "residual":
            probes = [m.adapters[0].controller[0].weight,
                      m.adapters[0].log_a, m.adapters[0].log_c,
                      m.adapters[0].log_alpha]
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in probes):
                raise RuntimeError(f"controller gradient missing for {controller}")
        if any(len(layer._forward_hooks) for layer in layers): raise RuntimeError("hook leak")
        del m
    if len(set(counts.values())) != 1: raise RuntimeError(f"parameter mismatch {counts}")
    print("PREFLIGHT PASS", {"trainable": counts, "layers": L, "hidden": D})
    if torch.cuda.is_available(): torch.cuda.empty_cache()

preflight()

@torch.no_grad()
def fit_validation_quantiles(model):
    """Fit monotone per-layer throttle maps using validation only."""
    for ad in model.adapters: ad.calibration_enabled = False
    values = [[] for _ in range(L)]
    model.eval()
    for x, y in loader("val"):
        x, y = x.to(device), y.to(device)
        model(x, y, False)
        for li, ad in enumerate(model.adapters):
            values[li].append(ad.last_lambda.float().cpu().reshape(-1))
    rows = []
    qprob = torch.tensor(cfg.calibration_quantiles)
    target = torch.tensor(cfg.calibration_targets)
    for li, ad in enumerate(model.adapters):
        if ad.controller_kind == "residual": continue
        src = torch.quantile(torch.cat(values[li]), qprob).float()
        # Enforce strict knots for stable interpolation.
        for j in range(1, src.numel()):
            src[j] = torch.maximum(src[j], src[j-1] + 1e-6)
        ad.source_quantiles.copy_(src.to(ad.source_quantiles))
        ad.target_quantiles.copy_(target.to(ad.target_quantiles))
        ad.calibration_enabled = True
        rows.extend({"layer": li, "quantile": float(q), "source": float(s), "target": float(t)}
                    for q, s, t in zip(qprob, src, target))
    return rows

results, budgets, audit_rows = [], [], []
calibration_rows = []
for seed in cfg.seeds:
    print("\n" + "#"*96 + f"\nPAIRED SEED {seed}")
    for controller in ("residual", "geometry", "generic", "entropy", "random_feature"):
        shared = initial_state(controller, seed)
        model, br, fallbacks = train_arm(controller, seed, shared)
        cell = f"geoflow_{controller}"
        for row in br: budgets.append({"seed": seed, "cell": cell, **row})
        cr = fit_validation_quantiles(model)
        for row in cr: calibration_rows.append({"seed": seed, "cell": cell, **row})
        # Replace the pre-calibration checkpoint with the inference-ready state.
        calibrated_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.save({"arm":cell,"seed":seed,"cfg":asdict(cfg),
                    "geoflow_commit":GEOFLOW_COMMIT,"state":calibrated_state,
                    "calibration_enabled":controller!="residual"},
                   Path(cfg.outdir)/f"{cell}_seed{seed}.pt")
        for split in ("val", "test", "ood"):
            metrics, ar = evaluate(model, split, audit=True)
            results.append({"seed": seed, "optimizer": "geoflow",
                            "controller": controller, "cell": cell,
                            "split": split, "fallbacks": fallbacks, **metrics})
            for q in ar: audit_rows.append({"seed": seed, "cell": cell, "split": split, **q})
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()

df = pd.DataFrame(results); bdf = pd.DataFrame(budgets); adf = pd.DataFrame(audit_rows)
df.to_csv(Path(cfg.outdir)/"metrics.csv", index=False)
bdf.to_csv(Path(cfg.outdir)/"product_budgets.csv", index=False)
adf.to_csv(Path(cfg.outdir)/"controller_audit.csv", index=False)
pd.DataFrame(calibration_rows).to_csv(Path(cfg.outdir)/"validation_quantile_calibration.csv", index=False)
summary = df.groupby(["optimizer", "controller", "split"])[["loss", "accuracy", "ece"]].agg(["mean", "std"])
summary.to_csv(Path(cfg.outdir)/"summary.csv")
print("\nMEAN ± SD\n", summary.to_string(float_format=lambda x: f"{x:.6f}"))

TCRIT = {2:4.303, 3:3.182, 4:2.776, 5:2.571, 6:2.447, 7:2.365, 8:2.306, 9:2.262}
def paired_stat(values):
    x = np.asarray(values, float); n = len(x); mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    half = TCRIT.get(n-1, 1.96) * sd / math.sqrt(max(1, n))
    return {"values": x.tolist(), "mean": mean, "sd": sd,
            "ci95_t": [mean-half, mean+half], "negative_wins": int((x < 0).sum()), "n": n}

comparisons = {}
for split in ("test", "ood"):
    p = df[df.split == split].pivot(index="seed", columns="cell", values="loss")
    for control in ("generic", "entropy", "random_feature", "residual"):
        comparisons[f"{split}:geometry-{control}"] = paired_stat(
            p.geoflow_geometry - p[f"geoflow_{control}"])

budget_means = bdf.groupby("cell").budget.mean().to_dict()
budget_spread = max(budget_means.values()) / max(min(budget_means.values()), 1e-30) - 1
budget_matched = bool(budget_spread <= cfg.budget_match_rtol)

adaptive = adf[adf.cell.isin(["geoflow_geometry", "geoflow_generic",
                              "geoflow_entropy", "geoflow_random_feature"])]
dist = adaptive.groupby(["cell", "split"])[["lambda_mean", "lambda_std"]].mean().reset_index()
test_dist = dist[dist.split=="test"].set_index("cell")
mean_spread = float(test_dist.lambda_mean.max()-test_dist.lambda_mean.min())
std_spread = float(test_dist.lambda_std.max()-test_dist.lambda_std.min())
throttle_matched = bool(mean_spread <= cfg.calibration_tolerance and
                        std_spread <= cfg.calibration_tolerance)

primary_controls = ("generic", "entropy", "random_feature")
id_directional = all(comparisons[f"test:geometry-{c}"]["mean"] < 0 and
                     comparisons[f"test:geometry-{c}"]["negative_wins"] >= 2
                     for c in primary_controls)
ood_noninferior = all(comparisons[f"ood:geometry-{c}"]["mean"] < cfg.ood_tolerance_nats
                      for c in primary_controls)
beats_residual = (comparisons["test:geometry-residual"]["mean"] < 0 and
                  comparisons["test:geometry-residual"]["negative_wins"] >= 2)

# Directional three-seed screen only.  Confirmation must use ten new seeds and
# CI gates; this screen deliberately does not pretend n=3 is definitive.
go_screen = bool(
    budget_matched and throttle_matched and id_directional
    and ood_noninferior and beats_residual
)

audit = {
    "protocol": "OPT-125M; fixed GeoFlow; gauge-compatible lambda(h)BAh; 3 untouched screening seeds; validation-only per-layer quantile calibration",
    "geoflow_commit": GEOFLOW_COMMIT,
    "adaptive_arms": ["geometry", "generic", "entropy", "random_feature"],
    "residual_anchor": "residual",
    "comparisons": comparisons,
    "mean_product_budget_by_cell": budget_means,
    "calibrated_throttle_summary": dist.to_dict("records"),
    "calibrated_test_lambda_mean_spread": mean_spread,
    "calibrated_test_lambda_std_spread": std_spread,
    "G1_product_budget_matched_within_5pct": budget_matched,
    "G2_throttle_mean_std_matched_after_validation_calibration": throttle_matched,
    "G3_geometry_beats_generic_entropy_random_ID_directional_2of3": id_directional,
    "G4_geometry_beats_residual_ID_directional_2of3": beats_residual,
    "G5_OOD_noninferior_to_adaptive_controls": ood_noninferior,
    "GO_TO_10_NEW_SEEDS": go_screen,
    "claim_boundary": "A GO supports only a preregistered controller-specificity replication under fixed GeoFlow. It does not establish physical collapse, decoherence, or universal geometry superiority.",
}
print("\n" + "="*108 + "\nK1 V5.8 GEOFLOW CONTROLLER-SPECIFICITY SCREEN\n" + "="*108)
print(json.dumps(audit, indent=2))
with open(Path(cfg.outdir)/"audit_v58.json", "w") as f: json.dump(audit, f, indent=2)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.barplot(data=df[df.split=="test"], x="controller", y="loss", ax=axes[0])
axes[0].tick_params(axis="x", rotation=20); axes[0].set_title("Fixed-GeoFlow held-out loss")
delta_rows=[]
for c in primary_controls:
    for seed,value in zip(cfg.seeds,comparisons[f"test:geometry-{c}"]["values"]):
        delta_rows.append({"seed":seed,"control":c,"geometry_minus_control":value})
sns.barplot(data=pd.DataFrame(delta_rows),x="control",y="geometry_minus_control",ax=axes[1])
axes[1].axhline(0,color="black",lw=1); axes[1].set_title("Geometry − control (<0 favors geometry)")
plt.tight_layout(); plt.savefig(Path(cfg.outdir)/"v58_controller_specificity.png", dpi=180, bbox_inches="tight")
plt.show()
print("Saved to", cfg.outdir)
