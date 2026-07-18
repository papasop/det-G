# K=1 Geometric Deep Learning and Chronogeometrodynamics

This repository currently contains the LaTeX manuscript source for:

**K=1 Chronogeometrodynamics: Lorentzian Signature as a Geometric Precondition for the Schrodinger Representation**

It also serves as the theoretical anchor for a proposed K=1 geometric deep
learning program for language models: a pseudo-Riemannian framework in which
Transformer hidden states are treated as evolving near Lorentzian metric
structures rather than as purely Euclidean statistical features.

The central theorem-level claim is conditional: given a physically selected
non-degenerate local metric block `G`, the Ornstein-Uhlenbeck/Fokker-Planck
construction supports a Schrodinger-type wavefunction representation exactly
when the local metric block has Lorentzian signature, equivalently `det G < 0`.

## Repository contents

```text
k=1 quantum.TEX                                  Main LaTeX manuscript
experiments/k1_throttle_v43_negative_replication.py
                                                 Standalone V4.3 DistilGPT-2
                                                 negative-Lorentz replication
experiments/k1_throttle_v44_gpt2_screen.py       V4.4 GPT-2 three-seed screen
experiments/k1_v45_gpt2_failure_diagnostic.py    V4.5 read-only diagnosis
experiments/k1_v46_gpt2_mask_holdout.py          V4.6 mask/full-Lorentz holdout
experiments/k1_v47_gpt2_pooled_audit.py          V4.7 pooled GPT-2 audit
experiments/k1_v56_geometry_specificity_matched_controls.py
                                                 V5.6 geometry specificity
                                                 matched-controls screen
experiments/k1_v57_geoflow_detg_2x2_screen.py    V5.7 GeoFlow/det-G 2x2 screen
experiments/k1_v58_geoflow_controller_specificity_screen.py
                                                 V5.8 controller-specificity
                                                 screen
experiments/k1_v59_geometric_incremental_feature_screen.py
                                                 V5.9 incremental-feature
                                                 screen
results/audit_v43.json                           Preregistered ten-seed audit
results/audit_v47_summary.json                   Compact V4.7 evidence summary
results/raw_v5/                                  Raw V5.7-V5.9 Colab transcripts
docs/V43_RESULT_BOUNDARIES.md                    Interpretation boundary
docs/V47_EVIDENCE_AND_CLAIM_BOUNDARY.md          V4.7 claim boundary
docs/V5_EVIDENCE_BOUNDARY.md                     V5 engineering evidence boundary
docs/K1_LLM_EVIDENCE.md                          Research status through V4.7
```

The TeX file references several PDF figures:

- `fig1_signature_gate.pdf`
- `fig2_oneway_barrier.pdf`
- `fig3_activetime_clock.pdf`
- `fig4_mu2_diagnostic.pdf`

These figure files are not currently included in the repository, so a direct
PDF build will require adding those files or temporarily removing/commenting the
corresponding `\includegraphics` lines.

## Main ideas

- A `K=1` stability condition gives a critical damping parameter
  `d_c > 0` if and only if `det G < 0`.
- In the local OU reduction near the `K=1` surface, the Fokker-Planck equation
  can be mapped by a similarity transform to an imaginary-time
  Schrodinger-type structure.
- Since the OU construction requires real positive damping, the
  Schrodinger-type representation is supported only on the Lorentzian side.
- Treating `Delta = det G` as an effective dynamical variable gives an idealized
  one-way signature-boundary protection mechanism through Lorentzian-side noise
  scaling `D_L(Delta) proportional to sqrt(|Delta|)`.
- The active-time and measurement-channel sections are presented as downstream
  operational closures, not as evidence for physical metric signature switching.

## Geometric deep learning interpretation

The language-model interpretation takes the manuscript's signature gate as an
engineering hypothesis:

- Hidden representations should be regularized or parameterized so their
  effective local metric remains Lorentzian (`det G < 0`).
- Token output may be studied as a boundary event near a critical determinant
  surface, rather than only as unconstrained softmax sampling. This repository
  does not establish physical wave-function collapse in language models.
- Optimization can be viewed as flow along a pseudo-Riemannian structure,
  replacing flat Euclidean updates with geometry-aware dynamics.

In this reading, the model update has the schematic form:

```text
h_{l+1} = h_l + P_up (z_{l+1} - z_l)
z_{l+1} = z_l + dt (J_G - d_c I) grad V
```

where:

- `G` is the local effective metric of hidden states.
- `d_c` is the critical damping parameter, real and positive only on the
  Lorentzian side.
- `J_G = alpha G^-1 J` is the symplectic generator.
- `det G < 0` marks the Lorentzian regime in which the wavefunction
  representation is supported.
- `det G > 0` marks the Euclidean regime in which this representation is not
  supported by the K=1 OU-FP construction.

## K1 LLM evidence through V4.7

A pre-mix scale-matched, full-layer negative-Lorentz adapter produced a
small, reproducible cross-entropy improvement over matched residual,
negative-Euclidean, and negative-random controls in DistilGPT-2 and in a
secondary ten-seed pooled GPT-2 analysis. The GPT-2 pooled Test improvements
were significant at paired 95% intervals and showed 9/10 wins against each
control.

The effective direction was negative and is not OU attraction toward `K=1`.
These experiments do not validate physical wave-function collapse or a
Token-collapse transition. Absolute improvements were small and remain limited
to the tested models, datasets, and adapter protocol.

### Experiment history

| Experiment | Status | Meaning |
| --- | --- | --- |
| V4.3 DistilGPT-2 10 seeds | Passed | Negative-Lorentz engineering effect in this setting |
| V4.4 GPT-2 3-seed screen | Failed | Screening GO rule did not pass |
| V4.5 diagnosis | Exploratory | Suggested local seed/layer sensitivity |
| V4.6 fixed Mask | Failed | Fixed layer-mask hypothesis did not generalize |
| V4.6 full-Lorentz control | Positive held-out result | 7/7 Test wins over residual |
| V4.7 GPT-2 pooled 10 seeds | Passed, secondary | Pooled common-control replication |

V4.7 is a secondary pooled analysis of two consecutive GPT-2 experiments, not
a newly preregistered standalone ten-seed experiment. The V4.4 screen failure
and V4.6 fixed Mask failure are part of the evidence record and should remain
visible.

See `docs/K1_LLM_EVIDENCE.md` and
`docs/V47_EVIDENCE_AND_CLAIM_BOUNDARY.md` for the full evidence boundary.

## K1 LLM evidence through V5.9

V5.6--V5.9 archive engineered low-rank controller screens in frozen language
models without changing the manuscript's conditional theorem. The tested
gauge-compatible GeoFlow direction substantially improved the three-seed
OPT-125M screen relative to the matched-budget factor-Adam path, and
token-adaptive throttles improved over a constant residual anchor in the tested
protocol.

The geometry-specificity gates did not pass. Explicit geometric controllers
beat entropy-only, random-feature, and residual controls in V5.8, but did not
beat the generic learned scalar gate. In V5.9, an explicit Lorentz-derived
seventh feature supplied no measurable incremental benefit to a strong shared
MLP. V5.7, V5.8, and V5.9 all report `GO_TO_10_NEW_SEEDS=false`.

The raw V5.7--V5.9 Colab transcripts are stored under `results/raw_v5/`. No
compact V5.6 audit JSON is included or inferred from rounded transcript text.
See `docs/V5_EVIDENCE_BOUNDARY.md` for the V5 claim boundary.

### Reproducibility entry points

The scripts are standalone Colab/GPU experiments:

```bash
python experiments/k1_throttle_v43_negative_replication.py
python experiments/k1_throttle_v44_gpt2_screen.py
python experiments/k1_v45_gpt2_failure_diagnostic.py
python experiments/k1_v46_gpt2_mask_holdout.py
python experiments/k1_v47_gpt2_pooled_audit.py
python experiments/k1_v56_geometry_specificity_matched_controls.py
python experiments/k1_v57_geoflow_detg_2x2_screen.py
python experiments/k1_v58_geoflow_controller_specificity_screen.py
python experiments/k1_v59_geometric_incremental_feature_screen.py
```

V4.3 compares frozen DistilGPT-2 with parameter-matched adapter branches under
a fixed negative-sign protocol:

- Model: `distilgpt2`
- ID dataset: `Salesforce/wikitext`, `wikitext-2-raw-v1`
- OOD dataset: `fancyzhx/ag_news`, `test`
- Block counts: `train=500`, `val=160`, `test=500`, `ood=500`
- Sequence length: `96`
- Adapter rank: `2 * planes = 16`
- Ten paired seeds:
  `10103, 10301, 10501, 10709, 10903, 11113, 11311, 11503, 11701, 11909`
- Matched parameter budget across residual, negative-Euclidean,
  negative-random, and negative-Lorentz branches
- Pre-mix branch normalization by per-token RMS over planes/components
- No post-hoc sign selection

The preregistered audit is stored in `results/audit_v43.json`.
The compact V4.7 pooled GPT-2 summary is stored in
`results/audit_v47_summary.json`.

### Selected results

- V4.3: `PASS_NEGATIVE_LORENTZ_SPECIFIC = true`.
- V4.3 Test Lorentz-negative minus Euclid-negative:
  mean `-0.0003550461`, 95% CI `[-0.0004488782, -0.0002612139]`, wins `10/10`.
- V4.3 Test Lorentz-negative minus Random-negative:
  mean `-0.0004974693`, 95% CI `[-0.0005776457, -0.0004172929]`, wins `10/10`.
- V4.3 OOD Lorentz-negative minus residual:
  mean `-0.0000541615`, 95% CI `[-0.0000815838, -0.0000267391]`, wins `9/10`.
- V4.6 full-Lorentz held-out control: Test Lorentz minus residual mean
  `-0.000601889`, paired 95% CI `[-0.000800628, -0.000403150]`, wins `7/7`.
- V4.7 GPT-2 pooled Test Lorentz minus residual:
  mean `-0.000583545`, paired 95% CI `[-0.000814308, -0.000352782]`,
  wins `9/10`.
- V4.7 GPT-2 pooled Test Lorentz minus Euclid:
  mean `-0.000545595`, paired 95% CI `[-0.000762316, -0.000328873]`,
  wins `9/10`.
- V4.7 GPT-2 pooled Test Lorentz minus Random:
  mean `-0.000613526`, paired 95% CI `[-0.000836661, -0.000390391]`,
  wins `9/10`.

Suggested dependencies:

```bash
pip install torch transformers datasets numpy pandas matplotlib seaborn
```

## Building the manuscript

Install a LaTeX distribution such as TeX Live or MacTeX, then run:

```bash
pdflatex "k=1 quantum.TEX"
pdflatex "k=1 quantum.TEX"
```

Run `pdflatex` twice so cross-references are resolved.

If the referenced figure PDFs are absent, LaTeX will stop at the first missing
figure. To build a text-only draft, either add placeholder PDFs with the
expected names or comment out the four `\includegraphics` commands in the TeX
source.

## Claim status

The manuscript separates its claims into three layers:

1. **Theorem-level result:** the conditional signature gate for the
   Schrodinger-type representation within the stated `K=1` OU-FP construction.
2. **Effective-dynamics consequences:** one-way boundary protection and related
   equilibrium identities inside the effective model.
3. **Operational closures:** active-time reconstruction and the `mu = 2`
   measurement-channel diagnostic.

The paper should be evaluated primarily on the first layer.

## Citation

No formal citation metadata is included yet. If you use or discuss this work,
please cite the manuscript title and author listed in the TeX source.

## References

- Y.Y.N. Li, *K=1 Chronogeometrodynamics: Lorentzian Signature as a Geometric
  Precondition for the Schrodinger Representation*.
- S. Amari, "Natural gradient works efficiently in learning," Neural
  Computation, 1998.
- P.-A. Absil, R. Mahony, and R. Sepulchre, *Optimization Algorithms on Matrix
  Manifolds*, 2008.
- J. Anandan and Y. Aharonov, "Geometry of quantum evolution," Physical Review
  Letters, 1990.

## License

No license file is included yet. Add a `LICENSE` file before distributing or
reusing this work under a formal open-source license.
