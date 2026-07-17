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
results/audit_v43.json                           Preregistered ten-seed audit
docs/V43_RESULT_BOUNDARIES.md                    Interpretation boundary
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

## V4.3 reproducibility: negative-Lorentz DistilGPT-2 replication

This repository includes a standalone V4.3 experiment:

```bash
python experiments/k1_throttle_v43_negative_replication.py
```

The script is designed as a standalone Colab/GPU run. It compares frozen
DistilGPT-2 with parameter-matched adapter branches under a fixed
negative-sign protocol:

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

### Exact V4.3 result

- `PASS_NEGATIVE_LORENTZ_SPECIFIC = true`
- Pre-mix normalization passed.
- Lorentz-negative beat residual, Euclid-negative, and Random-negative at the
  preregistered paired 95% CI threshold.
- Test Lorentz-negative minus Euclid-negative:
  mean `-0.0003550461`, 95% CI `[-0.0004488782, -0.0002612139]`, wins `10/10`.
- Test Lorentz-negative minus Random-negative:
  mean `-0.0004974693`, 95% CI `[-0.0005776457, -0.0004172929]`, wins `10/10`.
- OOD Lorentz-negative minus residual:
  mean `-0.0000541615`, 95% CI `[-0.0000815838, -0.0000267391]`, wins `9/10`.

Allowed interpretation:

> In the tested frozen DistilGPT-2 small-data adapter setting, a pre-mix
> scale-matched negative-Lorentz branch produced a small, reproducible,
> geometry-specific cross-entropy improvement over matched residual,
> negative-Euclidean, and negative-random controls across ten paired seeds.

Required limitations:

- The useful branch used the negative direction and exhibited negative
  estimated K1 regression; this is not evidence for OU attraction to `K=1`.
- This does not validate physical wave-function collapse or a critical token
  commit transition.
- The absolute loss improvement is small.
- Evidence is currently limited to DistilGPT-2 with WikiText ID evaluation and
  AG News OOD evaluation in this experiment.
- GPT-2-scale and independently implemented replication remain future work.

See `docs/V43_RESULT_BOUNDARIES.md` for the full interpretation boundary.

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
