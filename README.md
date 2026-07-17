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
k=1 quantum.TEX   Main LaTeX manuscript
```

At the moment, the repository contains the manuscript source only. The TeX file
references several PDF figures:

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
engineering principle:

- Hidden representations should be regularized or parameterized so their
  effective local metric remains Lorentzian (`det G < 0`).
- Token output can be modeled as a boundary event near a critical determinant
  surface, rather than only as unconstrained softmax sampling.
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

## Reported implementation results

The broader implementation notes associated with this project describe several
validation layers:

| Layer | Purpose | Reported result |
| --- | --- | --- |
| V1 | Geometric probe for next-token correctness | AUROC improved from `0.7491` to `0.8380` |
| V2 | Early-exit ablation | Negative/informative result: shallow classification did not isolate the geometry |
| V3 | Lorentz-constrained adapter training | Test loss `4.7280` vs. frozen baseline `4.7508` |
| V3.2 | Null-flow / critical algebra audit | `det A_c = 0` to machine precision |
| V3.3 | Initialization sensitivity | Near-critical starts reduce the need for explicit constraints |

Summary table from the implementation notes:

| Variant | Test loss | Test PPL | OOD loss | ECE |
| --- | ---: | ---: | ---: | ---: |
| Frozen base | 4.7508 | 115.68 | 4.7995 | 0.056 |
| AdamW/free | 4.7323 | 113.56 | 4.7932 | 0.053 |
| Lorentz | 4.7280 | 113.07 | 4.7919 | 0.052 |

These implementation scripts and result files are not yet present in this
repository. Until they are added, the empirical claims above should be read as
project notes rather than independently reproducible repository artifacts.

## Planned implementation layout

The associated implementation notes refer to the following target structure:

```text
k1_geometric_llm/
|-- k1_token_collapse_v1.py      # Geometric probe / AUROC validation
|-- k1_token_collapse_v2.py      # Matched ablation
|-- k1_token_collapse_v3.py      # Lorentz-constrained fine-tuning
|-- k1_token_collapse_v32.py     # Null-flow audit
|-- k1_token_collapse_v33.py     # Initialization sensitivity
|-- h14c3_optimizer/
|   |-- compact_klr.py
|   |-- retraction.py
|   `-- response_operator.py
|-- metrics/
`-- results/
```

Suggested runtime dependencies for that future implementation:

```bash
pip install torch transformers datasets numpy matplotlib
```

Example commands from the implementation notes:

```bash
python k1_token_collapse_v3.py \
  --seed 20260719 \
  --epochs 3 \
  --lr 0.0008 \
  --planes 8

python k1_token_collapse_v32.py \
  --seeds 10103,10301,10501 \
  --epochs 3 \
  --outdir k1_results
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
