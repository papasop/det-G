# Principle R to Lorentzian Law I

This repository is a reproducible computational companion to the
realizability programme introduced in
[*A Principle of Physical Realizability: Attainability and Local Zero Modes*](https://doi.org/10.5281/zenodo.21782618)
and used as upstream context for
[*K=1 Chronogeometrodynamics*](https://doi.org/10.5281/zenodo.21770348).

Its scope is deliberately narrower than either paper. It studies one
intermediate bridge:

\[
\text{Principle R}
+\mathcal A_{\mathrm{Law\,I}}
\Longrightarrow
\text{two-dimensional Lorentzian Law I}.
\]

## Principle R in this repository

Principle R states that a fundamental physical law must possess physically
attainable realizations. This repository uses only its local-zero-mode form:
at a realizable stationary configuration, at least one nonzero tangent
variation is required to have vanishing leading realization cost.

In this local reading:

- **attainable** means admitted by the selected physical process model;
- **realization cost** is the leading local cost assigned to a tangent
  process variation;
- **nonzero zero-cost mode** means a nontrivial tangent direction \(v\ne0\)
  whose leading cost vanishes.

This local-zero-mode statement is the Principle-R premise used here. The
repository does not derive the selected model, selected coordinates, or TESC
functional from Principle R alone.

## Law I: local zero-cost geometry

Law I is the claim that, on a selected real two-dimensional local process
plane, all and only the physical zero-cost tangent directions are represented
by the null set of a real, symmetric, nondegenerate quadratic form:

\[
q(v)=v^T Gv.
\]

Equivalently, inside the selected tangent plane,

\[
v\text{ is a physical zero-cost tangent direction}
\quad\Longleftrightarrow\quad
v^TGv=0.
\]

The double implication is essential: Law I is not merely the existence of some
vector satisfying \(v^TGv=0\). It says the complete local zero-cost tangent
structure is represented by the null geometry of \(G\).

## Auxiliary assumptions

The conditional theorem uses the following assumptions
\(\mathcal A_{\mathrm{Law\,I}}\):

| Assumption | Role | Current source |
|---|---|---|
| selected real two-dimensional process space | In two dimensions, a real nondegenerate indefinite form has signature `(1,1)` | selected reduction; not derived from R |
| real \(C^2\) cost near the base point | ensures the Hessian exists | regularity assumption |
| stationary base point | removes the first-order cost term | base-point choice |
| Hessian quadratic completeness | identifies the Hessian null set with the physical local zero-cost directions | key Law-I structural assumption |
| symmetry of \(G\) | Hessians of \(C^2\) real costs are symmetric | follows from the \(C^2\) setting |
| nondegeneracy of \(G\) | excludes \(\det G=0\) and collapsed null structure | Law-I structural assumption |

These assumptions are not cosmetic. They specify exactly what must be supplied
in addition to the local Principle-R zero-mode premise before the Lorentzian
conclusion follows.

## Conditional theorem

Under Principle R in the local-zero-mode form and
\(\mathcal A_{\mathrm{Law\,I}}\), there exists \(v\ne0\) such that

\[
v^TGv=0,\qquad G=G^T,\qquad \det G\ne0.
\]

A positive-definite or negative-definite real symmetric form has no nonzero
null vector. Nondegeneracy excludes the degenerate case. Therefore the two
real eigenvalues of \(G\) have opposite signs, and

\[
\det G<0,\qquad \operatorname{sig}(G)=(1,1).
\]

The null set of \(q\) is then the union of two distinct real rays:

\[
\operatorname{Null}(q)=\ell_+\cup\ell_-.
\]

This is an analytic two-dimensional linear-algebra theorem. It is not proved
by Python or by numerical experiments.

## TESC operational witness

The frozen operational witness is TESC: Task-Exposure Signed Cost. The process
coordinates are

\[
z=(\delta\Omega,\delta\Delta),
\]

where \(\delta\Omega\) is a Rabi-amplitude perturbation and
\(\delta\Delta\) is a detuning perturbation. The frozen cost is

\[
\mathcal C_{\mathrm{TESC}}(z)
=\mathcal E_{\mathrm{task}}(z)
-\lambda\,\widetilde{\mathcal E}_{\mathrm{exposure}}(z),
\qquad \lambda=1,
\]

where the exposure term is centred so that its value and linear tangent at the
base point are removed. The local metric candidate is

\[
G_{\mathrm{TESC}}=D^2\mathcal C_{\mathrm{TESC}}(0).
\]

For the frozen protocol, the public run recomputes

\[
G_{\mathrm{TESC}}\approx
\begin{pmatrix}
-1.47535118 & -0.04866380\\
-0.04866380 & 0.26618815
\end{pmatrix},
\qquad
\det G_{\mathrm{TESC}}\approx -0.39508917.
\]

Thus the TESC Hessian supplies a concrete operational witness for the
conditional Law-I structure in the declared two-dimensional model.

## Rays and finite branches

There are two related geometric objects:

- the local tangent rays \(\ell_\pm\), defined by the quadratic equation
  \(v^TGv=0\);
- the finite nonlinear zero branches \(\Gamma_\pm\), defined by the full
  equation \(\mathcal C_{\mathrm{TESC}}(z)=0\) inside the frozen bounded
  domain.

The intended relation is

\[
T_0\Gamma_\pm=\ell_\pm.
\]

In words: the finite zero-cost branches may curve at finite scale, but at the
base point their tangent directions agree with the two Hessian null rays. The
code checks this bounded-domain relationship numerically; it does not prove a
global analytic continuation theorem.

## What mathematics proves and what the code checks

| Claim or object | Source in this repository |
|---|---|
| \(R+\mathcal A_{\mathrm{Law\,I}}\Rightarrow\det G<0\) | analytic linear-algebra proof |
| \(G_{\mathrm{TESC}}\), \(\det G\), signature and \(\ell_\pm\) | numerical recomputation from the frozen TESC protocol |
| finite nonlinear branches \(\Gamma_\pm\) | bounded numerical search in `r_to_law1.finite_zero_set` |
| no extra branch in the frozen rectangle | bounded numerical search, not a global theorem |
| `GL(2)` covariance, unit rescaling and \(\lambda\) sensitivity | numerical audit in `r_to_law1.covariance` |
| native Principle-R selection of TESC | not proved |

The public command

```bash
python run_r_to_law1.py
```

recomputes the TESC Hessian, null rays, finite zero branches, covariance
checks, unit rescalings, \(\lambda\)-sensitivity scan and native-selection
gate. It then emits the unified report under `reference_results/v0.1.0/`.

## Current status

| Layer | Current status | Meaning |
|---|---|---|
| Conditional mathematical theorem | Proved | \(R+\mathcal A_{\mathrm{Law\,I}}\Rightarrow\det G<0\) in the selected real two-dimensional setting |
| TESC operational witness | Passed | The frozen model gives `(1,1)` signature, two null rays and finite two-branch zero structure in the declared domain |
| Native Principle R selection of TESC | Not proved | The repository has not derived task-minus-exposure, `lambda=1`, two-dimensionality or global completeness from Principle R alone |

## Scientific context and companion publications

The conceptual and dynamical context is developed in two separate works:

1. **Foundational source of Principle R**

   Y.Y.N. Li,  
   *A Principle of Physical Realizability: Attainability and Local Zero Modes*,  
   Zenodo, 2026.  
   <https://doi.org/10.5281/zenodo.21782618>

   This work supplies the upstream realizability principle: physical
   admissibility requires attainability and permits nontrivial local
   zero-realization-cost modes.

2. **Downstream K=1 dynamical framework**

   Y.Y.N. Li,  
   *K=1 Chronogeometrodynamics*,  
   Zenodo, 2026.  
   <https://doi.org/10.5281/zenodo.21770348>

   This work develops the subsequent K=1 critical dynamics, including rank
   reduction and G-null-flow structures.

The present repository does not computationally validate either paper in full.
It sits between them:

```text
Principle R paper
|
|  Principle R:
|  attainable nonzero zero-cost mode
v
det-G repository
|
|  R + 2D/C^2/completeness/symmetry/nondegeneracy
|  => detG < 0 => signature (1,1) => two null rays
v
K=1 Chronogeometrodynamics
|
|  critical damping, rank reduction, G-null flow
v
Law II / Law III / later quantum extensions
```

## Evidence hierarchy

| Layer | Source | Status in this repository |
|---|---|---|
| Principle R and attainability | *A Principle of Physical Realizability* | theoretical premise; not proved by code |
| Conditional Lorentzian Law I | this repository | exact conditional theorem plus operational TESC witness |
| Native selection of TESC from R | open | not yet derived |
| K=1 critical rank-one/G-null dynamics | *K=1 Chronogeometrodynamics* | downstream theory; not validated here |
| Law II/III, spacetime and wavefunction claims | outside scope | not established by this repository |

## How to read the central status

A correct reference run reports:

```text
conditional_theorem_premises_gate = true
theorem_proof_kind = analytic_linear_algebra
theorem_numerically_proved = false
operational_TESC_witness.gate = true
native_R_selection.gate = false
conditional_R_to_LawI_supported = true
unconditional_R_alone_to_LawI_proved = false
all_scientific_gates_pass = false
```

The final `false` does not mean that the conditional theorem or TESC witness
failed. It means that the stronger, unconditional native-selection claim
remains open.

## Quick start

```bash
python -m pip install -r requirements.txt
python run_r_to_law1.py
```

The historical one-file audits are preserved under `archive/`. They are not
the public interface for this release.

## Repository layout

```text
run_r_to_law1.py                 public entry point
src/r_to_law1/                   recomputable theorem and witness modules
protocols/frozen_tesc_protocol.json
certificates/native_r_selection.template.json
reference_results/v0.1.0/        reference report and generated evidence
archive/                         historical one-file audits
```

See `MATHEMATICAL_STATEMENT.md`, `VALIDATION_REPORT.md`, and
`CLAIM_BOUNDARY.md` before citing the result.

## Repository summary

> This repository supplies a rigorous conditional theorem and a reproducible
> operational witness for the route from Principle R to two-dimensional
> Lorentzian Law I. It does not establish that Principle R alone uniquely
> selects the witness or derives physical spacetime.

det-G connects the Principle-R foundation to the local Lorentzian Law-I
structure through a rigorous conditional theorem and a reproducible TESC
witness. It does not establish the complete K=1 dynamics.
