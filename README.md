# Principle R to Lorentzian Law I

This repository is a reproducible computational companion to the
realizability programme introduced in
[*A Principle of Physical Realizability: Attainability and Local Zero Modes*](https://doi.org/10.5281/zenodo.21782618)
and used as upstream context for
[*K=1 Chronogeometrodynamics*](https://doi.org/10.5281/zenodo.21770348).

Its scope is deliberately narrower than either paper.

This repository studies one specific implication in the realizability
programme:

\[
\text{Principle R}
+\mathcal A_{\mathrm{Law\,I}}
\Longrightarrow
\mathrm{Law\,I}.
\]

Principle R requires at least one attainable, nonzero local process direction
with zero realization cost.

Law I states that, on a selected real two-dimensional process tangent space,
the complete local zero-cost structure is represented by a real, symmetric and
nondegenerate quadratic form

\[
q(v)=v^T Gv,
\]

whose nonzero null set is therefore the union of two distinct real rays.

The additional assumptions \(\mathcal A_{\mathrm{Law\,I}}\) are:

1. a selected real two-dimensional process space;
2. a real \(C^2\) cost near a stationary base point;
3. completeness of the Hessian quadratic form for local zero-cost directions;
4. symmetry and nondegeneracy of \(G\).

Under these assumptions, the existence of a nonzero null vector excludes
positive-definite, negative-definite and degenerate forms. Hence

\[
\det G<0,\qquad \operatorname{sig}(G)=(1,1).
\]

Version `0.1.0` packages the mathematical statement, reproducible operational
audits, reference results and claim-boundary certificate for this conditional
route.

## Scientific context and companion publications

This repository is the computational companion for one limited bridge in the
realizability programme:

\[
\text{Principle R}
+\mathcal A_{\mathrm{Law\,I}}
\Longrightarrow
\text{two-dimensional Lorentzian Law I}.
\]

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
Its scope is restricted to the intermediate Law-I bridge:

\[
R+\mathcal A_{\mathrm{Law\,I}}
\Longrightarrow
\det G<0,\quad
\operatorname{sig}(G)=(1,1),\quad
\operatorname{Null}(q)=\ell_+\cup\ell_-,
\]

together with the frozen TESC operational witness.

## Evidence hierarchy

| Layer | Source | Status in this repository |
|---|---|---|
| Principle R and attainability | *A Principle of Physical Realizability* | Theoretical premise; not proved by code |
| Conditional Lorentzian Law I | This repository | Exact conditional theorem plus operational TESC witness |
| Native selection of TESC from R | Open | Not yet derived |
| K=1 critical rank-one/G-null dynamics | *K=1 Chronogeometrodynamics* | Downstream theory; not validated here |
| Law II/III, spacetime and wavefunction claims | Outside scope | Not established by this repository |

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

## What mathematics proves and what the code checks

The implication

\[
R+\mathcal A_{\mathrm{Law\,I}}
\Longrightarrow
\det G<0
\]

is a two-dimensional linear-algebra theorem. It is not proved by numerical
experiments.

The code performs three separate tasks:

1. `v2.4` numerically follows the two finite nonlinear zero branches of the
   frozen TESC cost inside a declared bounded domain;
2. `v2.5` checks coordinate covariance, zero-set covariance, unit rescaling and
   coefficient sensitivity;
3. `v2.7` aggregates these results, constructs the local null rays and audits
   the logical claim boundary.

Therefore, `v2.7` is an evidence aggregator and logical-boundary audit. It is
not a computer proof of the conditional theorem.

## Current status

| Layer | Current status | Meaning |
|---|---|---|
| Conditional mathematical theorem | Proved | \(R+\mathcal A_{\mathrm{Law\,I}}\Rightarrow\det G<0\) in the selected real two-dimensional setting. |
| TESC operational witness | Passed | The frozen model gives `(1,1)` signature, two null rays and finite two-branch zero structure in the declared domain. |
| Native Principle R selection of TESC | Not proved | The repository has not derived task-minus-exposure, `lambda=1`, two-dimensionality or global completeness from Principle R alone. |

## What is established

1. **Exact conditional theorem.** A real symmetric nondegenerate quadratic form
   in two dimensions that has a nonzero null vector is necessarily indefinite,
   hence has negative determinant, signature `(1,1)`, and two distinct real
   null rays.
2. **Operational TESC witness.** The frozen task-minus-centred-exposure Hessian
   has eigenvalues approximately `(-1.476710, 0.267547)` and determinant
   `-0.395089`.
3. **Finite zero branches.** In the frozen domain `|x|<=0.08`,
   `|y|<=0.40`, two finite nonlinear zero branches were recovered with maximum
   residual about `9.84e-14`; no additional branch was found in that domain.
4. **Representation diagnostics.** The signature and zero-set construction
   survive the declared `GL(2)` covariance and unit-rescaling audits.

## What is not established

Principle R alone has not been proved to select TESC, the coefficient
`lambda=1`, a two-dimensional process space, or a globally complete two-branch
zero set. The package makes no Law II/III, spacetime, light-cone, wavefunction,
Born-rule, Cloud, or QPU claim.

## How to read the central status

A correct reference run reports:

```text
conditional_theorem_premises_gate = true
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
python src/principle_r_to_law1_strengthened_audit_v2_7.py
```

For the historical component audits:

```bash
python src/k1_pasqal_law1_adaptive_zero_set_v2_4.py
python src/k1_pasqal_law1f_covariance_v2_5.py
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
