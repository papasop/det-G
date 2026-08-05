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

In the selected two-dimensional setting, the corresponding local chain is

\[
R+\mathcal A_{\mathrm{Law\,I}}
\Longrightarrow
\det G<0
\Longrightarrow
\operatorname{sig}(G)=(1,1)
\Longrightarrow
\ell_+\cup\ell_-.
\]

## Scientific context and companion publications

The conceptual and dynamical context is developed in two separate works:

1. **Foundational source of Principle R**

   Y.Y.N. Li,  
   *A Principle of Physical Realizability: Attainability and Local Zero Modes*,  
   Zenodo, 2026.  
   DOI: [10.5281/zenodo.21782618](https://doi.org/10.5281/zenodo.21782618)

   This work supplies the upstream realizability principle: physical
   admissibility requires attainability and motivates the local
   zero-realization-cost mode studied here.

2. **Downstream K=1 dynamical framework**

   Y.Y.N. Li,  
   *K=1 Chronogeometrodynamics*,  
   Zenodo, 2026.  
   DOI: [10.5281/zenodo.21770348](https://doi.org/10.5281/zenodo.21770348)

   This work develops the downstream K=1 critical dynamics, including rank
   reduction and G-null-flow structures.

The present repository does not computationally validate either paper in full.
Its scope is restricted to the intermediate conditional Law-I bridge and the
frozen signed-TESC zero-set representative.

## Mapping status

| Mapping | Type | Status |
|---|---|---|
| General Principle R -> local zero-mode premise | specialization | adopted, not derived |
| Local zero mode + \(\mathcal A_{\mathrm{Law\,I}}\) -> Law-I quadratic representation | conditional structural bridge | assumed/defined |
| Law-I zero-set representation -> Lorentzian signature | analytic theorem | proved |
| Lorentzian signed representative -> TESC zero-contrast witness | operational realization | numerically supported |
| Principle R -> unique TESC selection | native provenance | open |

## Evidence hierarchy

| Layer | Source | Status in this repository |
|---|---|---|
| Principle R and attainability | *A Principle of Physical Realizability* | Upstream theoretical premise; not proved by code |
| Conditional Lorentzian Law I | This repository | Analytic conditional theorem plus signed-TESC zero-set representative |
| Native selection of TESC from Principle R | Open | Not derived |
| K=1 critical rank-one/G-null dynamics | *K=1 Chronogeometrodynamics* | Downstream theory; not validated here |
| Law II/III, spacetime and wavefunction claims | Outside scope | Not established by this repository |

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
plane, the complete zero set of the nonnegative realization cost is
independently represented by the null set of \(q\), an independently declared
real, symmetric, nondegenerate signed quadratic form:

\[
q(v)=v^T Gv.
\]

Equivalently, inside the selected tangent plane,

\[
F(v)=0
\quad\Longleftrightarrow\quad
v^TGv=0.
\]

The double implication is essential: Law I is not merely the existence of some
vector satisfying \(v^TGv=0\). It says the complete local zero set of the
nonnegative cost is represented by the null geometry of \(G\). It does not say
that \(F=q\), and it does not say that \(D^2F=G\).

## Auxiliary assumptions

The conditional theorem uses the following assumptions
\(\mathcal A_{\mathrm{Law\,I}}\):

| Layer | Assumption | Role | Current source |
|---|---|---|---|
| Principle-R layer | nonnegative realization cost \(F\ge0\) is predeclared | prevents confusing the physical cost with the signed representative | open without source-bound certificate |
| Principle-R layer | R supplies an attained nonzero direction in \(Z(F)\) | gives the physical zero-mode premise | adopted local form |
| Law-I representation layer | selected real two-dimensional process space \(V\) | in two dimensions, a real nondegenerate indefinite form has signature `(1,1)` | selected reduction; not derived from R |
| Law-I representation layer | signed representative \(C\) is real \(C^2\) | permits the quadratic germ \(q(v)=v^TGv\) | frozen TESC protocol |
| Law-I representation layer | \(Z(F)\cap V=Z(q)\cap V\) | binds the physical zero set to the signed representative null set | key open certificate |
| Law-I representation layer | symmetry and nondegeneracy of \(G\) | gives a real noncollapsed quadratic null geometry | audited for signed TESC |

These assumptions are not cosmetic. They specify exactly what must be supplied
in addition to the local Principle-R zero-mode premise before the Lorentzian
conclusion follows.

## Conditional theorem

Under Principle R in the local-zero-mode form, plus the zero-set binding
\(Z(F)\cap V=Z(q)\cap V\) and the remaining assumptions
\(\mathcal A_{\mathrm{Law\,I}}\), there exists \(v\ne0\) such that

\[
F(v)=0,\qquad q(v)=v^TGv=0,\qquad G=G^T,\qquad \det G\ne0.
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

## TESC signed zero-set representative

The frozen operational object is TESC: Task-Exposure Signed Cost. It is a
signed contrast, not the nonnegative Principle-R realization cost \(F\). The
process coordinates are

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

Thus the TESC Hessian supplies a concrete operational witness for the signed
quadratic zero-set representative in the declared two-dimensional model. It
does not establish \(F=\mathcal C_{\mathrm{TESC}}\), and it does not establish
\(D^2F=G_{\mathrm{TESC}}\).

TESC supplies a signed quadratic zero-set candidate. Its physical zero-set
binding to the nonnegative Principle-R realization cost remains open.

## Rays and finite branches

There are two related geometric objects:

- the local tangent rays \(\ell_\pm\), defined by the quadratic equation
  \(v^TGv=0\);
- the finite signed-TESC zero-contrast branches \(\Gamma_\pm\), defined by the full
  equation \(\mathcal C_{\mathrm{TESC}}(z)=0\) inside the frozen bounded
  domain.

The intended relation is

\[
T_0\Gamma_\pm=\ell_\pm.
\]

In words: the finite signed zero-contrast branches may curve at finite scale,
but at the base point their tangent directions agree with the two Hessian null
rays. The code checks this bounded-domain relationship numerically; it does not
prove a global analytic continuation theorem or certify these branches as
physical zero-cost branches of \(F\).

## What mathematics proves and what the code checks

| Claim or object | Source in this repository |
|---|---|
| \(R+\mathcal A_{\mathrm{Law\,I}}\Rightarrow\det G<0\), where \(\mathcal A_{\mathrm{Law\,I}}\) includes \(Z(F)\cap V=Z(q)\cap V\) | analytic linear-algebra proof |
| \(G_{\mathrm{TESC}}\), \(\det G\), signature and \(\ell_\pm\) | numerical recomputation from the frozen signed-TESC protocol |
| finite nonlinear signed zero-contrast branches \(\Gamma_\pm\) | bounded numerical search in `r_to_law1.finite_zero_set` |
| no extra signed zero-contrast branch in the frozen rectangle | bounded numerical search, not a global theorem |
| `GL(2)` covariance, unit rescaling and \(\lambda\) sensitivity | numerical audit in `r_to_law1.covariance` |
| physical binding \(Z(F)\cap V=Z(q)\cap V\) | not proved |
| native Principle-R selection of TESC | not proved |

The public command

```bash
python run_r_to_law1.py
```

recomputes the TESC Hessian, null rays, finite zero-contrast branches, covariance
checks, unit rescalings, \(\lambda\)-sensitivity scan and native-selection
gate. It then emits the unified report under `reference_results/v0.1.0/`.

## Current status

| Layer | Current status | Meaning |
|---|---|---|
| Conditional mathematical theorem | Proved | \(R+\mathcal A_{\mathrm{Law\,I}}\Rightarrow\det G<0\), conditional on zero-set binding |
| Signed TESC zero-set representative | Passed | The frozen signed contrast gives `(1,1)` signature, two null rays and finite two-branch zero-contrast structure in the declared domain |
| Physical zero-set binding | Open | \(Z(F)\cap V=Z(q)\cap V\) has not been certified |
| Native Principle R selection of TESC | Not proved | The repository has not derived task-minus-exposure, `lambda=1`, two-dimensionality or global completeness from Principle R alone |

## Theory-chain position

The repository sits between the cited works:

```text
Principle R paper
|
|  Principle R:
|  attainable nonzero zero-cost mode
v
det-G repository
|
|  R + 2D/signed C^2 representative/zero-set binding/symmetry/nondegeneracy
|  => detG < 0 => signature (1,1) => two null rays
v
K=1 Chronogeometrodynamics
|
|  critical damping, rank reduction, G-null flow
v
Law II / Law III / later quantum extensions
```

## How to read the central status

A correct reference run reports:

```text
analytic_theorem_logic_gate = true
conditional_theorem_premises_gate = false
theorem_proof_kind = analytic_linear_algebra
theorem_numerically_proved = false
signed_TESC_zero_set_representative.gate = true
physical_zero_set_binding_certificate_gate = false
native_R_selection.gate = false
conditional_R_to_LawI_supported = false
unconditional_R_alone_to_LawI_proved = false
all_scientific_gates_pass = false
```

The final `false` does not mean that the analytic theorem or signed-TESC
representative failed. It means that the physical zero-set binding and the
stronger native-selection claim remain open.

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
> signed-TESC zero-set representative for the route from Principle R to
> two-dimensional Lorentzian Law I. It does not establish the physical zero-set
> binding, that Principle R alone uniquely selects the representative, or that
> physical spacetime is derived.

det-G connects the Principle-R foundation to the local Lorentzian Law-I
structure through a rigorous conditional theorem and a reproducible signed
TESC representative. It does not establish the complete K=1 dynamics.
