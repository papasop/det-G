# Principle R to Lorentzian Law I

This repository is a reproducible computational companion for one limited
bridge in the realizability programme:

\[
\text{Principle R}
+\mathcal A_{\mathrm{Law\,I}}
\Longrightarrow
\text{two-dimensional Lorentzian Law I}.
\]

Equivalently, in the selected two-dimensional setting it studies

\[
R+\mathcal A_{\mathrm{Law\,I}}
\Longrightarrow
\det G<0
\Longrightarrow
\operatorname{sig}(G)=(1,1)
\Longrightarrow
\ell_+\cup\ell_-.
\]

It supplies a conditional analytic theorem and a reproducible signed-TESC
zero-set representative. It does not prove that Principle R alone uniquely
selects TESC, `lambda=1`, physical spacetime, or a wavefunction.

## Result at a glance

| Question | Status |
|---|---|
| Does the conditional 2D theorem force `det(G)<0`? | Yes, analytically |
| Does frozen TESC exhibit signature `(1,1)` and two null rays? | Yes, numerically |
| Is `Z(F)=Z(q)` physically certified? | No |
| Does Principle R uniquely select TESC or `lambda=1`? | No |
| Does this validate Law II/III, spacetime, or a wavefunction? | No |

## v0.2.0 final audit bundle

The v0.2.0 bundle adds three fail-closed route and falsification audits:

| Audit | Question | Default result |
|---|---|---|
| v3.0 bidirectional zero-set binding | Does an independently defined nonnegative \(F\) satisfy both \(F=0\Rightarrow q=0\) and \(q=0\Rightarrow F=0\)? | Calibration only; independent \(F\) data required |
| v3.1 two-channel origin | Does scalar Information-Time fail as a single channel, and does an independent two-channel product give `det(G)<0`? | Single-channel obstruction and conditional mechanism certified; native channel provenance open |
| v3.2 cross-protocol cone naturality | Do independently sourced protocols recover one common unordered zero-cone class under frozen maps? | Calibration only; independent protocol data required |

These audits freeze the next theory boundary. They are not additional
parameter scans, and they do not replace `python run_r_to_law1.py`.

## Single-channel obstruction and two-channel candidate

For the scalar Information-Time realization

\[
F_{\rm IT}(x,v)=\frac{|D\Phi_x(v)|}{H(x)},
\]

the local zero set on a two-dimensional process plane is the single linear
subspace \(\ker D\Phi_x\). It therefore cannot equal the union of two distinct
null lines of a nondegenerate Lorentzian quadratic form.

A complete two-line zero set can instead arise conditionally from two
independently defined nonparallel realization channels:

\[
F_\times(x,v)=\frac{|L_+(v)L_-(v)|}{H(x)}.
\]

Then

\[
Z(F_\times)=\ker L_+\cup\ker L_-,
\]

and the induced quadratic form has negative determinant. The repository has
certified this algebraic mechanism, but has not derived the two channels or
their product law from Principle R. Defining the channels from the already
computed TESC null rays would be circular.

## Evidence status

| Layer | Current status | Meaning |
|---|---|---|
| Conditional mathematical theorem | Proved | \(R+\mathcal A_{\mathrm{Law\,I}}\Rightarrow\det G<0\), conditional on zero-set binding |
| Signed TESC zero-set representative | Passed | The frozen signed contrast gives `(1,1)` signature, two null rays and finite two-branch zero-contrast structure in the declared domain |
| Scalar Information-Time origin | Obstructed | A single scalar channel supplies only one kernel line on a two-dimensional plane |
| Two-channel product mechanism | Conditional | Independent \(L_+,L_-\) channels would give two zero lines and `det(G)<0`, but native provenance is not supplied |
| Physical zero-set binding | Open | \(Z(F)\cap V=Z(q)\cap V\) has not been certified |
| Native Principle-R selection of TESC | Not proved | The repository has not derived task-minus-exposure, `lambda=1`, two-dimensionality or global completeness from Principle R alone |

| Mapping | Type | Status |
|---|---|---|
| General Principle R -> local zero-mode premise | specialization | adopted, not derived |
| Local zero mode + \(\mathcal A_{\mathrm{Law\,I}}\) -> Law-I quadratic representation | conditional structural bridge | assumed/defined |
| Law-I zero-set representation -> Lorentzian signature | analytic theorem | proved |
| Lorentzian signed representative -> TESC zero-contrast witness | operational realization | numerically supported |
| Principle R -> unique TESC selection | native provenance | open |

## Core conditional theorem

Principle R is used here only in its local-zero-mode form: an attained
nonzero direction exists in the zero set of a nonnegative realization cost
\(F\).

Law I is represented through a separate signed quadratic zero-set
representative

\[
q(v)=v^TGv,\qquad G=G^T,\qquad \det G\ne0.
\]

The key additional binding assumption is

\[
Z(F)\cap V = Z(q)\cap V.
\]

Under this binding, the nonzero Principle-R zero mode gives a nonzero null
vector of \(q\). In two real dimensions, a nondegenerate symmetric quadratic
form with a nonzero null vector cannot be positive definite, negative
definite, or degenerate. Therefore

\[
\det G<0,\qquad \operatorname{sig}(G)=(1,1),
\]

and the null set is two distinct real rays \(\ell_+\cup\ell_-\).

This is a two-dimensional linear-algebra theorem. It is not proved by Python
or by numerical experiments. The full statement is in
`MATHEMATICAL_STATEMENT.md`.

## Quick start

```bash
python -m pip install -r requirements.txt
python run_r_to_law1.py
```

To test a separate provenance certificate without editing the frozen protocol:

```bash
python run_r_to_law1.py --certificate certificates/native_r_selection.template.json
```

Useful options:

```bash
python run_r_to_law1.py --protocol protocols/frozen_tesc_protocol.json
python run_r_to_law1.py --outdir reference_results/v0.1.0
```

The v0.2.0-preflight route audit is separate from the public v0.1.0 entry:

```bash
python audits/r_law1_bidirectional_zero_set_audit_v3_0.py --outdir reference_results/v0.2.0/v3_0
python audits/r_law1_two_channel_origin_audit_v3_1.py --outdir reference_results/v0.2.0/v3_1
python audits/r_law1_cross_protocol_cone_audit_v3_2.py --outdir reference_results/v0.2.0/v3_2
```

## Expected run result

A successful reproduction may still report:

```text
analytic_theorem_logic_gate = true
signed_TESC_zero_set_representative.gate = true
physical_zero_set_binding_certificate_gate = false
physical_zero_set_binding_provenance.gate = false
native_unique_TESC_selection_gate = false
conditional_R_to_LawI_supported = false
unconditional_R_alone_to_LawI_proved = false
all_scientific_gates_pass = false
```

The final `false` values do not mean that the code failed. They mean that the
signed TESC witness reproduced correctly while physical provenance remains
uncertified.

The run writes:

| File | Contents |
|---|---|
| `reference_results/v0.1.0/run_summary.json` | central status, gates, Hessian, rays and metrics |
| `reference_results/v0.1.0/finite_zero_branches.json` | bounded finite signed zero-contrast branch records |
| `reference_results/v0.1.0/covariance_records.json` | `GL(2)`, unit-rescaling and sensitivity records |
| `reference_results/v0.2.0/v3_0/run_summary.json` | bidirectional physical zero-set binding calibration |
| `reference_results/v0.2.0/v3_1/run_summary.json` | scalar obstruction and conditional two-channel mechanism |
| `reference_results/v0.2.0/v3_2/run_summary.json` | cross-protocol cone naturality calibration |

## File navigation

| File | Purpose |
|---|---|
| `run_r_to_law1.py` | public entry point |
| `src/r_to_law1/theorem.py` | conditional theorem gate and null-ray construction |
| `src/r_to_law1/tesc.py` | frozen TESC Hessian recomputation |
| `src/r_to_law1/finite_zero_set.py` | finite signed zero-contrast branch tracing |
| `src/r_to_law1/covariance.py` | covariance, unit-rescaling and lambda sensitivity audits |
| `src/r_to_law1/provenance.py` | physical zero-set binding and native unique-TESC provenance gates |
| `audits/r_law1_bidirectional_zero_set_audit_v3_0.py` | prospective independent \(F\) bidirectional binding audit |
| `audits/r_law1_two_channel_origin_audit_v3_1.py` | v0.2.0-preflight single-channel obstruction and two-channel mechanism audit |
| `audits/r_law1_cross_protocol_cone_audit_v3_2.py` | cross-protocol zero-cone naturality audit |
| `MATHEMATICAL_STATEMENT.md` | detailed theorem statement and proof boundary |
| `VALIDATION_REPORT.md` | numerical result details |
| `CLAIM_BOUNDARY.md` | unsupported claims and scientific boundary |
| `docs/TWO_CHANNEL_ORIGIN.md` | v3.1 route decision note |
| `docs/V0_2_0_FINAL_AUDIT_BUNDLE.md` | v0.2.0 bundle summary and boundary |
| `certificates/native_r_selection.template.json` | template for source-bound provenance certificates |
| `archive/` | historical one-file audits, not the public interface |

## Citations and context

This repository is narrower than the two companion publications. It is an
intermediate Law-I bridge between the upstream Principle-R framework and the
downstream K=1 dynamical framework.

1. Y.Y.N. Li,
   *A Principle of Physical Realizability: Attainability and Local Zero Modes*,
   Zenodo, 2026.
   DOI: [10.5281/zenodo.21782618](https://doi.org/10.5281/zenodo.21782618)

2. Y.Y.N. Li,
   *K=1 Chronogeometrodynamics*,
   Zenodo, 2026.
   DOI: [10.5281/zenodo.21770348](https://doi.org/10.5281/zenodo.21770348)

If you use this repository, see `CITATION.cff`.

## Not supported by this repository

This repository does not establish:

- `Principle R => Law I` without auxiliary assumptions;
- physical certification of \(Z(F)\cap V=Z(q)\cap V\);
- unique derivation of TESC, the relative sign, or `lambda=1` from Principle R;
- Law II/III;
- physical spacetime or a spacetime metric;
- physical light rays;
- wavefunctions, Born rule, collapse, or physical time.

## Repository summary

> This repository supplies a rigorous conditional theorem and a reproducible
> signed-TESC zero-set representative for the route from Principle R to
> two-dimensional Lorentzian Law I. It does not establish the physical zero-set
> binding, that Principle R alone uniquely selects the representative, or that
> physical spacetime is derived.
