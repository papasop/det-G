# Candidate 3: Operational Lorentzian Signed-Cost Geometry

Version: `candidate3-law1-operational-v0.1`

## Frozen functional

The operational candidate is

\[
F_{\mathrm{signed}}(z)
=F_{\mathrm{endpoint\ infidelity}}(z)
-\bigl(E_{\mathrm{Rabi}}(z)-E_{\mathrm{Rabi}}(0)
-\nabla E_{\mathrm{Rabi}}(0)^Tz\bigr),
\qquad z=(\delta\Omega,\delta\Delta).
\]

The subtraction coefficient is fixed to one. The four-segment public two-level
model and all numerical domain parameters are frozen in the scripts and result
protocol hashes.

## Supported result

Within the declared no-account two-level model:

- the signed-cost Hessian is real, symmetric, nondegenerate and Lorentzian;
- its two local Hessian-null tangents continue to finite zero-cost branches;
- adaptive transverse enumeration finds exactly two branches throughout the
  frozen rectangle `|x| <= 0.08`, `|y| <= 0.40`;
- the two roots missing from the original `|y| <= 0.20` rectangle are recovered
  by expansion to `|y| <= 0.32`;
- no additional zero branch is found in the frozen maximum rectangle;
- maximum reported zero-level residual is approximately `9.84e-14`.

This supports an operational Law-I(d/e) candidate in the declared bounded
model. It is stronger than a Hessian-sign test because it audits the finite
zero level set.

## Open gates

- Law-I(f): no dependency-closed derivation of the task-minus-exposure
  functional from Principle R is supplied.
- Law II: no native PASQAL transport identified with `J_G H` is certified.
- Law III: rank-one roots found in an isotropic-shift proxy fail the `G`-null
  image and frozen critical-scale gates.
- No production compiler, Cloud EMU, QPU, Born rule, collapse, physical clock,
  or physical-wavefunction claim is made.

## Reproduction

Run in order:

```bash
python src/k1_pasqal_public_data_generator_v2_1.py
python src/k1_pasqal_law1_def_v2_3.py
python src/k1_pasqal_law1_adaptive_zero_set_v2_4.py
```

Notebook environments may run the same files with `%run`.

## Package layout

```text
src/                 frozen executable audits
reference_results/   reference JSON artifacts
docs/                claim and interpretation boundary
```

