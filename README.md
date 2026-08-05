# Principle R to Lorentzian Law I

Version `0.1.0` is a reproducible evidence package for the conditional chain

\[
R+\mathcal A_{\mathrm{Law\,I}}
\Longrightarrow \det G<0
\Longrightarrow \operatorname{sig}(G)=(1,1)
\Longrightarrow \ell_+\cup\ell_-.
\]

Here Principle R requires an attainable nonzero zero-realization-cost direction. The auxiliary assumptions are: a selected real two-dimensional process space, a real `C2` cost at a stationary base point, a complete quadratic tangent representation, symmetry, and nondegeneracy.

## What is established

1. **Exact conditional theorem.** A real symmetric nondegenerate quadratic form in two dimensions that has a nonzero null vector is necessarily indefinite, hence has negative determinant, signature `(1,1)`, and two distinct real null rays.
2. **Operational TESC witness.** The frozen task-minus-centred-exposure Hessian has eigenvalues approximately `(-1.476710, 0.267547)` and determinant `-0.395089`.
3. **Finite zero branches.** In the frozen domain `|x|<=0.08`, `|y|<=0.40`, two finite nonlinear zero branches were recovered with maximum residual about `9.84e-14`; no additional branch was found in that domain.
4. **Representation diagnostics.** The signature and zero-set construction survive the declared `GL(2)` covariance and unit-rescaling audits.

## What is not established

Principle R alone has not been proved to select TESC, the coefficient `lambda=1`, a two-dimensional process space, or a globally complete two-branch zero set. The package makes no Law II/III, spacetime, light-cone, wavefunction, Born-rule, Cloud, or QPU claim.

## Quick start

```bash
python src/principle_r_to_law1_strengthened_audit_v2_7.py
```

For the historical component audits:

```bash
python src/k1_pasqal_law1_adaptive_zero_set_v2_4.py
python src/k1_pasqal_law1f_covariance_v2_5.py
```

See `MATHEMATICAL_STATEMENT.md`, `VALIDATION_REPORT.md`, and `CLAIM_BOUNDARY.md` before citing the result.

