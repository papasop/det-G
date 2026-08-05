# Validation report

## Frozen TESC Hessian

\[
G_{\rm TESC}=\begin{pmatrix}
-1.4753511884402215 & -0.048663800766846066\\
-0.048663800766846066 & 0.2661881493004614
\end{pmatrix}.
\]

- eigenvalues: `-1.4767099400044166`, `0.26754690086465627`
- determinant: `-0.39508916792421417`
- quadratic discriminant: `1.5803566716968567`
- normalized null rays:
  - `(0.3658136532, 0.9306881170)`
  - `(0.4171750929, -0.9088261340)`
- maximum null-ray residual in the v2.7 run: approximately `2.31e-17`

## Finite zero-set audit

- frozen domain: `|x|<=0.08`, `|y|<=0.40`
- recovered finite branches: `2`
- maximum finite zero residual: approximately `9.84e-14`
- no additional branch found inside the frozen domain

This is bounded-domain numerical completeness, not a global analytic theorem.

## Covariance audit

- `GL(2)` Hessian covariance maximum relative residual: approximately `1.65e-7`
- finite zero-set covariance maximum residual: approximately `9.33e-14`
- signature preserved under the declared extreme unit rescalings
- Lorentzian behavior persists throughout the frozen coefficient scan `lambda in [0.25,4]`; therefore it is not a `lambda=1` knife-edge, but this does not derive the coefficient.

## Status

`CONDITIONAL_R_PLUS_STRUCTURE_IMPLIES_LAW_I_TESC_WITNESS_SUPPORTED_NATIVE_SELECTION_OPEN`

