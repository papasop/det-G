# Validation report

## Frozen signed-TESC Hessian

\[
G_{\rm TESC}=\begin{pmatrix}
-1.4753511828891064 & -0.04866380215462485\\
-0.04866380215462485 & 0.2661881493004614
\end{pmatrix}.
\]

- eigenvalues: `-1.476709934535062`, `0.26754690094641703`
- determinant: `-0.39508916658164234`
- quadratic discriminant: `1.5803566663265691`
- normalized null rays:
  - `(0.3658136531, 0.9306881171)`
  - `(0.4171750943, -0.9088261334)`
- maximum null-ray residual in the public run: approximately `2.22e-17`

## Finite signed zero-contrast audit

- frozen domain: `|x|<=0.08`, `|y|<=0.40`
- recovered finite zero-contrast branches: `2`
- maximum finite zero-contrast residual: approximately `9.90e-14`
- no additional branch found inside the frozen domain

This is bounded-domain numerical completeness for the signed representative,
not a global analytic theorem and not a physical zero-cost certificate.

## Covariance audit

- `GL(2)` Hessian covariance maximum relative residual: approximately `1.49e-7`
- finite signed-zero-set pullback identity maximum residual: approximately `7.79e-14`
- signature preserved under the declared extreme unit rescalings
- Lorentzian behavior persists throughout the frozen coefficient scan `lambda in [0.25,4]`; therefore it is not a `lambda=1` knife-edge, but this does not derive the coefficient.

The finite zero-set calculation here is a pullback identity check on already
found zero-contrast points. It is not an independent root search in transformed
coordinates.

## Status

`SIGNED_TESC_ZERO_SET_REPRESENTATIVE_SUPPORTED_PHYSICAL_ZERO_SET_BINDING_OPEN`
