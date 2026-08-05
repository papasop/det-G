# Changelog

## 0.2.0-preflight — 2026-08-05

- added the v3.1 two-channel origin audit;
- recorded the scalar Information-Time obstruction: one scalar channel gives
  only one kernel line on a two-dimensional plane;
- recorded the conditional two-channel product mechanism
  `F_cross=abs(L_plus(v)L_minus(v))/H(x)`, which gives two zero lines and
  `det(G)<0` when the channels are independent;
- kept native provenance for `L_plus`, `L_minus`, and the product law open.

## 0.1.0 — 2026-08-05

- separated the exact conditional theorem from numerical witness validation;
- formalized Principle R and the auxiliary Law-I assumptions;
- recorded the frozen TESC Hessian, local null rays, finite zero branches, and covariance diagnostics;
- added a native-selection certificate template;
- made all unsupported spacetime, Law II/III, and wavefunction claims explicit.
- reorganized the public interface around `python run_r_to_law1.py`, with
  historical single-file audits preserved under `archive/`.
