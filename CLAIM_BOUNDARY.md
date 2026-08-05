# Claim boundary

## Supported wording

> Principle R together with the declared two-dimensional complete-zero-set representation assumptions forces the independently representing quadratic form to have Lorentzian `(1,1)` signature. TESC is a frozen signed zero-set representative with two local null rays and two finite zero-contrast branches in the audited domain.

## Relationship to the cited works

The cited Principle-R paper supplies the upstream theoretical motivation. The
cited K=1 paper supplies the downstream dynamical context. Inclusion of these
references does not mean that this repository validates either paper in full.

This repository establishes only:

\[
R+\mathcal A_{\mathrm{Law\,I}}
\Longrightarrow
\det G<0,\quad
\operatorname{sig}(G)=(1,1),
\]

together with the bounded signed-TESC zero-contrast representative. TESC
supplies a signed quadratic zero-set candidate. Its physical zero-set binding
to the nonnegative Principle-R realization cost remains open.

## Upstream zero-mode witness boundary

The repository now includes an end-to-end software interface for a path-level
realizability witness:

```text
path-level realization witness
-> ZeroModeCertificate
-> R-to-Law-I input
```

This interface can certify that a supplied upstream witness is source-bound,
nonnegative, attained, nonconstant, zero-cost and positive-measure according to
the frozen realizability protocol. The committed reference run does not include
independent physical path data, so this upstream production gate remains closed
by default.

Even a valid upstream witness does not certify the separate physical zero-set
binding:

\[
Z(F)\cap V = Z(q)\cap V.
\]

The evidence levels remain distinct:

| Level | Status boundary |
|---|---|
| A. upstream audit interface implemented | software interface only |
| B. source-bound upstream witness certified | requires independent path/cost data |
| C. cross-protocol physical zero-set binding certified | still open |
| D. conditional Law-I representation certified | requires C plus declared local structure |
| E. K=1 Law II/III | outside current scope |

## Unsupported wording

Do not claim that:

- Principle R alone proves Law I;
- the path-level zero-mode certificate proves Principle R as a universal physical law;
- a certified upstream witness substitutes for \(Z(F)\cap V=Z(q)\cap V\);
- Principle R uniquely selects TESC;
- `lambda=1` has been derived;
- scalar Information-Time by itself derives the complete Lorentzian two-ray null cone;
- the two-channel product mechanism has native \(L_+\), \(L_-\) provenance from Principle R;
- v3.0 calibration without independent \(F\) data proves physical zero-set binding;
- v3.2 calibration without independent protocol data proves cross-protocol naturality;
- finite-domain root searches prove global zero-set completeness;
- signed TESC zero-contrast branches have been certified as physical zero-cost branches;
- `F=q` or `D^2F=G_TESC` has been established for the nonnegative realization cost;
- the TESC cost metric is a physical spacetime metric;
- the null branches are physical light rays;
- four-dimensional `(1,3)` spacetime has been derived;
- Law II, Law III, a physical wavefunction, Born rule, collapse, or physical time follows;
- PASQAL hardware or Cloud observations were used.
