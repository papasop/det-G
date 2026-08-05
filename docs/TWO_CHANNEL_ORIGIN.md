# Two-Channel Origin Audit

This note records the v3.1 preflight result. It is a structural audit for the
next theory route, not a replacement for the public v0.1.0 TESC reproduction.

## Single-Channel Obstruction

For the scalar Information-Time realization

\[
F_{\rm IT}(x,v)=\frac{|D\Phi_x(v)|}{H(x)},
\qquad H(x)>0,
\]

the zero set on a real two-dimensional process plane is

\[
Z(F_{\rm IT})=\ker D\Phi_x.
\]

If \(D\Phi_x\ne0\), this is one linear subspace: a single line through the
origin. It cannot equal the union of two distinct null lines of a
nondegenerate Lorentzian quadratic form. Therefore scalar Information-Time
does not by itself supply the complete two-ray Law-I null cone.

## Conditional Two-Channel Mechanism

A complete two-line zero set can arise from two independently defined
nonparallel realization channels:

\[
F_\times(x,v)=\frac{|L_+(v)L_-(v)|}{H(x)}.
\]

Then

\[
Z(F_\times)=\ker L_+\cup\ker L_-.
\]

The signed quadratic representative

\[
q_\times(v)=L_+(v)L_-(v)
\]

has Hessian matrix

\[
G_\times=\frac12(L_+^TL_-+L_-^TL_+),
\]

and in two dimensions

\[
\det G_\times=-\frac14\det
\begin{pmatrix}
L_+\\
L_-
\end{pmatrix}^2<0
\]

whenever \(L_+\) and \(L_-\) are independent.

## Boundary

This mechanism is conditional. The repository has not derived \(L_+\),
\(L_-\), or the product law from Principle R. Defining the channels from the
already computed TESC null rays would be circular.

The next admissible work is source-text provenance search for independently
defined channels, not additional numerical parameter scans.

## Next Source-Text Search Targets

Future work should check the original theoretical sources for noncircular
two-channel candidates defined before inspecting \(G_{\rm TESC}\):

- whether task and exposure can be derived as separate linear realization
  channels;
- whether forward and reverse realizability form two independent channels;
- whether preparation and recovery form two independent channels;
- whether the product law follows from composition, capacity normalization or
  symmetry;
- whether the selected two-dimensional plane is fixed before the TESC null
  rays are computed.

Only a source-bound noncircular candidate should be promoted to a provenance
certificate.
