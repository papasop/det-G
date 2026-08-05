# Mathematical statement

## Principle R premise

Let \(F\ge 0\) denote a nonnegative realization cost. In the local form used by
this repository, Principle R supplies an attainable nonzero local process
direction \(v\) with

\[
F(v)=0.
\]

## Law-I structural assumptions

The Lorentzian conclusion is not obtained by taking the Hessian of the
nonnegative cost \(F\). A real \(C^2\) nonnegative cost at a zero-valued local
minimum has a positive-semidefinite Hessian.

Instead, the complete zero set of the nonnegative realization cost is
independently represented by the null set of \(q\), an independently declared
signed quadratic zero-set representative on a selected real two-dimensional
process space:

\[
q(v)=v^T Gv,
\qquad G=G^T,
\qquad \det G\ne0.
\]

The key binding assumption is equality of local zero sets:

\[
Z(F)\cap V = Z(q)\cap V.
\]

Equivalently, within the selected plane \(V\), the physical zero-cost
directions of \(F\) are represented by the null directions of the signed form
q. This does not assert \(F=q\) and does not assert \(D^2F=G\).

## Conditional theorem

If Principle R and the above zero-set representation assumptions hold, then

\[
\det G<0,
\qquad \operatorname{sig}(G)=(1,1),
\]

and the null set is the union of two distinct real rays.

## Proof

By Principle R there is \(v\ne0\) with \(F(v)=0\). By the zero-set binding
\(Z(F)\cap V=Z(q)\cap V\), the same direction satisfies \(q(v)=v^TGv=0\). A
positive- or negative-definite real symmetric form has no nonzero null vector.
Nondegeneracy excludes a zero eigenvalue. Thus the two real eigenvalues of
\(G\) have opposite signs, so their product \(\det G\) is negative.
Diagonalization then gives a difference of two squares, whose zero set factors
into two distinct real lines. This is a linear-algebra theorem; numerical
scripts only audit concrete signed representatives and bounded zero-contrast
sets.

## Logical boundary

The theorem is

\[
R+\mathcal A_{\mathrm{Law\,I}}\Rightarrow\mathrm{Law\,I},
\]

where \(\mathcal A_{\mathrm{Law\,I}}\) includes the zero-set binding between
the nonnegative realization cost and the signed quadratic representative. It is
not an unconditional proof of `R => Law I`.
