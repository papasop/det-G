# Mathematical statement

## Principle R premise

There exists an attainable nonzero local process direction `v` of zero realization cost.

## Law-I structural assumptions

At a stationary base point, the local tangent cost is completely represented on a selected real two-dimensional process space by

\[
q(v)=v^T Gv,
\qquad G=G^T,
\qquad \det G\ne0,
\]

and `q(v)=0` is equivalent to being a physical zero-cost tangent direction.

## Conditional theorem

If Principle R and the above assumptions hold, then

\[
\det G<0,
\qquad \operatorname{sig}(G)=(1,1),
\]

and the null set is the union of two distinct real rays.

## Proof

By Principle R and completeness there is `v != 0` with `q(v)=0`. A positive- or negative-definite real symmetric form has no nonzero null vector. Nondegeneracy excludes a zero eigenvalue. Thus the two real eigenvalues have opposite signs, so their product `det G` is negative. Diagonalization then gives a difference of two squares, whose zero set factors into two distinct real lines. This is a linear-algebra theorem; numerical scripts only audit concrete witnesses and implementations.

## Logical boundary

The theorem is

\[
R+\mathcal A_{\mathrm{Law\,I}}\Rightarrow\mathrm{Law\,I},
\]

not an unconditional proof of `R => Law I`.

