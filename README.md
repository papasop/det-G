# Operational Lorentzian Task–Exposure Signed Cost

Release: `v0.1.0-law1-operational`  
Object name: **TESC** (Task–Exposure Signed Cost)  
Metric candidate: `G_TESC`

This package supersedes the exploratory archive
`candidate3-law1-operational-v0.1.zip`. “Candidate 3” is retained only as a
historical alias; the scientific object is called TESC throughout this release.

## 1. Frozen definition

Let

\[
z=(\delta\Omega,\delta\Delta)
\]

denote Rabi-amplitude and detuning perturbations. Define

\[
\mathcal C_{\rm TESC}(z)
=\mathcal E_{\rm task}(z)
-\lambda\widetilde{\mathcal E}_{\rm exp}(z),
\qquad \lambda=1,
\]

with

\[
\mathcal E_{\rm task}(z)=1-\mathcal F_{\rm endpoint}(z),
\]

and centred dimensionless Rabi exposure

\[
\widetilde{\mathcal E}_{\rm exp}(z)
=\mathcal E_{\rm Rabi}(z)-\mathcal E_{\rm Rabi}(0)
-\nabla\mathcal E_{\rm Rabi}(0)^Tz.
\]

The local metric candidate is

\[
G_{\rm TESC}=D^2\mathcal C_{\rm TESC}(0),
\]

and the finite zero set is

\[
\mathcal Z_{\rm TESC}=\{z:\mathcal C_{\rm TESC}(z)=0\}.
\]

## 2. Main result

In the frozen no-account two-level control model:

- `G_TESC` is real, symmetric, nondegenerate and Lorentzian;
- the two Hessian-null tangents continue to finite zero-cost branches;
- every nonzero transverse section in the frozen adaptive rectangle contains
  exactly two roots;
- roots absent from the original `|y| <= 0.20` box are recovered by expansion
  to `|y| <= 0.32`;
- no extra branch is found through the frozen maximum boundary `|y| <= 0.40`;
- the largest zero-level residual is about `9.84e-14`;
- 256 frozen `GL(2)` trials preserve Hessian covariance, finite zero points and
  Lorentzian signature;
- extreme unit rescalings preserve the signature;
- all nine frozen weights `lambda = 0.25, ..., 4` remain Lorentzian.

This supports an **operational Law-I(a–f) candidate**. It does not derive TESC
uniquely or necessarily from Principle R.

## 3. Quick start

### Colab / Jupyter

Upload or extract the archive, change into its root directory, then run:

```python
%run src/k1_pasqal_public_data_generator_v2_1.py
%run src/k1_pasqal_law1_def_v2_3.py
%run src/k1_pasqal_law1_adaptive_zero_set_v2_4.py
%run src/k1_pasqal_law1f_covariance_v2_5.py
```

The kernel `-f ...json` notice is harmless; each script ignores notebook
arguments.

### Command line

```bash
python src/k1_pasqal_public_data_generator_v2_1.py
python src/k1_pasqal_law1_def_v2_3.py
python src/k1_pasqal_law1_adaptive_zero_set_v2_4.py
python src/k1_pasqal_law1f_covariance_v2_5.py
```

Only NumPy and Python's standard library are required.

## 4. What each audit establishes

### v2.1 — signed cost and transport diagnostic

Constructs the frozen TESC functional and estimates its Hessian. It also shows
that full-gradient calibration is not rank one, while a detuning-only policy is
rank one by controller design. The policy result is not labelled native K=1
dynamics.

Expected key fields:

```text
signed_cost_Lorentzian = true
full_gradient_transport_rank_one = false
detuning_only_policy_rank_one = true
native_K1_PASQAL_bridge_supported = false
```

### v2.3 — finite continuation

Tests whether the two local Hessian-null directions continue into finite exact
zero-level branches. The original fixed rectangle misses boundary roots, so
v2.3 is retained as the documented fail-closed precursor to v2.4.

### v2.4 — adaptive bounded completeness

Expands the transverse boundary according to a frozen rule and checks for
missing or additional roots.

Expected key fields:

```text
Law_I_d_finite_extension_supported = true
Law_I_e_complete_in_frozen_adaptive_domain = true
Law_I_f_Principle_R_origin_supported = false
```

### v2.5 — operational Law-I(f)

Checks coordinate covariance, unit invariance, finite-zero-set covariance and
coefficient robustness.

Expected key fields:

```text
operational_I_f_supported = true
native_Principle_R_certificate_pass = false
complete_I_f_supported = false
```

## 5. Reading the evidence correctly

Current status:

| Gate | Status |
|---|---|
| Law I(a–c): local Lorentzian quadratic geometry | Supported |
| Law I(d): finite zero branches | Supported |
| Law I(e): complete two-branch zero set in frozen bounded domain | Supported |
| Law I(f), operational covariance/non-fine-tuning | Supported |
| Law I(f), native derivation from Principle R | Open |
| Law II: native `J_G H` transport | Open |
| Law III: critical rank-one `G`-null flow and scale | Not supported by current proxy |
| OU/FP same-model closure | Open |
| Physical wavefunction / Born rule / physical clock | Not claimed |

Allowed summary:

> In a frozen public two-level control model, the predeclared TESC functional
> has a nondegenerate Lorentzian Hessian and a numerically complete two-branch
> finite zero set within the declared bounded domain. Its Hessian and finite
> zero set are representation-covariant, and its signature is robust over the
> frozen coefficient range.

Do not claim that this release proves Principle R, native K=1 dynamics, PASQAL
hardware realization, a physical light cone, a physical wavefunction, Born's
rule or collapse.

## 6. Output locations

Each script writes a sibling result directory in the working directory. Frozen
reference outputs are included under:

```text
reference_results/v2_1/
reference_results/v2_3/
reference_results/v2_4/
reference_results/v2_5/
```

Compare scientific status, protocol hash and gate fields before comparing
floating-point diagnostics across Python/NumPy versions.

## 7. Next research steps

1. Produce a source-bound analytic derivation or explicit operational axiom for
   the subtraction, relative normalization and realizability meaning of TESC.
2. Prospectively replicate Law-I(d/e/f-operational) on a new frozen pulse
   family without changing gates.
3. Independently identify a Law-II transport; do not reuse the calibration
   gradient as `J_G H`.
4. Test Law III only after Law II is bound to the same state space and time
   parameter.

## 8. Package tree

```text
README.md
docs/
  CLAIM_BOUNDARY.md
  OPERATION_GUIDE.md
src/
  k1_pasqal_public_data_generator_v2_1.py
  k1_pasqal_law1_def_v2_3.py
  k1_pasqal_law1_adaptive_zero_set_v2_4.py
  k1_pasqal_law1f_covariance_v2_5.py
reference_results/
  v2_1/
  v2_3/
  v2_4/
  v2_5/
```

