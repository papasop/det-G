# TESC operation guide

## A. Extract and enter the package

Colab:

```python
from zipfile import ZipFile
from pathlib import Path

archive = Path("operational-lorentzian-task-exposure-cost-v0.1.0.zip")
with ZipFile(archive) as z:
    z.extractall("tesc_v0_1_0")

%cd tesc_v0_1_0/operational-lorentzian-task-exposure-cost-v0.1.0
```

## B. Run the confirmatory chain

```python
%run src/k1_pasqal_public_data_generator_v2_1.py
%run src/k1_pasqal_law1_adaptive_zero_set_v2_4.py
%run src/k1_pasqal_law1f_covariance_v2_5.py
```

Run v2.3 only when reproducing the historical fixed-boundary diagnostic:

```python
%run src/k1_pasqal_law1_def_v2_3.py
```

## C. Minimal acceptance checklist

The frozen reference chain should show:

```text
v2.1 signed_cost_Lorentzian = true
v2.4 Law_I_d_finite_extension_supported = true
v2.4 Law_I_e_complete_in_frozen_adaptive_domain = true
v2.5 operational_I_f_supported = true
```

It should also continue to show:

```text
v2.1 native_K1_PASQAL_bridge_supported = false
v2.4 Law_I_f_Principle_R_origin_supported = false
v2.5 native_Principle_R_certificate_pass = false
```

Changing those negative fields manually invalidates the audit.

## D. Reproducing the original narrow-boundary issue

v2.3 uses the initial rectangle and should report incomplete bounded coverage.
v2.4 then applies the frozen adaptive expansion and recovers the two missing
roots. This pair documents that the earlier failure was caused by truncation,
not silently removed from the record.

## E. Extending the project safely

- Copy the package to a new release before changing pulse segments, coefficient
  grids, root thresholds or domains.
- Give every changed protocol a new version and hash.
- Never select a pulse family after inspecting whether its Hessian is
  Lorentzian.
- Never tune the exposure coefficient to force `det(G) < 0`.
- Keep operational Law I separate from native Principle-R provenance.
- Keep calibration iteration separate from physical quantum time.

## F. Troubleshooting

- `ignored notebook/kernel arguments`: expected in Colab/Jupyter.
- Small last-digit differences: compare gates and protocol hashes first.
- Missing output directory: confirm the working directory is writable.
- Missing roots after modifying the domain: treat as a new result; do not widen
  the boundary after looking at individual failures without a new protocol.

