# Operation guide

## 1. Reproduce the central audit

```bash
python src/principle_r_to_law1_strengthened_audit_v2_7.py \
  --outdir principle_r_law1_strengthened_v2_7_results
```

Expected central fields:

```text
exact_conditional_theorem.gate = true
operational_TESC_witness.gate = true
native_R_selection.gate = false
conditional_R_to_LawI_supported = true
unconditional_R_alone_to_LawI_proved = false
```

## 2. Work on the open native-selection gate

Copy and fill `certificates/r_law1_native_derivation_template.json`, preserving source hashes and freezing definitions before examining outcomes. Then run:

```bash
python src/principle_r_to_law1_strengthened_audit_v2_7.py \
  --certificate certificates/r_law1_native_derivation_template.json
```

Changing booleans is not a proof. Each positive provenance field must correspond to a dependency-closed derivation in the hashed source.

## 3. Recommended citation level

Call this release a **conditional theorem and operational existence-witness package**. Do not describe it as an unconditional derivation of spacetime or quantum mechanics.

