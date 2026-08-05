# Operation guide

## 1. Reproduce the central audit

```bash
python run_r_to_law1.py
```

Expected central fields:

```text
analytic_theorem_logic_gate = true
conditional_theorem_premises_gate = false
theorem_proof_kind = analytic_linear_algebra
theorem_numerically_proved = false
signed_TESC_zero_set_representative.gate = true
physical_zero_set_binding_certificate_gate = false
native_R_selection.gate = false
conditional_R_to_LawI_supported = false
unconditional_R_alone_to_LawI_proved = false
```

## 2. Work on the open native-selection gate

Copy and fill `certificates/native_r_selection.template.json`, preserving
source hashes and freezing definitions before examining outcomes. Then run:

```bash
python run_r_to_law1.py
```

Changing booleans is not a proof. Each positive provenance field must
correspond to a dependency-closed derivation in the hashed source.

## 3. Recommended citation level

Call this release a **conditional theorem and signed zero-set representative package**. Do not describe it as an unconditional derivation of spacetime, quantum mechanics, or physical zero-cost branches.
