# Prospective Audits

These scripts are prospective falsification interfaces. Their calibrated
self-tests are not empirical evidence for Principle R or Law I.

When a certificate or manifest claims source binding, it must provide both a
source path and a SHA-256 digest. The audit recomputes the digest from the file;
a filled 64-character hash string alone is not accepted as source-bound.

The public reproduction entry point remains:

```bash
python run_r_to_law1.py
```

The v0.2.1-preflight audits are fail-closed tools:

| Audit | Role | Default status |
|---|---|---|
| `r_law1_bidirectional_zero_set_audit_v3_0.py` | tests prospective independent \(F\) data against both zero-set inclusions | calibration only |
| `r_law1_two_channel_origin_audit_v3_1.py` | records scalar single-channel obstruction and conditional two-channel mechanism using protocol-derived \(G_{\rm TESC}\) | native channel provenance open |
| `r_law1_cross_protocol_cone_audit_v3_2.py` | tests cross-protocol unordered cone naturality under frozen maps with manifest-declared criteria | calibration only |
| `rc_zero_structure_to_law1_audit_v0_1.py` | exposes the RC zero-structure interface and five-gate bridge audit | physical binding open |

Do not interpret passing self-tests as a derivation of `R => Law I`.
