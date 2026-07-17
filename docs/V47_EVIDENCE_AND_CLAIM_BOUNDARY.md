# K1 LLM evidence through V4.7

## Supported engineering result

Within the frozen adapter protocol, the useful intervention was a full-layer
negative-Lorentz flow whose residual and geometric branches were normalized to
matched RMS before mixing.

- DistilGPT-2 V4.3 passed its ten-seed matched-control audit.
- GPT-2 V4.6 provided a positive seven-seed held-out full-Lorentz control:
  Test Lorentz minus residual mean `-0.000601889`, paired 95% CI
  `[-0.000800628, -0.000403150]`, wins `7/7`.
- GPT-2 V4.7 pooled the common full-layer controls from V4.4 and V4.6. It
  passed its secondary ten-seed audit with no detected two-cohort
  heterogeneity (`I²=0%`, descriptive with only two cohorts).

### GPT-2 pooled Test results

| Comparison | Mean loss difference | Paired 95% CI | Wins |
|---|---:|---:|---:|
| Lorentz − residual | -0.000583545 | [-0.000814308, -0.000352782] | 9/10 |
| Lorentz − Euclid | -0.000545595 | [-0.000762316, -0.000328873] | 9/10 |
| Lorentz − Random | -0.000613526 | [-0.000836661, -0.000390391] | 9/10 |

### GPT-2 pooled OOD results

| Comparison | Mean loss difference | Paired 95% CI | Wins |
|---|---:|---:|---:|
| Lorentz − residual | -0.000109573 | [-0.000188304, -0.000030841] | 8/10 |
| Lorentz − Euclid | -0.000102179 | [-0.000182121, -0.000022236] | 8/10 |
| Lorentz − Random | -0.000117842 | [-0.000195787, -0.000039898] | 8/10 |

The absolute effect is small. GPT-2 pooled Test accuracy changed from roughly
`0.288446` for residual to `0.288560` for negative Lorentz, approximately
`0.0115` percentage points.

## Failed or exploratory results that must remain visible

- V4.4's original three-seed GO/PASS rule failed.
- V4.5 was a read-only exploratory layer diagnosis.
- V4.6's fixed layer Mask failed because it significantly underperformed the
  full-layer Lorentz control. It must not be presented as validated.
- V4.7 is a secondary pooled audit, not an independent preregistered ten-seed
  experiment.

## Unsupported physical interpretation

The useful flow was the negative direction and showed negative estimated K1
regression rather than OU attraction to K=1. None of these experiments establish:

- OU regression toward K=1 inside an LLM;
- physical wave-function collapse;
- a critical Token-commit transition;
- Born-rule sampling from geometric signature change;
- general improvements across LLM families or production tasks.

The evidence supports a small structured optimization bias, not the physical
Token-collapse interpretation.
