# K1 LLM evidence status through V4.7

This page records the engineering evidence currently included in the repository
for negative-Lorentz adapter experiments. It separates passed results, failed
hypotheses, exploratory diagnosis, and unsupported physical interpretations.

## Supported engineering result

A pre-mix scale-matched, full-layer negative-Lorentz adapter produced a small,
reproducible cross-entropy improvement over matched residual,
negative-Euclidean, and negative-random controls in DistilGPT-2 and in a
secondary ten-seed pooled GPT-2 analysis. The GPT-2 pooled Test improvements
were significant at paired 95% intervals and showed 9/10 wins against each
control.

The effective direction was negative and is not OU attraction toward `K=1`.
These experiments do not validate physical wave-function collapse or a
Token-collapse transition. Absolute improvements were small and remain limited
to the tested models, datasets, and adapter protocol.

## Experiment history

| Experiment | Status | Meaning |
| --- | --- | --- |
| V4.3 DistilGPT-2 10 seeds | Passed | Negative-Lorentz engineering effect in this setting |
| V4.4 GPT-2 3-seed screen | Failed | Screening GO rule did not pass |
| V4.5 diagnosis | Exploratory | Suggested local seed/layer sensitivity |
| V4.6 fixed Mask | Failed | Fixed layer-mask hypothesis did not generalize |
| V4.6 full-Lorentz control | Positive held-out result | 7/7 Test wins over residual |
| V4.7 GPT-2 pooled 10 seeds | Passed, secondary | Pooled common-control replication |

V4.7 is a secondary pooled analysis of two consecutive experiments, not a newly
preregistered standalone ten-seed experiment.

## V4.7 pooled GPT-2 results

The V4.7 audit pools the common full-layer controls from V4.4 and V4.6.
It reported no detected two-cohort heterogeneity (`I^2 = 0%`), which is
descriptive because there were only two cohorts.

### Test

| Comparison | Mean loss difference | Paired 95% CI | Wins |
| --- | ---: | ---: | ---: |
| Lorentz - residual | -0.000583545 | [-0.000814308, -0.000352782] | 9/10 |
| Lorentz - Euclid | -0.000545595 | [-0.000762316, -0.000328873] | 9/10 |
| Lorentz - Random | -0.000613526 | [-0.000836661, -0.000390391] | 9/10 |

### OOD

| Comparison | Mean loss difference | Paired 95% CI | Wins |
| --- | ---: | ---: | ---: |
| Lorentz - residual | -0.000109573 | [-0.000188304, -0.000030841] | 8/10 |
| Lorentz - Euclid | -0.000102179 | [-0.000182121, -0.000022236] | 8/10 |
| Lorentz - Random | -0.000117842 | [-0.000195787, -0.000039898] | 8/10 |

The absolute effect is small. GPT-2 pooled Test accuracy changed from roughly
`0.288446` for residual to `0.288560` for negative Lorentz, approximately
`0.0115` percentage points.

## Failed or exploratory results that remain part of the record

- V4.4's original three-seed GO/PASS rule failed.
- V4.5 was a read-only exploratory layer diagnosis.
- V4.6's fixed layer Mask failed because it significantly underperformed the
  full-layer Lorentz control.
- The fixed Mask result must not be presented as validated.

## Unsupported physical interpretation

The useful flow was the negative direction and showed negative estimated K1
regression rather than OU attraction toward `K=1`. None of these experiments
establish:

- OU regression toward `K=1` inside an LLM;
- physical wave-function collapse;
- a critical Token-commit transition;
- Born-rule sampling from geometric signature change;
- general improvements across LLM families or production tasks.

The evidence supports a small structured optimization bias, not the physical
Token-collapse interpretation.
