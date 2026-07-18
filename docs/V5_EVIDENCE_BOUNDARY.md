# K1 LLM engineering evidence boundary through V5.9

## Scope

These experiments test engineered low-rank interventions in frozen language
models. They do not measure a physical metric, quantum state, physical
decoherence, or physical wavefunction collapse.

## Evidence progression

| Version | Primary question | Result |
|---|---|---|
| V5.6 | Does the dynamic geometric controller beat matched general controls? | No. It beat residual/fixed controls but was dominated by generic scalar and learned-direction gates. |
| V5.7 | Does a gauge-compatible GeoFlow direction interact beneficially with a Token throttle? | GeoFlow strongly improved both geometric and generic arms, but generic remained better; preregistered screen was NO-GO. |
| V5.8 | Under fixed GeoFlow and matched throttle distributions, is geometry controller-specific? | Geometry beat entropy, random, and residual controls but lost to the generic learned scalar gate; NO-GO. |
| V5.9 | Does an explicit Lorentz-derived feature add information to a shared generic MLP? | No measurable incremental effect; all adaptive candidate features were effectively tied; NO-GO. |

## V5.7 three-seed screen

Protocol: frozen OPT-125M, gauge-compatible `lambda(h) B A h`, exact matched
first-order product-displacement budget, three untouched seeds.

- GeoFlow geometry minus AdamW geometry Test loss: mean `-0.0570195`, 3/3
  wins, paired 95% t interval `[-0.0581083, -0.0559306]`.
- GeoFlow generic minus AdamW generic: mean `-0.0565966`, 3/3 wins.
- Geometry minus generic under GeoFlow: mean `+0.0001304`, 0/3 wins for
  geometry; paired interval entirely above zero.
- The interaction was negative because GeoFlow reduced geometry's disadvantage,
  not because geometry became the best controller.
- `GO_TO_10_NEW_SEEDS=false`.

This is promising evidence for the quotient direction plus adaptive-throttle
engineering combination. It is not yet an independent confirmation of the
public `CapacityAdaptiveQuotientFlow` optimizer.

## V5.8 controller-specificity screen

Protocol: fixed GeoFlow, four parameter-matched adaptive controllers,
validation-only per-layer quantile calibration, matched product budget.

Held-out Test loss ranking:

| Controller | Mean loss |
|---|---:|
| generic | 4.368912 |
| geometry | 4.369786 |
| entropy | 4.374298 |
| residual | 4.381452 |
| random feature | 4.382124 |

- Geometry minus generic: `+0.0008738`, 0/3 geometry wins, paired interval
  `[+0.0007845, +0.0009630]`.
- Geometry minus entropy: `-0.0045124`, 3/3.
- Geometry minus random feature: `-0.0123381`, 3/3.
- Geometry minus residual: `-0.0116668`, 3/3.
- `GO_TO_10_NEW_SEEDS=false`.

Geometry therefore carried useful structured Token information beyond entropy,
random features, and a constant throttle, but did not outperform the general
learned scalar controller.

## V5.9 incremental-feature screen

Protocol: fixed GeoFlow, shared seven-input MLP, one candidate feature changed,
validation-only quantile calibration.

- Geometry-feature minus generic-base Test loss: `+4.60e-6`, interval crossed
  zero.
- Geometry-feature minus entropy-feature: `+1.96e-5`, interval crossed zero.
- Geometry-feature minus random-feature: `+9.75e-7`, interval crossed zero.
- Geometry-feature minus residual: `-0.0126254`, 3/3.
- `GO_TO_10_NEW_SEEDS=false`.

The strong shared MLP made the seventh candidate feature effectively
interchangeable. A read-only feature-use audit would be needed to distinguish
feature redundancy from the MLP ignoring its seventh input.

## Supported engineering statement

Within the tested frozen OPT-125M, low-rank, matched-product-budget protocol:

1. the tested GeoFlow quotient direction was much more effective than the
   matched factor-Adam direction;
2. Token-adaptive throttling was better than a constant residual throttle;
3. Lorentz-derived control was interpretable and useful but not uniquely or
   incrementally superior to a strong generic controller.

## Unsupported statements

The experiments do not establish:

- physical Token collapse;
- determinant-boundary crossing as the cause of generation;
- physical quantum decoherence;
- Lorentz-signature specificity in LLM performance;
- general superiority across models, datasets, training regimes, or scales.

## Recommended next experiment

Run a preregistered V6.0 with ten untouched seeds and at least OPT-125M plus
GPT-2. Compare only AdamW/GeoFlow by residual/adaptive throttle, use the public
GeoFlow optimizer API, match compute and represented product displacement, and
report Test loss, OOD loss, calibration, runtime, and all paired intervals.

