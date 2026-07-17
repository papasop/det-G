# V4.3 result and interpretation boundary

## What passed

The preregistered ten-seed DistilGPT-2 experiment passed all configured gates:

- branches were normalized to matched RMS before mixing;
- the negative-Lorentz variant beat the residual control at paired 95% CI;
- it beat matched negative-Euclidean and negative-random controls;
- OOD performance was non-inferior to all controls;
- the minimum effect and 7/10 win-rate requirements passed.

This supports a small geometry-specific engineering effect in this exact
adapter and dataset configuration.

## What did not become established

The useful direction was the negative-Lorentz direction. Its estimated K1
regression was negative rather than an OU attraction toward K=1. Therefore the
experiment does not establish:

- OU K=1 regression inside an LLM;
- critical wave-function collapse;
- a physical Token-collapse transition;
- broad improvements across model families or tasks.

The next scientific milestone is a frozen-protocol replication on another
model size and an independent implementation, not further post-hoc tuning on
the same DistilGPT-2 setting.
