# Experiment Card: Mel Frontend Metamer Comparison

## Mode

- Mode: executor
- Route: TEST
- Acceptance source: the owner-provided 2026-08-18 request to implement and
  run an isolated gradient-based waveform/Mel-metamer comparison.
- Source snapshot / evidence grade: canonical frontend source inspection plus
  a local real-audio segment scan; no metamer result exists yet.

## Hypothesis

At comparably low normalized log-Mel error, the 24 kHz / 128-bin / 40 ms
frontend may constrain perceptually meaningful waveform variation more strongly
than the current 16 kHz / 80-bin / 25 ms frontend. The experiment must also be
able to show no meaningful difference or greater ambiguity for the candidate.

## Root Objective

Make the audible null spaces of two explicitly defined Mel frontends directly
comparable without changing training or inference architecture.

## Goal Decomposition

- Reproduce the existing frontend's exact differentiable semantics and verify
  them against `compute_log_mel_10ms`.
- Optimize several noise initializations against the same physical excerpt and
  make convergence quality visible and comparable.
- Emit listening artifacts, diagnostics, exact configurations, and metrics
  sufficient for an owner-led perceptual comparison.

## Candidate Variants

- A: directly optimize bounded waveform samples with Adam against normalized
  log-Mel MSE.
- B: use Griffin-Lim from a Mel pseudo-inverse.
- C: optimize an intermediate complex STFT and synthesize a waveform.
- D: use a learned vocoder or diffusion model.

## Local Verification Matrix

| Variant | Matches the question | Adds another inverse prior | Directly exposes Mel null space | Decision |
| --- | --- | --- | --- | --- |
| A | Yes | No | Yes | Select |
| B | Partly | Yes, phase-recovery dynamics | No | Reject |
| C | Partly | Yes, STFT consistency/parameterization | Indirectly | Reject |
| D | No | Yes, learned generator prior | No | Reject |

## Selected Variant and Selection Pressure

- Selected: direct waveform-sample Adam optimization from three fixed random
  seeds.
- Primary pressure: reach the same strict normalized log-Mel RMSE target for
  both frontends.
- Guard pressure: no waveform-domain optimization term, learned generator, or
  training/inference integration.
- Runtime pressure: early-stop a run after reaching the frozen target; cap each
  run at a fixed explicit step count.
- Kill pressure: stop interpretation if baseline parity fails, any run is
  non-finite, or convergence quality is not comparable.

## Exact Frontends

Both use nnAudio `MelSpectrogram`, Hann window, `center=False`, power `2.0`,
`norm=1`, non-trainable STFT and Mel filters, natural logarithm, a `1e-5`
linear-Mel floor, and the repository's right-padding/frame-count semantics.

| Name | Sample rate | Mel bins | Hop | Window | `n_fft` | `win_length` | `fmin` | `fmax` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `existing` | 16,000 Hz | 80 | 160 samples / 10 ms | 400 samples / 25 ms | 400 | 400 | 20 Hz | 8,000 Hz |
| `candidate_24k_128mel_40ms` | 24,000 Hz | 128 | 240 samples / 10 ms | 960 samples / 40 ms | 960 | 960 | 20 Hz | 12,000 Hz |

Using `n_fft=win_length=960` keeps the candidate consistent with the existing
frontend's unpadded-window convention while representing exactly 40 ms at
24 kHz.

## Minimal Change and Scope

- Add one self-contained notebook:
  `notebooks/mel_frontend_metamer_experiment.ipynb`.
- The notebook owns the input cell, exact frontend definitions, differentiable
  feature path, optimizer, WAV/metric/plot output, validation, and final ordered
  audio-path table. It must not depend on experiment code in another tracked
  file.
- Read-only context: `src/pulsefield_model/features/audio.py` and
  `src/pulsefield_model/features/mel_base.py`.
- Do not change production features, configs, training, or inference.

## Dataset Slice

- Default reproducibility example: seconds 11.0--17.0 of
  `dataset/0/2183073/audio.ogg`.
- The fixed six-second slice was selected before implementation from a
  one-second-grid scan for high positive canonical log-Mel flux while retaining
  non-silent energy. This is an excerpt-selection heuristic, not an outcome
  metric.
- The CLI accepts any owner-selected local source, start, and duration.

## Metrics and Verification

- Primary metric: log-Mel normalized RMSE,
  `sqrt(mean((prediction-target)^2) / var(target))`, with frozen target `0.10`.
- Secondary Mel metrics: raw RMSE, MAE, maximum absolute residual, and linear
  Mel spectral convergence.
- Report-only waveform metrics: RMS, peak, reference correlation, waveform
  RMSE, and SI-SDR. None enters the optimization loss.
- Verify: execute the notebook end to end on a real-audio smoke slice, then run
  its owner-facing six-second entry configuration with three seeds and the
  frozen convergence target. Notebook assertions replace a separate tracked
  experiment test file.
- Guard: baseline differentiable output must match
  `compute_log_mel_10ms` within float32 numerical tolerance; all outputs and
  gradients must be finite; source and reconstruction durations must match.
- Qualitative check: listen to the source and every final reconstruction, with
  attention to transient timing/character, smearing, harmonic and instrument
  identity, texture, and between-seed variation.

## Signals, Kill Criteria, and Failure Modes

- Positive: all runs reach comparable error and candidate metamers are
  consistently closer to the source and to each other in task-relevant audible
  properties.
- Negative: all runs reach comparable error but the candidate does not reduce
  perceptual variation.
- Ambiguous: any frontend misses the target, achieved error differs materially,
  or listener judgments vary by seed/content.
- Kill interpretation (not artifact generation) on parity failure, non-finite
  optimization, unequal segment durations, or incomparable convergence.
- Expected failures: Adam plateaus, learning-rate sensitivity, sample clipping,
  nnAudio device/backend differences, random-seed variance, and one excerpt not
  generalizing to other musical material.

## Runtime Budget and Confounders

- Full default: 3 seeds x 2 frontends, at most 3,000 Adam steps per run, saving
  every 250 steps, with learning rate `0.003`; expected runtime is
  hardware-dependent and is recorded.
- Smoke gate: a sub-second excerpt and a few steps only; it validates mechanics,
  not the hypothesis.
- Confounders: the candidate changes sample rate, bandwidth, bin count, and
  window length together; both retain a 10 ms frame grid, but `center=False`
  means window support and timestamp interpretation differ; source resampling,
  optimizer stochasticity, clipping, convergence mismatch, and listening setup
  can all affect interpretation.

## Result Interpretation Plan

- Positive result suggests a narrower audible null space for this excerpt and
  optimizer, not improved mapper performance.
- Negative result weakens the combined candidate but does not isolate which
  changed frontend parameter is responsible.
- Ambiguous result requires convergence repair before any frontend conclusion.
- The human owner decides perceptual outcome and whether a later controlled
  sweep separates sample rate, bandwidth, Mel bins, and window length.

## Result Log Template

- Experiment / date / commit or run id:
- Source slice and exact frontend configs:
- Runtime and device:
- Per-seed primary and secondary metrics:
- Convergence target reached:
- Baseline parity / guard results:
- Listening observations:
- Confounders or failed checks:
- Interpretation: positive / negative / ambiguous:
- Next-loop action: KILL / MUTATE / TEST:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes, explicitly requested by owner
- Closed loop complete: yes
- Remaining ambiguity: perceptual outcome is intentionally reserved for owner
  listening after artifact generation.

## Pre-Execution Mutation 001: Stable Adam and Attainable Common Gate

- Route: MUTATE back to TEST; the direct-waveform variant and research question
  are unchanged.
- Trigger: a 0.5-second real-audio convergence probe showed oscillation at the
  initially proposed learning rate `0.03`. At `0.003`, the existing frontend
  improved monotonically enough to reach normalized RMSE `0.079973` after
  3,000 steps, but did not reach the originally frozen `0.05` gate. The
  candidate reached `0.099741` before its 3,000-step cap at the stable rate.
- Mutation: freeze Adam learning rate `0.003` and a common normalized-RMSE gate
  of `0.10`, meaning reconstruction RMSE is no more than 10% of the target
  log-Mel standard deviation.
- Why bounded: optimizer family, loss, metrics, seeds, step cap, frontends,
  source slice, artifacts, and interpretation guards are unchanged.
- Local verification outcome: both frontends reached the mutated common gate;
  proceed to the full six-second, three-seed run. If any full run misses it,
  mark the comparison ambiguous rather than interpreting frontend differences.

## Pre-Execution Mutation 002: Notebook-Only Delivery

- Route: MUTATE back to TEST; frontend definitions, optimizer, metrics,
  convergence gate, dataset slice, and research interpretation are unchanged.
- Trigger: owner explicitly requires one self-contained notebook entrypoint and
  no separate tracked experiment script or `tests/` file.
- Mutation: consolidate all experiment code and executable validation into
  `notebooks/mel_frontend_metamer_experiment.ipynb`; delete the earlier
  standalone script and experiment test file. Outside the notebook, only the
  two `docs/research/` records may point to the experiment.
- Required notebook output: for each frontend and seed, print and return audio
  file paths in increasing optimization-step order alongside normalized
  log-Mel RMSE, so the audible approach to the target can be inspected
  directly.
- Why bounded: this changes packaging and entrypoint ergonomics only; it does
  not change experimental behavior or evidence thresholds.

## Pre-Execution Mutation 003: Four Listening Paths per Setting

- Route: MUTATE back to TEST; optimization still runs three seeds and retains
  complete convergence metrics internally.
- Trigger: owner judged the previous 33-path listening manifest too large and
  requested exactly four paths for each Mel setting.
- Mutation: choose one explicit representative listening seed (default seed 0)
  and select four evenly spaced ranks from its checkpoints after sorting by
  normalized log-Mel RMSE from high to low. This guarantees each setting's
  returned paths are ordered from lower to higher Mel similarity and includes
  the endpoints of the available similarity range.
- Output contract: exactly eight rows total -- four `existing` and four
  `candidate_24k_128mel_40ms` -- while the summary continues to gate
  comparability on all three seeds for both settings.
- Why bounded: only the owner-facing listening manifest changes; optimization,
  saved checkpoints, metrics, and interpretation remain unchanged.
