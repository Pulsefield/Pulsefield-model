# Result Log: Mel Frontend Metamer Notebook

## Mode

- Mode: executor
- Experiment Card existed before execution: yes,
  `docs/research/mel_frontend_metamer_experiment.md`
- Route entering executor: TEST
- Recorded mutations: stable Adam/common convergence gate, followed by the
  owner-requested notebook-only delivery.
- Source snapshot / evidence grade: canonical frontend parity assertions, one
  completed six-run real-audio notebook execution, and completed owner
  listening evaluation.

## Experiment

- Date: 2026-08-19
- Sole executable entrypoint:
  `notebooks/mel_frontend_metamer_experiment.ipynb`
- Input contract: edit `AUDIO_PATH`, `OUTPUT_DIR`, segment bounds, and optional
  optimizer controls in the tagged input cell, then run all cells.
- Dataset slice used for verification: seconds 11.0--17.0 of
  `dataset/0/2183073/audio.ogg`.
- Baseline / comparator: current 16 kHz / 80 Mel / 25 ms frontend versus the
  24 kHz / 128 Mel / 40 ms candidate; both use a 10 ms frame grid.
- Runtime: 18.561 seconds on CPU for all six optimization runs.
- Tracked experiment files: this result log, the Experiment Card, and the one
  notebook. There is no standalone experiment script or experiment test file.
- Generated artifacts: ignored `artifacts/mel_metamer_notebook_4paths/`.

## Result

The notebook's internal assertions passed for exact geometry, finite gradients,
and parity with the repository frontend at `rtol=1e-5`, `atol=1e-6`. Both
targets have exactly 600 frames: `[600, 80]` for the current frontend and
`[600, 128]` for the candidate.

| Frontend | Seed | Adam steps | Normalized log-Mel RMSE |
| --- | ---: | ---: | ---: |
| existing | 0 | 956 | 0.099818 |
| existing | 1 | 938 | 0.099758 |
| existing | 2 | 953 | 0.099823 |
| candidate | 0 | 1,173 | 0.099990 |
| candidate | 1 | 1,167 | 0.099851 |
| candidate | 2 | 1,240 | 0.099697 |

- All six runs passed the common `<= 0.10` normalized log-Mel RMSE gate.
- Median current error: `0.099818`.
- Median candidate error: `0.099851`.
- Convergence summary status: `comparable`.
- The notebook retains 33 saved checkpoint rows internally for convergence
  evidence, but emits exactly eight owner-facing listening rows to
  `artifacts/mel_metamer_notebook_4paths/audio_path_progression.json`: four for
  each frontend, all from representative seed 0. Each group is ordered from
  higher error/lower similarity to lower error/higher similarity and includes
  optimization step, error, and an absolute WAV path.
- Every listed WAV is mono and exactly 6.0 seconds. Each run includes initial
  noise, periodic checkpoints, final audio, metrics, and a diagnostic plot.
- No waveform-domain metric entered the optimization target.
- Qualitative result: owner listening found a large, immediately audible
  difference in high-frequency onset retention between the two matched-error
  metamer groups.

## Owner Listening Conclusion

Metamers constrained by the existing 16 kHz / 80-bin / 25 ms frontend lose or
blur substantially more high-frequency attack information than metamers
constrained by the 24 kHz / 128-bin / 40 ms frontend. The difference remains
clearly audible at matched log-Mel convergence, so it cannot be explained by
one configuration receiving a weaker optimization result.

The existing representation therefore has a perceptually significant null
space that includes high-frequency transient detail relevant to onset character
and rhythmic articulation. Its compression profile is better suited to speech
and vocal recognition than to general music beatmap generation. It should not
remain the default audio frontend for the mapper.

This experiment changes several frontend properties together, including sample
rate, bandwidth, Mel-bin count, and window length. It does not identify which
individual property causes the improvement. It does establish that the current
combined configuration discards task-relevant audible information and needs to
be replaced or redesigned before further mapper work treats the frontend as a
fixed input representation.

## Closed-Loop Outcome

- Notebook-only packaging passed: all experiment implementation and executable
  validation live in one notebook.
- Direct-waveform Adam passed local verification and the common convergence
  control across both frontends and all three seeds.
- The four-path-per-setting output makes the approach toward each target
  directly listenable without exposing the full checkpoint/seed inventory.
- Next-loop action: MUTATE. Retire the existing configuration as the assumed
  general-music frontend and use a separate controlled experiment to determine
  which candidate changes are necessary before freezing its replacement.

## Reproduction Command

Run interactively by opening the notebook and editing its input cell, or execute
the current input cell non-interactively:

```sh
uv run --extra mps jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  notebooks/mel_frontend_metamer_experiment.ipynb \
  --ExecutePreprocessor.timeout=600 \
  --ExecutePreprocessor.kernel_name=python3
```

The verified local run used the repository `.venv` equivalent. Notebook
execution completed successfully. The sandbox emitted an IPykernel child-process
inspection warning during shutdown, after all cells and artifacts had completed;
the command returned zero and the executed notebook validates as nbformat v4.

## Verification and Interpretation

- Checks performed inside the notebook: exact frontend geometry, differentiable
  finite-gradient smoke checks, optional production-frontend parity, fail-closed
  output directory, finite optimization, exact WAV duration, shared convergence
  gate, and ordered path manifest generation.
- Confounders: sample rate, bandwidth, Mel-bin count, and window length change
  together; one excerpt and three seeds do not establish generality; resampling
  and listening setup can affect judgments.
- Supported: the notebook produces comparably converged audible progressions;
  owner listening finds that the candidate retains substantially more
  high-frequency onset information; the existing frontend is unsuitable as the
  default representation for general music beatmap generation.
- Not supported: attribution of the difference to one candidate parameter,
  downstream mapper-performance improvement, or generalization across songs.
- Classification: positive representation-level result. Reject the existing
  frontend for the intended general-music use case and MUTATE toward a
  controlled replacement study.
