# Mel Frontend Metamer Comparison

## Question and scope

This experiment tested whether the 24 kHz / 128-bin / 40 ms Mel frontend
constrains perceptually meaningful waveform variation more strongly than the
legacy 16 kHz / 80-bin / 25 ms frontend at comparably low normalized log-Mel
error. A valid result could show a narrower audible null space for the
candidate, no meaningful difference, or an ambiguous comparison caused by
unequal convergence.

The objective was to compare the audible null spaces of two explicitly defined
frontends without changing mapper training or inference architecture. Direct
waveform-sample optimization was selected because it exposes each frontend's
null space without adding a phase-recovery, STFT-consistency, learned-vocoder,
or diffusion-model prior. The experiment did not test downstream mapper
quality.

## Compared frontends

Both frontends use nnAudio `MelSpectrogram` with a Hann window,
`center=False`, power `2.0`, `norm=1`, non-trainable STFT and Mel filters,
natural logarithms, a `1e-5` linear-Mel floor, and the repository's
right-padding and frame-count semantics.

| Name | Sample rate | Mel bins | Hop | Window | `n_fft` | `win_length` | `fmin` | `fmax` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `existing` | 16,000 Hz | 80 | 160 samples / 10 ms | 400 samples / 25 ms | 400 | 400 | 20 Hz | 8,000 Hz |
| `candidate_24k_128mel_40ms` | 24,000 Hz | 128 | 240 samples / 10 ms | 960 samples / 40 ms | 960 | 960 | 20 Hz | 12,000 Hz |

Using `n_fft=win_length=960` preserves the legacy frontend's unpadded-window
convention while representing exactly 40 ms at 24 kHz. Both configurations
retain a 10 ms frame grid.

## Input and method

The comparison used seconds 11.0--17.0 of
`dataset/0/2183073/audio.ogg`. The fixed six-second slice was selected before
the metamer optimization from a one-second-grid scan for high positive
canonical log-Mel flux while retaining non-silent energy. This was an
excerpt-selection heuristic, not an outcome metric.

For each frontend, bounded waveform samples were optimized directly with Adam
from seeds 0, 1, and 2. The fixed protocol used learning rate `0.003`, a cap of
3,000 steps per run, and checkpoints every 250 steps. A 0.5-second real-audio
pre-run probe had oscillated at the initially considered learning rate `0.03`.
At `0.003`, the legacy frontend reached normalized RMSE `0.079973` after 3,000
steps and the candidate reached `0.099741`, so the originally considered
`0.05` gate was not attainable by both frontends within the cap. The learning
rate and a common `0.10` gate were fixed before the six-run comparison.

The optimizer minimized normalized log-Mel MSE. The reported primary metric was
normalized log-Mel RMSE:

```text
sqrt(mean((prediction - target)^2) / var(target))
```

Every run had to reach `<= 0.10`, meaning reconstruction RMSE was no more than
10% of the target log-Mel standard deviation. Raw log-Mel RMSE, MAE, maximum
absolute residual, and linear-Mel spectral convergence were secondary metrics.
Waveform RMS, peak, reference correlation, waveform RMSE, and SI-SDR were
report-only diagnostics; no waveform-domain term entered the optimization
loss.

Each differentiable notebook frontend had to match `compute_log_mel_10ms`
instantiated with the corresponding repository `MelCacheConfig` within float32
numerical tolerance. The notebook also required exact frame geometry, finite
outputs and gradients, equal source and reconstruction durations, and a shared
convergence gate across all six runs. A missed gate, non-finite run, or unequal
duration would make the frontend comparison ambiguous. A parity failure would
invalidate the comparison before optimization.

The qualitative check compared the source with four reconstruction checkpoints
for each frontend from representative seed 0. After sorting checkpoints from
higher error to lower error, the notebook selected four evenly spaced ranks,
including both endpoints. Convergence comparability still depended on all three
seeds. Listening focused on transient timing and character, smearing, harmonic
and instrument identity, and texture. The perceptual judgment reported below
was made by the human experiment owner.

## Results

The experiment ran on CPU on 2026-08-19. All six optimization runs completed in
18.561 seconds. The notebook's internal assertions passed for exact geometry,
finite gradients, and parity between both differentiable frontends and
`compute_log_mel_10ms` under their corresponding repository configs at
`rtol=1e-5` and `atol=1e-6`. Both targets contained exactly 600 frames:
`[600, 80]` for the legacy frontend and `[600, 128]` for the candidate.

| Frontend | Seed | Adam steps | Normalized log-Mel RMSE |
| --- | ---: | ---: | ---: |
| existing | 0 | 956 | 0.099818 |
| existing | 1 | 938 | 0.099758 |
| existing | 2 | 953 | 0.099823 |
| candidate | 0 | 1,173 | 0.099990 |
| candidate | 1 | 1,167 | 0.099851 |
| candidate | 2 | 1,240 | 0.099697 |

All runs passed the common gate. Median normalized log-Mel RMSE was `0.099818`
for the legacy frontend and `0.099851` for the candidate. The final seed-0 pair
was also closely matched at `0.099818` and `0.099990`, respectively. Every
compared reconstruction was mono and exactly 6.0 seconds. The intermediate
rank-spaced checkpoints were ordered by error but were not pairwise
error-matched across frontends.

Across the unblinded seed-0 progressions and final pair, owner listening found a
large, immediately audible difference in high-frequency onset retention.
Reconstructions constrained by the legacy frontend lost or blurred
substantially more high-frequency attack information than those constrained by
the candidate. Under this excerpt, optimizer, and listening setup, this is a
single-listener qualitative observation of task-relevant audible null-space
differences, not a blinded perceptual significance result.

## Interpretation and limits

The owner used the result to reject the 16 kHz / 80-bin / 25 ms configuration
as the assumed frontend for general-music beatmap generation and select the
24 kHz / 128-bin / 40 ms configuration as the general-music follow-up. The
repository implements that selection as `MUSIC_MEL_CACHE_CONFIG`, while
`DEFAULT_MEL_CACHE_CONFIG` remains available for legacy Stage 2 use. The
comparison does not establish that the candidate is a final frozen frontend.

The candidate changes sample rate, bandwidth, Mel-bin count, and window length
together. The experiment does not identify which property caused the audible
difference. It also does not establish downstream mapper-performance
improvement or generalization across songs: the evidence covers one excerpt,
three optimization seeds, one optimizer configuration, and the owner's
listening setup. Source resampling, clipping, optimizer stochasticity, and
listening conditions remain possible influences. This report does not record
the CPU model, operating system, or dependency versions; backend and hardware
differences therefore limit runtime comparisons and may affect reproduction.

A controlled follow-up should separate sample rate, bandwidth, Mel-bin count,
and window length before freezing a replacement frontend. The result supports
that narrower comparison; it does not by itself select which individual
parameter changes are necessary.

## Reproduction

The self-contained executable experiment is
`notebooks/mel_frontend_metamer_experiment.ipynb`. Edit `AUDIO_PATH`,
`OUTPUT_DIR`, segment bounds, and optional optimizer controls in its tagged
input cell, then run all cells. Set `DEVICE="cpu"` to reproduce the recorded
device choice; `DEVICE="auto"` may select MPS or CUDA. `OUTPUT_DIR` must not
already exist: the notebook fails instead of overwriting an earlier run. To
execute the current input cell non-interactively:

```sh
uv run --extra mps --group dev jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  notebooks/mel_frontend_metamer_experiment.ipynb \
  --ExecutePreprocessor.timeout=600 \
  --ExecutePreprocessor.kernel_name=python3
```

Use `--extra cuda` instead of `--extra mps` on Linux with NVIDIA CUDA.
