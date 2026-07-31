# Timing Module

This module turns audio into a compact osu!-style timing grid for the rest of
the Pulsefield generation stack.

The current path is:

```text
audio
  -> BeatThis beat/downbeat probabilities at 50 Hz
  -> GridFitter compact timing segments
  -> optional dense timing features for mapper/control conditioning
```

The intended product role is real-time beatmap timing support: it should provide
a usable beat grid quickly enough that the mapper can start from musical timing
instead of raw seconds. It is not claimed to be a final human-quality timing
oracle.

## Current Claims

### Good enough for real-time timing support

The strongest defensible version of the claim is:

> The timing module is good enough and fast enough to provide a real-time timing
> prior for osu!mania-like beatmap generation on local bounded eval slices.

Evidence:

- In the original 20-map mixed slice, the module reported mean phase error
  `36.14 ms`; in the 20-map targeted multi-BPM slice, it reported `46.15 ms`.
- After split-candidate and alias-selection work, the same local slices improved
  to `33.56 ms` mixed and `43.32 ms` targeted multi-BPM.
- On the later 100-map final0 guard slice, semantic alias promotion improved
  mean phase error from `48.69 ms` to `42.54 ms`, p90 phase error from
  `74.35 ms` to `64.79 ms`, and mean alias-aware BPM MAE from `2.40` to
  `2.05`.
- The output is compact. The 100-map guard slice averaged about `2.10`
  predicted segments after alias semantic promotion, so the result is usable as
  a timing grid rather than a dense overfit redline trace.

Claim boundary:

- This supports a fast timing prior for generation and playback sync.
- It does not prove production-quality musical timing for every song.
- `.osu` red timing is a noisy comparator; it is useful for diagnostics, not a
  final ground-truth label.

### Real multi-BPM recall is high on audited labels

The multi-BPM claim should be read against the web-audited label table, not raw
red timing alone.

The audit split 200 local redline multi-BPM candidates into:

- `58` likely real multi-BPM audio rows.
- `97` likely mapper or alias artifact rows.
- `45` ambiguous or unresolved rows kept out of binary metrics.

On the 155 confirmed binary rows, the timing module's default post-refinement
multi-family read was:

| metric | value |
|---|---:|
| confirmed real recall | `47/58 = 81.0%` |
| confirmed artifact false-positive rate | `29/97 = 29.9%` |
| confirmed binary accuracy | `74.2%` |
| likely-real post-refinement collapse | `0/58 = 0.0%` |

Interpretation:

- The positive signal is real: likely real multi-BPM rows are much more often
  predicted as multi-family than confirmed artifacts.
- The main miss mode is not post-refinement collapsing true multi-BPM grids.
  In the 58 likely-real rows, default and no-refine both recognized `47`.
- The current weak spot is distinguishing true audio tempo changes from mapper
  redline artifacts. A `29.9%` artifact false-positive rate is too high for a
  production real-vs-artifact classifier.

Known weak class:

- Songs with a section whose BPM slowly varies, rubato timing, or dense local
  tempo drift are weak for the current module. The fitted representation is a
  compact piecewise-constant `TimingSegment` grid, with default minimum segment
  duration and merge/refinement rules. It is designed to recover stable timing
  regions, not continuous tempo curves.
- This is why the claim is "high recall for source-verified real multi-BPM
  candidates", not "robust handling of all tempo variation".

The supporting 200-row label index was a local evaluation artifact and is
intentionally not retained in the repository.

### Phase error is acceptable as a mapper prior

The phase-error claim is defensible only with scope:

> The observed phase error is acceptable for an osu!mania mapping prior and
> interactive generation loop, especially when the error behaves like a stable
> per-map phase offset rather than local drift.

Evidence:

- Local mixed and multi-BPM slices are usually in the `33-50 ms` mean phase-error
  range after current fitting changes.
- The 100-map guard slice reached `42.54 ms` mean phase error and `64.79 ms`
  p90 phase error after alias semantic promotion.
- The web-audited real multi-BPM bucket reported mean phase error `50.28 ms`,
  median `46.70 ms`, and p90 `67.52 ms`.
- Qualitative eval logs repeatedly show that many failures are segment or alias
  interpretation errors, not complete loss of beat phase. For example,
  Ops:Code-Rapture- had low alias-aware BPM error but high phase error, pointing
  to phase/segment placement rather than raw beat detection.

Claim boundary:

- This is acceptable for providing timing context to a mapper and for fast
  preview/playback experiments.
- It is not enough to claim final export-quality osu! timing without additional
  drift checks, listening checks, and per-map inspection.
- The repo does not yet have a dedicated "phase offset consistency across the
  whole map" metric. The current justification is based on compact fitted grids,
  mean/p90 phase diagnostics, and qualitative failure analysis. A future audit
  should measure per-map phase-error variance or local drift directly.

### Fast enough for the real-time loop

The single-pass timing path is fast once the model is loaded.

Evidence:

- 20-map mixed slice: mean `2.05 s/map`, p90 `2.40 s/map`.
- 20-map targeted multi-BPM slice: mean `2.37 s/map`, p90 `2.83 s/map`.
- 100-map final0 baseline: mean `2.25 s/map`, with mean prediction `1.44 s`
  and mean fitting `0.81 s`.
- Checkpoint divergence audit: each checkpoint pass averaged about
  `2.40-2.44 s/map`.

Interpretation:

- A single pass typically lands in the `2-3 s` range on the local MPS setup for
  short-to-medium songs after model load.
- Super-timing shifted runs are diagnostic and slower because they run multiple
  shifted passes. They should not be used as the headline latency number unless
  the product deliberately pays that cost.

## API Notes

Important entry points:

- `BeatThisTimingProvider`: loads BeatThis and returns frame-level beat and
  downbeat probabilities.
- `GridFitter`: converts frame probabilities into compact timing segments.
- `fit_audio_file`: command/API helper that combines provider and fitter.
- `render_dense_timing_v2`: renders fitted segments into dense mapper/control
  timing features.

Example CLI:

```sh
uv run python -m pulsefield_model.timing.fit_audio path/to/audio.mp3 --device mps --json
```

Use `--super-timing-shifts` only when you want diagnostic shifted evidence, not
the normal latency path.
