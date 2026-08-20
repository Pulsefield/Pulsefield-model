# Timing v3 Experiment 028: stable256 causal source corpus v2

- Status: TEST completed / positive; durable v2 corpus validated
- Date: 2026-08-16
- Result: `docs/research/timing_v3_experiment_028_result.md`

## Hypothesis

Audio-consensus stable sources can produce a source-disjoint causal corpus whose identity, persistent-jump, and linear-time-ramp variants have analytic transform truth without treating mapper `.osu` redlines or absolute BPM labels as ground truth.

## Root objective

Create a frozen, resumable 256-source, 768-route real-audio corpus for Timing v3 causal validation.

## Goal decomposition

1. Use the frozen full5050 audio-consensus stable-source cohort as the only source list.
2. Render three real-audio variants per source: renderer-matched identity, persistent jump, and linear-time ramp.
3. Keep opaque inference routes physically separate from analytic transform truth.
4. Verify source uniqueness, split separation, route uniqueness, causal map roundtrip, and transform persistence.

## Candidate variants

1. Reuse random full5050 sources: rejected because identity would not be a stable-source control.
2. Use `.osu` redline-derived sources: rejected because mapper timing is not ground truth.
3. Use the frozen audio-consensus stable256 cohort: selected because source qualification is multi-estimator audio consensus and already split into 128 train + 128 untouched holdout.

## Local verification matrix

| Check | Pass condition |
| --- | --- |
| Source count | 256 rows |
| Route count | 768 WAV routes |
| Split separation | no source key appears in both train and untouched holdout |
| Opaque boundary | route manifest exposes exactly schema, route ID, and audio path |
| Transform truth | truth receipt contains only whitelisted source locator/split metadata, relative transform params, and mapping checks |
| Causal maps | every identity/jump/ramp mapping roundtrips within configured tolerance |
| Persistence | jump post-seam rate is constant; ramp endpoint and monotonic direction match plan |
| Render duration | WAV frame count equals the analytic target frame count for every route |

## Selected variant

Generate the full 256×3 corpus from `artifacts/reports/timing/timing_v3_full5050_audio_consensus_constant_sources_256.jsonl` with one frozen renderer and deterministic balanced transforms:

- decode to mono 16 kHz float audio;
- render identity, jump, and ramp through the same STFT/ISTFT variable-rate phase vocoder (`n_fft=1024`, `hop_length=256`, Hann window);
- cycle the relative target-rate ratio over `{0.8, 0.875, 1.125, 1.25}` independently within each split, yielding 32 sources per ratio in train and 32 per ratio in untouched holdout;
- spread jump output-time seam fractions deterministically over the closed interval `[0.30, 0.70]` within each split;
- make jump local rate `1.0` before the seam and the assigned ratio after it;
- make ramp local rate linear in output time from `1.0` to the assigned ratio.
- assign every route an independent cryptographically random opaque ID; the route-to-source/transform assignment exists only in truth-side state and is sealed before holdout inference.

For source duration `D`, ratio `q`, output-time seam fraction `f`, and output duration `T`:

- identity: `T=D`, `s(t)=t`;
- jump: `T=D/(f+q(1-f))`, seam `t_s=fT`, then `s(t)=t` before `t_s` and `s(t)=t_s+q(t-t_s)` after it;
- ramp: `T=2D/(1+q)`, `s(t)=t+(q-1)t^2/(2T)`.

The stored inverse is the analytic inverse of `s(t)`. Roundtrip, endpoint, monotonicity, assigned-rate, and rendered-frame-count checks are mandatory for every route.

## Selection pressure

This variant directly addresses the current failure mode: `.osu` redlines are diagnostic comparators, not truth. The only truth in this corpus is the transform applied to each waveform.

## Minimal change

Update this Experiment Card and write a resumable builder plus all generated state under the ignored workspace artifact directory. Do not change model or inference source.

## Files likely to change

- `docs/research/timing_v3_experiment_028_stable256_causal_corpus.md`
- `artifacts/local/timing_v3/stable256_causal_v2/truth/build_corpus.py.sealed`
- `artifacts/local/timing_v3/stable256_causal_v2/rekey_and_split.py`
- `artifacts/local/timing_v3/stable256_causal_v2/opaque/`
- `artifacts/local/timing_v3/stable256_causal_v2/truth/`
- `artifacts/local/timing_v3/stable256_causal_v2/state/`

## Dataset slice

Frozen cohort: all 128 train sources and all 128 untouched-holdout sources from full5050 audio-consensus stable selection. No smaller execution slice is authorized.

## Baseline / comparator

The prior random-source causal128 corpus remains valid only for causal-warp sensitivity. It is not a stable-source constant-control corpus.

## Primary metric

Generate 256 unique real sources and 768 unique opaque WAV routes with zero validation errors.

## Secondary metric

Balanced deterministic transform assignment across relative-rate ratio families and directions; seam fractions span 0.30–0.70 in both splits.

## Verify command or evaluation procedure

Run the corpus builder directly over all 256 sources. It reports at 64-source boundaries, resumes only from per-source validated state, then validates the summary, route manifests, split truth receipts, WAV headers, analytic maps, and exact manifest field sets. Before holdout prediction, consumers may read only `opaque/train_routes.jsonl`, `opaque/holdout_routes.jsonl`, and `truth/train_truth.jsonl`; holdout truth, combined truth, assignment state, all per-source build state, and the generator are mode-000 sealed.

## Guard check

The builder must not use `.osu`, mapper redline data, confidence scores, or absolute tempo labels. Although the upstream selection receipt contains qualification-only fields, the streaming JSON decoder must immediately project each row through the exact whitelist `{resolved_audio_path, source_key, duration_seconds, split, split_index, row_index}`. Only these six fields may enter a source record; no other input field may be copied into state or truth. Opaque route rows have exactly `{schema, route_id, audio_path}`. Opaque IDs must not be recoverable from a public seed, source index, transform order, filename order, or modification time.

## Qualitative check

Identity variants are rendered through the same STFT/ISTFT phase-vocoder function as jump/ramp variants, not symlinked or copied from source audio.

## Positive signal

The corpus can be consumed by the stable causal evaluator as a full 768-route real-audio test.

## Negative signal

Any duplicate source, split leak, route leak, failed causal mapping, or missing route kills the corpus.

## Kill criteria

- Fewer than 256 usable sources.
- Any `.osu`/redline/BPM-label field is required.
- Any generated route fails mapping or persistence validation.
- Opaque route manifest contains truth fields.

## Expected failure modes

- Source audio missing from local dataset.
- Phase-vocoder render failure on long or unusual files.
- Interrupted output before a source-level completion record is atomically committed.
- Accidental leakage of upstream qualification fields into the truth side.

## Expected runtime / runtime budget

Full 256-source generation is expected to take tens of minutes on the local machine, depending on CPU contention. Stop on the first validation error. An interruption may resume from atomically committed, revalidated source state, but it may not weaken or skip any route check.

## Confounders

The three rendered variants from a source share audio content, so route-level scores are not source-independent. Downstream evaluation must aggregate by source and split.

## Result interpretation plan

If generation passes, the corpus becomes the stable-source causal validation set. If downstream identity still fails, the failure is attributable to Timing v3 extraction/selection rather than mapper redline disagreement.
