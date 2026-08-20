# Timing v3 Experiment 029: fresh128 causal external corpus

- Status: TEST completed / positive; external corpus sealed
- Date: 2026-08-16
- Result: `docs/research/timing_v3_experiment_029_result.md`

## Hypothesis

The frozen full5050 three-view audio-consensus results contain at least 128 additional stable real-audio sources, disjoint from Experiment 028 stable256, that can support a fully sealed external causal evaluation corpus with analytic identity, jump, and linear-ramp truth.

## Root objective

Create one new source-disjoint 128-source, 384-route external corpus without exposing any route label before prediction freeze.

## Goal decomposition

1. Reapply the frozen Experiment 027 accepted-constant and confidence ranking to the full5050 result receipt.
2. Exclude every source key in `timing_v3_full5050_audio_consensus_constant_sources_256.jsonl` before selecting the next 128 unique sources.
3. Render identity, persistent jump, and output-time-linear ramp for every selected source with the Experiment 028 renderer.
4. Use independent random opaque IDs, neutralize file-order metadata, and expose only 384 opaque route rows.
5. Seal source selection, route assignment, all truth, all state, and the builder at mode 000 before publishing the opaque manifest.

## Candidate variants

1. Reuse any stable256 source: rejected because it violates external source disjointness.
2. Select arbitrary remaining full5050 sources: rejected because identity would not be a stable-source control.
3. Select the next 128 unique Experiment 027 accepted-constant sources after excluding stable256: selected because it preserves the frozen audio-only qualification and ranking.

## Local verification matrix

| Check | Pass condition |
| --- | --- |
| Available fresh sources | at least 128 after exclusion |
| Source overlap | zero source-key overlap with stable256 |
| Source count | exactly 128 unique existing audio paths |
| Route count | exactly 384 unique WAVs and opaque IDs |
| Balance | identity/jump/ramp each 128; each nonidentity ratio 32 |
| Analytic maps | all duration, endpoint, monotonicity, inverse-roundtrip, and assigned-rate checks pass |
| Opaque fields | exactly `schema`, `route_id`, `audio_path` |
| Seal boundary | source receipt, assignment, truth, state, and builder are mode 000 before opaque publication |

## Selected variant

Use the same mono 16 kHz STFT/ISTFT variable-rate phase vocoder as Experiment 028 (`n_fft=1024`, `hop_length=256`, Hann window). Cycle relative target-rate ratios over `{0.8, 0.875, 1.125, 1.25}` with 32 sources per ratio. Spread jump output-time seam fractions deterministically over `[0.30, 0.70]`. Ramp local rate changes linearly in output time from `1.0` to the assigned ratio. Every route ID is generated independently with `secrets.token_hex` and its mapping is sealed.

## Selection pressure

The selected corpus must be genuinely source-disjoint, use only the frozen audio-consensus stable qualification, and make pre-freeze label recovery impossible from paths, IDs, ordering, or filesystem timestamps.

## Minimal change

Write one self-sealing resumable generator and its output under `artifacts/local/timing_v3/fresh128_causal_final_v2/`. Do not alter model code, Experiment 028 artifacts, or `fresh128_causal_final_v1`.

## Files likely to change

- `docs/research/timing_v3_experiment_029_fresh128_causal_external.md`
- `artifacts/local/timing_v3/fresh128_causal_final_v2/build_corpus.py` during execution
- `artifacts/local/timing_v3/fresh128_causal_final_v2/opaque/`
- `artifacts/local/timing_v3/fresh128_causal_final_v2/sealed/`

## Dataset slice

Exactly the next 128 unique accepted-constant sources from the frozen full5050 consensus ranking after excluding all 256 Experiment 028 source keys. No smaller execution slice is authorized.

## Baseline / comparator

Experiment 028 stable256 is the renderer/protocol comparator but is excluded from the source set.

## Primary metric

128 unique fresh sources and 384 valid opaque WAV routes with zero validation error and zero overlap with stable256.

## Secondary metric

Balanced ratio families, 128 unique jump seam fractions spanning 0.30–0.70, and exact rendered frame counts.

## Verify command or evaluation procedure

Run the builder directly over all 128 sources with two workers. It reports every 32 completed sources. After generation, verify only public opaque counts/fields/WAV headers and permission metadata; do not open sealed truth.

## Guard check

Source qualification may read only row locator/duration, `consensus.accepted_constant_source`, and `consensus.confidence_score`; it must not read or persist absolute tempo estimates, mapper timing, titles, artists, or confidence payloads. The selected source receipt persists only locator, duration, row index, and external split index. Opaque rows contain only the exact three allowed fields.

## Qualitative check

Identity uses the same phase-vocoder function as jump and ramp. No output is copied from Experiment 028 and no source is shared with it.

## Positive signal

The sealed corpus can be handed to a fresh external evaluator as 384 opaque routes with no per-route truth available before freeze.

## Negative signal

Any source overlap, missing audio, reversible ID, metadata-order leak, failed analytic map, malformed WAV, unsealed truth/state/builder, or fewer than 384 routes kills the corpus.

## Kill criteria

- Fewer than 128 qualified source-disjoint candidates.
- Any stable256 source key appears in the new selection.
- Any route or mapping validation failure.
- Any route label remains readable outside the sealed directory.

## Expected failure modes

- Missing source audio despite a completed consensus row.
- Decode or phase-vocoder failure on an unusual file.
- Interrupted generation before source-level state is committed.
- Accidental publication of assignment-bearing state.

## Expected runtime / runtime budget

The complete 128-source render is expected to finish within several minutes on the current machine. Stop on the first validation error; an interrupted run may resume from validated source state. Report progress every 32 sources.

## Confounders

The three transforms for one source share musical content, so downstream results must aggregate by source. Audio-consensus views are correlated same-waveform estimators, not independent recordings.

## Result interpretation plan

Passing construction validates the external evaluation substrate only. It does not establish Timing v3 accuracy, absolute tempo truth, or runtime targets. The next action is one frozen-prediction external evaluation over all 384 opaque routes, followed by controlled unsealing.

## Result log template

- Source count / overlap:
- Route and transform counts:
- Ratio and seam coverage:
- Mapping / WAV validation:
- Seal validation:
- Runtime:
- Kill criteria:
- Next-loop action:

## Next-loop action

If all construction gates pass: TEST the frozen high-resolution boundary model on all 384 opaque routes. Otherwise: KILL the corpus without weakening selection or validation gates.

## Closest analogies and novelty layer

Closest analogies are source-disjoint external evaluation, causal time-warp augmentation, and blinded challenge-set packaging. This is protocol engineering, not a novelty claim.
