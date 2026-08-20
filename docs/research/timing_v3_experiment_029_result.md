# Timing v3 Experiment 029 result: fresh128 causal external corpus

## Mode

- Mode: executor
- Experiment Card existed before execution: yes
- Route entering executor: TEST
- Stopped and returned to planner mode: no
- Source evidence: frozen full5050 three-view audio-consensus result receipt

## Experiment

- Experiment Card: `docs/research/timing_v3_experiment_029_fresh128_causal_external.md`
- Date: 2026-08-16
- Root objective: create one new source-disjoint 128-source, 384-route external causal corpus with no readable pre-freeze truth.
- Selected variant: next 128 frozen accepted-constant sources after excluding all stable256 source keys; Experiment 028 renderer and relative-transform plan; independent random opaque IDs.
- Dataset slice: all 128 selected external sources; no smaller run.
- Runtime: 36.485 seconds for selection, 128×3 rendering, validation, metadata neutralization, publication, and sealing with two workers.
- Artifact root: `artifacts/local/timing_v3/fresh128_causal_final_v2/`

## Result

- Primary metric: 128/128 unique fresh sources, 0 overlap with stable256, 384/384 opaque WAV routes, 0 validation errors.
- Transform counts: identity 128, jump 128, linear ramp 128.
- Balance: every jump/ramp relative-rate ratio has 32 sources; jump seams have 128 unique values spanning 0.30–0.70.
- Mapping/WAV validation: all routes passed analytic duration, monotonicity, endpoint, inverse-roundtrip, assigned-rate, output-frame-count, 16 kHz mono PCM16 WAV, and finite-audio checks.
- Opaque validation: 384 unique random IDs; every public row has exactly `schema`, `route_id`, and `audio_path`; no legacy sequential ID; all WAV modification times equalized and inodes recreated in random order.
- Seal validation: public builder and work directory are absent; selection, assignment, truth, all state, builder, and bytecode are inside a non-readable, non-traversable mode-000 sealed directory.
- Kill criteria triggered: none.

## Closed-loop outcome

- Selected variant passed local verification: yes.
- Subgoals satisfied: source disjointness, complete rendering, balanced transforms, analytic truth, opaque publication, metadata neutralization, and physical sealing.
- Selection pressure confirmed: the public artifact exposes routes but no recoverable route-to-source or route-to-transform assignment.
- Next-loop action: TEST the frozen model once over all 384 opaque routes, freeze predictions, then authorize controlled truth unsealing.

## Commands

- Full run: `uv run --extra mps python artifacts/local/timing_v3/fresh128_causal_final_v2/build_corpus.py --workers 2` before the builder self-sealed.
- Public verification: opaque line count, exact fields, random-ID format and uniqueness, WAV count/header checks, uniform modification-time check, absence of work/builder, and sealed directory permission checks.
- Command failures: none in the authoritative run.

## Interpretation

- Supported: the external causal evaluation substrate is complete and source-disjoint from stable256.
- Not supported: Timing v3 accuracy, absolute tempo truth, boundary accuracy, phase/offset accuracy, or runtime targets.
- Confounder: three transformed routes from one source share content; downstream reporting must aggregate by source.
- Classification: positive for corpus construction.
- Human owner decision: pending external frozen-prediction run.
