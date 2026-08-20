# Timing v3 Experiment 028 result: durable stable256 causal corpus v2

## Mode

- Mode: executor
- Experiment Card existed before execution: yes
- Route entering executor: TEST
- Stopped and returned to planner mode: no
- Source snapshot / evidence grade: frozen full5050 audio-consensus selection; direct local execution evidence

## Experiment

- Experiment Card: `docs/research/timing_v3_experiment_028_stable256_causal_corpus.md`
- Date: 2026-08-16
- Root objective: create a resumable 256-source, 768-route real-audio causal corpus with analytic transform truth.
- Goal decomposition: exact source projection; renderer-matched identity/jump/ramp; opaque/truth separation; full mapping and WAV validation.
- Candidate variants considered: random sources, mapper-timing-derived sources, and the selected frozen audio-consensus stable256 cohort.
- Selected variant: 16 kHz mono variable-rate phase vocoder with deterministic balanced relative-rate assignments.
- Selection pressure from card: obtain transform truth from the waveform operation rather than from mapper timing annotations or absolute tempo labels.
- Dataset slice: all 128 train plus all 128 untouched-holdout real sources; no smaller execution slice.
- Baseline / comparator: prior random-source causal corpus is not a stable-source control.
- Runtime: 61.759 seconds for generation plus complete validation with two workers; 0.290 seconds for an independent validate-only restart; 0.339 seconds for mechanical random-ID rekeying, physical split, and sealing without rerendering.
- Files changed: Experiment Card, this result log, sealed ignored corpus builder, rekey/split utility, and generated corpus under `artifacts/local/timing_v3/stable256_causal_v2/`.
- Read-only context files consulted: `README.md`, the research-triage skill, the frozen stable256 source receipt, and relevant audio-loading code.

## Result

- Primary metric value: 256/256 unique real sources and 768/768 unique WAV routes; zero validation errors.
- Secondary metric value: identity/jump/linear-ramp each 256; every relative-rate ratio has 32 jump and 32 ramp sources in each split; jump seams span 0.30 to 0.70 with 128 unique values per split.
- Verify command / result: the pre-seal builder validate-only pass completed with 256 sources and 768 routes; post-rekey physical checks found 384 train routes, 384 holdout routes, 768 unique random opaque IDs, 768 WAVs, and no legacy sequential route filenames.
- Guard command / result: exact-whitelist input projection and exact manifest-field validation passed; public deterministic route assignment was removed; all WAV modification times were equalized and their inodes were recreated in a random order; holdout truth, combined truth, assignment state, all 256 per-source state files and their directory, and the generator are mode-000 sealed.
- Qualitative observations: identity traverses the same STFT/ISTFT renderer as jump and ramp; jump and ramp use one continuous phase accumulator rather than audio splices.
- Positive signal observed: the full durable corpus is ready for train-supervised and physically sealed holdout causal Timing v3 evaluation.
- Negative signal observed: none for corpus construction.
- Kill criteria triggered: none.

## Closed-Loop Outcome

- Local verification outcomes: all 768 analytic maps passed monotonicity, endpoint, inverse roundtrip, and assigned-rate checks; all 768 WAV headers and target frame counts passed.
- Selected variant passed local verification: yes, after replacing the initially reversible sequential route assignment with random opaque IDs.
- Subgoals satisfied: exact stable256 cohort; full three-way rendering; opaque/truth separation; resume validation; balanced parameters; zero-error materialization.
- Subgoals failed: none within corpus construction.
- Selection pressure confirmed: the corpus has causal relative-transform truth without importing upstream qualification values into route state or truth.
- Selection pressure contradicted: no.
- Variant should be kept, mutated, or killed: kept as the causal corpus.
- Next-loop action: TEST the frozen Timing v3 selector on all 768 opaque routes and evaluate only after prediction freeze.

## Commands

- Commands run: full builder; independent validate-only restart; random-ID mechanical rekey/split; WAV and manifest counts; exact-field, permission, filename, and metadata checks.
- Command failures: one pre-execution dynamic-import helper failed because its ad-hoc module was not registered; direct module import then validated all 768 analytic maps. Corpus execution was unaffected.
- Reproduction command: the durable generator is sealed at `artifacts/local/timing_v3/stable256_causal_v2/truth/build_corpus.py.sealed`; reproduction is intentionally unavailable to pre-freeze consumers.

## Verification / Failure Modes

- Checks performed: 256 source uniqueness and split separation; 768 route uniqueness; whitelist-only source state; exact opaque fields; transform/ratio/seam balance; analytic duration and map checks; rendered WAV sample rate, channel count, subtype, and frame count; resumable state revalidation; random opaque-ID uniqueness; equalized file metadata; 384/384 physical train/holdout split; mode-000 sealed holdout labels.
- Failed checks: none in the authoritative run.
- Suspected confounders: the three routes from one source share musical content, so downstream statistics must aggregate by source and split.
- Expected failure modes observed: none.
- Unexpected failure modes: the first materialization used a public deterministic route shuffle, which made labels reconstructable from route IDs. It was corrected before any model consumer received the corpus by mechanically randomizing all 768 IDs and sealing the assignment.
- Reproducibility notes: waveform transforms and source-level commits are reproducible; opaque route IDs are intentionally random and their assignment receipt is truth-side sealed.
- Evidence gaps: construction validity does not establish Timing v3 classification, boundary, ramp, offset, or runtime accuracy.

## Interpretation

- What the result supports: a full, real-audio, causal evaluation substrate whose labels are the applied relative time maps.
- What the result does not support: any absolute-tempo ground-truth claim or any Timing v3 quality target.
- Alternative explanations: downstream route-level gains may reflect shared source content unless evaluated source-disjoint and aggregated by source.
- Positive / negative / ambiguous classification: positive for corpus construction.
- Recommended next step: TEST the frozen evaluator on all 768 opaque routes before opening the separated truth receipt.
- Human owner decision: pending.
