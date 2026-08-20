# Timing v3 Experiment 029: high-resolution BeatThis+raw seam boundary v2

## Mode

- Mode: planner
- Route: TEST
- Source idea: Rebuild the lost durable high-resolution seam experiment for the stable causal jump corpus.
- Acceptance source, if any: User directive on 2026-08-16: use pointwise 0.1 s candidates, BeatThis beat/downbeat probabilities plus raw multiband flux local relative-window features, no synthetic or small-sample experiment, no absolute-time or identity leakage, train on at least 128 real jump routes, and evaluate on at least 128 source-disjoint real jump routes. Follow-up directive froze causal corpus parameters as train/holdout `128/128`, jump/ramp ratios `{0.8, 0.875, 1.125, 1.25}`, seam fraction `0.30..0.70`, ratio-bucket reporting, seam-only primary gate `holdout jump pass >=103/128`, and joint class/ratio/ramp reporting as separate from the seam-only gate.
- Source snapshot / evidence grade: Local repository context from README, Timing v3 current code/evaluation, Experiment 027 audio-consensus source screen, Experiment 028 stable256 causal corpus card, and owner-reported previous real `dev512` result for the frozen highres MLP (`512/512 <=1s`). The route/truth manifest contract from `causal_corpus_rebuild` is still pending, so execution is contract-blocked until schemas are bound.

## Hypothesis

A pointwise boundary MLP over 0.1 s candidate centers can recover persistent-jump seam times better than the current coarse Timing v3 jump proposal path when it uses the frozen 100-dimensional local feature contract from BeatThis beat/downbeat and raw flux around each candidate. The model must not see absolute candidate time, normalized position, route duration, path, route ID, source ID, `.osu`, redlines, metadata labels, ratio, seam fraction, or production candidate-lane identity.

## Root Objective

Build a durable, fail-closed experiment runner that is ready to train and evaluate on the forthcoming stable256 causal jump corpus without starting a data run before the route/truth contract is frozen.

## Goal Decomposition

- Subgoal 1: Define a contract-bound loader for opaque routes plus separate transform truth, with route/truth schemas supplied explicitly when the corpus contract is available.
- Subgoal 2: Generate 0.1 s pointwise seam candidates for real jump routes using an 8 s boundary margin, because the frozen far windows require `[-8,-4]` and `[4,8]` seconds without zero padding or edge shortcuts.
- Subgoal 3: Enforce hard leakage guards: no absolute time, normalized position, duration, path, route, source, `.osu`, redline, or metadata identity field enters the model feature matrix or feature names.
- Subgoal 4: Refuse to train unless at least 128 real jump routes are available in the training split.
- Subgoal 5: Refuse final evaluation unless at least 128 real jump routes are source-disjoint from the training sources and report every frozen jump ratio bucket separately.

## Candidate Variants

- Variant A: Reuse current Timing v3 jump candidates and score their seams. Rejected because the failure being isolated is high-resolution seam localization, not the existing candidate generator. Also rejected because current production code contains absolute-ms proposal lanes that are not allowed in this experiment.
- Variant B: Pointwise classifier with absolute candidate time, normalized position, duration, and route identity features. Rejected because it can learn corpus/rendering position priors instead of audio evidence.
- Variant C: Selected. Frozen pointwise MLP ranker using the previous 100-dimensional local feature contract, architecture `100 -> 96 ReLU -> 48 ReLU -> 1`, per-song masked softmax cross entropy, 450 epochs, AdamW `lr=1.5e-3`, `weight_decay=1e-3`, and fixed seed.
- Variant D: Sequence model over full-route feature tracks. Rejected for this loop because it adds model capacity, batching, and runtime risk before proving the local signal is present.

## Local Verification Matrix

- Variant A: Static review says it cannot test dense seam candidates independently of current proposal recall.
- Variant B: Leakage guard fails by construction.
- Variant C: Static self-check must prove feature names and plan output contain no forbidden model inputs, route/truth loading is fail-closed, min-count/source-disjoint gates block small fixtures, candidate margin is 8 s, and the MLP config is frozen.
- Variant D: Scope review rejects it until Variant C produces a positive or interpretable negative signal.

## Selected Variant

- Selected: Variant C.
- Rejected: A, B, and D.
- Why this is the smallest useful test: It changes only the evaluation runner and a local research model, leaves production Timing v3 untouched, and restores the owner-reported highres MLP rather than downgrading to an unvalidated classifier.

## Selection Pressure

- Primary pressure: At least 103 of 128 source-disjoint holdout jump routes pass the seam-only tolerance gate, with ratio-bucket metrics reported for each of `{0.8, 0.875, 1.125, 1.25}`.
- Guard pressure: No forbidden feature names or model inputs; no `.osu`, redline, mapper labels, path, route ID, source ID, absolute time, normalized position, or route duration in the model feature matrix.
- Runtime pressure: Feature extraction is resumable through ignored local artifacts and can run route-by-route once the corpus exists.
- Kill pressure: Any count shortfall, source leak, schema mismatch, forbidden feature, or route/truth mismatch blocks execution.

## Research Question

Can the frozen 100-feature high-resolution seam MLP recover jump boundary placement from BeatThis beat/downbeat and raw multiband flux evidence alone on real transformed audio?

## Closest Analogies / Novelty Layer

- Closest analogies: Pointwise change-point classification, local audio-feature boundary detection, logistic ranking over dense candidates, and source-disjoint corpus evaluation.
- Relevant taxonomy bucket: Evaluation-time auxiliary model and seam-localization experiment.
- Novelty layer, if any: None claimed.
- Representation novelty vs engineering variation: Engineering variation around a timing evaluation probe, not a new Timing v3 representation.

## Minimal Change

Add one plan-only-by-default durable module plus focused tests. The module owns manifest preflight, frozen 100-dimensional feature extraction, the `100->96->48->1` MLP trainer, model serialization, final evaluation, and static self-check. It does not change production inference, does not call Timing v3 candidate generation, and does not consume any production lane identity.

## Files Likely to Change

- `docs/research/timing_v3_experiment_029_highres_boundary_v2.md`
- `src/pulsefield_model/timing/evaluation/highres_boundary_v2.py`
- `tests/timing/test_timing_v3_highres_boundary_v2.py`

## Read-Only Context Files

- `README.md`
- `.agents/skills/research-triage/SKILL.md`
- `docs/research/timing_v3_experiment_027_full5050_audio_consensus_constant_sources.md`
- `docs/research/timing_v3_experiment_028_stable256_causal_corpus.md`
- `src/pulsefield_model/timing/v3/inference.py`
- `src/pulsefield_model/timing/v3/audio_evidence.py`
- `src/pulsefield_model/timing/evaluation/full5050_audio_consensus.py`
- `src/pulsefield_model/timing/evaluation/full5050_shadow_runner.py`

## Dataset Slice

Deferred until `causal_corpus_rebuild` publishes the route/truth contract. The adapter is implemented against the expected Exp028 shape: route rows `{schema, route_id, audio_path}` and truth rows with at least `{route_id, source_key, split, transform_class, ratio, seam_seconds}`. The intended slice is the stable256 causal corpus, jump transform only: exactly 128 train jump routes and exactly 128 holdout jump routes when the frozen corpus is complete, source-disjoint between train and holdout. Jump and ramp ratios are frozen to `{0.8, 0.875, 1.125, 1.25}` and seam fraction to `0.30..0.70`; the seam runner consumes the realized seam time and ratio from transform truth for labels and reporting but does not feed either ratio, absolute seam time, seam fraction, or route duration into the model.

No synthetic data, tiny pilot, mechanism row, or small-sample run is authorized as experiment evidence.

## Baseline / Comparator

Current Timing v3 jump behavior and any coarse seam estimate from existing evaluators are diagnostic comparators only. The first accepted result for this card is the source-disjoint real-jump holdout metric from this runner.

## Primary Metric

Seam-only holdout pass count within `1.0 s`, requiring at least `103/128` source-disjoint jump routes. The selected candidate is the maximum-logit pointwise candidate per route. Median and p90 absolute seam error in milliseconds are reported but do not replace the pass-count gate.

## Secondary Metric

Within-100 ms, within-250 ms, within-500 ms, and within-1000 ms rates; per-ratio count/error/pass-rate metrics for ratios `0.8`, `0.875`, `1.125`, and `1.25`; positive-candidate recall availability; train/final route counts; train/final source counts; candidate count distribution; feature extraction failure counts. Class, ratio, and ramp behavior use existing production output or a separately frozen translation-invariant evaluator and are reported as joint diagnostics only; they cannot make this seam-only experiment pass.

## Verify Command / Evaluation Procedure

Static checks now:

```sh
uv run --extra mps --group dev pytest -q tests/timing/test_timing_v3_highres_boundary_v2.py
uv run --extra mps python -m pulsefield_model.timing.evaluation.highres_boundary_v2 --self-check
uv run --extra mps python -m pulsefield_model.timing.evaluation.highres_boundary_v2
```

Deferred after `causal_corpus_rebuild` contract is available:

```sh
uv run --extra mps python -m pulsefield_model.timing.evaluation.highres_boundary_v2 \
  --preflight \
  --route-manifest <contract route manifest> \
  --truth-receipt <contract truth receipt> \
  --route-schema <frozen route schema> \
  --truth-schema <frozen truth schema>

uv run --extra mps python -m pulsefield_model.timing.evaluation.highres_boundary_v2 \
  --run \
  --route-manifest <contract route manifest> \
  --truth-receipt <contract truth receipt> \
  --route-schema <frozen route schema> \
  --truth-schema <frozen truth schema>
```

## Guard Check

The runner defaults to plan-only and does not open the corpus unless `--preflight` or `--run` is passed. `--run` requires explicit route/truth schema names, exact route/truth ID matching, no forbidden `.osu`/redline fields, at least 128 training jump routes, at least 128 source-disjoint final jump routes, and frozen-ratio reporting. The feature contract is exactly 100 dimensions: 10 signals `{beat, downbeat, abs_grad_beat, abs_grad_downbeat, raw_sum, raw_max, raw_low, raw_mid, raw_high, abs_grad_raw_sum}` times 10 derived local-relative features. Every signal is robust-normalized per route by median and `p90-p10`, then sampled at 50 ms. The feature contract rejects forbidden tokens in feature names and never includes candidate time, normalized position, duration, path, route ID, source ID, ratio, seam fraction, or labels in `X`.

## Qualitative Check

After a valid run, inspect aggregate error histograms and the highest-error routes by anonymized route index only. Do not listen manually or use source/path names to tune thresholds.

## Positive Signal

The deferred full run completes with at least 128 training jump routes and at least 128 source-disjoint final jump routes, no leakage guard failures, per-ratio metrics for all frozen ratios, and at least 103 of 128 holdout jump routes within 1.0 s.

## Negative Signal

The contract cannot supply enough real source-disjoint jump routes, any frozen ratio bucket is missing from reporting, feature extraction fails on a large fraction of routes, or fewer than 103 of 128 holdout jump routes are within 1.0 s.

## Kill Criteria

- Route/truth schemas remain unavailable or incompatible.
- Fewer than 128 real jump routes are available for training.
- Fewer than 128 source-disjoint real jump routes are available for final evaluation.
- Any forbidden feature or identity/time/duration leak reaches model input.
- Any `.osu`, redline, mapper label, path identity, route ID, or source ID is needed to improve the model.
- Final metrics are reported from a synthetic, tiny, or non-source-disjoint slice.

## Expected Failure Modes

- The causal corpus contract changes field names beyond the loader aliases, requiring a small contract adapter patch before execution.
- BeatThis caches are missing for rendered routes.
- Raw mel extraction is expensive for the full rendered corpus.
- The seam is near the edge of the feature margin and has no eligible 0.1 s candidate.
- Local evidence is insufficient because phase-vocoder seams are not strongly visible in BeatThis or raw flux.

## Confounders

The route is rendered from a known source transform, so the truth seam is analytic but the audio content is still shared across transform variants from the same source. Evaluation must aggregate by source-disjoint split, not by route count alone.

## Expected Runtime / Runtime Budget

Current task: card, implementation, static tests, and plan/self-check only. Deferred run should be treated as a data run and scheduled separately after contract freeze. Stop immediately on preflight failure.

## Result Interpretation Plan

- Positive result would suggest: local BeatThis+raw seam evidence is worth integrating into a later Timing v3 boundary proposal or scoring mutation.
- Negative result would suggest: the seam-localization signal is weak under local pointwise features, or the transform/corpus does not expose the boundary acoustically.
- Ambiguous result would require: reporting failure buckets without adding absolute-time or identity features, then writing a separate mutation card.
- Human owner decides: whether the error distribution is worth productizing.
- Next-loop action if positive: design a production-safe candidate/scoring mutation that uses the same allowed feature family.
- Next-loop action if negative: `KILL` this high-resolution local ranker and test a different non-leaky boundary signal.
- Next-loop action if ambiguous: wait for failure-bucket review and create a narrower follow-up card.

## Result Log Template

- Experiment: timing_v3_exp029_highres_boundary_v2
- Date:
- Commit / run id:
- Dataset slice:
- Baseline / comparator:
- Runtime:
- Primary metric value:
- Secondary metric value:
- Verify command / result:
- Guard command / result:
- Qualitative observations:
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
- Checks performed:
- Failed checks:
- Suspected confounders:
- Selected variant: Variant C
- Candidate variants rejected before execution: A, B, D
- Local verification outcomes:
- Selection pressure observed:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes, for static self-checks and preflight only until the route/truth contract is available
- Closed loop complete: yes
- Remaining ambiguity: `causal_corpus_rebuild` route/truth manifest schemas are not yet frozen.

## Next-Loop Action

- If positive: create an integration card for a production-safe seam proposal or selector mutation.
- If negative: return `KILL` and do not add forbidden priors.
- If ambiguous: summarize failure buckets and write one bounded mutation card.

## Novelty Notes

- Closest analogies: Pointwise boundary classification and source-disjoint change-point evaluation.
- Novelty layer, if any: None claimed.
- Representation novelty vs engineering variation: Engineering variation in a research evaluator.
