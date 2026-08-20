# Timing v3 Experiment 027: full5050 audio consensus source screens

## Mode

- Mode: planner
- Route: TEST
- Source idea: Select real constant-like and natural persistent-change sources from full5050 using only audio-derived estimators before building the next causal corpora.
- Acceptance source, if any: User directive on 2026-08-16: `.osu` redlines are not ground truth, mapper redlines may encode intent, 1 ms quantization adds small error, and BPM should be interpreted from multiple audio sources rather than one `.osu` redline. :codex-annotation{index="1"}
- Source snapshot / evidence grade: Local full5050 manifest plus three same-waveform estimators. Evidence is audio-only but correlated, not independent-recording evidence.

## Hypothesis

A strict audio-only stability screen over BeatThis probabilities, raw log-mel flux, and BeatNet model 3 offline DBN can find at least 256 full5050 sources whose tempo is a defensible constant octave-family approximation. Within each view, stability and persistent-change detection use that view's direct BPM trajectory, so a sustained 2× or 1/2× change is not folded away. Across views, constant consensus still allows octave-family equivalence because different trackers may choose different tactus. The same run can also report a natural persistent-change cohort where all three views agree on a sustained direct-BPM change boundary and direct left/right log2 ratio. The experiment must not read mapper labels, redlines, metadata BPM, title, or artist.

## Root Objective

Produce source lists for new causal constant-control and natural-change corpora where tempo evidence is audio-estimator consensus, not a mapper/redline oracle.

## Goal Decomposition

- Subgoal 1: Project `artifacts/reports/timing/timing_v3_labels_v1.jsonl` into locator-only rows: row index, audio path, cache key, duration, and cache status.
- Subgoal 2: For each source, estimate full-track and sliding-window direct BPM plus folded octave-family independently for BeatThis probabilities, raw log-mel flux, and BeatNet model 3 DBN beat times.
- Subgoal 3: Accept constant sources only when every view has no persistent direct-BPM change and all three global octave-families agree.
- Subgoal 4: Independently report natural persistent-change sources when every view has a persistent direct-BPM change, the three boundary intervals intersect, and the three signed direct left/right log2 ratios agree within the frozen tolerance. Ratios are not folded before comparison; sustained ±1-octave ratio changes are valid changes.
- Subgoal 5: Before consensus full run, freeze a complete locator-bound BeatNet model 3 DBN events JSONL with exactly one row per full5050 locator row. The consensus runner must fail closed if that file is missing, incomplete, or has a completed row with an empty event stream.

## Candidate Variants

- Variant A: Mapper-redline consensus. Rejected because mapper intent and 1 ms `.osu` quantization are not physical audio ground truth.
- Variant B: BeatThis-only stability. Rejected because one model can be overconfident in octave aliases or tracking artifacts.
- Variant C: BeatThis + raw flux. Rejected because both are still missing an explicit beat-tracker DBN view.
- Variant D: BeatThis probabilities + raw log-mel flux + BeatNet model 3 offline DBN, all same waveform, with both constant and natural-change consensus screens from the same computed views. Selected.

## Local Verification Matrix

- Variant A: Would pass many mapper-consistent cases but violates the input-scope guard.
- Variant B: Software check can be simple, but it cannot detect single-estimator alias failures.
- Variant C: Better than B, but still weaker on periodic beat decoding.
- Variant D: Requires more runtime, but gives three distinct estimator families while preserving the no-redline guard.

## Selected Variant

- Selected: Variant D.
- Rejected: A, B, C.
- Why this is the smallest useful test: It adds only one bounded batch runner and selector. It does not modify production Timing v3 or train a model.

## Selection Pressure

- Primary pressure: At least 256 unique constant sources accepted by the frozen screen.
- Guard pressure: No `.osu`, redline, label, metadata BPM, title, or artist fields are read.
- Runtime pressure: Runner is resumable JSONL and defaults to plan-only; full execution is deliberately not started until scheduled.
- Kill pressure: If fewer than 256 unique constant sources pass, return `KILL` and do not loosen thresholds post hoc. If fewer than 100 natural-change sources pass, report the count without loosening thresholds.

## Research Question

Can correlated audio estimators provide cleaner constant and natural-change source sets than mapper-redline timing without turning any one estimator into ground truth?

## Closest Analogies / Novelty Layer

- Closest analogies: Autocorrelation tempo estimation, DBN beat tracking, ensemble agreement filters, and octave-tempo equivalence classes.
- Relevant taxonomy bucket: Evaluation-set curation and evidence screening.
- Novelty layer, if any: None claimed.
- Representation novelty vs engineering variation: Engineering variation; the useful part is the source-selection protocol.

## Minimal Change

Add a research evaluation module and tests:

- plan-only CLI;
- resumable full5050 JSONL runner;
- strict precomputed BeatNet events JSONL input mode, with no inline BeatNet fallback during full consensus runs;
- fixed three-view screen: direct-BPM stability/change inside each view, folded octave-family agreement only for cross-view constant consensus;
- constant selector that writes exactly 256 unique sources only if at least 256 pass, split as 128 train and 128 untouched holdout;
- natural-change selector that writes exactly 100 unique sources only if at least 100 pass, split as 50 train and 50 untouched holdout.

## Files Likely to Change

- `docs/research/timing_v3_experiment_027_full5050_audio_consensus_constant_sources.md`
- `src/pulsefield_model/timing/evaluation/full5050_audio_consensus.py`
- `tests/timing/test_timing_v3_full5050_audio_consensus.py`

## Read-Only Context Files

- `README.md`
- `.agents/skills/research-triage/SKILL.md`
- `src/pulsefield_model/timing/evaluation/full5050_shadow_runner.py`
- `src/pulsefield_model/timing/v3/audio_evidence.py`
- `artifacts/reports/timing/timing_v3_labels_v1.jsonl` as locator-only input

## Dataset Slice

Exact full5050 manifest rows from `artifacts/reports/timing/timing_v3_labels_v1.jsonl`. The implementation validates the expected row count but the current task does not run the full extraction.

## Baseline / Comparator

No previous accepted audio-only full5050 constant-source selector exists. Existing mapper/redline audits are diagnostic comparators only and are not read by this experiment.

## Primary Metric

Number of unique accepted constant-like source paths after three-view consensus, with a hard gate of 256.

## Secondary Metric

Natural persistent-change accepted-source count, per-view direct stable/change-window counts, confidence floor, cross-view constant octave-family distance, cross-view direct signed-ratio distance, boundary intersection, and failure reasons.

## Verify Command / Evaluation Procedure

Plan-only and tests now:

```sh
uv run --extra mps --group dev pytest -q tests/timing/test_timing_v3_full5050_audio_consensus.py
uv run --extra mps python -m pulsefield_model.timing.evaluation.full5050_audio_consensus --expected-row-count 5050
/private/tmp/beatnet-venv/bin/python /private/tmp/timing_v3_full5050_beatnet_model3_dbn_extract.py plan --expected-row-count 5050
```

Deferred full run:

```sh
/private/tmp/beatnet-venv/bin/python /private/tmp/timing_v3_full5050_beatnet_model3_dbn_extract.py extract --expected-row-count 5050
uv run --extra mps python -m pulsefield_model.timing.evaluation.full5050_audio_consensus --run
uv run --extra mps python -m pulsefield_model.timing.evaluation.full5050_audio_consensus --select
uv run --extra mps python -m pulsefield_model.timing.evaluation.full5050_audio_consensus --select-change
```

## Guard Check

The locator loader projection ignores mapper labels, `.osu` fields, redlines, representative grids, metadata BPM, title, and artist. Row output contains no content-derived hash, redline, label, title, artist, or mapper-derived timing payload. Constant train and untouched holdout are source-disjoint. Natural-change train and untouched holdout are source-disjoint.
The BeatNet precompute file is bound to locator row index, row id, resolved audio path, and duration. Every row must be `completed` with non-empty beat times or an explicit error row. The consensus runner does not silently run BeatNet inline if the precompute file is absent or incomplete.

## Qualitative Check

Use only pre-registered full5050 aggregate statistics and failure buckets. Do not perform row-level spot checks for tuning, and do not use mapper outcomes to adjust thresholds.

## Positive Signal

At least 256 unique constant sources pass with all three views direct-stable and cross-view global octave-family distance at or below 0.06 octaves. If at least 100 natural-change sources also pass, the same run yields a fixed 50/50 source-disjoint train/untouched-holdout change cohort whose direct signed ratios agree within 0.06 octaves.

## Negative Signal

Fewer than 256 unique constant sources pass, or many failures are caused by missing BeatThis/BeatNet outputs rather than genuine instability. Fewer than 100 natural-change sources is a report-only negative for that cohort, not permission to relax thresholds.

## Kill Criteria

If fewer than 256 unique constant sources pass, mark the constant-source experiment `KILL`; do not relax window size, confidence thresholds, or octave-family tolerance based on mapper/audit outcomes. If fewer than 100 unique natural-change sources pass, report the shortfall and do not write a train/holdout cohort.

## Expected Failure Modes

- BeatNet package or model 3 DBN dependency is unavailable.
- Sparse BeatNet beat times produce low-confidence windows.
- Highly syncopated or weak-onset music fails raw flux confidence despite constant tempo.
- Correlated estimator agreement still misses true tempo-family changes.
- Duplicate audio paths reduce accepted unique-source count below accepted-row count.
- Natural changes with broad or ambiguous boundaries may fail the boundary-intersection gate even when each individual view detects a direct-BPM change.
- If a tracker internally reports a different but stable tactus from another tracker, constant consensus can still pass through octave-family agreement; if that same tracker changes from one tactus to its double/half inside the track, the direct-BPM view gate rejects constant and may feed the natural-change screen.

## Confounders

All three views are derived from the same waveform. Agreement is stronger than a single view, but it is not independent-recording evidence. BeatThis and raw log-mel features also share front-end assumptions.

## Expected Runtime / Runtime Budget

Current task: software tests and plan-only CLI only. Deferred full5050 run is expected to be dominated by BeatNet model 3 DBN and raw mel cache misses; it must run resumably and should not be started alongside timing runtime profiling.

## Result Interpretation Plan

- Positive result would suggest: the next causal constant corpus can use 128 train + 128 untouched holdout sources, and if the change gate passes, a natural-change corpus can use 50 train + 50 untouched holdout sources, without treating mapper redlines as ground truth.
- Negative result would suggest: the full5050 corpus may not contain enough robust constant-like sources under strict audio-only agreement, or BeatNet/raw evidence availability is insufficient.
- Ambiguous result would require: summarizing full5050 failure buckets without changing thresholds, then designing a separate mutation card.
- Human owner decides: whether accepted sources are sufficient for the next causal corpus.
- Next-loop action if positive: generate causal corpora from frozen source-disjoint train/holdout lists.
- Next-loop action if negative: `KILL` this selector and propose a new source-acquisition or estimator-availability card.
- Next-loop action if ambiguous: summarize failure buckets and ask for a constrained mutation.

## Result Log Template

- Experiment: timing_v3_exp027_full5050_audio_consensus_constant_sources
- Date:
- Commit / run id:
- Dataset slice: full5050 locator-only manifest
- Baseline / comparator: no audio-only selector
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
- Selected variant: Variant D
- Candidate variants rejected before execution: A, B, C
- Local verification outcomes:
- Selection pressure observed:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes, for software tests and plan-only CLI only in the current task
- Closed loop complete: yes
- Remaining ambiguity: actual full5050 pass count remains unknown until the deferred run.

## Next-Loop Action

- If positive: build causal corpora from selected source-disjoint train/holdout paths.
- If negative: return `KILL` and do not weaken thresholds.
- If ambiguous: create a new mutation card from failure buckets only.

## Novelty Notes

- Closest analogies: Autocorrelation tempo-family estimation, DBN beat tracking, ensemble source filtering.
- Novelty layer, if any: None claimed.
- Representation novelty vs engineering variation: Engineering variation in evaluation-set curation.
