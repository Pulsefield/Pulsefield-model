# BPM Ramp Timing Detection Experiment

## Mode

- Mode: planner
- Route: TEST
- Source idea: improve the timing module so it recognizes true ramp BPM maps.
- Acceptance source, if any: user target: recall >= 85%, mean runtime < 0.2s on this machine, false positives 0% on 500 non-ramp BPM maps.
- Source snapshot / evidence grade: local committed ramp audit artifacts in `artifacts/evals/bpm_ramp_candidate_mining`; medium-strength labels because they are redline/web-audit derived, not final musical ground truth.

## Hypothesis

A cheap structural detector over a `FittedTimingGrid` can recognize real continuous BPM ramp shapes from the committed ramp audit with >= 85% recall while producing 0 false positives on a deterministic 500-map non-ramp guard slice, without running audio inference.

## Root Objective

Add ramp-pattern recognition to the timing module as a bounded post-fit/oracle-grid classifier, not a rewrite of BeatThis prediction or grid fitting.

## Goal Decomposition

- Subgoal 1: detect continuous monotonic BPM movement with enough points, span, duration, and temporal continuity.
- Subgoal 2: reject ordinary multi-BPM jumps, short mapper timing corrections, and noisy redline churn.
- Subgoal 3: expose the detector through timing APIs/eval artifacts and verify recall, false positives, and runtime locally.

## Candidate Variants

- Variant A: classify ramp by unique BPM count only.
- Variant B: copy the committed audit thresholds directly.
- Variant C: compact-grid-aware monotonic-run detector with continuity, span, duration, jumpiness, and coverage guards.
- Variant D: rerun the audio timing model or add super-timing passes.

## Local Verification Matrix

- Variant A: likely high recall, likely poor false-positive guard because many mapper artifacts have many unique BPMs.
- Variant B: good for redline labels, but may be brittle on compact fitted grids and duplicate audit logic too closely.
- Variant C: expected to retain redline-pass recall while remaining cheap and usable on predicted grids.
- Variant D: too slow for the <0.2s target and outside the minimal structural question.

## Selected Variant

- Selected: Variant C.
- Rejected: A for weak specificity; B for redline overfitting risk; D for runtime and scope.
- Why this is the smallest useful test: it answers whether ramp recognition can be added as a fast timing-grid read before any fitter or model rewrite.

## Selection Pressure

- Primary pressure: recall >= 85% on `audit_status == pass` ramp beatmapsets.
- Guard pressure: 0/500 false positives on deterministic non-ramp beatmaps.
- Runtime pressure: mean classification time per map < 0.2s on this machine.
- Kill pressure: do not expand to audio inference or large fitter changes if the structural detector misses the target.

## Research Question

Can a fast timing-grid shape classifier recognize true BPM ramp maps well enough to be useful as timing-module metadata?

## Closest Analogies / Novelty Layer

- Closest analogies: local redline ramp audit scripts; timing multi-BPM structural diagnostics; osu! red timing monotonic-run inspection.
- Relevant taxonomy bucket: evaluation/diagnostic engineering for timing representation.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: this is engineering variation over existing timing-grid representation, not representation novelty.

## Minimal Change

Add a small ramp detector module, expose it in timing reports, add focused tests, and add a local eval script/result log.

## Files Likely to Change

- `src/pulsefield_model/timing/ramp_detection.py`
- `src/pulsefield_model/timing/__init__.py`
- `src/pulsefield_model/timing/fit_audio.py`
- `tests/timing/test_ramp_detection.py`
- `artifacts/evals/bpm_ramp_timing_detection/run_bpm_ramp_timing_detection_eval.py`
- `artifacts/evals/bpm_ramp_timing_detection/result_log.md`

## Read-Only Context Files

- `artifacts/evals/bpm_ramp_candidate_mining/real_ramp_beatmapset_audit_unique_bpm_gt5.parquet`
- `artifacts/evals/bpm_ramp_candidate_mining/candidate_index_unique_bpm_gt5.parquet`
- `artifacts/indexes/beatmap_index_4k_no_timing_anomalies_2to6.parquet`
- `src/pulsefield_model/timing/providers/oracle.py`
- `src/pulsefield_model/timing/grid_fitting/*`

## Dataset Slice

- Positives: beatmapset representatives from `real_ramp_beatmapset_audit_unique_bpm_gt5.parquet` where `audit_status == "pass"`.
- Negatives: 500 deterministic rows from `beatmap_index_4k_no_timing_anomalies_2to6.parquet`, excluding beatmap sets present in `candidate_index_unique_bpm_gt5.parquet` and excluding positive beatmap sets.

## Baseline / Comparator

- Comparator labels come from the committed ramp audit, not raw unique-BPM count.
- Baseline behavior is no dedicated ramp metadata in the timing module.

## Primary Metric

Recall on true ramp positives: `recognized_ramp_pass / pass_count`.

## Secondary Metric

- False positives on the 500-map guard slice.
- Mean, p95, and max classification time per map.
- Recognized ramp family distribution and rejected reasons for misses.

## Verify Command / Evaluation Procedure

- `uv run pytest tests/timing/test_ramp_detection.py tests/timing/test_providers_and_cli.py`
- `uv run python artifacts/evals/bpm_ramp_timing_detection/run_bpm_ramp_timing_detection_eval.py`

## Guard Check

No audio model inference is part of the eval. The detector runs on parsed timing grids and must keep false positives at zero on the guard slice.

## Qualitative Check

Inspect false negatives from the pass set and any guard false positives with titles, BPM run shape, reasons, and beatmap path.

## Positive Signal

Recall >= 85%, false positives = 0/500, and mean runtime < 0.2s/map.

## Negative Signal

Recall below 85% at the zero-FP threshold, any guard false positive, or runtime above budget.

## Kill Criteria

- Do not add audio inference or super-timing to meet this target.
- Stop and report `MUTATE` if structural thresholds cannot hit recall and zero-FP together.
- Stop if >10% of evaluation rows fail due to parser/data errors.

## Expected Failure Modes

- True ramps with very few red points or abrupt short sweeps may be missed.
- Some mapper gimmicks may look structurally identical to true ramps.
- Fitted grids with very low `max_segments` may compress a continuous ramp into too few points.

## Confounders

Audit labels are imperfect; multiple difficulties in a beatmapset can have different timing; redline timing may encode mapper intent rather than audio tempo; canonicalization can fold high BPM ramps unless raw grids are used.

## Expected Runtime / Runtime Budget

Unit tests under 10 seconds. Full structural eval under 60 seconds total and mean detector runtime below 0.2 seconds per map.

## Result Interpretation Plan

- Positive result would suggest: timing-grid ramp metadata is worth keeping as a cheap diagnostic/control signal.
- Negative result would suggest: this needs a different label definition or a fitter/audio experiment, not more threshold tuning.
- Ambiguous result would require: manual review of misses/false positives before changing research direction.
- Human owner decides: whether this is a useful ramp definition for downstream generation.
- Next-loop action if positive: consider using ramp metadata in downstream timing/control conditioning.
- Next-loop action if negative: mutate to a smaller fitter-output recall experiment.
- Next-loop action if ambiguous: audit borderline cases and revise the label slice.

## Result Log Template

- Experiment:
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
- Selected variant:
- Candidate variants rejected before execution:
- Local verification outcomes:
- Selection pressure observed:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes
- Closed loop complete: yes
- Remaining ambiguity: exact thresholds are selected by local eval against the stated recall/false-positive/runtime targets.

## Next-Loop Action

- If positive: keep the detector and document/use the metadata.
- If negative: return `MUTATE` with observed failure cases.
- If ambiguous: inspect borderline cases before expanding scope.

## Novelty Notes

- Closest analogies: local redline ramp audit and multi-BPM structural diagnostics.
- Novelty layer, if any: none claimed.
- Representation novelty vs engineering variation: engineering variation only.
