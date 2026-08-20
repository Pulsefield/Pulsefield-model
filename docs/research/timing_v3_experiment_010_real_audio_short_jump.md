# Timing v3 Experiment 010: Raw-audio alias and short-jump repair

Status: mutated after exposed mechanism verification

## Question

Can one bounded candidate generator, followed by a deterministic raw-audio
reranker, fix the two dominant exposed failures without weakening the global
phase-continuity contract?

1. choose the direct musical tempo rather than a half/double alias; and
2. retain a 2--8 second tempo excursion that the existing 8-second boundary
   merge and minimum-section rules cannot represent.

This is a new behavior-affecting experiment. Experiments 001--009 remain
immutable evidence.

## Hypothesis

A compact local tempo track can propose constant, paired-boundary jump, and
diagnostic linear-ramp analytic curves. Scoring no more than 64 such curves
against four-band onset flux will choose the correct phase/alias family while
keeping one absolute integer beat axis.

## Inference inputs

- the existing shift-0 BeatThis `final0` frame cache;
- deterministic 16 kHz, 80-bin, 10 ms log-mel from the same recording;
- no `.osu`, metadata BPM, title, catalog value, or manual choice.

Weak `.osu` redlines are read only after prediction freezes for evaluation.

## Frozen first implementation

- local tempo windows must be bounded independently of song duration;
- short excursion duration range: 2--8 seconds;
- output candidates: at most 64;
- every candidate uses `PhaseContinuousTimingCurve` and one global integer beat
  axis;
- candidate ranking uses the frozen Experiment 009 raw-audio score unchanged;
- ramp candidates are diagnostic in this Phase-1 goal and cannot become a
  production success claim.

## Data and exposure

Development begins with the already exposed pilot80 only. The three initial
mechanism probes are:

- `dataset/0/2300685/audio.mp3` (stable/direct-alias probe);
- `dataset/0/618173/audio.mp3` (4.17-second jump excursion probe);
- `dataset/0/829383/audio.mp3` (ambiguous ramp-like stress probe).

The protected holdout100-v2 is not opened while this card is being mutated.
After the smoke gate, evaluation expands to all high/medium, non-ambiguous
pilot80 stable/jump rows. Ramp/dense rows remain diagnostic-only.

## Metrics

- direct BPM coverage and alias-aware BPM coverage, reported separately;
- stable false-boundary song rate and boundary count;
- jump boundary precision/recall at 500 ms and 1,000 ms;
- direct left/right tempo-pair accuracy for matched boundaries;
- signed initial phase, mean phase, and p90 phase in milliseconds;
- maximum serialized seam discontinuity;
- candidate count and extraction/rerank/end-to-end runtime;
- ramp direction/endpoints/slope only, with ramp accuracy recorded as `null`.

The primary weak-oracle decision slice after pilot smoke is all 227
high/medium non-ambiguous jump rows plus 227 deterministic duration/confidence
matched stable controls. This slice is development evidence, not fresh
holdout.

## Guards

- no `.osu` or metadata access before the selected fingerprint is frozen;
- no local phase reset and no non-integer section boundary;
- serialized seam error at most 5 ms;
- at most 64 candidates;
- no hard failure;
- 10-minute algorithmic candidate generation plus reranking target: at most 5
  seconds, measured separately from one-time audio decode/mel cache creation;
- final product/runtime gates from the goal remain unchanged.

## TEST / MUTATE / KILL

- `TEST`: synthetic constant, short excursion, and analytic time-linear ramp;
  then the three exposed real mechanism probes; then pilot80.
- `MUTATE`: if the correct candidate is absent, change only local trajectory or
  paired-boundary proposal; if present but not selected, change only the
  reranking family in a new card.
- `KILL`: hard failure, phase reset, more than 64 candidates, or failure to
  represent the 4.17-second exposed excursion after two mechanism-distinct
  proposal mutations.

No fresh holdout, broad500, or full5050 result may be claimed from this running
card.

## Result

- Date: 2026-08-13
- Synthetic verification: constant 200 BPM, short `175 -> 143 -> 175` ABA,
  analytic time-linear ramp, deterministic replay, candidate cap, and phase
  continuity passed.  The focused analytic/raw/tempo-track suite reached
  `47 passed`; the later joint focused Timing-v3 suite reached `62 passed`.
- Runtime: a 10-minute synthetic 200 BPM candidate generation run completed in
  `2.39 s` with 12 candidates.  On the exposed `618173` row, existing BeatThis
  and mel caches produced 48 candidates and raw scores in about `0.69 s`.
- Candidate recall: positive.  The exposed 4.172-second jump probe contains a
  generated `143.964 BPM / 4.168 s` candidate with boundaries approximately
  `359/363 ms` before the weak comparator, both inside the 750 ms mechanism
  gate, and exact analytic seam continuity.
- Reranking: negative.  The frozen Experiment 009 common-domain scorer selected
  a `148.265 BPM / 2.428 s` candidate.  A half-tempo member of the same batch
  reduced the common retained domain from the expected full track to only 88
  beats, so the decision-relevant late excursion was largely absent from every
  candidate score.
- Stable probe: the exposed `2300685` row retained the direct 200 BPM constant
  candidate; no stable success is generalized beyond this mechanism row.
- Interpretation: Experiment 010 succeeds at representation and short-jump
  candidate recall but fails its top-1 hypothesis.  Per its frozen rule, the
  next change is reranking only in a new card.  No pilot80 aggregate,
  holdout100-v2, broad500, or new full5050 replay was opened or claimed.
