# Timing v3 Experiment 001: Evaluation Foundation

## Mode

- Mode: critic
- Route: TEST
- Route rationale: the 5,050-cache corpus is complete, but the repository has
  no durable full-corpus timing evaluator and the retained slice claims cannot
  attribute BeatThis, fitting, assembly, and comparator error separately.
- Source snapshot / evidence grade: strong for local cache/index facts; medium
  for `.osu` timing as musical truth; weak-to-medium for API BPM; external
  sources are primary papers, official docs, and official repositories.

## 1. Core Question

- Summary: can the existing local sources form a reproducible, confidence-
  weighted evaluation set that is strong enough to select timing-v3 algorithms
  without leaking `.osu` or online metadata into inference?
- Claimed contribution: an evaluation and error-attribution foundation, not a
  timing-algorithm novelty claim.
- Research question: does multi-difficulty redline consensus plus independent
  object-grid evidence yield useful stable/jump/ramp strata while preserving an
  explicit ambiguous bucket?
- Idea quality: worth a bounded test because 5,050 activations and 14,689 maps
  already exist, while full-corpus baseline provenance is missing.

## 2. Implementation Families

- Family A: one `.osu` redline per audio as oracle. Fast, but confounds mapper
  artifacts with audio timing and wastes multi-difficulty evidence.
- Family B: consensus over redlines for all difficulties of an audio. Better,
  but correlated mapper conventions can still agree on the wrong grid.
- Family C: audio-grouped redline consensus plus object-placement residual,
  local official API metadata, and an explicit ambiguity policy.
- Family D: manually/web-audited labels only. Highest precision, but too small
  and expensive to be the primary 5,050-song framework.
- Lowest-risk family: Family C, with Family D reserved for a small gold audit.
- Minimal experiment implied: inventory and label all audio groups, run the
  unchanged v2 fitter from the single cache on a stratified pilot, and measure
  label coverage/agreement and metric observability.
- Baseline / comparator: current `GridFitter` on the exact `final0`, shift-0
  cache plus current redline-rendered diagnostics.
- Verify gate: the report can reproduce every row from immutable provenance,
  separates audio groups, and exposes enough high-confidence strata for later
  selection without forcing ambiguous rows.
- Guard gate: `.osu`, hit objects, and metadata are inaccessible to the fitter;
  all comparisons are paired on the same cached activation.

## 3. Closed-Loop Decomposition

- Root objective: establish a trustworthy selection surface before changing the
  timing algorithm.
- Checkable subgoals:
  1. build one canonical audio-group inventory linking cache, maps, metadata,
     redline stats, object stats, and duration;
  2. assign reproducible stable/jump/ramp/artifact/ambiguous evidence fields,
     retaining raw evidence rather than only a final label;
  3. run v2 from cache and emit per-layer metrics and resumable results;
  4. prove that splits and summaries operate on 5,050 audio groups, not 14,689
     difficulty rows.
- Candidate variants: A redline-only; B redline consensus; C consensus + object
  + metadata tiers; D manual-only.
- Local verification matrix:
  - A: compare labels across difficulties; fail if nominal truth disagrees
    within an audio group.
  - B: measure consensus coverage and object-grid residual; fail if consensus
    does not predict independent object support.
  - C: measure coverage, agreement, ambiguity rate, and manually inspect the
    highest/lowest-confidence examples in each stratum.
  - D: estimate annotation time and use it only to audit precision of C.
- Selection pressure: maximize auditable label precision and error attribution,
  subject to useful coverage, zero inference leakage, bounded runtime, and an
  explicit ambiguous class.
- Selected candidate: Family C.
- Rejected candidates and why: A ignores known redline noise; B lacks an
  independent phase signal; D cannot cover the corpus alone.
- Next-loop action if selected candidate fails: MUTATE to redline-consensus
  strata plus a manually audited 200-audio benchmark; do not tune timing v3
  against low-confidence object evidence.

## 4. Novelty Source

- Closest analogies: BeatThis postprocessing, DBN/DP beat tracking, osu! timing
  semantics, and Osu2MIR timing-group curation.
- Relevant taxonomy bucket: benchmark/evaluation infrastructure and structured
  postprocessing.
- What is probably not new: tempo priors, DP, redline parsing, grid residuals,
  consensus labels, or change-point detection.
- Possible novelty layer: using multi-difficulty `.osu` weak annotators and
  rational object placement to attribute error in a phase-continuous
  BeatThis-to-section pipeline.
- Representation novelty vs engineering variation: this experiment is
  engineering infrastructure; beat-index sections are evaluated here but not
  implemented by this card.

## 5. Feasibility Risks

- Weak assumptions: objects concentrate on a small rational grid; difficulties
  are partly independent annotators; official API BPM identifies the same
  recording and a meaningful tempo family.
- Simplest counterargument: all difficulties may copy one faulty timing setup,
  so consensus is correlated rather than independent.
- Confounders: syncopation, sparse intros/outros, deliberate off-grid placement,
  hold-release shaping, half/double-time aliases, redline micro-corrections,
  beatmapset duplicates, audio edits, and API metadata derived from the same
  maps.
- Runtime / data risk: current fitting is roughly seconds per ordinary song;
  a full v2 baseline is hours unless batching is resumable. The 72-minute
  maximum track needs a timeout and progress checkpoint.
- Missing source evidence: exact jump/ramp precision is unknown until the
  confidence audit and small manual review are complete.

## 6. Observability / Debuggability

- Required instrumentation: cache/config hashes; audio group key; all raw
  redlines; per-difficulty agreement; object start/end residuals; API metadata
  provenance; BeatThis support; v2 timing metrics; runtime; failure stage; and
  label reasons.
- Verification checks: exact 5,050 cache/audio linkage; deterministic inventory;
  group-disjoint splits; schema validation; resume equivalence; synthetic grids;
  and manual spot checks from every evidence bucket.
- Guard checks: current timing unit tests, no production-path dependency on
  `.osu`, paired v2 metrics, no overwritten user artifacts, and no secret/API
  credential persistence.
- Likely false positives: dense mapper redlines interpreted as ramps and common
  subdivision placements supporting an alias grid.
- Likely false negatives: true ramps with weak BeatThis peaks, sparse objects,
  or copied constant redlines.
- Failure modes: missing map/cache, duplicate identifiers, inconsistent audio
  grouping, invalid red points, no objects, pathological redline density,
  timeout, and ambiguous alias family.

## 7. Recommendation

- Final route: TEST.
- Why this route: the inputs are complete and the missing durable evaluator is
  a smaller, necessary risk-reduction step before a new fitter.
- How local verification drove the route: 5,050/5,050 cache coverage, 2.91 maps
  per audio on average, 813 varying-redline audio groups, and 48 permissive ramp
  candidates make a bounded audit feasible while demonstrating redline noise.
- Selection rule used: choose the smallest family that adds an independent
  phase signal and retains ambiguity without putting weak labels into inference.

## 8. Experiment Card

- Hypothesis: a deterministic audio-group inventory combining redline consensus,
  rational object-placement residuals, and source-tiered metadata can create
  useful stable/jump/ramp/ambiguous evaluation strata and reproduce a paired v2
  cache baseline without inference leakage.
- Root objective: make later timing-v3 selection measurable and attributable.
- Goal decomposition:
  1. inventory/provenance;
  2. weak-evidence extraction and confidence tiers;
  3. cache-backed v2 runner;
  4. aggregate/stratified report and manual audit queue.
- Candidate variants: A redline-only; B redline consensus; C consensus + object
  + metadata; D manual-only.
- Local verification matrix: as defined in section 3; run synthetic unit tests,
  then an 80-audio stratified slice, then a label-only 5,050 inventory.
- Selected variant: C.
- Selection pressure: precision/attribution first, coverage second, then runtime;
  zero inference leakage and group leakage are hard constraints.
- Minimal change: add a durable inventory/label module, cache-backed evaluation
  runner, tests, and JSON/JSONL report schemas; do not change `GridFitter`.
- Files likely to change:
  - `src/pulsefield_model/timing/evaluation/` (new);
  - `tests/timing/` focused evaluation tests;
  - `docs/research/` experiment and result logs.
- Read-only context files:
  - `src/pulsefield_model/timing/grid_fitting/`;
  - `src/pulsefield_model/timing/providers/beatthis_cache.py`;
  - `src/pulsefield_model/osu_core/timing.py`;
  - `src/pulsefield_model/osu_core/hitobjects.py`;
  - local 5,050 cache, index, dataset maps, and metadata snapshots.
- Dataset slice: synthetic cases, then 80 audio groups: 20 single-redline normal
  duration, 20 2-16-redline jump candidates, 20 dense/ramp candidates, 10 long
  tracks, and 10 duplicate/grouping anomalies. Deduplicate by audio and allow a
  row to belong to only one priority stratum. Next run label-only inventory on
  all 5,050.
- Baseline / comparator: unchanged current `GridFitter` using the same cached
  activation; retained local claims are contextual only until reproduced.
- Primary metric: evaluation-label audit precision on a stratified manual sample
  plus high-confidence label coverage by audio group.
- Secondary metric: within-audio redline agreement, object-grid residual/inlier
  rate, API BPM alias agreement, ambiguity rate, inventory coverage, v2 phase
  and alias-aware BPM metrics, and runtime/failure rate.
- Verify command / evaluation procedure: unit tests -> deterministic 80-audio
  manifest -> two identical inventory runs compared byte-for-byte -> full 5,050
  inventory -> resumable v2 pilot run -> aggregate report.
- Guard check: existing timing tests remain green; fitter imports no evaluation
  source; all splits are audio-grouped; report paths are explicit and resumable;
  raw labels include source/reason/confidence.
- Qualitative check: inspect activation/redline/object overlays for at least five
  highest-confidence and five lowest-confidence examples in each nonempty
  stable/jump/ramp bucket.
- Positive signal: useful high-confidence stable and jump strata, a nonempty
  ramp audit queue, object residual improves confidence discrimination over
  redline consensus alone, and all 5,050 groups are reproducible.
- Negative signal: object support is unrelated to consensus/manual judgments,
  most variable-tempo audio is forced ambiguous, or grouping/provenance cannot
  be made deterministic.
- Kill criteria: fewer than 30 credible jump groups or fewer than 10 credible
  ramp audit candidates after manual review; object evidence fails to improve
  precision over consensus; any inference leakage; or nondeterministic grouping.
- Expected failure modes: correlated copied redlines, alias ambiguity,
  dense-redline false ramps, sparse/off-grid objects, invalid maps, duplicates,
  and pathological runtimes.
- Confounders: mapper and API sources are correlated; object density depends on
  difficulty; current v2 metrics compare against noisy redlines.
- Expected runtime / runtime budget: inventory under 10 minutes; 80-audio v2
  pilot under 10 minutes after optimization or under 30 minutes as a hard stop;
  full baseline is deferred until the runner is resumable and a pilot runtime
  estimate exists.
- Result interpretation plan:
  - positive: accept the foundation and create Experiment 002 for the v3 schema
    plus constant/jump global path;
  - negative: drop object evidence or narrow it to an audit metric and MUTATE to
    a 200-audio manual benchmark;
  - ambiguous: expand only the manual audit, not the timing algorithm;
  - human owner decides final trust thresholds and whether web/manual evidence
    is sufficient for headline ramp claims.
- Result log template:
  - experiment/date/commit/config and data hashes;
  - selected variant and rejected variants;
  - audio counts by evidence/label/ambiguity stratum;
  - label audit confusion table and confidence calibration;
  - inventory determinism and coverage;
  - v2 pilot metrics/runtime/failures;
  - qualitative examples and suspected confounders;
  - kill criteria status, interpretation, and next-loop decision.
- Next-loop action: if positive, TEST the beat-index schema and global
  constant/jump path; if negative, MUTATE the benchmark; if ambiguous, collect
  only enough manual labels to decide.

## Result Interpretation

- Positive result would suggest that later v3 changes can be selected without
  treating `.osu` as unquestioned truth.
- Negative result would suggest that a smaller manually audited benchmark is a
  prerequisite and that full-corpus redline metrics are guards only.
- Ambiguous result would require confidence calibration and targeted manual
  review, not extra fitter complexity.
- Human owner decides label trust thresholds, acceptable ambiguity, and whether
  the next Experiment Card is accepted.
