# Timing v3 Experiment 004: BeatThis Global Constant/Jump Assembly

Status: pre-registered; not run. This card freezes the experiment before any
Experiment 004 implementation or evaluator outcome is inspected.

## Mode

- Mode: planner
- Route: TEST
- Source idea: replace v2-derived local split/projection with a cache-only
  global constant/jump assembler over one BeatThis frame prediction.
- Acceptance source: [Timing v3 task definition](timing_v3_task_definition.md)
  delivery gate 5, after the evaluation foundation and constant-section v3
  schema from Experiments 001-003.
- Source snapshot / evidence grade: strong for local cache/schema/API facts;
  medium for `.osu` redlines and object grids as evaluation comparators; weak
  for ramp labels, which remain audit-only.

## Hypothesis

A global beam or dynamic program that assembles phase-continuous constant BPM
sections directly from one cached BeatThis `beat_prob`/`downbeat_prob` sequence
will reduce jump-boundary and accumulated drift errors compared with the current
source-owned v2 grid fitter, without regressing stable tracks or requiring
`.osu`, metadata, network evidence, ramps, or long-track decomposition.

## Root Objective

Test whether the next timing-v3 step should be a BeatThis-supported global
constant/jump path rather than another adapter from v2 segments.

The public output remains the existing constant-section representation:

```text
beat [a,b) bpm q
```

with one global beat axis, shared boundary beats, and derived boundary times.

## Goal Decomposition

- Subgoal 1: extract beat, downbeat, tempo, phase, and change-boundary
  candidates from exactly one cached BeatThis prediction.
- Subgoal 2: assemble one global constant/jump path with explicit transition
  costs for BeatThis support, boundary evidence, tempo alias stability, sparse
  changes, and beat-count feasibility.
- Subgoal 3: emit the existing `TimingV3Grid` invariants and diagnostics for
  every accepted path or a tagged fallback for every rejected path.
- Subgoal 4: compare the selected global path, exact ablations, and the current
  baseline on the frozen repair -> holdout100 -> broad500 -> full5050
  progression without post-outcome tuning.

## Candidate Variants

Only these candidates are allowed in Experiment 004:

- Variant A, `CJ0 constant_only`: one global constant section selected from the
  frozen whole-track tempo/phase candidates. No interior jumps.
- Variant B, `CJ1 beat_only_global`: global constant/jump beam using BeatThis
  beat support and hard feasibility only. No downbeat, alias, sparse-change, or
  beat-count soft priors.
- Variant C, `CJ2 priors_no_downbeat`: Variant B plus frozen BPM, alias,
  sparse-change, section-duration, and beat-count priors. No downbeat or meter
  term.
- Variant D, `CJ3 selected_global`: Variant C plus frozen downbeat-boundary and
  downbeat-phase terms. This is the selected experiment hypothesis.

The current v2 fitter is the primary baseline comparator, not a candidate
family. Frozen projection or fallback controls from Experiments 002-003 may be
reported only as non-primary controls when their source is already frozen, but
they must not provide candidate sections, weights, or boundary choices to this
experiment.

## Local Verification Matrix

- `CJ0 constant_only`: pass local verification if synthetic constant tracks and
  single-tempo cache fixtures produce one valid section with deterministic
  origin, coverage, and JSON round-trip seams.
- `CJ1 beat_only_global`: pass local verification if synthetic on-lattice jumps,
  half/double aliases, empty-signal fallbacks, and no-path cases are reproduced
  by beat support and hard feasibility alone.
- `CJ2 priors_no_downbeat`: pass local verification if alias-switch, short
  weak-boundary, and sparse-change synthetic cases choose the preregistered
  lower-cost path and remain deterministic under exact-score ties.
- `CJ3 selected_global`: pass local verification if downbeat-supported
  boundaries receive the expected preference, downbeat-free fixtures reduce
  exactly to `CJ2`, global modulo-4 phase propagates by absolute beat count
  without edge resets, and metadata traps prove `.osu`, title, API, and network
  fields are not read.

No real-audio metric may change candidate extraction, weights, tie-breaks,
thresholds, or gates after these synthetic checks are frozen.

## Selected Variant

- Selected: Variant D, `CJ3 selected_global`.
- Rejected: `CJ0` is too weak for jumps, `CJ1` cannot test priors, and `CJ2`
  omits the only frozen downbeat evidence available in the cache.
- Why this is the smallest useful test: all four variants share one candidate
  extractor and one global assembler, so the experiment attributes any benefit
  to the exact cost terms rather than to unrelated infrastructure.

## Selection Pressure

- Primary pressure: improve or preserve paired phase and drift metrics on
  comparison-eligible audio groups, especially jump and long strata.
- Guard pressure: zero inference leakage, exact v3 continuity, bounded fallback,
  no section explosion, and no stable-stratum regression.
- Runtime pressure: keep a single audio group under a frozen timeout and keep
  full-corpus evaluation resumable.
- Kill pressure: if `CJ3` does not beat its simpler ablations or cannot pass the
  staged gates unchanged, do not add more weights or ramp logic.

## Research Question

Can the single BeatThis cache support a globally assembled constant/jump
section graph that is better than local splitting plus projection, while
remaining reproducible, phase-continuous, and evaluation-leakage free?

## Closest Analogies / Novelty Layer

- Closest analogies: DBN/Viterbi beat tracking, dynamic-programming tempo
  tracking, change-point search, BeatThis postprocessing, and osu! redline
  weak-label evaluation.
- Relevant taxonomy bucket: structured postprocessing and benchmark-coupled
  evaluation.
- Novelty layer, if any: Pulsefield-specific coupling of one BeatThis cache,
  a half-open beat-index section representation, and audio-grouped osu! weak
  comparators.
- Representation novelty vs engineering variation: the DP/beam, tempo priors,
  and change costs are known families. The experiment is an engineering
  selection test, not a novelty claim.

## Inference Boundary

The implementation may read only:

- `FrameTimingPrediction.beat_prob`;
- `FrameTimingPrediction.downbeat_prob`;
- `FrameTimingPrediction.frame_rate_hz`;
- cache provenance needed for diagnostics and reproducibility.

The implementation must not read `.osu` files, hit objects, red timing,
beatmap titles, artist/title metadata, official API BPM, label rows, previous
evaluator outputs, local notes, network sources, or audio samples. No network
access is permitted during inference or candidate extraction.

Evaluation code may read frozen weak labels and stored comparators, but that
layer must be import-separated from the fitter path and covered by metadata
trap tests.

## Frozen Candidate-Extraction Contract

Given frame count `F` and frame rate `f`, evaluated cache support is:

```text
coverage_start_ms = 0
coverage_end_ms   = 1000 * F / f
frame_time_ms[i]  = 1000 * i / f
```

All calculations that affect a candidate use float64. Input arrays are only
validated and converted; they are not smoothed by a tunable filter in this
experiment.

### Peak materialization

Materialize beat and downbeat peaks independently from the corresponding
probability vector:

1. A candidate peak at frame `i` must satisfy
   `signal[i] >= signal[i-1]`, `signal[i] > signal[i+1]`, and
   `signal[i] >= max(mean(signal) + std(signal), 0.35 * max(signal))`.
2. Adjacent selected peaks must be at least one frame apart. If two candidates
   conflict, keep the one with higher probability, then lower frame index.
3. Refine a selected peak by a three-point parabolic offset when the parabola
   is finite and the offset is within `[-0.5, 0.5]` frames; otherwise keep the
   frame center.
4. Peak confidence is the original clipped probability at the selected frame.

### Frozen constants

The candidate contract version is `timing-v3-exp004-candidate-contract-v1`.
Implementation must serialize these constants and their JSON SHA-256. It must
not import `GridFitterConfig` for candidate limits, because future config edits
must not change this preregistered experiment.

- hard BPM guard: `[20, 1000]`;
- preferred BPM band: `[80, 240]`;
- cache frame rate expected by the current corpus: `50 Hz`; non-50 Hz caches may
  run only through the same formulas below and must report the realized period
  frame bounds;
- period frame bounds:
  `min_period_frames = ceil(f * 60 / 1000)` and
  `max_period_frames = floor(f * 60 / 20)`, which are `3` and `150` at `50 Hz`;
- autocorrelation top lag count: `16`;
- alias expansion multipliers: `{1/4, 1/3, 1/2, 1, 2, 3, 4}`;
- BPM grid step: `0.5 BPM`;
- local BPM window: `max(2 BPM, 0.08 * candidate_bpm)`;
- fractional BPM parts:
  `{1/9, 1/8, 2/9, 1/3, 3/8, 4/9, 5/9, 5/8, 2/3, 7/9, 7/8, 8/9}`;
- pulse width: `40 ms`;
- peak/grid matching tolerance: `min(45 ms, 0.15 * beat_length_ms)`;
- boundary support tolerance: `60 ms`;
- minimum section duration: `8000 ms`;
- maximum section count: `20`;
- beam width: `64`;
- maximum origin candidates: `16`;
- maximum interior boundary candidates: `192`;
- maximum tempo candidates retained: `256`;
- maximum beat-count candidates per edge: `16`.

### Tempo candidates

Tempo candidates are generated once per audio, not inside the edge loop.

1. Compute whole-track autocorrelation candidates from the centered beat signal
   over the period frame bounds above. Retain the top 16 lags by normalized dot
   score, breaking ties by smaller lag.
2. Convert each lag to BPM, expand by the alias multipliers, expand each valid
   BPM by the local BPM window on the 0.5 BPM grid, and add fractional BPMs
   using the frozen fractional parts when the integer part is at least `80`.
3. Deduplicate by `round(bpm, 6)`, keep candidates in first-seen order, and cap
   at 192 whole-track candidates.
4. Add up to 64 peak-interval BPM candidates from materialized beat peaks:
   compute the median interval over every run of 4 to 16 consecutive beat
   intervals, convert to BPM and alias-expanded BPMs, and keep the highest
   median peak-confidence candidates after deduplication.
5. The final tempo pool is capped at 256. Edges derive integer beat counts from
   this pool only. No autocorrelation, BPM-window enumeration, or 1000-candidate
   search is allowed per edge.

### Origin candidates

Origin candidates are generated once per audio from the final tempo pool.
For the first 64 tempo candidates in deterministic tempo-pool order:

1. Let `p = 60000 / bpm`.
2. Score offsets `o` in `[0,p)` on a `20 ms` grid using the frozen triangular
   pulse correlation defined in the cost section.
3. Refine the best offset for that BPM over `[o-10 ms,o+10 ms]` at `1 ms`
   resolution.
4. Convert the refined offset to the first grid beat at or after
   `coverage_start_ms`:

   ```text
   tau = offset + ceil((coverage_start_ms - offset) / p) * p
   ```

5. Reject `tau >= coverage_end_ms`.
6. Deduplicate origins within `20 ms`, keeping higher score, then earlier time,
   then lower BPM.

Keep at most 16 origin candidates by score, time, and BPM. If no origin
candidate remains, emit fallback reason `no_origin_candidate`.

### Boundary candidates

Interior boundary candidates use the versioned function
`boundary_candidate_score_v1`. It consumes only materialized beat peaks
`P = (p_0 ... p_n)` with confidences `w`, materialized downbeat peaks `D`, and
frame-derived times. It does not inspect any fitted section or `.osu` boundary.

For beat peak index `k`, ordinary evidence is defined only when four intervals
exist on both sides:

```text
left_intervals   = diff(P[k-4:k+1])
right_intervals  = diff(P[k:k+5])
left_period      = median(left_intervals)
right_period     = median(right_intervals)
interval_change  = abs(right_period - left_period)
left_fit_error   = median(abs(P[k-4:k] - (P[k] - [4,3,2,1] * left_period)))
right_fit_error  = median(abs(P[k+1:k+5] - (P[k] + [1,2,3,4] * right_period)))
phase_change     = abs(right_fit_error - left_fit_error)
phase_residual   = max(left_fit_error, right_fit_error)
ordinary_score   =
    interval_change / 20
  + phase_change / 10
  + phase_residual / max(min(left_period, right_period), 1e-6)
```

The ordinary candidate is eligible when
`interval_change >= 20 ms`, `phase_change >= 10 ms`, or
`phase_residual >= 20 ms`.

Super-timing evidence is defined when three intervals exist on both sides:

```text
left_period       = median(diff(P[k-3:k+1]))
right_period      = median(diff(P[k:k+4]))
left_fit_error    = median(abs(P[k-3:k] - (P[k] - [3,2,1] * left_period)))
right_fit_error   = median(abs(P[k+1:k+4] - (P[k] + [1,2,3] * right_period)))
phase_residual    = max(left_fit_error, right_fit_error)
relative_change   = abs(right_period - left_period) / max(min(left_period, right_period), 1e-6)
boundary_strength = w[k]
super_score =
    relative_change / 0.025
  + abs(right_period - left_period) / 20
  + phase_residual / 10
  + 0.25 * boundary_strength
```

The super candidate is eligible when `relative_change >= 0.025`.

For either path, the boundary time is `P[k]` unless a downbeat peak lies within
`min(0.5 * min(left_period, right_period), 1000 ms)`; in that case use the
nearest downbeat peak and add:

```text
downbeat_bonus = 0.5 * (1 - distance_to_downbeat / tolerance)
```

If ordinary evidence is unavailable, `ordinary_score = -inf`. If super evidence
is unavailable, `super_score = -inf`. A boundary candidate exists only when at
least one score is finite. The winning evidence mode carries its own
`left_period` and `right_period`; those periods are used for downbeat tolerance,
edge diagnostics, and tie-breaks. If ordinary and super scores tie exactly,
choose ordinary evidence, then earlier boundary time, then lower source peak
index.

Final boundary rank score is `winner_score + downbeat_bonus`. Merge candidates
within `8000 ms` by keeping higher rank score, then ordinary over super, then
earlier time, then lower source peak index. Keep the highest-scoring candidate
in every half-open 60-second coverage bin before filling remaining slots by
score, evidence mode, time, and peak index. The interior boundary cap is
`min(192, max(16, ceil(duration_s / 4)))`.

Required boundary tests include insufficient-window edges:

- `k < 4` and `k > n - 5` produce absent ordinary evidence;
- `k < 3` and `k > n - 4` produce absent super evidence;
- both absent yields no candidate;
- exact ordinary/super ties use the tie order above and preserve the winning
  periods in diagnostics.

Coverage start and end are support limits, not forced beat boundaries.
The first selected beat anchor may occur after 0 ms; the first section is
extended backward as the schema permits.

### Candidate caching and lazy complexity

Let `O <= 16` origin candidates, `B <= 192` interior boundaries, and
`A = O + B`. The implementation precomputes and caches:

- peak arrays once per audio;
- the final tempo pool once per audio;
- boundary candidates once per audio;
- per-edge beat-count candidate sets keyed by `(left_anchor_id,right_anchor_id)`;
- lazy per-edge section scores keyed by
  `(left_anchor_id,right_anchor_id,N,variant)`.

For each ordered anchor pair satisfying the 8000 ms minimum duration, derive
`N0 = floor(delta_ms * bpm_candidate / 60000 + 0.5)` for the 256 tempo
candidates, add `{N0-1,N0,N0+1}`, reject invalid counts/BPMs, deduplicate, rank
by nearest tempo-pool support and beat-support score, and retain at most 16
counts. The assembler must not score every theoretical edge. It scores only
outgoing edges from states that survive into the current beam. Expansion order
is frozen:

```text
beam bucket order: (anchor_index, section_count)
state order:       objective, fewer sections, alias switches, replay fingerprint
outgoing anchors:  boundary time, boundary rank descending, anchor id
beat counts:       count-rank, N, bpm
```

Section scoring must use vectorized or prefix-backed computations over the
half-open score interval; no implementation may run a Python loop over all
frames for every attempted edge. The hard cap is `120000` scored section/tail
attempts per audio. Reaching it yields fallback reason
`edge_attempt_cap_exceeded`.

The implementation must include a synthetic stress benchmark with `16` origins,
`192` boundaries, `20` section cap, and dense feasible count sets. It passes
only if the run stays under the attempt cap, returns a deterministic tagged
fallback or path, and completes within the 180-second per-audio timeout on the
recorded machine.

## Global DP / Beam State

The assembler builds a path of section starts, not independent osu!-style
redline anchors. A complete path has:

- one origin anchor `tau_0`, selected from the origin-candidate cache;
- zero or more interior boundary anchors `tau_1 ... tau_m`, selected from the
  boundary-candidate cache in strictly increasing time;
- one terminal virtual endpoint at the exclusive cache end `E = coverage_end_ms`;
- one BPM `q_j` for every section beginning at `tau_j`.

The origin beat is always `0`. For an interior edge from `tau_j` to
`tau_(j+1)`, define:

```text
delta_ms = tau_(j+1) - tau_j
N_j      = positive integer beat count
q_j      = 60000 * N_j / delta_ms
b_(j+1)  = b_j + N_j
```

An interior edge is feasible only when `delta_ms >= 8000`, `N_j >= 1`, `q_j`
is in `[20,1000]`, and `N_j` survived the cached per-edge count cap. The
half-up rule for every count proposal is mandatory; Python banker's rounding is
forbidden.

For a terminal tail beginning at `tau_m`, `E` must not define the BPM. The
terminal BPM `q_m` is chosen from a frozen terminal tempo set:

- the BPM carried by the reached state, so a constant path can keep its origin
  tempo;
- every BPM in the final tempo pool;
- deduplicated by `round(bpm, 6)` and capped at 256 in deterministic order.

For each terminal BPM:

```text
p_m             = 60000 / q_m
tail_beat_count = max(1, ceil((E - tau_m) / p_m))
end_beat        = b_m + tail_beat_count
```

The terminal candidate is feasible only when `tau_m < E`, `q_m` is in
`[20,1000]`, and the final derived beat boundary covers `E`. Coverage is
half-open: frames represent `[0,E)`. The derived final boundary may equal `E`
or be after `E`; normally `E` lies inside the last beat interval. It must never
be before `E`, and no score term may reward making `E` an exact beat.

The first section is extended backward over the coverage prefix using its
section BPM:

```text
p_0        = 60000 / q_0
start_beat = floor((coverage_start_ms - tau_0) / p_0)
start_time = tau_0 + start_beat * p_0
```

This guarantees `start_time <= coverage_start_ms` and
`start_beat <= origin_beat < first_end_beat`. If `tau_0 <= coverage_start_ms`,
`start_beat` may be `0`; if `tau_0 > coverage_start_ms`, it is negative.

State stored in the beam:

- current anchor index and anchor time;
- section count so far;
- absolute integer beat at the current anchor;
- previous section BPM;
- previous alias-family representative;
- global downbeat phase modulo 4, or `none`;
- cumulative objective and deterministic replay fingerprint.

The global downbeat phase is selected only on the origin branch. If the centered
downbeat signal norm is zero, the phase is `none` and every downbeat cost is
zero. Otherwise the origin branch enumerates phases `{0,1,2,3}` relative to
absolute beat `0`. Interior edges propagate the phase by absolute beat counts:
when an edge adds `N_j`, the next boundary beat is `b_j + N_j`. No edge may
reset downbeat phase or choose a local replacement phase.

Beam width is `64` states per `(anchor_index, section_count)` bucket. Section
count cap is `20` for every track, including long tracks. Ties are broken by:

1. lower total objective;
2. fewer sections;
3. lower alias-switch count;
4. lower maximum boundary displacement from a materialized peak;
5. earlier first origin time;
6. lexicographically smaller edge tuple `(left_frame,right_frame,N,bpm_rounded_1e-6)`.

If no feasible global path exists, the result is a tagged fallback with reason
`no_global_constant_jump_path`.

## Frozen Cost Families

All candidate scores are minimized. Edge terms are duration-normalized so that
splitting a track does not reduce cost by creating more short sections.

For a section interval `[a_ms,b_ms)` with duration `L = b_ms - a_ms`, define:

```text
C_edge =
  1.00 * C_beat_support
+ 0.25 * C_peak_recall_precision
+ 0.20 * C_downbeat_phase
+ 0.10 * C_bpm_prior
+ 0.10 * C_beat_count_prior
+ 0.08 * C_section_duration
```

The scored intervals form one exact half-open partition of `[0,E)`:

- prefix interval: `[0, min(tau_0,E))`, using `q_0`;
- interior section interval `j`: `[max(tau_j,0), min(tau_(j+1),E))`,
  using `q_j`;
- tail interval: `[max(tau_m,0), E)`, using terminal `q_m`.

No frame can be scored twice. Empty intervals contribute zero. The mathematical
grid may extend before 0 ms and after `E`; score normalization only uses the
cache support `[0,E)`.

Transition cost between adjacent sections is:

```text
C_transition =
  0.18 * C_change_sparsity
+ 0.12 * C_alias_switch
+ 0.10 * C_jump_size
+ 0.15 * C_boundary_support
```

The total path objective is:

```text
D = coverage_end_ms - coverage_start_ms
J_duration =
  sum_partition_intervals(L_i * C_edge_i) / D
J_transition =
  sum_k C_transition_k / max(1, D / 60000)
J_total = J_duration + J_transition
```

The cost terms are frozen as follows:

- `C_beat_support`: `0.5 * (1 - grid_correlation)`, using the triangular pulse
  template with pulse width `40 ms`, clipped to `[0,1]`.
- `C_peak_recall_precision`: average of missed predicted grid beats and extra
  materialized peaks, using tolerance `min(45 ms, 0.15 * beat_length_ms)`.
- `C_downbeat_phase`: downbeat pulse correlation over beats whose absolute beat
  satisfies `(beat - global_downbeat_phase) mod 4 == 0`; equals `0` when global
  phase is `none`.
- `C_boundary_support`: `1 - max(beat_peak_confidence, downbeat_peak_confidence)`
  at the selected boundary, with linear falloff to zero support at `60 ms`; if
  no peak is within `60 ms`, support is `0`. This is a transition term only and
  is never weighted by the left section duration.
- `C_bpm_prior`: hard range only plus a soft penalty outside the preferred band
  `[80,240]`, computed as `abs(log2(bpm / clamp(bpm,80,240)))`.
- `C_beat_count_prior`: `0` for section beat counts in `[16,384]`, otherwise
  `min(1, abs(log2(N / clamp(N,16,384))))`.
- `C_section_duration`: `0` for durations in `[8000 ms, 180000 ms]`, otherwise
  the same clipped log penalty.
- `C_change_sparsity`: `1` per interior boundary.
- `C_alias_switch`: `0` when adjacent BPMs are within `0.5%`; `0.5` when they
  are compatible under the frozen alias orbit; `1` otherwise.
- `C_jump_size`: `min(1, abs(log2(bpm_right / bpm_left)))`.

`CJ1` includes only `C_beat_support`, `C_peak_recall_precision`, hard
feasibility, and tie-breaks. `CJ2` adds all non-downbeat priors. `CJ3` adds the
downbeat terms. No weights, formulas, caps, or tie-breaks may be tuned after
real-audio inspection.

The triangular pulse correlation is versioned as `pulse_correlation_v1`.
For frames in the half-open score interval and section lattice `(tau,bpm)`:

```text
p             = 60000 / bpm
phase_ms[i]   = mod(frame_time_ms[i] - tau, p)
distance_ms   = min(phase_ms[i], p - phase_ms[i])
template[i]   = max(0, 1 - distance_ms / 40)
grid_correlation = dot(center(signal), center(template)) /
                   (norm(center(signal)) * norm(center(template)))
```

If either centered norm is zero, the correlation is `-1` and the section must
fall back unless another accepted path avoids that interval.

## Priors

- BPM range: `[20,1000]` is a hard candidate-generation guard, not a musical
  truth claim.
- Preferred musical band: `[80,240]` is a soft prior only.
- Alias orbit: `{1/4, 1/3, 1/2, 1, 2, 3, 4}` is used for candidate expansion
  and alias-switch cost.
- Change sparsity: every jump must pay the frozen change cost; there is no free
  boundary solely because a local split score exists.
- Beat-count prior: very short and very long sections are penalized but not
  hidden unless they violate hard duration, BPM, or section-count constraints.
- Empirical BPM priors are out of scope for Experiment 004. Adding one requires
  a new card and a tuning-only fit before any holdout metric is inspected.

## Output Invariants

Every accepted `CJ*` result must construct a `TimingV3Grid` with:

- integer `start_beat` and `end_beat` for every `[a,b)` section;
- contiguous sections where `section[i].end_beat == section[i+1].start_beat`;
- `origin_beat == 0` and `origin_time_ms` equal to the first selected beat
  anchor;
- `origin_beat` inside the first half-open section;
- first `start_beat = floor((0 - origin_time_ms) / first_period_ms)`;
- final `end_beat = last_start_beat + max(1, ceil((coverage_end_ms -
  last_start_time_ms) / last_period_ms))`;
- terminal BPM is selected from the frozen terminal tempo set, never inferred
  from `coverage_end_ms`;
- finite BPM in `[20,1000]`;
- strict derived boundary times by prefix scan;
- `coverage_start_ms == 0` and `coverage_end_ms == 1000 * F / f`;
- coverage start and end covered by the derived section support;
- JSON round-trip boundary deltas no greater than `max(1e-6 ms, 8 ulp(time))`;
- no serialized adjacent independent time anchors;
- all diagnostics finite and JSON-safe.

A BPM jump is a derivative discontinuity at a shared beat/time boundary. It is
not a phase reset. The final section extends only far enough to cover the
exclusive cache end; audio end is not forced to be a beat.

## Minimal Change

Add a new source-owned global constant/jump assembly path and evaluator hooks.
Do not change the BeatThis cache format, existing v2 fitter behavior, ramp
renderer, mapper features, or inference service defaults in this experiment.

## Files Likely to Change

- `src/pulsefield_model/timing/v3/global_constant_jump.py` or equivalent new
  source module for candidate extraction, global assembly, result dataclasses,
  and diagnostics.
- `src/pulsefield_model/timing/v3/fitter.py` only after holdout100, broad500,
  and full5050 all pass unchanged; before then the global path remains an
  evaluator-only candidate.
- `src/pulsefield_model/timing/evaluation/` for an Experiment 004 evaluator,
  manifest selection, summaries, and source-owned provenance.
- `tests/timing/` for synthetic candidate extraction, DP, leakage, schema,
  resume, and evaluator tests.
- `docs/research/timing_v3_problem_log.md` only after results or decisions
  exist.

## Read-Only Context Files

- [timing_v3_task_definition.md](timing_v3_task_definition.md)
- [timing_v3_experiment_001_evaluation_foundation.md](timing_v3_experiment_001_evaluation_foundation.md)
- [timing_v3_experiment_001_result.md](timing_v3_experiment_001_result.md)
- [timing_v3_experiment_002_phase_continuous_projection.md](timing_v3_experiment_002_phase_continuous_projection.md)
- [timing_v3_experiment_002_result.md](timing_v3_experiment_002_result.md)
- [timing_v3_experiment_003_joint_phase_projection.md](timing_v3_experiment_003_joint_phase_projection.md)
- [timing_v3_problem_log.md](timing_v3_problem_log.md)
- `src/pulsefield_model/timing/providers/beatthis_cache.py`
- `src/pulsefield_model/timing/schema.py`
- `src/pulsefield_model/timing/grid_fitting/`
- `src/pulsefield_model/timing/v3/schema.py`
- `src/pulsefield_model/timing/v3/projection.py`
- `src/pulsefield_model/timing/v3/joint_projection.py`

## Dataset Slice

All splits are by cache audio key first. Difficulties of the same audio may
never cross stages.

### Repair set

Use the existing exposed 80-audio timing-v3 pilot from Experiment 001/002 as a
repair and regression set. It includes stable, jump, dense, ramp-audit, long,
and anomaly rows. Comparator-unavailable rows count in projection/fallback
denominators but not oracle-dependent phase metrics. Repair80 can expose bugs,
but it cannot accept the candidate and cannot justify threshold, weight,
candidate, or tie-break changes. Any algorithmic change made after repair80
inspection restarts repair80 before holdout creation.

### Holdout100

Create a new Experiment 004 holdout only after code, candidate extraction,
weights, tie-breaks, and synthetic tests are frozen. The exclusion set is every
audio key for which an implementer has viewed candidate-relative or individual
Timing-v3 `.osu` oracle/comparator rows before the Experiment 004 holdout
manifest is frozen. A precomputed full-corpus v2 baseline does not by itself
exclude all 5,050 audio keys; exclusion is caused by human/agent inspection of
individual oracle rows or candidate-relative oracle metrics.

The selector must read only an exposure-exclusion manifest, never metric
values. The required manifest schema is:

```text
pulsefield_model.timing_v3_exp004_oracle_exposure_exclusion_manifest_v1
```

It must contain sorted unique entries with `cache_audio_key`,
`exposure_reason`, `exposure_source`, and `first_exposed_at_or_run_id`.
Manifest-level fields `schema_id`, `generated_from_commit`,
`exposure_scan_source_sha256`, `generated_at_utc`, `entries_sha256`, and
`minimum_required_exclusion_keys` are also required. Missing manifest, stale
source hash, duplicate key, malformed key, missing required minimum entry, or
incomplete provenance is fail-closed: no holdout may be selected.

The minimum required exclusion count before ad hoc additions is 183:

- the 80-audio repair/pilot set;
- the 3 Experiment 003 protocol-exposed audio keys;
- the 100 Experiment 003 v2 holdout audio keys whose individual oracle rows were
  exposed for protocol preparation.

Any later ad hoc debug audio where Timing-v3 oracle phase, drift, boundary, or
candidate-relative comparator rows were viewed must be appended before
selection. If exposure status is uncertain, exclude the audio key. Store the
exact exclusion list and SHA-256 before selecting holdout100.

Selector tests must prove that metric-valued fields are absent from the manifest
and that adding one causes schema rejection rather than silent ignore.

Rank within each quota by:

```text
sha256("timing-v3-exp004-holdout100-v1\0" + cache_audio_key)
```

Exclusive priority order:

```text
ramp_audit -> anomaly -> long -> dense -> jump -> stable
```

Quotas:

| Quota | Count | Definition |
| --- | ---: | --- |
| stable | 40 | `label.stratum == stable` |
| jump | 25 | `label.stratum == jump_candidate` |
| dense | 10 | `label.stratum == dense` |
| ramp audit | 5 | `label.stratum == ramp_candidate` |
| long | 10 | `source.long_track == true` after higher priorities |
| anomaly | 10 | `label.stratum == ambiguous` after ramp priority |

If a quota is underfilled after exclusions, use the preregistered degraded
selection rule instead of stopping:

1. Select all available rows in the underfilled quota.
2. Record the quota as `degraded_underfilled` with requested and available
   counts.
3. Fill the remaining holdout slots from nonselected rows ranked by:

   ```text
   sha256("timing-v3-exp004-holdout100-deficit-v1\0" + cache_audio_key)
   ```

4. Apply fixed deficit-fill class priority:
   `jump -> dense -> long -> anomaly -> ramp_audit -> stable`.
5. If the final holdout has fewer than 100 available audio groups, stop and
   return to planner mode.

A degraded quota cannot produce a positive stratum-specific decision for that
quota. If jump has fewer than 15 comparison-eligible audio groups or long has
fewer than five, the stage can be at most ambiguous even when aggregate metrics
pass.

### Broad500 and Full5050

If holdout100 passes unchanged, materialize broad500 as the holdout100 plus the
400 lowest-ranked new audio keys under:

```text
sha256("timing-v3-exp004-broad500-v1\0" + cache_audio_key)
```

Exclude every key in the frozen oracle-metric exposure set and every holdout100
key, then deduplicate by cache audio key. No quota is applied to the added 400.
If fewer than 400 new keys remain, broad500 is degraded underfilled and the
experiment can be at most ambiguous until a new card defines a smaller broad
stage.

If broad500 passes unchanged, evaluate all 5,050 cached audio groups with the
same source/config hashes. The full5050 stage includes previously exposed audio
because it is a final replay and tail-risk report, not a new holdout. Any
algorithm, weight, candidate, gate, or comparator-policy change after holdout
inspection restarts with a new card and a new audio-disjoint holdout.

## Baseline / Comparator

Primary comparator: the current source-owned cache-backed v2 `GridFitter`
baseline at the Experiment 004 run commit. Experiment 003 has not accepted a
production v3 candidate for this card, so no v3 projection is the primary
comparator. The comparator must consume the same cached BeatThis prediction.

Secondary comparators:

- `CJ0`, `CJ1`, and `CJ2` ablations;
- source-relative comparison to the current v2 grid, clearly marked as source
  comparison rather than oracle truth;
- `.osu` redline/object-grid evidence only in the evaluation layer;
- any Experiment 002/003 projection controls only as non-primary context when
  their source/config hashes are recorded.

## Weak Comparator Contract

`.osu` redlines and objects remain weak evaluation evidence. They may not define
section truth. Experiment 004 reports weak-boundary agreement, not true jump
precision/recall or true section IoU.

For each valid difficulty:

- parse only uninherited red timing points;
- reject a difficulty comparator when red timing has nonpositive beat length,
  nonfinite fields, or no valid red point;
- a weak redline boundary is an adjacent uninherited timing-point change with
  `abs(log2(bpm_right / bpm_left)) >= log2(1.005)` and absolute time inside
  `[coverage_start_ms, coverage_end_ms)`;
- dense and ramp-audit difficulties are reported as robustness strata only.

Boundary matching is one-to-one greedy by increasing absolute time error, after
sorting candidate pairs by `(abs_error_ms, predicted_time_ms, oracle_time_ms)`.
A pair is eligible when:

```text
abs_error_ms <= min(750, 0.5 * min(pred_left_period_ms,
                                   pred_right_period_ms,
                                   oracle_left_period_ms,
                                   oracle_right_period_ms))
```

Report matched signed error, absolute error, unmatched predicted boundary
count, and unmatched weak-redline boundary count. Name these metrics
`weak_boundary_*`; do not name them jump precision, jump recall, or IoU.

Cross-difficulty aggregation is audio-first:

1. compute difficulty-level weak metrics for every valid comparator;
2. aggregate continuous values by median across valid difficulties for the same
   audio key;
3. aggregate counts by sum across difficulties and also report per-difficulty
   means;
4. mark a predicted boundary `weak_consensus_supported` only if it matches at
   least half of valid difficulties, or at least two difficulties when three or
   more valid difficulties exist.

Alias-aware BPM error for a predicted BPM `p` and weak comparator BPM `r` is:

```text
min(abs(p * m - r) for m in {1/4,1/3,1/2,1,2,3,4}
    if 20 <= p * m <= 1000)
```

Alias-normalized drift is strictly bound to the existing source-owned
`canonical_bpm_80_160` / `TIMING_CANONICALIZATION_BPM_80_160` behavior and the
canonicalization source SHA at the run commit. Experiment 004 must not add a
separate closest-to-120 rule, alternate tie-break, or local alias policy. If the
canonicalization source SHA changes after holdout selection, all stale results
are invalid.

## Phase Sampling Contract

Phase, drift, and active-section metrics use `phase_sampling_v1`:

- sample times are `t_k = 20 * k ms` for all integers `k` with
  `coverage_start_ms <= t_k < coverage_end_ms`;
- predicted and weak-comparator sections are half-open `[start,end)`; a sample
  exactly on an interior boundary belongs to the right section, and no sample is
  taken at the exclusive coverage end;
- cumulative beat functions `B_pred(t)` and `B_ref(t)` are evaluated from their
  active sections after the required alias-normalization path is selected;
- circular phase error in beats is
  `abs(((B_pred(t) - B_ref(t) + 0.5) mod 1) - 0.5)`;
- phase error in milliseconds is circular phase error in beats multiplied by
  the active reference beat length at `t`;
- endpoint and prefix drift use the same sample partition plus the exact
  half-open endpoint limit, never a rounded beat at `coverage_end_ms`.

Any existing evaluator function used by Experiment 004 must be wrapped or
versioned so these sampling and partition semantics are enforced.

## Primary Metric

On pure `CJ3` matched rows only:

- pure `CJ3` versus current v2 mean phase-error ratio;
- pure `CJ3` versus current v2 p90 phase-error ratio.

Fallback-selected comparator rows are product-safety reporting only. They are
not allowed to make the primary phase denominator look better. The primary
decision is made at the audio-group level. Difficulty-level metrics are
secondary detail.

## Secondary Metric

Report all of the following for `CJ0`, `CJ1`, `CJ2`, `CJ3`, current v2, and any
non-primary projection control explicitly frozen before real-audio metric
inspection:

- mean, p50, p90, and max phase error in ms and beats;
- raw and alias-aware local BPM error;
- raw and alias-normalized endpoint drift;
- raw and alias-normalized max prefix drift;
- raw and alias-normalized drift slope in ms/min;
- p90 30-second and 60-second local drift;
- weak-boundary match rate, signed error, absolute error, unmatched predicted
  boundary count, unmatched weak-redline boundary count, and
  weak-consensus-supported boundary count;
- redline-section disagreement and active-section disagreement against weak
  comparators, with dense and ramp-audit rows reported as robustness only;
- predicted section count, jump count, and section duration distribution;
- original/source-relative boundary disagreement and active-section
  disagreement;
- fallback rate and fallback reason distribution;
- cache load, candidate extraction, no-path, schema, serialization, timeout,
  and comparator-unavailable counts;
- runtime seconds per audio, peak memory when available, candidate peak count,
  boundary candidate count, edge count, beam-pruned state count, and deterministic
  replay hash;
- selected-fallback product-safety phase ratios, reported separately from pure
  `CJ3` acceptance metrics.

## Verify Command / Evaluation Procedure

The implementation card must freeze exact commands before running real audio.
The intended procedure is:

1. Unit tests for schema, peak extraction, candidate caps, cost terms, DP/beam
   tie-breaks, fallback reasons, metadata traps, and deterministic replay.
2. Synthetic fixtures for constant tempo, exact jumps, half/double aliases,
   weak or absent boundaries, empty beat signal, downbeat/no-downbeat
   equivalence, terminal `E` inside the final beat interval, and JSON
   round-trip.
3. Exposure-exclusion manifest schema tests for missing, stale, duplicate,
   incomplete, malformed, metric-valued, and minimum-count-invalid manifests.
4. Boundary evidence edge tests for absent ordinary/super windows, exact
   ordinary/super ties, carried winner periods, and downbeat tolerance.
5. Lazy-DP stress benchmark with maximum origin/boundary candidates and dense
   feasible counts.
6. Repair80 run for all four `CJ*` variants plus comparators.
7. Holdout100 run without changing code/config.
8. Broad500 run without changing code/config.
9. Full5050 run without changing code/config.
10. Byte-identical replay of manifests and mathematical grids from stored
   source/config hashes.

## Guard Check

Hard guards:

- no inference access to `.osu`, hit objects, metadata, labels, network, or raw
  audio;
- no cache/config mismatch or stale resume reuse;
- every accepted `CJ*` grid passes v3 schema and serialization invariants;
- fallback is explicit and reason-tagged;
- section count <= 20;
- projection/fallback denominator is separate from comparator availability;
- all stage metrics are grouped by audio key;
- all result rows are finite JSON and deterministic under replay.

Denominators are frozen for every stage:

- `stage_audio_count`: selected audio keys in the manifest;
- `cache_valid_count`: selected keys whose declared BeatThis cache loads and
  matches config/provenance;
- `projection_evaluable_count`: cache-valid rows where the candidate returns an
  accepted grid or tagged fallback before comparator reading;
- `comparison_eligible_count`: projection-evaluable rows with at least one valid
  weak `.osu` comparator;
- `pure_CJ3_phase_count`: comparison-eligible rows where `CJ3` itself produced
  a grid and both pure `CJ3` and current v2 have matched phase metrics;
- `pure_CJ3_phase_coverage`: `pure_CJ3_phase_count / comparison_eligible_count`;
- `selected_safety_phase_count`: comparison-eligible rows scored by
  `selected_CJ3_or_v2_fallback`, where a `CJ3` fallback contributes the current
  v2 comparator grid and still counts against fallback rate. This is product
  safety only, not primary acceptance.

Stage gates for holdout100, broad500, and full5050 use the same pass,
ambiguous, and kill bands:

| Metric | Pass | Ambiguous | Kill |
| --- | --- | --- | --- |
| mean phase ratio | `<= 1.05` | `(1.05, 1.10]` | `> 1.10` |
| p90 phase ratio | `<= 1.10` | `(1.10, 1.15]` | `> 1.15` |
| pure `CJ3` phase coverage | `>= 95%` | `[90%, 95%)` | `< 90%` |
| stable mean or p90 ratio, if `n >= 5` | `<= 1.10` | `(1.10, 1.20]` | `> 1.20` |
| jump mean ratio, if `n >= 15` | `<= 1.00` or drift mean `<= 0.90` | otherwise up to kill band | mean `> 1.10` and drift mean `> 1.10` |
| long max-prefix drift mean and p90, if `n >= 5` | `<= 1.15` | `(1.15, 1.30]` | `> 1.30` |
| fallback rate on projection-evaluable rows | `<= 5%` | `(5%, 10%]` | `> 10%` |
| no-path plus candidate-extraction failure rate | `<= 3%` | `(3%, 5%]` | `> 5%` |
| p90 runtime per audio | `<= 30 s` | `(30 s, 60 s]` | `> 60 s` |

Hard leakage, schema-invalid accepted grids, stale resume reuse, nonfinite JSON,
or cache/config mismatch is immediate kill for the run. Comparator-unavailable
rows do not enter phase ratios but remain in fallback and failure denominators.

Stage actions:

- repair80: debug/regression only; no accept/kill decision unless a hard guard
  fails.
- holdout100 pass: freeze source/config and materialize broad500.
- holdout100 ambiguous: stop, audit only failure attribution, and create a new
  card before mutation.
- holdout100 kill: kill this candidate family for the loop.
- broad500 pass: freeze source/config and run full5050.
- broad500 ambiguous: stop; no production fitter switch; create a new card.
- broad500 kill: kill this candidate family for the loop.
- full5050 pass: accept the candidate for later production integration.
- full5050 ambiguous: keep the result as research-only and create a new card.
- full5050 kill: do not switch the production fitter; write a negative result.

## Qualitative Check

For repair80 and holdout100 only, inspect representative diagnostic overlays
after metrics are frozen:

- five best and five worst stable rows by phase ratio;
- five best and five worst jump rows by boundary error;
- every fallback reason with at least two examples;
- all catastrophic regressions where phase ratio exceeds `1.5`;
- at least three long-track examples if present.

Qualitative inspection may classify comparator uncertainty, but it may not
change the frozen prediction, thresholds, weights, or holdout membership.

## Positive Signal

- `CJ3` passes holdout100, broad500, and full5050 unchanged under the stage
  gates above;
- `CJ3` beats `CJ0`, `CJ1`, and `CJ2` on the primary jump/long signals without
  stable regression;
- fallback stays <= `5%` and is concentrated in low BeatThis evidence or weak
  comparator cases;
- boundary and drift improvements are visible by module attribution rather than
  only as aggregate phase averages;
- full5050 replay is deterministic from recorded hashes.

## Negative Signal

- `CJ3` enters a kill band on holdout100, broad500, or full5050;
- `CJ3` is not better than `CJ1` or `CJ2`, making downbeat or prior terms
  unjustified;
- jump or long strata regress while aggregate metrics pass;
- fallback, no-path, schema, or timeout rates exceed guards;
- qualitative review shows improvements are mostly comparator artifacts.

## Kill Criteria

Kill the global constant/jump path for this loop if any of these occur:

- any holdout100, broad500, or full5050 metric enters the kill band in the stage
  gate table;
- `CJ3` does not beat both `CJ1` and `CJ2` on either jump phase or jump drift on
  holdout100 when the jump denominator is sufficient;
- no-path or candidate-extraction failures exceed the stage kill band;
- p90 runtime exceeds the stage kill band on the recorded machine;
- implementation requires `.osu`, metadata, network, raw audio, ramps, learned
  priors, or decomposition to pass.

Do not mutate weights or add rescue heuristics inside this card.

## Ambiguous Decision Rules

- Any metric in the ambiguous band is ambiguous, not negative and not positive.
  Audit only failed attribution layers and create a new card before mutation.
- Aggregate passes but jump or long denominator is below the minimum after
  preregistered quota degradation: ambiguous.
- `CJ3` passes but differs from `CJ2` by less than `1%` relative on every
  headline metric: accept `CJ2` as the smaller candidate only if all
  downbeat-specific diagnostics are neutral; otherwise mark ambiguous for human
  owner decision.
- Metrics improve but comparator-unavailable or weak-label rows dominate the
  apparent gain: ambiguous comparator uncertainty, not an algorithm win.
- Dense or ramp-audit rows regress catastrophically while formal constant/jump
  guards pass: ambiguous review stop. Ramp truth is not inferred.

## Expected Failure Modes

- BeatThis peaks are weak, syncopated, or half/double-time ambiguous.
- Downbeat probabilities prefer mapper conventions rather than audio structure.
- Real jumps are unsupported by a local activation change and are penalized
  away by sparsity.
- Dense redline or ramp-audit comparators punish a constant/jump-only model.
- A global beam keeps the wrong early alias and accumulates drift.
- Long tracks exceed candidate caps without decomposition.
- The current weak comparator is wrong or unavailable.

## Confounders

- `.osu` redlines and official BPM snapshots are map-derived, correlated
  comparators.
- Object-grid evidence validates phase more reliably than tempo alias.
- Ramp-audit rows are deliberately weak and cannot establish ramp precision.
- Stable tracks dominate the full corpus, while repair and holdout strata are
  deliberately harder.
- Runtime depends on machine, Python, NumPy, and BLAS behavior; those versions
  must be recorded.

## Error Attribution

Every result row must assign failures and regressions in this order:

1. cache validity and provenance;
2. BeatThis evidence quality;
3. peak materialization;
4. tempo candidate generation;
5. boundary candidate generation;
6. edge feasibility and beat-count selection;
7. global assembly and beam pruning;
8. v3 schema and serialization;
9. fallback selection;
10. comparator availability and uncertainty;
11. runtime or resource limit.

A weak comparator remains comparator uncertainty unless cache evidence, source
comparison, and object/redline evidence all indicate an algorithm failure.

## Strata

Report every metric by:

- `label_stratum`: stable, jump_candidate, dense, ramp_candidate, ambiguous;
- `label_confidence`;
- `label_ambiguous`;
- `source_long_track`;
- duration bins: `<60s`, `60-180s`, `180-600s`, `>=600s`;
- comparator availability;
- BeatThis evidence quality tercile by peak support;
- predicted section count: `1`, `2-4`, `5-10`, `>10`;
- predicted jump count: `0`, `1`, `2-4`, `>4`;
- primary BPM band: `<80`, `80-160`, `160-240`, `>240`;
- alias-switch count: `0`, `1`, `>1`;
- fallback reason.

## Expected Runtime / Runtime Budget

- Synthetic and unit tests: under 2 minutes.
- Repair80: under 30 minutes hard stop.
- Holdout100: under 45 minutes hard stop.
- Broad500: under 4 hours hard stop.
- Full5050: under 36 hours hard stop with resumable checkpoints.
- Per-audio timeout: 180 seconds.
- Beam/candidate cap failures are reported as tagged fallbacks, not silent
  truncation successes.

## Reproducibility Hashes

Every manifest, row, summary, and result log must record:

- git commit SHA and dirty-file list;
- this Experiment Card SHA-256;
- implementation source module SHA-256 values;
- evaluator source module SHA-256 values;
- candidate-extraction/config JSON SHA-256;
- cache config fingerprint, cache version, checkpoint path, `shift_ms`, and
  frame rate;
- input inventory SHA-256 and label JSONL SHA-256;
- required exposure-exclusion manifest SHA-256, schema, source hash, entry
  count, and selected/excluded key-set SHA-256;
- repair, holdout100, broad500, and full5050 manifest SHA-256 values;
- selected and excluded cache audio key set SHA-256 values;
- per-row cache audio key hash and cache file hash when available;
- Python, NumPy, Torch, platform, and accelerator metadata;
- deterministic replay fingerprint for each mathematical grid and search path.

No result number belongs in this pre-registration section before the run.

## Out of Scope

- Ramp representation, ramp detection, and ramp precision/recall.
- Long-track decomposition, block overlap sweeps, or posterior handoff.
- Multiple shifted BeatThis passes, raw-audio inference, model retraining, or
  learned priors.
- `.osu` or network evidence in inference.
- Mapper/control-feature changes.
- Relaxing Experiment 002/003 schema invariants.

## Result Interpretation Plan

- Positive result would suggest: after holdout100, broad500, and full5050 all
  pass unchanged, implement `CJ3` or the accepted simpler ablation behind the
  `TimingV3Fitter` interface in a separate integration change.
- Negative result would suggest: kill cache-only global constant/jump assembly
  for this loop and return to candidate extraction or BeatThis evidence quality,
  not weight tuning.
- Ambiguous result would require: one new card targeted only at the failed
  stratum or comparator uncertainty.
- Human owner decides: whether `CJ2` is preferable to `CJ3` when downbeat terms
  are neutral, and whether weak comparator uncertainty blocks promotion.
- Next-loop action if positive: update the timing problem log, then make a
  scoped production-fitter integration change; only after that should ramp or
  decomposition work be planned.
- Next-loop action if negative: write a result log and either kill or mutate to
  a smaller candidate-extraction experiment.
- Next-loop action if ambiguous: audit failed strata without changing
  predictions, then create a new card.

## Result Log Template

- Experiment: Timing v3 Experiment 004
- Status: pre-registered | repair run | holdout run | broad run | full run |
  accepted | negative | ambiguous | killed
- Date:
- Commit / run id:
- Dirty worktree:
- Experiment card SHA-256:
- Implementation source hashes:
- Evaluator source hashes:
- Candidate/config hash:
- Cache fingerprint:
- Inventory / label hashes:
- Manifest hashes:
- Exposure-exclusion manifest hash:
- Exposure-exclusion entry count:
- Dataset slice:
- Baseline / comparator:
- Candidate variants evaluated:
- Runtime:
- Projection-evaluable audio count:
- Comparison-eligible audio count:
- Comparator-unavailable audio count:
- Primary metric values:
- Secondary metric values:
- Guard results:
- Fallback reasons:
- Module attribution:
- Stratified outcomes:
- Qualitative observations:
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
- Ambiguous decision rule triggered:
- Failed checks:
- Suspected confounders:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes, only for the scoped Experiment
  004 implementation and verifier surface.
- Closed loop complete: yes
- Remaining ambiguity: the current repository may not yet contain an accepted
  Experiment 003 result. Experiment 004 therefore uses current v2 as its
  primary comparator unless a new preregistration supersedes this card before
  any real-audio metrics are inspected.

## Next-Loop Action

- If positive: promote the accepted `CJ*` candidate behind the timing-v3 fitter
  interface only after holdout100, broad500, and full5050 all pass unchanged.
- If negative: kill this candidate family or mutate only candidate extraction
  under a new Experiment Card.
- If ambiguous: audit the failed stratum or comparator layer and do not add
  ramps, decomposition, or learned priors until the constant/jump decision is
  closed.

## Novelty Notes

- Closest analogies: DBN/DP beat trackers, tempo Viterbi decoders, and
  change-point search over beat activations.
- Novelty layer, if any: Pulsefield-specific representation/evaluation
  coupling rather than the algorithmic family.
- Representation novelty vs engineering variation: the half-open
  phase-continuous v3 grid is the representation constraint; this experiment is
  an engineering selection of a global assembler under that constraint.
