# Timing v3 Experiment 005: Phase-Continuous Local Frontier Decomposition

Status: pre-registered; not run. This card freezes the bounded jump-only
experiment before Experiment 005 implementation or real-audio output is
inspected. It does not authorize holdout selection until the scheduler sweep,
synthetic verification, and repair80 gate are complete.

## Mode

- Mode: planner
- Route: TEST
- Source idea: replace Experiment 004's whole-track boundary/tempo/count graph
  with fixed-lag local inference that carries a bounded posterior over global
  beat phase and tempo across common time cuts.
- Acceptance source: the Timing v3 task definition's long-track decomposition
  gate and the negative Experiment 004 runtime result.
- Source snapshot / evidence grade: strong for the one-cache input, v3 schema,
  current-v2 baseline, candidate extraction, runtime failure, and existing
  evaluator contracts; medium for `.osu` redline/object weak comparators; weak
  for ramp semantics and external catalog BPM.
- Task definition SHA-256:
  `1fdde4a16d061aeb009bdfb1f622d9ede966fe9273b0b6cf18b8684117e47c73`
- Experiment 004 result SHA-256:
  `69e8eaf27583cf1b9b1d8bd9cae4fbcd218c72ad86880646407695a661bf9f79`
- Problem log SHA-256:
  `1907330b256d39bce46cee5f0a4a724eaab26f2002657be015e9bcfd6f9d4766`

## Hypothesis

A fixed-lag local search can preserve one absolute beat axis while making
runtime approximately linear in track duration. Carrying the top 16 distinct
phase/tempo frontier states across blocks, and using right-side lookahead to
rank those states, will avoid the phase resets of independent windows and the
premature alias lock of a single best prefix. It should therefore remove the
whole-track timeouts observed in Experiment 004 without materially regressing
current-v2 phase or cumulative drift.

The hypothesis is only about constant BPM sections and instantaneous BPM
jumps. Analytic ramp sections remain out of scope.

## Root Objective

Determine whether one shift-zero BeatThis cache can support a bounded,
phase-continuous constant/jump decoder whose output remains:

```text
beat [a,b) bpm q
```

on one global integer beat axis, while satisfying the existing runtime,
fallback, phase, drift, section-count, and serialization guards through a fresh
100 -> 500 -> 5,050 progression.

## Goal Decomposition

- Subgoal 1: define a decomposition cut state that carries exact beat phase,
  tempo/alias, downbeat residue, open-section state, and deterministic history
  without turning an internal cut into a false output boundary.
- Subgoal 2: bound each local candidate graph and export a small multi-state
  posterior so cost scales with the number of blocks instead of the square of
  whole-track boundary candidates.
- Subgoal 3: select block core/lookahead duration from the prescribed
  `30/10`, `60/20`, `90/30`, and adaptive `64/16` candidates on an exposed,
  deterministic scheduler slice before any fresh holdout is selected.
- Subgoal 4: attribute error separately to cache evidence, bootstrap,
  frontier pruning, local jump search, overlap disagreement, assembly,
  comparator uncertainty, and resource limits.
- Subgoal 5: keep inference cache-only and retain current v2 as the unchanged
  safety fallback and paired comparator.

## Candidate Variants

The core candidate extraction, local beam width, BPM priors, output schema, and
resource guards are shared. These four architecture variants differ only in
how state crosses a decomposition cut:

- Variant A, `LF0 independent_windows`: every core window chooses a new local
  origin/alias. It is the required negative control and cannot be selected
  because independent phase resets violate the task contract.
- Variant B, `LF1 scalar_prefix`: carry only the single best phase/tempo state
  at each cut. This tests early alias/phase lock but is too brittle to be the
  selected family.
- Variant C, `LF2 frontier16_no_lookahead`: carry 16 states but rank them only
  on committed core evidence. This isolates the value of the posterior from
  the value of overlap.
- Variant D, `LF3 frontier16_fixed_lag`: carry 16 states and rank the cut
  posterior with right-side lookahead whose score is not committed twice.
  This is the selected family.

For Variant D, block duration is a tuning configuration with exactly four
allowed arms:

- `S30`: 30 s core, 10 s right lookahead;
- `S60`: 60 s core, 20 s right lookahead;
- `S90`: 90 s core, 30 s right lookahead;
- `S64`: bootstrap with 90/30 s, then target 64 beats of core and 16 beats of
  lookahead using the best-ranked frontier BPM only for scheduling. Core is
  clamped to 30-90 s and lookahead to 10-30 s.

The scheduler sweep is part of the selected Variant D protocol. It returns one
frozen configuration by the total ordering below. No arm may be chosen by
visual preference or changed after a fresh holdout is selected.

## Local Verification Matrix

- `LF0 independent_windows`: on a synthetic constant track spanning at least
  four cuts, it must expose phase/alias disagreement when local evidence is
  intentionally ambiguous. It is retained only as an attribution control.
- `LF1 scalar_prefix`: on an early half/double-tempo ambiguity resolved by
  later evidence, it must demonstrate the expected irreversible wrong-prefix
  failure while remaining deterministic.
- `LF2 frontier16_no_lookahead`: on the same fixture, it must retain the correct
  alias/phase state through the cut; a jump placed just after the cut may still
  be localized poorly without lookahead.
- `LF3 frontier16_fixed_lag`: it must retain the correct state and localize
  jumps immediately before, inside, and after the lookahead region without an
  artificial output seam.
- `S30/S60/S90/S64`: all schedules must reproduce a long synthetic constant
  grid and the same well-supported jump grid. `S30` is expected to have more
  cut overhead; `S90` a larger local graph; `S64` a more stable musical-evidence
  count across BPM. Any schedule that changes the mathematical grid on an
  unambiguous fixture is invalid rather than merely slower.
- Frontier-width check: internal local buckets retain width 64, while only 16
  states cross a cut. Frontier classes use the frozen alias-family
  representative plus downbeat phase. Reserve the best state from each present
  class before filling remaining slots by global state order; if more than 16
  classes are present, retain the 16 class representatives with the best state
  order. A tied fixture with exactly 16 present classes must preserve all 16.
  Downbeat-free fixtures use the single `none` phase class. This reservation
  does not protect multiple exact-phase hypotheses inside one class; pruning a
  later-correct same-class phase is a recorded negative signal and returns to
  `MUTATE`. Width 64 across blocks is rejected for this card because it permits
  up to four times as many incoming states and may increase joint local work
  substantially without local evidence that 16 loses a needed class.
- Cut-state check: fixed time cuts with an exact phase/open-section state are
  required. The rejected control either gives different states path-specific
  cut times or re-anchors every state's lattice so the common cut becomes an
  integer beat/output boundary. Merely scheduling a common cut at the best
  state's natural beat while other states retain exact fractional phase is not
  this control. Synthetic verification records whether the rejected control
  violates phase, section, or boundary invariants without presuming which
  failure must occur.

If Variant D fails its synthetic local verification, stop and return `MUTATE`.
Do not promote A-C or tune D on the failing fixture in this card.

## Selected Variant

- Selected: `LF3 frontier16_fixed_lag`, with block schedule chosen once by the
  pre-registered four-arm scheduler protocol on the exposed schedule16 slice.
- Rejected: independent windows reset phase; scalar prefix lock cannot recover
  from early alias ambiguity; no-lookahead frontier lacks evidence across the
  decision boundary; an unrestricted final whole-track replay recreates the
  Experiment 004 failure.
- Why this is the smallest useful test: it changes only search decomposition
  and posterior handoff while reusing the one-cache candidate evidence,
  constant-section schema, cost families, weak metrics, and v2 comparator.

## Selection Pressure

- Primary pressure: eliminate repair-row timeouts and keep selected-vs-v2 mean
  phase, p90 phase, and long-track cumulative drift within frozen guards.
- Guard pressure: exact beat/phase continuity, no inference leakage, no stale
  resume, bounded sections/frontier/candidates, explicit fallback, and finite
  deterministic JSON.
- Runtime pressure: bound local graphs, keep every row under 180 s, and keep
  p90 runtime under 30 s on formal stages.
- Duration pressure: prefer weak change-boundary agreement and low cumulative
  drift first, then low overlap disagreement, runtime, and memory.
- Kill pressure: if multi-frontier fixed-lag cannot pass unchanged, do not add
  ramps, learned priors, more frontier states, a larger timeout, or a global
  repair pass inside this card.

## Research Question

Can bounded fixed-lag inference carry enough phase/tempo posterior information
to assemble a single-cache, globally phase-continuous jump grid, without the
whole-track graph cost that killed Experiment 004?

## Closest Analogies / Novelty Layer

- Closest analogies: fixed-lag Viterbi smoothing, streaming beam search,
  receding-horizon decoding, boundary-conditioned dynamic programming, and
  overlap-save processing.
- Relevant taxonomy bucket: structured postprocessing and long-sequence
  decomposition.
- Novelty layer, if any: Pulsefield-specific state and evaluation coupling for
  one BeatThis cache, half-open integer-beat sections, and audio-grouped osu!
  weak comparators.
- Representation novelty vs engineering variation: fixed-lag search and beam
  handoff are established techniques. This is an engineering selection test,
  not a novelty claim.

## Inference Boundary

Prediction may read only:

- one `FrameTimingPrediction.beat_prob` array;
- the matching `downbeat_prob` array;
- `frame_rate_hz`;
- cache identity/config/provenance required for integrity and replay.

It must not read `.osu`, redlines, objects, difficulty names, audio/title/artist
metadata, label rows, official API BPM, network data, previous evaluator
results, or raw audio. The runner must construct a restricted prediction with
`source_path=None`, make signal arrays read-only, and verify memory sharing and
cache identity before and after each row.

Candidate extraction runs once over the cache, before decomposition, using the
Experiment 004 candidate contract v1. Blocks filter that immutable candidate
set; they do not independently peak-pick or produce window-dependent global
tempo candidates. Candidate and restricted-signal fingerprints are carried
into every row and resume key.

This is fixed-lag scoring over a whole-cache proposal set, not a causal online
decoder. Whole-cache extraction and the global candidate rank may reveal that
a proposal exists from evidence beyond the current lookahead; only evidence in
the active core-plus-lookahead interval contributes local support and path
score. The experiment therefore claims bounded assembly cost and phase handoff,
not strict future-signal isolation.

## Bootstrap Contract

Block 0 creates its initial open states directly from the immutable Experiment
004 origin-candidate tuple, in tuple order. For every origin
`(anchor_id, tau_ms, q_origin, score)`:

- set the global origin to `(origin_beat=0, origin_time_ms=tau_ms)`;
- set the logical open-section start to `(0, tau_ms, q_origin)`, with
  `previous_bpm=none`, zero committed objectives, zero alias switches, and
  `real_section_count=1` because the open first section already counts toward
  the global 20-section cap;
- when centered downbeat signal norm is zero, enumerate only phase `none`;
  otherwise enumerate phases `0,1,2,3` in that order;
- compute `next_beat_index` as the smallest integer whose origin lattice time
  is on or after `coverage_start_ms`, using the frozen half-open ULP rule, and
  derive `next_beat_time_ms` from that same lattice;
- compute `serialized_first_start_beat` as the greatest integer whose origin
  lattice time is on or before `coverage_start_ms`. This may be negative and
  only extends the mathematical first section backward to cover the cache; it
  does not change the logical prior anchor at beat 0;
- initialize the replay key from origin tuple position, exact origin time/BPM,
  and downbeat phase.

All bootstrap states enter one joint first-block beam. Deduplicate, order, and
retain the first 64 exactly as for later local buckets, recording dominance and
width prunes. Missing origins, no valid retained bootstrap state, or an invalid
coverage extension produces the corresponding tagged fallback. For the first
real section, the beat-count prior always uses
`closure_beat - origin_beat`; it never uses
`closure_beat - serialized_first_start_beat`.

## Frontier and Cut Contract

Decomposition uses common time cuts. A cut is an inference checkpoint, not an
output section boundary and not necessarily an integer beat.

Every exported frontier state contains at least:

```text
cut_time_ms
next_beat_index
next_beat_time_ms
current_section_start_beat
current_section_start_time_ms
serialized_first_start_beat
current_bpm
previous_bpm
alias_family
global_downbeat_phase in {0,1,2,3,none}
committed_duration_objective_numerator
committed_transition_objective
real_section_count
alias_switch_count
max_boundary_displacement_ms
open_section_state
prefix_sections_or_backpointer
deterministic_replay_key
```

`next_beat_index` and `next_beat_time_ms` define the first beat on or after the
cut. Together with `current_bpm`, they represent exact phase at an arbitrary
cut without rounding the cut into a beat. The prior beat is derived from the
same lattice. ULP ties use one frozen half-open convention and receive
`nextafter` tests for negative and large beat indices.

All states in a frontier share `cut_time_ms` but may carry different absolute
beat indices, phases, BPMs, aliases, and downbeat phases. States are deduplicated
only by this complete future-equivalence key:

```text
(float.hex(cut_time_ms),
 next_beat_index, float.hex(next_beat_time_ms),
 current_section_start_beat, float.hex(current_section_start_time_ms),
 float.hex(current_bpm), none | float.hex(previous_bpm),
 alias_family, global_downbeat_phase,
 real_section_count, alias_switch_count,
 float.hex(max_boundary_displacement_ms),
 open_section_identity_fingerprint)
```

Accumulated objective and replay history are not part of the equivalence key.
When two paths have the same key, retain the better state under the complete
ordering below and keep its backpointer. This dominance rule is what prevents
the frontier from growing with every distinct prefix. The exported width is
16. Local search buckets keep width 64.

The alias-family representative is the Experiment 004 v1 mapping: enumerate
the frozen alias multipliers, keep valid candidates in `[20,1000]`, prefer the
lowest candidate in `[80,160)`, otherwise use the lowest valid candidate, and
round only that class label to 6 decimals. This rounded label is never used as
the authoritative BPM or phase lattice.

`open_section_identity_fingerprint` hashes exactly
`(current_section_start_beat, float.hex(current_section_start_time_ms),
float.hex(current_bpm), float.hex(cut_time_ms), "closure_prior_unpaid")`.
Block-local evidence is already in the committed objective and is not hidden
inside this fingerprint. The unpaid marker is replaced only when a real
closure is committed, at which point the right section receives a new open
identity.

A real BPM jump may occur only at an integer `boundary_beat`. Its time is
derived from the left lattice and is also the first time of the right section.
Internal cuts never pay a change cost, increment section count, or appear in
the serialized grid. An unchanged open section crosses a cut and is merged
without introducing a synthetic redline.

### Open-section closure

For an authoritative open section `(b0, t0_ms, q_left)` and immutable boundary
candidate time `u_ms`, compute the non-negative nominal count with half-up
rounding:

```text
x       = (u_ms - t0_ms) * q_left / 60000
N0      = floor(x + 0.5)
b_jump  = b0 + N
t_jump  = t0_ms + N * 60000 / q_left
```

The Experiment 005 closure candidates are exactly the positive, deduplicated
values from `(N0, N0 - 1, N0 + 1)` in that order; there are at most three.
Python banker's rounding is forbidden. A closure is feasible only when
`N > 0`, `t_jump - t0_ms >= 8000 ms`, the derived `t_jump` belongs to the
active core/lookahead ownership interval, and

```text
abs(t_jump - u_ms) <= min(60 ms, 0.15 * 60000 / q_left).
```

The observed candidate time supplies boundary support; `t_jump` is the only
mathematical boundary authority. The right BPM is selected from the frozen
local tempo shortlist, and the right section starts at exactly
`(b_jump, t_jump)`. A same-BPM continuation is not a real jump and remains the
same open section.

At the coverage tail `E`, close the final open section at the smallest integer
`b_end > b0` whose left-lattice time is greater than or equal to `E`, using the
same half-open ULP convention as `next_beat_index`. Tail closure needs no
observed boundary candidate and cannot change `q_left`.

Once a core cut is committed, no later block may change the committed beat
indices, BPM, phase, or a real boundary before that cut. Lookahead-only choices
may be revised when they become core. An open section that began before the cut
may close later, but its stored start, left BPM, and lattice are immutable; the
later block may only choose a future `b_jump/t_jump` under the formula above.

The deterministic state ordering is:

1. lower fixed-lag rank objective;
2. lower committed objective;
3. fewer real sections;
4. fewer alias switches;
5. smaller maximum boundary displacement;
6. lexicographically smaller complete replay key.

NaN, infinity, incomplete phase state, impossible next-beat order, invalid
BPM, more than 16 exported states, or any duplicate equivalence key remaining
after dominance fails closed before the next block.

## Block and Overlap Contract

For a cut `s`, core duration `C`, lookahead `O`, and coverage end `E`, a block
evaluates:

```text
window = [s, min(E, s + C + O))
core   = [s, min(E, s + C))
right_lookahead = [min(E, s + C), min(E, s + C + O))
```

Only core evidence is added to the committed path objective. Lookahead evidence
ranks the frontier at the next cut and is recomputed as core evidence by the
next block; it is never counted twice in the final objective. The final block
has no uncommitted suffix.

A cut state represents committed coverage `[coverage_start_ms, cut_time_ms)`.
A real boundary whose authoritative time equals the cut is uncommitted and is
owned by the next core. The prior block may retain it only in its fingerprinted
provisional lookahead trace.

Let `D = E - coverage_start_ms`. For the intersections of candidate sections
with one core interval, define:

```text
C_local_i = 1.00 * C_beat_support
          + 0.25 * C_peak_recall_precision
          + 0.20 * C_downbeat_phase
          + 0.10 * C_bpm_prior

P_close_j = L_section_j
            * (0.10 * C_beat_count_prior_j
             + 0.08 * C_section_duration_j) / D

A_core = sum(L_i * C_local_i) / D
       + sum(P_close_j for real sections closed by core)
       + sum(C_transition_k for real boundaries owned by core)
         / max(1, D / 60000)

J_committed_next = J_committed_previous + A_core
J_rank            = J_committed_next + A_lookahead
```

`L_section_j` is the complete real section duration clipped to cache coverage;
its count prior uses the final `end_beat - start_beat`. The count and duration
priors are deferred and charged exactly once by the core that owns the real
closure, or by the final tail closure at `E`. They are never estimated from a
non-integer internal cut and never written back into an earlier committed
objective.

`A_lookahead` uses the same formula over the right-lookahead interval but is
discarded after ranking. A closure prior or transition inside lookahead is
provisional and is recomputed if that boundary later enters core. A real
boundary at the exact next cut belongs to the next half-open core, so every
closure and transition cost is counted once. An open mathematical section may
have its local evidence terms scored on two adjacent core intersections; this
changes nonlinear support normalization but not BPM, phase, section count, or
transition cost. The row reports block-local evidence, deferred closure priors,
and frontier-pruning effects separately.

For every future-equivalence state at the next cut, enumerate its feasible
lookahead continuations under the same local graph. Reduce them to the one
smallest `(J_rank, lookahead_replay_key)`. If several committed prefixes reach
the same state key, retain the prefix attached to the best reduced continuation
under the complete state ordering. Class reservation and the frontier-width
cut are applied only after this reduction. The retained lookahead suffix is
diagnostic and is discarded after the next block recomputes that interval. It
must first be serialized as ordered tuples
`(absolute_beat, float.hex(beat_time_ms), float.hex(bpm))` plus real boundary
records and a SHA-256 fingerprint. The next block records the recomputed trace,
comparison domain, residual-vector fingerprint, and unavailable reason before
the provisional suffix may be released from memory.

Overlap disagreement compares the prior block's retained lookahead path with
the next block's recomputed path. For every absolute integer beat present in
both paths whose two beat times lie inside their common overlap, record
`abs(t_previous(k) - t_next(k))` in milliseconds and divide by the smaller
local period for the beat-unit value. Aggregate beats within audio first, then
audio across a stage. Fewer than eight comparable beats for an overlap makes
that overlap unavailable rather than zero; a stage with fewer than five
available audio groups is ambiguous for the overlap gate.

This metric is versioned `timing_v3_overlap_disagreement_v1`. For each audio,
concatenate all comparable beat residuals from all its overlaps and compute the
linear-interpolated p90 with `h=(n-1)*0.9`. The stage value is the same p90 over
the per-audio p90 values. It is a raw absolute-beat-identity metric; no tempo
alias canonicalization or nearest-beat rematching is permitted.

For `S64`, after the initial 90/30 bootstrap:

```text
q_ref = current_bpm of the best-ranked exported frontier state
C_ms  = clamp(64 * 60000 / q_ref, 30000, 90000)
O_ms  = clamp(16 * 60000 / q_ref, 10000, 30000)
```

`q_ref` schedules a common cut only; it cannot delete, rephase, or reweight any
of the other 15 states directly. Because the common cut changes block-local
score partitioning, `q_ref` does change the pre-registered objective intervals
for every state; this scheduler coupling is reported explicitly. The next cut
is `min(E, s + C_ms)`. A zero-length or non-advancing cut is a tagged internal
failure.

The final result is selected by deterministic traceback through retained local
backpointers. Traceback may choose among already retained paths but may not
create new candidates or perform an unrestricted global rescore.

## Frozen Local Graph Bounds

All 16 incoming frontier states enter one joint local search; they do not
receive independent width-64 beams. Interior states are bucketed by
`(right_boundary_candidate_id, resulting_real_section_count)`, and the final
cut uses `(cut_sentinel, resulting_real_section_count)`. Within a bucket,
deduplicate exact future-equivalence states first, retaining the dominant
prefix/continuation. Then sort by the complete deterministic state order and
keep the first 64. Prune counts include every valid state removed by dominance
or width, with the two reasons reported separately. The class-reservation rule
is applied only when exporting the 16-state cut frontier, after all local
buckets and lookahead continuations have been reduced.

- maximum exported frontier states: 16;
- local beam width: 64;
- maximum real output sections: 20 over the whole audio;
- maximum decomposition blocks: 192;
- maximum locally retained boundary candidates: the first 32 candidates in the
  immutable Experiment 004 boundary-candidate order after filtering to the
  block window; incoming/open-section anchors and coverage endpoints are
  mandatory and do not consume this cap;
- maximum local tempo candidates: 64 after deduplication at 6 decimals;
- maximum closure count candidates per observed boundary: 3 in the frozen
  `(N0, N0-1, N0+1)` order;
- maximum unique section-score misses per block: 30,000;
- maximum unique section-score misses per audio: 500,000;
- per-audio wall timeout: 180 s, including cache load, current-v2 fit,
  candidate extraction, all blocks, traceback, and serialization;
- per-worker peak resident memory hard guard: 4 GiB.

The tempo shortlist is phase-independent and uses no partially specified
`pulse_correlation_v1` call. Reserve source quotas and append, in this exact
order:

1. up to 16 exact incoming BPMs by frontier rank;
2. up to 16 new boundary-derived BPMs, iterating retained boundary candidates
   in immutable order and appending `60000/right_period_ms` then
   `60000/left_period_ms`;
3. up to 16 new BPMs from the immutable global candidate tuple order;
4. fill every remaining slot by iterating the already retained base BPMs in
   shortlist order and the frozen alias multipliers in multiplier order,
   rounding products to 6 decimals and retaining only `[20,1000]`.

Deduplicate by rounded-6 identity, keep the first exact value as the
authoritative binary64 BPM, and stop at 64. A candidate excluded only by the
cap is recorded in diagnostics. The same ordering and caps apply to every
scheduler arm. Local beat/downbeat support is evaluated only after a candidate
has a phase-bearing section start; it does not participate in shortlist
construction.

Unused quota does not let incoming/alias candidates consume the reserved
boundary or global slots. A synthetic non-alias jump must place its supported
right-side boundary-derived BPM in the shortlist and select it over persistence
when its frozen objective is better.

Section-score caches are row-scoped and use complete interval, BPM, phase, and
variant identity. The 30,000 block counter advances only when the active block
causes a unique cache miss; the 500,000 row counter advances only on the first
miss for that row. Reusing a score first computed in the prior block's
lookahead consumes neither counter, but the score is applied to the current
core objective under the current ownership rule. The next miss beyond either
cap is rejected before scoring.

Exceeding a block, row, section, score-miss, timeout, or memory bound yields a
tagged fallback. It never returns the current best path as an accepted result.

## Frozen Score and Prior Contract

Experiment 005 reuses the Experiment 004 `CJ3` cost definitions and weights:

- beat pulse support and peak recall/precision;
- optional modulo-4 downbeat phase support;
- soft BPM band `[80,240]` inside the hard `[20,1000]` guard;
- alias orbit `{1/4,1/3,1/2,1,2,3,4}`;
- expected integer beat counts use the open-section closure formula and exact
  Experiment 005 `(N0,N0-1,N0+1)` order above. The prior count is
  `closure_beat - logical_open_section_start_beat`; the serialized span is
  reported separately as `end_beat - serialized_start_beat`. They differ only
  for the backward coverage extension of the first section;
- section-duration, change-sparsity, alias-switch, jump-size, and boundary
  support terms;
- existing deterministic score and path tie-breaks.

No empirical, title-derived, catalog-derived, or learned BPM prior is added.
The broad candidate path remains available so the soft 80-240 preference
cannot hide rare tempi.

Scores are accumulated over disjoint committed core intervals. Splitting score
measurement at an internal cut is part of this experiment's local objective,
but splitting the measurement interval must not split the mathematical output
section or add a transition cost. Diagnostics report both per-block objective
components and the duration-weighted total so this approximation can be
attributed separately from frontier pruning.

## Assembly and Fallback Contract

Traceback produces one `TimingV3Grid` with one beat-zero origin. Adjacent
committed pieces with the same binary64 BPM and the same continued lattice are
merged. Every retained real boundary has the same integer beat and derived
time from both adjacent sections. Sections are contiguous half-open intervals,
cover the complete cache, remain within 20-1000 BPM, and survive finite-JSON
round-trip validation.

The selected product path is:

```text
selected_LF3_or_current_v2_fallback
```

Every fallback counts in the projection denominator and retains an exact
reason and stage: cache, current-v2, candidate extraction, bootstrap, scheduler,
local graph, frontier, lookahead, traceback, schema, serialization, timeout,
memory, or source integrity.

## Minimal Change

Add a source-owned local-frontier core and synthetic tests first. Reuse the
existing cache loader, restricted-prediction boundary, candidate extractor,
v3 schema, cost primitives, weak metrics, and baseline rows. Add an Experiment
005 runner/evaluator only after core local verification passes. Do not edit the
production fitter in this experiment until holdout100, broad500, and full5050
all pass unchanged.

## Files Likely to Change

- `src/pulsefield_model/timing/v3/local_frontier.py`;
- `src/pulsefield_model/timing/v3/__init__.py` for public experiment APIs only;
- `tests/timing/test_timing_v3_local_frontier.py`;
- `src/pulsefield_model/timing/evaluation/exp005_runner.py`;
- `src/pulsefield_model/timing/evaluation/exp005_protocol.py`;
- `src/pulsefield_model/timing/evaluation/exp005_metrics.py` only for assembly
  diagnostics not already present in Experiment 004 metrics;
- focused Experiment 005 runner/protocol/metrics tests;
- this result document's future companion and the timing-v3 problem log.

`src/pulsefield_model/timing/v3/fitter.py` remains unchanged until every formal
stage passes.

## Read-Only Context Files

- `docs/research/timing_v3_task_definition.md`;
- `docs/research/timing_v3_experiment_004_result.md`;
- `src/pulsefield_model/timing/v3/schema.py`, SHA-256
  `019ed7e942ef4a994d7ff0869de433e410dfc932c320dd7231bde83173a37e75`;
- `src/pulsefield_model/timing/v3/global_constant_jump.py`, initial SHA-256
  `736e7c47e57d8567b47a56fa576f0187cf5b25f2dadd33a72ba59c278080528d`;
- `src/pulsefield_model/timing/evaluation/exp004_metrics.py`, initial SHA-256
  `f88366562d35645fa4264cdb8710d2ea4781c6615a2ea23c2c3c87b0524dc21e`;
- the frozen current-v2 full baseline, SHA-256
  `e31089b0aa5688e6cdad9b11f53efb36a7be7147552d76418ae4133eff1239b3`.

If reusable code must move from the Experiment 004 core, exact old/new parity
tests and both source hashes are required. The Experiment 004 formal artifacts
remain immutable.

## Dataset Slice

All grouping and selection occur by cache audio key before any difficulty-level
metric is read.

### Synthetic local verification

Use source-owned arrays only: long constant tracks; one and several supported
jumps; jumps on both sides of a cut; early alias ambiguity resolved after the
cut; downbeat-free input; sparse/flat signal; BPM boundaries at 20 and 1000;
large/negative beat indices; missing local candidates; section-cap and resource
fallbacks; and metadata traps.

### Exposed scheduler16

Derive a 16-audio tuning slice only from the existing exposed repair80 identity
and frozen source labels. Under the seed
`timing-v3-exp005-schedule16-v1`, select four rows each with exclusive priority
`long -> dense -> jump -> stable`, ranking by

```text
sha256("timing-v3-exp005-schedule16-v1\0" + cache_audio_key)
```

If a quota is underfilled, fill from the next class in that fixed order, then
from remaining repair80 rows by the same hash. The manifest may contain identity,
stratum, duration, and source hashes, but no candidate-relative metric values.

Run `LF3` under S30/S60/S90/S64. First eliminate any arm with leakage, cache or
source mismatch, stale resume, invalid accepted schema, nonfinite JSON,
timeout, memory violation, nondeterministic replay, phase mean ratio above
`1.10`, phase p90 ratio above `1.15`, p90 overlap disagreement above `90 ms`,
or more than 20 real sections. Also eliminate an arm that enters any other
applicable kill band in the formal gate table with a valid denominator,
including pure coverage below 90%, fallback above 10%, candidate/no-path
failure above 5%, or p90 row runtime above 60 s. Select the remaining arm by
this exact total order:

1. fewer selected fallbacks;
2. higher audio-grouped weak change-boundary F1;
3. lower alias-normalized maximum-prefix drift ratio versus current v2;
4. lower p90 overlap disagreement in milliseconds;
5. lower p90 row runtime;
6. lower maximum normalized worker peak RSS for the arm;
7. tie order `S64`, `S90`, `S60`, `S30`.

The ordering surface is frozen and arm-independent:

- fallback, failure, and runtime use all 16 schedule identities; memory uses
  the arm-level maximum over its four fresh workers;
- paired phase and drift ordering use the intersection where current v2 and all
  four pure LF3 arms are accepted and matched to the same usable weak
  comparator; fallback-product metrics never enter the schedule order;
- boundary ordering further restricts that common set to audio with at least
  one weak redline change across valid difficulties. For one audio, sum
  one-to-one match counts across its valid difficulties as `TP`, unmatched
  predicted as `FP`, and unmatched weak redlines as `FN`, then compute
  `2TP/(2TP+FP+FN)`. The denominator is positive by construction. The ordering
  value is the arithmetic mean of this per-audio F1 over the common boundary
  set;
- the drift ordering value is
  `mean(LF3 alias-normalized max-prefix error) /
  mean(current-v2 alias-normalized max-prefix error)` on that same set;
- a ratio is `1` when both numerator and denominator are zero, and positive
  infinity when only the denominator is zero;
- p90 values use the same linear interpolation
  `h=(n-1)*0.9` defined by overlap v1;
- a missing, nonfinite, or provenance-mismatched ordering value eliminates the
  arm before total ordering.

Weak boundaries use the existing 0.5% log-BPM change threshold, one-to-one
matching, and tolerance `min(750 ms, 0.5 * minimum adjacent period)`. At least
five comparison-eligible audio groups with weak changes are required. If that
denominator is smaller, or every arm is eliminated, scheduler selection is
ambiguous and this card stops before repair80.

An arm in an ambiguous but non-kill band may win the schedule16 tuning order;
this does not authorize a holdout. The mechanically selected arm must still
pass every applicable repair80 pass band before the experiment can advance.

The selected schedule and all four arm hashes/metrics are written to one
immutable config-selection artifact. Outcome-driven changes to frontier width,
caps, costs, candidates, metrics, or ordering require a new card, not a second
scheduler choice.

Any behavior-affecting change to the core, runner, evaluator, metrics,
candidate ordering, resource accounting, or fallback semantics invalidates
that artifact. Synthetic verification and all four schedule16 arms must restart
to new immutable paths, and the total order must be recomputed before repair80.
Only a proven byte-equivalent provenance/serialization repair may retain the
winner, and that exception requires differential equality of the deterministic
row projection plus an unchanged mathematical/replay fingerprint on all four
arms. Volatile telemetry is excluded under the replay contract below.

### Repair80

Run only the frozen selected schedule on all existing exposed repair80 rows.
Repair80 may expose implementation defects. Any behavior-affecting repair
returns to synthetic verification and the complete four-arm scheduler sweep,
then restarts all 80 rows with the newly selected immutable config. A proven
byte-equivalent provenance/serialization repair may restart repair80 alone
under the exception above. Repair80 cannot accept the candidate, but a
hard-guard, runtime, fallback, phase, drift, or section-count failure stops the
experiment before holdout.

Repair80 advances only when every applicable gate is `pass`. An ambiguous or
kill classification blocks holdout creation; ambiguity cannot be resolved by
choosing a different scheduler arm after the fact.

### Fresh holdout100

No Experiment 005 holdout identity may be selected or materialized until the
selected schedule, core, config, runner, evaluator, metrics, source closure,
and repair80 result hashes are frozen.

The exposure manifest must include at least the 183 previously exposed audio
keys: pilot/repair80 (80), the Experiment 003 protocol-exposed rows (3), and
the Experiment 003 protocol-v2 holdout (100), plus every later audio key whose
candidate batch aggregate, prediction, metric, diagnostic, runtime, failure,
trace, or rendering was observed, whether or not an individual weak row was
opened. Before holdout materialization, real-audio candidate execution is
limited to synthetic data and keys already present in this exposure manifest;
an accidental run on any additional key appends every key in that observed
batch before selection. The selector reads identities and source labels, never
metric values.

Rank under:

```text
sha256("timing-v3-exp005-holdout100-v1\0" + cache_audio_key)
```

The selectable pool is exactly all declared cached audio keys minus every key
in the frozen exposure manifest and minus any independently frozen protocol
exclusion manifest. Every quota and deficit fill draws only from this pool.
The holdout selector must prove
`selected_holdout_keys ∩ exposure_keys = empty` and the same for protocol
exclusions. The manifest records the full exposure/protocol source provenance,
their SHAs, the sorted selectable-key-set SHA, and the selected-key-set SHA;
any mismatch or overlap fails closed before candidate execution.

Use exclusive priority and quotas:

| Quota | Count |
| --- | ---: |
| anomaly / ambiguous | 10 |
| long | 10 |
| dense | 20 |
| jump | 30 |
| stable | 30 |

Ramp has no quota because all known ramp-audit candidates were previously
exposed and ramps are not modeled here. If a quota is underfilled, select every
available row, mark the quota degraded, then fill by
`jump -> dense -> long -> anomaly -> stable` under the independent deficit seed
`timing-v3-exp005-holdout100-deficit-v1`. Fewer than five comparison-eligible
long rows or fifteen jump rows makes the stage at most ambiguous.

### Broad500 and full5050

Only an unchanged holdout100 pass may materialize broad500: the holdout plus
400 unexposed, non-holdout keys ranked under
`timing-v3-exp005-broad500-v1`. Only an unchanged broad500 pass may run all
5,050 cached audio groups. The final replay includes exposed rows for tail-risk
reporting, not holdout selection. Any behavior, gate, or comparator-policy
change after holdout inspection requires a new card and fresh audio-disjoint
holdout.

## Baseline / Comparator

- Primary product comparator: the frozen current-v2 full baseline, evaluated
  on the identical cache and paired eligible audio.
- Source-only controls: `LF0`, `LF1`, `LF2`, selected `LF3`, and constant-only
  `CJ0` where it completes without changing its frozen source.
- Experiment 004 `CJ3` is diagnostic only; its timed-out repair result cannot
  become a primary baseline or supply candidates.
- Product safety result: selected LF3 grid or tagged current-v2 fallback.

`.osu` remains a weak evaluation comparator. Redlines propose period families,
phase anchors, and possible change boundaries. Hit-object starts measure
rational-grid phase concentration; they do not independently determine BPM
alias, and all difficulties are aggregated within audio before corpus metrics.
Comparator-unavailable and cross-difficulty-conflicting rows remain explicit.

Source-stamped catalog/API/network BPM may be audited only after predictions
are frozen and only when exact recording identity is established. A scalar BPM
can corroborate an alias family but cannot establish constant versus jump or
ramp. External values never enter inference, scheduler selection, or a primary
acceptance denominator.

## Denominator Contract

All counts are audio-grouped. They are recomputed from row payloads rather than
trusted from a prior summary:

- `stage_audio_count`: identities in the frozen stage manifest;
- `cache_valid_count`: stage rows whose declared cache and config pass;
- `projection_evaluable_count`: cache-valid rows where LF3 returns either an
  accepted grid or a deterministic tagged fallback before weak comparison;
- `lf3_accepted_count`: projection-evaluable rows with an accepted LF3 grid;
- `current_v2_accepted_count`: cache-valid rows with an accepted current-v2
  grid;
- `product_grid_available_count`: rows with accepted LF3, or tagged LF3
  fallback plus accepted current v2;
- `current_v2_phase_matched_count`: current-v2-accepted rows with a usable weak
  comparator and current-v2 phase metrics;
- `pure_lf3_phase_matched_count`: LF3-accepted rows that are also in
  `current_v2_phase_matched_count` and have LF3 phase metrics;
- `selected_safety_phase_matched_count`: product-grid-available rows that are
  also in `current_v2_phase_matched_count` and have selected-product metrics;
- stratum denominators: exact intersections of the relevant matched or
  cache-valid denominator with the frozen audio-level stratum.

The mandatory invariants are:

```text
pure_lf3_phase_matched       => current_v2_phase_matched
selected_safety_phase_matched => current_v2_phase_matched
pure_lf3_phase_matched       => lf3_accepted
product_grid_available       => lf3_accepted or
                                (lf3_tagged_fallback and current_v2_accepted)
```

Assuming the weak comparator is usable, row truth is:

| LF3 | Current v2 | Product grid | LF3 fallback numerator | Pure paired | Selected-safety paired |
| --- | --- | --- | ---: | --- | --- |
| accepted | accepted | LF3 | 0 | yes | yes |
| accepted | unavailable | LF3 | 0 | no | no |
| tagged fallback | accepted | current v2 | 1 | no | yes |
| tagged fallback | unavailable | unavailable | 1 | no | no |

A protocol/integrity hard failure is not silently converted to a tagged
fallback or product grid. Method-level LF3 weak metrics may be retained when
current v2 is unavailable, but no paired ratio may be constructed from them.
Comparator-unavailable rows remain in cache, projection, fallback, failure,
and runtime denominators but not phase/boundary denominators.

## Primary Metric

The primary formal comparison is audio-paired selected-LF3 versus current v2:

- mean phase-error ratio;
- p90 phase-error ratio;
- alias-normalized long-track maximum-prefix drift mean and p90 ratios;
- pure-LF3 phase coverage;
- selected fallback rate and exact reasons;
- row timeout rate and p90 runtime.

Pure LF3 and selected-LF3-or-v2 safety denominators are reported separately.

## Secondary Metric

- raw and alias-normalized endpoint drift, drift slope, and max-prefix error;
- BeatThis beat/downbeat support and local peak recall/precision;
- raw and alias-aware BPM error and alias switches;
- weak boundary precision, recall, F1, matched error, and unmatched counts;
- object-grid residual/inlier rate, without using it as alias truth;
- overlap disagreement in ms and beats;
- frontier width, class coverage, score margin, and prune reason per cut;
- cut/block count, candidates, score misses, local beam prunes, and runtime per
  block;
- real section count, artificial-cut merge count, and stable-track section
  inflation;
- peak resident memory and deterministic replay fingerprints;
- every failure stage and comparator-availability denominator.

## Error Attribution

Every non-pass row is assigned to the earliest supported layer:

1. cache/config/integrity;
2. BeatThis signal or candidate extraction;
3. current-v2 comparator availability;
4. bootstrap origin/phase/alias posterior;
5. scheduler/cut construction;
6. local tempo/boundary/count shortlist;
7. local path search or resource cap;
8. frontier deduplication/pruning;
9. lookahead disagreement;
10. traceback/open-section assembly;
11. v3 schema/serialization;
12. product fallback;
13. `.osu`/object/external comparator uncertainty.

Later evidence cannot overwrite an earlier confirmed failure. A poor or
ambiguous redline remains comparator uncertainty rather than an algorithm
failure.

## Verify Command / Evaluation Procedure

1. Freeze this card bytes and source snapshots.
2. Implement the core and pass synthetic/local differential tests.
3. Build the schedule16 identity manifest without reading metrics into that
   manifest selector.
4. Run all four schedules, publish immutable outputs, and apply the frozen
   total order mechanically.
5. Freeze the winner config and all behavior/source fingerprints.
6. Run selected LF3 on repair80; after any behavior repair, return to synthetic
   verification, rerun all four schedule16 arms, mechanically reselect, and
   then restart repair80.
7. If repair passes, freeze evaluator/source closure and only then select and
   run holdout100.
8. Run broad500 only after holdout pass, and full5050 only after broad pass.
9. Re-run all 5,050 after the final code/config freeze before production
   integration.

Every stage is resumable by row content fingerprint and rejects stale source,
config, cache, selection, metric, exposure, or comparator provenance.

The deterministic row projection includes candidate/order fingerprints,
mathematical grid, frontier/lookahead traces, fallback/schema fields, and
source/config identities. It excludes wall/CPU time, RSS, timestamps, absolute
paths, process IDs, and other volatile telemetry. Resume and mathematical
replay fingerprints bind only the deterministic projection. Telemetry must be
finite and provenance-complete but is not required to be byte-identical across
runs.

## Guard Check

Hard guards at every stage:

- zero inference access to `.osu`, metadata, network, labels, or raw audio;
- exact cache/config/source identity before and after each row;
- one global beat-zero origin and exact phase state across cuts;
- accepted grids pass v3 schema and finite JSON round trip;
- maximum 20 real sections and no artificial cut in serialized sections;
- byte-stable deterministic row projection and replay fingerprint, excluding
  volatile telemetry as defined above;
- explicit tagged fallback, no truncated success;
- per-row time <180 s and per-worker peak RSS <=4 GiB;
- projection/fallback and comparator denominators remain separate.

Stage gates preserve the Experiment 004 bands while binding every value to one
denominator. `n` always means the denominator in that row:

| Metric | Denominator | Pass | Ambiguous | Kill |
| --- | --- | --- | --- | --- |
| pure LF3 mean phase ratio | `pure_lf3_phase_matched_count` | `<=1.05` | `(1.05,1.10]` | `>1.10` |
| pure LF3 p90 phase ratio | `pure_lf3_phase_matched_count` | `<=1.10` | `(1.10,1.15]` | `>1.15` |
| pure LF3 phase coverage | `pure_lf3_phase_matched_count / current_v2_phase_matched_count` | `>=95%` | `[90%,95%)` | `<90%` |
| max of stable mean/p90 phase ratios, `n>=5` | `pure_lf3_phase_matched ∩ stable` | `<=1.10` | `(1.10,1.20]` | `>1.20` |
| jump mean phase ratio, `n>=15` | `pure_lf3_phase_matched ∩ jump` | `<=1.05` | `(1.05,1.15]` | `>1.15` |
| jump alias-normalized max-prefix drift mean ratio, `n>=15` | same jump paired set | `<=0.90` | `(0.90,1.15]` | `>1.15` |
| max of long max-prefix drift mean/p90 ratios, `n>=5` | `pure_lf3_phase_matched ∩ long` | `<=1.15` | `(1.15,1.30]` | `>1.30` |
| LF3 tagged fallback rate | `cache_valid_count` | `<=5%` | `(5%,10%]` | `>10%` |
| candidate/no-path failure rate | `cache_valid_count` | `<=3%` | `(3%,5%]` | `>5%` |
| p90 row runtime | every `cache_valid_count` row; missing runtime is hard failure | `<=30 s` | `(30 s,60 s]` | `>60 s` |

Decomposition-specific guards:

- adjacent serialized sections share one derived boundary scalar; maximum
  discontinuity must equal `0.0 ms` before and after JSON round trip, and any
  nonzero value is a hard failure. The task-level 5 ms ceiling is therefore a
  looser external guard, not implementation tolerance;
- p90 lookahead-to-next-core phase disagreement on LF3-accepted audio with an
  available `timing_v3_overlap_disagreement_v1` value is `<=45 ms` to pass,
  `(45,90] ms` ambiguous, and `>90 ms` kill;
- no stable row gains more than one real section over current v2; any violation
  makes the stable stratum fail without an evidence-dependent exception;
- any state/section/cut/candidate cap violation is a fallback and counts in the
  corresponding rate;
- schedule selection requires its minimum weak-boundary denominator and cannot
  silently skip a missing memory or runtime measurement.

Repair80 is debug/regression, but a failed hard guard or any kill-band result
blocks holdout creation. Holdout/broad/full actions otherwise match the frozen
pass/ambiguous/kill progression.

## Qualitative Check

Only on synthetic and exposed repair/schedule rows, render or inspect:

- a constant track across at least four internal cuts;
- an early alias ambiguity and its retained frontier classes;
- jumps before, inside, and after lookahead;
- the worst overlap disagreement;
- every fallback reason with at least two examples when available;
- one long/dense row's block runtime and prune trace.

The inspection may classify failure but may not change candidates, weights,
caps, schedule ordering, gates, or a fresh holdout.

## Positive Signal

- Variant D passes synthetic verification and the selected schedule is produced
  mechanically from complete schedule16 evidence;
- repair80 has no timeout, integrity, schema, or stale-resume hard guard;
- holdout100, broad500, and full5050 pass unchanged;
- phase and cumulative drift are not materially worse than current v2;
- fallback stays <=5%, internal cuts do not inflate sections, and runtime grows
  approximately linearly with block count;
- weak boundary/object evidence agrees without being required by inference;
- the final 5,050 replay is deterministic from recorded hashes.

## Negative Signal

- exported frontier16 loses the later-correct phase/alias state;
- overlap changes the committed phase by more than the guard;
- fixed-lag local scoring creates false jumps or stable section inflation;
- the selected schedule times out or exceeds memory on exposed long/dense rows;
- formal phase, drift, coverage, fallback, runtime, or section guards enter a
  kill band;
- performance remains superlinear in blocks or requires a final global replay;
- results depend on `.osu`, metadata, network, raw audio, or stale artifacts.

## Kill Criteria

Kill LF3 for this loop if:

- selected Variant D fails synthetic local verification;
- scheduler selection is undefined or all four schedules violate hard guards;
- repair80 retains any row timeout or exceeds the 30-minute hard budget after
  protocol-preserving implementation fixes;
- any formal stage enters a kill band;
- exact phase/open-section state cannot be serialized and replayed
  deterministically;
- passing requires widening frontier/candidate/resource caps, changing cost
  weights, or adding ramp/learned-prior/global-replay behavior after real-audio
  inspection.

## Expected Failure Modes

- the best scheduling BPM is initially the wrong alias, altering S64 duration;
- 16 exported states prune a later-correct low-ranked phase/tempo hypothesis;
- block-local score normalization disagrees with whole-section support;
- a jump inside lookahead changes when it becomes committed core;
- very sparse or silent blocks lack enough evidence to maintain phase;
- dense blocks hit boundary/tempo/score caps despite bounded duration;
- stable sections are split by numerical rather than musical differences;
- the global 20-section cap rejects genuinely complex long tracks;
- weak redline boundaries disagree across difficulties;
- object placement validates phase but not the intended BPM alias;
- external scalar BPM refers to a different recording or cannot identify
  variable tempo.

## Confounders

- only 29 corpus tracks are at least 600 s, so long-tail denominators are small;
- repair80 is exposed and may diagnose behavior but cannot provide acceptance;
- all known ramp-audit candidates are already exposed, and ramps are not part
  of the fresh holdout claim;
- scheduling and phase inference are coupled through `q_ref`, although only
  scheduling may use the best frontier state;
- local score partitioning changes a nonlinear correlation objective even when
  the output section remains open across a cut;
- `.osu` maps are correlated mapper annotations, not independent audio truth;
- runtime and RSS depend on Python, NumPy, platform, worker count, and cache
  state, all of which must be recorded.

## Expected Runtime / Runtime Budget

Formal executions use exactly four spawn workers with ordered
`imap(chunksize=1)`. Scheduler arms run as four separate executions in fixed
order `S30`, `S60`, `S90`, `S64`; they are not nested or run concurrently.
Aggregate wall time is parent-process monotonic time from before pool creation
through final row/summary fsync and pool cleanup. A worker-count, start-method,
machine, Python, NumPy, or cache-contract change invalidates the runtime
comparison and requires the applicable complete stage rerun.

Workers are fresh for each arm/stage and persistent within it with
`maxtasksperchild=None`. At worker initialization and after every row, record
`resource.getrusage(RUSAGE_SELF).ru_maxrss`. Normalize macOS values as bytes and
Linux values as KiB times 1024, recording the platform rule. This is a
process-lifetime high-water mark that includes mappings as reported by the OS;
it is not attributed to one row. The arm/stage memory value is the maximum
normalized value reported by its four workers. Missing telemetry, worker death,
or any report above 4 GiB is a hard failure. Scheduler ordering uses this
arm-level maximum, never a row-p90 or a delta from worker initialization.

- synthetic and unit verification: under 2 minutes;
- four-arm schedule16 sweep: under 20 minutes hard stop;
- selected-schedule repair80: under 30 minutes hard stop;
- holdout100: under 45 minutes;
- broad500: under 4 hours;
- full5050: under 36 hours with checkpoints;
- per-audio timeout: 180 seconds;
- full selected-config run only; controls and scheduler arms do not multiply
  holdout, broad, or full runtime.

## Result Interpretation Plan

- Positive result would suggest: bounded multi-frontier fixed-lag inference is
  a viable jump-only Timing v3 core and may be integrated behind the fitter
  interface after the final replay.
- Negative result would suggest: identify whether the limiting layer is cache
  evidence, local candidate quality, frontier compression, or local scoring;
  do not return to the rejected whole-track graph by default.
- Ambiguous result would require: one new card limited to the failed stratum or
  comparator uncertainty, with no post-holdout threshold change.
- Human owner decides: whether a successful jump-only LF3 result is sufficient
  to start the separate analytic-ramp experiment and whether external BPM audit
  coverage is adequate for reporting.
- Next-loop action if positive: integrate the frozen selected schedule, run the
  final 5,050 verification again, then pre-register ramps.
- Next-loop action if negative: `KILL` or `MUTATE` only the attributed failing
  layer.
- Next-loop action if ambiguous: audit without prediction changes, then create
  one narrower card.

## Result Log Template

- Experiment: Timing v3 Experiment 005
- Date:
- Card SHA-256:
- Source/config/evaluator hashes:
- Candidate contract/fingerprint:
- Dataset stage and manifest SHA-256:
- Exposure-manifest SHA-256:
- Scheduler16 arm outputs and selected config:
- Variant and schedule:
- Runtime / peak RSS:
- Cache/projection/comparator denominators:
- Phase and drift ratios:
- Boundary/object weak metrics:
- Overlap disagreement:
- Frontier/candidate/block/section diagnostics:
- Fallback and failure attribution:
- Deterministic replay result:
- Hard guards:
- Stage gates:
- Positive/negative/ambiguous classification:
- Checks performed and failed checks:
- Confounders/evidence gaps:
- Recommended next step: KILL | MUTATE | TEST
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes
- Code execution allowed after this card: yes, synthetic/core work first
- Scheduler16 execution allowed: only after synthetic verification and source
  fingerprint freeze
- Repair80 execution allowed: only after mechanical scheduler selection and
  selected-config freeze
- Holdout selection/execution allowed: no, until unchanged repair80 pass and
  full behavior/evaluator/source closure freeze
- Closed loop complete: yes
- Remaining ambiguity: empirical performance and weak-comparator coverage are
  experiment outcomes, not planner choices.

## Next-Loop Action

- If positive: freeze the selected scheduler, complete the staged progression,
  integrate only after final full replay, then create a separate ramp card.
- If negative: kill LF3 or mutate only the attributed local layer in a new
  card.
- If ambiguous: preserve artifacts, inspect attribution only, and do not open
  later stages.

## Novelty Notes

- Closest analogies: fixed-lag smoothing, streaming Viterbi/beam search,
  receding-horizon decoding, and overlap-save processing.
- Novelty layer, if any: state/output/evaluation coupling specific to
  Pulsefield Timing v3.
- Representation novelty vs engineering variation: the phase-continuous
  half-open beat schema is project-specific; the decomposition method is an
  engineering application of known structured-decoding ideas.
