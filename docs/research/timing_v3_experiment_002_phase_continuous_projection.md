# Timing v3 Experiment 002: Phase-Continuous Constant/Jump Projection

## Mode and route

- Mode: critic
- Route: TEST
- Reason: Experiment 001 established a reproducible paired evaluator and useful
  stable/jump strata. The smallest remaining uncertainty is whether the current
  v2 anchors can be expressed on one integer beat axis without materially
  degrading phase or cumulative drift.
- Scope: constant sections and BPM jumps only. Ramp representation and
  detection remain behind a later gate.

## Core question

Can one cached BeatThis prediction be converted from the current independently
anchored v2 segments into exact half-open sections

```text
beat [a, b) bpm q
```

such that every boundary has one shared beat index and one shared time, while
the paired `.osu` phase guards remain within the accepted regression budget?

The experiment is an adapter and representation test, not a claim of a new
beat tracker. The production candidate may call the unchanged current
`GridFitter`, but it may not read `.osu`, objects, metadata, or titles.

## Candidate families

Let adjacent v2 anchors be `o_i`, `o_(i+1)`, let the left period be `p_i`, and
define

```text
delta_i = o_(i+1) - o_i
x_i     = delta_i / p_i
N_i     = max(1, floor(x_i + 0.5))
r_i     = delta_i - N_i * p_i
```

The `floor(x + 0.5)` rule is versioned half-up rounding; Python banker's
rounding is not allowed.

### Family A: preserve BPM, move the next anchor

Keep `p_i` and derive the next boundary from the preceding section:

```text
tau_(i+1) = tau_i + N_i * p_i
```

This removes the seam but moves the change time and shifts the following phase.
It is retained as a negative/control candidate.

### Family B: preserve anchors, adjust the preceding BPM

Keep both v2 anchors and make the interval contain exactly `N_i` beats:

```text
p'_i = delta_i / N_i
q'_i = 60000 * N_i / delta_i
```

The boundary is exact. Relative BPM adjustment is

```text
(q'_i - q_i) / q_i = -r_i / delta_i
```

and projected-v2 phase deviation grows from zero to the old seam residual
inside the preceding section, then returns to the preserved right anchor.

### Family C: joint anchor/BPM optimization

Jointly move anchors and periods using confidence-weighted least squares while
enumerating nearby integer beat counts. This is more flexible, but it introduces
unfrozen weights and turns the adapter into a new fitter. It is deferred unless
Family B fails.

Selected candidate: Family B. Family A is the control; Family C is the
predeclared mutation path.

## Representation contract

The v3 grid has one authoritative timeline origin and ordered constant-tempo
sections. Machine data stores:

- integer `start_beat` and `end_beat` for every `[a,b)`;
- one positive finite BPM per section;
- a single origin beat/time pair;
- explicit evaluated cache coverage `[start_ms,end_ms)`;
- projection diagnostics and provenance outside the mathematical section.

All section start/end times are derived by a prefix scan. Independent adjacent
time anchors are not serialized as authorities.

Hard invariants:

1. each section has `end_beat > start_beat`;
2. `section[i].end_beat == section[i+1].start_beat`;
3. all BPMs are finite and in the configured 20-1000 BPM candidate guard;
4. derived time is strictly increasing;
5. the first/last derived boundaries cover the cache interval; and
6. JSON binary64 round-trip preserves every derived seam within `1e-6 ms` or
   eight ULPs, whichever is larger.

For cache frame count `F` and rate `f`, evaluated support is
`[0, 1000*F/f)`. Beat zero is the first v2 anchor. The first lattice is extended
backward to cover 0, so the serialized first beat may be negative; consumers
that require nonnegative beats may apply one global translation only. The final
section keeps the final v2 BPM and extends to the first beat boundary covering
the exclusive cache end. Audio end is never forced to be a beat.

The projection records the original seam `r_i`, implied beat count `x_i`,
chosen integer `N_i`, original/projected BPM, relative adjustment, source and
derived anchors, and fallback reason. A zero seam after projection must not
erase evidence that the source seam was large.

## Frozen guards and fallback

- Beat counts are restricted to the feasible integer interval implied by
  20-1000 BPM; an empty interval is a projection failure.
- A Family B section whose absolute relative BPM adjustment exceeds 5% is a
  frozen projection failure, not silently accepted.
- The experiment reports a tagged `fallback_v2`; it must not call an invalid
  v2 result a phase-continuous v3 grid.
- Family A may be emitted only as an explicit experiment/control result. A and
  B are not mixed boundary-by-boundary.
- If Family B fails the pilot aggregate guards or has more than 5% projection
  failures, mutate to Family C rather than tuning the frozen test slice.

## Implementation surface

Expected new source:

- `src/pulsefield_model/timing/v3/schema.py`: integer beat-axis grid, constant
  section math, lookup, and round-trip serialization;
- `src/pulsefield_model/timing/v3/projection.py`: Families A/B and diagnostics;
- `src/pulsefield_model/timing/v3/fitter.py`: one-cache wrapper around the
  unchanged v2 candidate generator;
- `src/pulsefield_model/timing/evaluation/v3_projection.py`: paired v2/A/B
  evaluation from durable baseline results.

Expected tests:

- schema rejection of empty, noncontiguous, reversed, nonfinite, uncovered, or
  invalid-BPM grids;
- exact constant and on-lattice jump cases;
- positive/negative off-lattice residuals and the Family B formula;
- half-beat ties and near-integer ULP cases;
- short intervals, infeasible counts, distortion fallback, and alias jumps;
- positive/negative first offsets and exact/non-exact final coverage;
- JSON round-trip at 20/1000 BPM and 4,358-second coverage;
- a `TimingV3Fitter` test proving exactly one shift-0 cached prediction is the
  only inference input; and
- evaluation-resume/provenance tests.

Existing v2 schema and output remain unchanged.

## Dataset and procedure

1. Run synthetic unit/property cases.
2. Evaluate v2, A, and B on the frozen 80-audio pilot using the same stored v2
   segments and oracle comparisons.
3. Inspect results by stable, jump, dense, ramp-audit, long, and anomaly pilot
   strata. Ramp-audit rows test robustness only; they are not ramp truth.
4. If B passes, run 100 and 500 deterministic audio groups before a full-corpus
   projection. No v3 parameters change after the 500 gate.

Primary metric: audio-group-weighted paired mean and p90 phase-error ratio of B
versus v2 on rows with a valid comparator.

Secondary metrics:

- raw and alias-aware local BPM error;
- raw and alias-normalized endpoint drift, slope, maximum prefix error, and
  30/60-second drift;
- original versus projected boundary seam;
- duration-weighted and max relative BPM adjustment;
- anchor displacement, section count, coverage, projection failure/fallback,
  deterministic replay, runtime, and serialization seam.

Required selection guards:

- mean phase error no more than 10% worse than paired v2;
- p90 phase error no more than 15% worse than paired v2;
- no serialized boundary discontinuity above 5 ms, with construction expected
  near machine precision;
- no section-count increase over the source v2 grid;
- projection failure/fallback at or below 5%;
- no material increase in cache/fit failure; and
- cumulative drift is reported even if the phase guards pass.

Positive result: Family B passes all guards and materially reduces boundary
seams. Negative result: phase/drift guard failure or excessive distortion/
fallback. Ambiguous result: aggregate passes but long/jump strata fail; expand
only the diagnostic slice and mutate to Family C without changing the frozen
80 predictions.

## Next-loop rule

- Positive: accept the constant/jump v3 representation, then create a new
  Experiment Card for a BeatThis-supported global jump path.
- Negative: reject the adapter and test Family C.
- Ambiguous: audit the failed strata and decide between C and a tagged v2
  fallback policy.
- Ramp work remains blocked on a section-level ramp audit and its own card.

