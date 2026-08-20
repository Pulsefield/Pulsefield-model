# Timing v3 Experiment 006: Boundary-Pair-Conditioned Transition

Date: 2026-08-12

## Mode

- Mode: planner
- Route: `TEST`
- Source idea: mutate only the transition potential that caused Experiment
  005 to skip an acoustically exact intermediate tempo.
- Acceptance source: independent blocker and scientific re-review of revision
  2 found no remaining blocker; the final bytes are frozen before code work.
- Source evidence: Experiment 005 synthetic result, exact objective ledger, and
  retained-path diagnostics. No real-cache result is used.

## Hypothesis

The Experiment 005 two-jump failure is an edge-identifiability failure rather
than a decomposition-capacity failure. Replacing part of its constant
per-change cost with an alias-aware mismatch between the selected left/right BPM pair and the
boundary candidate's observed left/right periods will rank the exact
`120 -> 150 -> 100 BPM` path above the one-jump shortcut without changing the
candidate graph, frontier widths, or any real-data gate.

## Root Objective

Preserve a single phase-continuous absolute beat axis across bounded local
blocks while distinguishing a real sequence of BPM jumps from a cheaper path
that skips an intermediate tempo.

## Goal Decomposition

- Subgoal 1: condition a transition on the two-sided tempo evidence already
  carried by its immutable boundary candidate.
- Subgoal 2: recover clean one- and two-jump grids under every frozen block
  schedule without adding sections to constant, noisy, short-island, or alias
  controls.
- Subgoal 3: prove through score and path ledgers that the change comes from
  the new transition potential, not a new candidate, cap, or global replay.
- Subgoal 4: leave real-cache schedule selection, repair80, weak comparison,
  and all fresh holdouts untouched until a later accepted card.

## Evidence Entering the Card

The frozen Experiment 005 card has SHA-256
`898ecef9bd88c1878713b5e5ad9fbce595b6bca3b23c247d53220ad27b312aea`.
Its selected `LF3` implementation returns `120 -> 100 BPM` on a clean
72-second `120 -> 150 -> 100 BPM` synthetic. The correct path is feasible,
survives the width-64 local search, and reaches cut export input. Retaining all
export states still selects the shortcut.

The terminal objective decomposes as follows:

| Path | Local evidence | Transition terms | Total |
| --- | ---: | ---: | ---: |
| correct `120 -> 150 -> 100` | `0.058970136` | `0.575574216` | `0.634544352` |
| shortcut `120 -> 100` | `0.210336493` | `0.271919534` | `0.482256026` |

Both paths have zero count, section-duration, and tail prior. The shortcut
saves one fixed change penalty, and that saving is larger than its middle-
section local-evidence loss. The boundary candidates already carry period
pairs `500/400 ms` at 12 seconds and `400/600 ms` at 36 seconds, but the
Experiment 005 transition score ignores pair compatibility.

## Candidate Variants

### E6-A: global transition rescaling

Multiply every inherited transition by one scalar `alpha`. The observed paths
would reverse only for approximately `alpha < 0.4985`. This variant is
rejected without implementation: it is an underidentified global weight tune,
weakens every false-section guard, and does not use the evidence that isolates
the failure.

### E6-B: hard boundary-pair gating

Allow a transition only when its selected BPMs match the boundary's left and
right period estimates. This variant is rejected before implementation because
it requires a new noise/alias tolerance and deletes paths rather than ranking
them. A one- or two-percent period error could make a valid transition
unreachable, confounding objective quality with candidate recall.

### E6-C: additive pair mismatch

Keep the inherited fixed `0.18` change cost and add `0.18 * C_pair`. On the
observed failure, the fixed cost remains paid twice by the correct path; with
the frozen coefficient the shortcut remains cheaper. Increasing the new
coefficient until this one fixture flips would be outcome-driven tuning. This
variant is rejected analytically.

### E6-D: pair-conditioned change potential with sparsity floor

Replace `C_change_sparsity = 1` with a convex combination
`C_change_exp006 = rho + (1 - rho) * C_pair`. Candidate floor values are
`rho in (1/2, 1/4, 1/8, 0)`. Choose mechanically, before implementation, the
largest floor whose exact frozen Experiment 005 path-ledger replay makes the
correct path strictly cheaper than the shortcut. This selects `rho = 1/4`:
`rho = 1/2` does not flip the order, while `1/4` does. The lower floors are
rejected because they discard more sparsity than the observed counterexample
requires. Keep every other score term, candidate, schedule, prior, tie-break,
and resource cap unchanged. This is the selected variant.

## Frozen Pair Potential

For a boundary candidate with finite positive periods `p_left` and `p_right`,
derive evidence BPMs without rounding:

```text
qhat_left  = 60000.0 / p_left
qhat_right = 60000.0 / p_right
```

Let the frozen alias multipliers, in their existing tuple order, be:

```text
A = (1/4, 1/3, 1/2, 1, 2, 3, 4)
```

For positive finite BPMs define the alias-aware octave distance:

```text
d_alias(q, qhat) = min(
    1.0,
    min(abs(log2(q / (m * qhat))) for m in A)
)
```

No BPM is rounded before this calculation. Alias multipliers whose product is
outside `[20,1000]` are still evaluated because `qhat` is evidence, not a
generated BPM; the selected `q` itself remains under the frozen hard guard.
Non-finite or nonpositive boundary periods remain an integrity failure under
the existing candidate validator and are not assigned a finite score.

The pair potential is:

```text
C_pair = min(
    1.0,
    d_alias(q_left, qhat_left) + d_alias(q_right, qhat_right)
)
```

The frozen sparsity floor and Experiment 006 change potential are:

```text
rho = 0.25
C_change_exp006 = rho + (1.0 - rho) * C_pair
```

The Experiment 006 transition cost is:

```text
C_transition_exp006 =
    0.18 * C_change_exp006
  + 0.12 * C_alias_switch
  + 0.10 * C_jump_size
  + 0.15 * C_boundary_support
```

The accumulation and duration normalizers remain exactly those of Experiment
005. `C_change_exp006` replaces the constant `C_change_sparsity = 1`; it is not
an additional term. Every transition therefore retains a raw sparsity cost of
at least `0.18 * 0.25 = 0.045`, even when its period pair matches exactly or
under an allowed alias. Boundary support still uses the observed candidate
time, while the snapped lattice time remains the authoritative half-open
section boundary.

Binary64 evaluation order follows the displayed formulas: generate `qhat`,
iterate `A` in tuple order, calculate `q / (m * qhat)`, then `log2`, absolute
value, inner minimum, individual clip, sum, outer clip, multiply `C_pair` by
`0.75`, and add `0.25`. A diagnostic ledger records the two un-clipped alias
minima, `C_pair`, `rho`, `C_change_exp006`, every other transition component,
raw transition cost, normalizer, and normalized increment.

For the exact correct transitions, `C_pair = 0` and
`C_change_exp006 = 0.25`. For the observed shortcut at
the first boundary, the nearest alias match between selected `100 BPM` and the
boundary's `150 BPM` right-side evidence has distance
`abs(log2(100 / 75)) = 0.415037499...`, so its change potential is
`0.561278124...`. A replay of the two already observed path ledgers, holding
local costs, closures, support, alias, jump terms, and path identities fixed,
gives approximately `0.409544352` for the correct path and `0.416447746` for
the shortcut. The analytic margin is only `0.006903394`; it mechanically
selects the highest feasible sparsity floor but is not a prediction that
search ordering or final output will remain fixed. The synthetic execution
must reconstruct the actual retained path ledgers independently.

## Local Verification Matrix

Every fixture runs under `S30`, `S60`, `S90`, and `S64`. Frame probabilities,
candidate tuples, RNG seed, expected section beats/times, and both objective
variants are materialized by source-owned tests before the first execution.

1. Clean constant: 72 seconds at 120 BPM, no real boundary. Both variants must
   return one section.
2. Noisy constant: the same grid with deterministic PCG64 seed `6006`,
   zero-mean Gaussian noise of standard deviation `0.03` added to both frame
   probabilities and clipped to `[0,1]`. The candidate set is constructed
   after the signal is frozen. Both variants must return one section.
3. Single jump: `120 -> 150 BPM` at 36 seconds. Run exact period evidence and
   two fixed perturbed arms: `(left * 1.02, right * 0.98)` and
   `(left * 0.98, right * 1.02)`. E6-D must return the exact two-section grid in
   all three arms; no hard gating is permitted.
4. Original two-jump kill: `120 [0,12s) -> 150 [12s,36s) -> 100 [36s,72s)`.
   The Experiment 005 comparator must reproduce its two-section shortcut and
   E6-D must return the exact three-section grid.
5. Direction control: `100 [0,12s) -> 150 [12s,36s) -> 120 [36s,72s)` with
   exact period pairs. E6-D must return the exact three-section grid.
6. Short false island: the signal remains constant 120 BPM. Inject
   `120 -> 150` and `150 -> 120` boundary-pair candidates at `24.0s` and
   `32.4s`, respectively; the latter source peak is included explicitly and
   the 8.4-second spacing satisfies the frozen merge and section-duration
   guards. E6-D must return one 120-BPM section.
7. Alias trap: the signal remains constant 120 BPM with tempo candidates
   `60/120/240`. Inject exact `120 -> 240` and `240 -> 120` pair candidates at
   `24.0s` and `32.5s`. E6-D must return one 120-BPM section and zero alias
   switches.
8. Dense compatible islands: the signal remains constant 120 BPM. Inject four
   successive, strictly merge-valid pairs at `12.0s`, `20.4s`, `28.8s`, and
   `37.2s`, alternating exact `120 -> 150` and `150 -> 120` evidence; include
   exact materialized source peaks at all four times. E6-D must return one
   120-BPM section. Diagnostics must show that every false transition retains
   the `0.045` raw sparsity floor.
9. Dense alias islands: the signal remains constant 120 BPM with tempo
   candidates `60/120/240`. Inject four successive pairs at the same times,
   alternating exact `120 -> 240` and `240 -> 120` evidence. All pair distances
   are zero under the alias orbit, so this directly tests the retained
   sparsity floor. E6-D must return one 120-BPM section with zero alias
   switches.

The matrix contains `4 * 11 = 44` schedule/fixture arms: one constant, one
noisy constant, three single-jump period arms, one original two-jump, one
direction control, one false island, one alias trap, one dense compatible
island, and one dense alias island. Fixture generation
and all expected results are frozen before execution. A fixture that violates
the existing strict candidate contract invalidates the card; it is not silently
normalized, weakened, or replaced after observing a score.

## Selected Variant

- Selected: `E6-D pair_conditioned_change_floor_1_4`.
- Rejected: global rescaling is underidentified; hard gating confounds recall;
  additive same-weight mismatch is analytically insufficient; a joint
  adjacent-boundary likelihood-ratio search is deferred because it adds new
  evidence windows and normalization choices.
- Why this is the smallest useful test: it changes one already-owned
  transition component, consumes only existing boundary fields, and leaves the
  entire local-frontier decomposition and candidate topology intact.

## Selection Pressure

- Primary pressure: exact expected section count, BPM sequence, integer beat
  boundaries, and section times on all 44 arms.
- Mechanism pressure: the original shortcut and correct path must both be
  scoreable; E6-D must reverse their terminal order through `C_pair` while
  every transition retains the `rho=1/4` floor; objective ledgers must
  reconstruct both totals.
- Guard pressure: no false section on constant/noisy/single/dense island or
  alias controls; no
  candidate, cap, schema, continuity, metadata-boundary, or deterministic
  replay regression.
- Runtime pressure: the complete CPU-only matrix finishes in under two minutes
  and E6-D adds no asymptotic search dimension.
- Kill pressure: no coefficient, alias set, clip, fixture, period perturbation,
  candidate cap, frontier width, or expected result may change after execution.

## Research Question

Does a two-sided, alias-aware boundary transition potential resolve the
multi-jump shortcut while preserving the bounded phase-continuous local search
and its false-section controls?

## Closest Analogies / Novelty Layer

- Closest analogies: pairwise transition potentials in HMMs/CRFs, change-point
  models conditioned on two-sided sufficient statistics, and edge potentials
  in structured prediction.
- Relevant taxonomy bucket: structured postprocessing and local objective
  calibration.
- Novelty layer: there is no algorithmic novelty claim. The contribution is a
  Pulsefield-specific evidence/transition contract for one BeatThis cache and
  contiguous half-open beat sections.
- Representation novelty versus engineering variation: this is an engineering
  correction to an edge potential, not a new dynamic-programming method.

## Minimal Change

Add a separately named Experiment 006 objective variant to the source-owned
local-frontier prototype. The default Experiment 005 entry point must retain
its exact baseline behavior so the known shortcut remains reproducible. Share
candidate extraction, block search, state schema, and all non-transition score
code; do not copy the 1,700-line core.

Before changing behavior, convert the intentionally red Experiment 005
two-jump test into a passing baseline-reproduction assertion that explicitly
expects the pinned `[120,100]` shortcut, and add a separate E6-D assertion for
the correct grid. This preserves the executable negative result while allowing
the complete suite to become green.

## Files Likely to Change

- `src/pulsefield_model/timing/v3/local_frontier.py`
- `tests/timing/test_timing_v3_local_frontier.py`
- `tests/timing/test_timing_v3_boundary_pair_transition.py`
- `docs/research/timing_v3_experiment_006_boundary_pair_transition.md`
- `docs/research/timing_v3_experiment_006_result.md`, only after execution
- `docs/research/timing_v3_problem_log.md`, only after a decision

No production fitter, provider, evaluator, split selector, runner, or packaged
configuration file changes in this card.

## Read-Only Context Files

- `docs/research/timing_v3_experiment_005_local_frontier_decomposition.md`
- `docs/research/timing_v3_experiment_005_result.md`
- `docs/research/timing_v3_task_definition.md`
- `src/pulsefield_model/timing/v3/global_constant_jump.py`
- `src/pulsefield_model/timing/v3/schema.py`

## Dataset Slice

Source-owned synthetic arrays only. No file in `artifacts/`, BeatThis cache,
audio, `.osu`, label manifest, schedule16, repair80, old holdout, new holdout,
broad500, full5050, API snapshot, or network source may be opened, selected,
materialized, or evaluated.

## Baseline / Comparator

The primary comparator is the pinned Experiment 005 objective on the identical
synthetic prediction and candidate set. It must reproduce the known shortcut
before E6-D is interpreted. `TimingV3Grid` schema continuity and the existing
Timing-v3 tests remain guards. Current v2 and weak `.osu` evidence are outside
this synthetic objective experiment.

## Primary Metric

Exact pass count over the 44 frozen schedule/fixture arms. Passing requires
`44/44`; there is no aggregate compensation. Each arm compares the serialized
section count, binary64 BPM sequence, integer half-open beat ranges, derived
boundary times within the existing schema seam tolerance, and alias-switch
count where specified.

## Secondary Metric

- strictly positive selected-versus-runner-up terminal objective margin;
- exact transition-ledger reconstruction within
  `max(1e-12, 8 * ulp(reconstructed_total))`;
- candidate fingerprint equality between Experiment 005 and E6-D for each
  paired arm;
- correct and shortcut path availability before terminal selection on the
  original kill fixture;
- frontier width, class coverage, pruning reason, section count, and every
  resource counter/cap;
- deterministic two-run grid, ledger, replay, and diagnostics fingerprints.

The two perturbed single-jump arms additionally require a strictly positive
margin over the best persistence and wrong-right-BPM alternatives. The
original kill fixture records four local counterfactual ledgers over the same
paths and candidates: Experiment 005's constant `1`, the rejected floor
`rho=1/2`, selected `rho=1/4`, and a pair-scrambled control formed by swapping
the two boundary candidates' `(p_left,p_right)` pairs while leaving times,
signals, and paths fixed. The expected order is: Experiment 005 and `rho=1/2`
prefer the shortcut; selected `rho=1/4` prefers the correct path; scrambling
removes that selected-variant advantage. These are ledger-only ablations and
do not create additional search arms or change the 44-arm primary denominator.

Different frontier identities or prune ordering caused by the intended score
change are diagnostic, not a guard failure. Candidate bytes, caps, and
non-transition formulas must remain identical.

## Verify Command / Evaluation Procedure

1. Freeze fixture bytes and candidate fingerprints in tests.
2. Run the Experiment 005 comparator first and require the known shortcut.
3. Run E6-D over all 44 arms in fixed fixture-major then schedule order.
4. Recompute every reported objective from its component ledger with an
   independent test helper.
5. Run every arm twice and compare deterministic mathematical payloads.
6. Stop on any kill criterion; do not inspect or generate real data.

Expected focused command:

```sh
.venv/bin/python -m pytest -q \
  tests/timing/test_timing_v3_local_frontier.py \
  tests/timing/test_timing_v3_boundary_pair_transition.py
```

## Guard Check

```sh
.venv/bin/python -m pytest -q tests/timing/test_timing_v3_*.py
.venv/bin/python -m py_compile \
  src/pulsefield_model/timing/v3/local_frontier.py \
  tests/timing/test_timing_v3_local_frontier.py \
  tests/timing/test_timing_v3_boundary_pair_transition.py
git diff --check
```

The full Timing-v3 test set must be green after the negative baseline is
represented as an expected baseline result rather than an intentionally
failing assertion.

## Qualitative Check

Inspect only source-owned synthetic ledgers for the original two-jump case,
the two perturbed single-jump cases, the false island, and the alias trap.
Confirm that `C_pair` has the expected side-specific owner and that every
selected section remains phase-contiguous. No audio, map, artifact, or plot is
permitted in this card.

## Positive Signal

- Experiment 005 reproduces its pinned shortcut;
- E6-D passes all 44 arms unchanged;
- the exact correct two-jump path becomes uniquely best through the pair
  transition ledger;
- constant/noisy/island/alias controls remain single-section;
- no candidate, cap, continuity, determinism, or runtime guard fails.

## Negative Signal

- the shortcut remains selected under any schedule;
- the correct path disappears before terminal ranking;
- a valid perturbed-period single jump is deleted or moved;
- E6-D creates a false constant/island/alias section;
- a result depends on a new tolerance, wider search, or coefficient change.

## Kill Criteria

Kill E6-D for this loop if any of the following occurs:

- baseline Experiment 005 cannot reproduce the known shortcut from the frozen
  fixture;
- any one of the 44 expected grids fails;
- any objective ledger cannot be reconstructed within the frozen tolerance;
- the candidate fingerprints differ between paired objective variants;
- success requires changing `0.18`, the alias tuple, either clip, a fixture,
  period perturbation, candidate order, beam/frontier width, resource cap,
  timeout, schema, or tie-break;
- any non-synthetic input is needed to choose or repair the variant.

## Expected Failure Modes

- the `rho=1/4` floor may remain too small to reject compatible dense islands;
- alias-aware zero mismatch may make a half/double-tempo island too cheap even
  with the retained sparsity floor;
- two-percent period error may reduce the correct margin below a shortcut;
- block-local nonlinear correlation may still change ordering by schedule;
- the correct path may become best but still be lost by a same-class export;
- a score cache that omits objective-variant or boundary-period identity may
  reuse the Experiment 005 transition incorrectly.

## Confounders

- the boundary periods are derived from the same BeatThis peaks as local
  evidence, so the two score families are correlated;
- exact synthetic pulses overstate period reliability relative to real caches;
- the alias orbit treats specified ratios as equivalent and cannot choose the
  musical metrical level by itself;
- passing a two-jump matrix does not establish real-audio boundary precision,
  phase quality, drift, fallback rate, runtime, or memory;
- schedule-local correlation is not additive across arbitrary partitions.

## Expected Runtime / Runtime Budget

CPU-only, under two minutes for the complete 44-arm matrix and focused
regression set. Any single arm exceeding 10 seconds or the matrix exceeding
two minutes is a synthetic hard stop and is recorded as a negative result.

## Result Interpretation Plan

- Positive result would suggest: the immediate Experiment 005 failure is
  caused by an unconditioned transition potential, while bounded fixed-lag
  decomposition remains viable enough for a new real-cache protocol card.
- Negative result would suggest: this pair potential is insufficient or
  creates false sections; kill it without tuning and return to a joint local
  split-gain or adjacent-boundary formulation.
- Ambiguous result would require: if the baseline fixture, strict candidate
  contract, or score ledger cannot be reproduced, invalidate Exp006 and repair
  the measurement fixture before selecting any variant.
- Human owner decides: whether a synthetic pass warrants a separately frozen
  schedule16/repair80 card.
- Next-loop action if positive: `TEST` a new real-cache protocol card; do not
  reuse Experiment 005's real-stage authorization implicitly.
- Next-loop action if negative: `MUTATE`; do not implement the rejected alpha,
  hard-gate, or additive variants on this evidence.
- Next-loop action if ambiguous: stop and restore measurement validity.

## Result Log Template

- Experiment: Timing v3 Experiment 006
- Date:
- Source and test SHA-256:
- Frozen card SHA-256:
- Synthetic fixture/candidate fingerprints:
- Baseline shortcut reproduced:
- Runtime:
- Arm pass count / 44:
- Exact-grid failures:
- Objective-ledger reconstruction failures:
- Original-kill baseline/correct/shortcut objectives and margin:
- Constant/noisy/island/alias false-section counts:
- Candidate fingerprint equality:
- Candidate/frontier/cap diagnostics:
- Deterministic replay result:
- Guard command / result:
- Kill criteria triggered:
- Interpretation:
- Recommended next step: `KILL | MUTATE | TEST`
- Human owner decision:

## Pre-Execution Gate

- Card complete: yes; revision 2 passed independent blocker and scientific
  re-review.
- Code execution allowed after this card: yes, only for the scoped synthetic
  Exp006 implementation, fixtures, objective ledgers, and verifier surface.
  Real-cache runner, schedule16, repair80, holdout, broad, full5050, production
  fitter, provider, evaluator, and split changes remain forbidden.
- Closed loop complete: yes at the planning layer.
- Remaining ambiguity: none before synthetic execution. The 44-arm result may
  still be negative and must not trigger a coefficient, floor, fixture, cap,
  or expected-output repair.

## Next-Loop Action

- If positive: write the Exp006 result, then create a new card for exposed
  scheduler selection and repair80.
- If negative: write the result and return `MUTATE` to the smallest attributed
  objective layer.
- If ambiguous: invalidate the execution and repair only the measurement
  contract before rerunning the same frozen question.

## Novelty Notes

- Closest analogies: pairwise structured-prediction potentials and two-sided
  change-point likelihoods.
- Novelty layer: repository-specific evidence accounting only.
- Representation novelty versus engineering variation: engineering variation;
  no novelty claim.
