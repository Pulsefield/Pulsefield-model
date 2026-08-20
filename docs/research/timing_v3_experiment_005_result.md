# Timing v3 Experiment 005 Result: Local Frontier Decomposition

Date: 2026-08-12

Decision: reject the frozen Experiment 005 local-frontier search before any
real-cache execution. The selected `LF3 frontier16_fixed_lag` implementation
passed constant-track, single-jump, seam, cap, and deterministic-replay checks,
but failed the pre-registered two-jump synthetic check. On a clean
`120 -> 150 -> 100 BPM` track, it returned `120 -> 100 BPM` and skipped the
24-second middle section. The correct path was present in the candidate graph
and survived the local beam, so the failure is in the frozen objective rather
than decomposition, candidate availability, or a resource limit.

The Experiment Card requires stopping when selected Variant D fails local
synthetic verification. Schedule selection, repair80, holdout100, broad500,
and full5050 were therefore not run. No production fitter was changed.

## Scope and measurement boundary

Experiment 005 tested whether a bounded fixed-lag search could replace the
whole-track replay that failed Experiment 004. Every synthetic input contained
only beat and optional downbeat frame probabilities, a frame rate, and a
strictly validated Experiment 004 candidate set. The core crossed an explicit
array-only boundary before fitting and did not read source paths, checkpoint
paths, `.osu` data, metadata, network sources, raw audio, or labels.

The Experiment Card was frozen before implementation at SHA-256
`898ecef9bd88c1878713b5e5ad9fbce595b6bca3b23c247d53220ad27b312aea`.
It selected `LF3` with a 16-state exported frontier, width-64 local buckets,
fixed-lag lookahead, exact open-section state, and the Experiment 004 `CJ3`
score definitions and weights. It also fixed this stop condition: a failure of
selected Variant D on the local verification matrix kills `LF3` for this loop;
the implementation may not compensate by widening the frontier, changing
weights, adding ramps, or running a final global replay.

Only synthetic tests were used. In particular, no schedule16, repair80,
holdout, broad, full-corpus, `.osu`, or external-BPM result was read or written
for this experiment.

## Implementation reached before the stop

The source-owned prototype implements the behavior needed to isolate the
selected hypothesis:

- one immutable Experiment 004 proposal set per cache;
- exact integer-beat/open-section state across fixed time cuts;
- `S30`, `S60`, `S90`, and adaptive `S64` block geometry;
- half-open core ownership and provisional right lookahead;
- 16-state cut export with alias/downbeat class reservation;
- width-64 local buckets keyed by the current right-boundary candidate and
  resulting real-section count;
- source-quota tempo shortlists, three half-up closure counts, the 8-second
  minimum section duration, and all frozen resource caps;
- contiguous `beat [a,b) bpm q` output through `TimingV3Grid`;
- candidate, state, grid, traceback, boundary-ownership, and deterministic
  replay diagnostics.

Review found and the prototype repaired four contract issues before the final
interpretation: snapped mathematical boundary time now owns the half-open
core; both persistence and jump successors share the current boundary bucket;
retained alternatives are not reported as final objective charges; and the
entry point no longer probes unrestricted prediction metadata. These are
protocol-conformance fixes and do not alter the scoring failure below.

## Local verification outcome

All tests other than the decisive two-jump case pass. The passing set includes
long constant tracks across multiple cuts, schedule-invariant unambiguous
grids, supported and too-short single jumps, boundaries immediately before,
at, and after a cut, an authoritative snapped boundary crossing a cut, rejected
boundary persistence, score and graph caps, BPM guard endpoints, metadata
traps, candidate immutability, and repeated deterministic JSON output.

The failing synthetic track is 72 seconds long with exact beat pulses and no
downbeat ambiguity:

| Real section | Interval | BPM |
| --- | --- | ---: |
| first | `[0 s, 12 s)` | 120 |
| middle | `[12 s, 36 s)` | 150 |
| tail | `[36 s, 72 s)` | 100 |

The strict candidate input contains an origin at `0 ms`, all three tempos, and
two supported boundaries. The first boundary reports periods `500/400 ms`
and the second reports `400/600 ms`. Both exact closures are feasible:

- 24 beats from beat 0 close the first section at `12,000 ms`;
- 60 beats at 150 BPM close the middle section at absolute beat 84 and
  `36,000 ms`.

The result nevertheless contains two sections, `120 BPM` followed directly by
`100 BPM` at 12 seconds.

## Objective accounting and root cause

The comparison below uses the actual `S30` core partitions and the frozen
Experiment 005 objective. Local costs are duration-normalized sums of beat,
peak, optional downbeat, and BPM-prior terms. Transition costs include the
fixed change penalty, alias, jump-size, and boundary support terms divided by
`max(1, D / 60000)`, where `D = 72,000 ms`.

| Candidate path | Local evidence | Transitions | Total |
| --- | ---: | ---: | ---: |
| correct `120 -> 150 -> 100` | `0.058970136` | `0.575574216` | `0.634544352` |
| shortcut `120 -> 100` at 12 s | `0.210336493` | `0.271919534` | `0.482256026` |

The false shortcut wins by `0.152288326`. Count, section-duration, and tail
priors are zero for both paths because every section lies within their frozen
preferred ranges. The terminal tie-break is therefore irrelevant.

This is not a missing-candidate or beam-width result:

- the tempo shortlist begins with `120`, `150`, and `100 BPM` and has only 19
  entries, below the cap of 64;
- each block has at most two boundary candidates, below the cap of 32;
- the correct three-section path survives the interior width-64 search and
  reaches the 60-second cut export input;
- retaining every export state in a diagnostic run still selects the false
  shortcut at terminal traceback.

The confirmed mechanism is that every supported boundary may transition to
every tempo in the local shortlist, while the transition score does not test
compatibility with that boundary candidate's left/right period estimates.
Skipping a real intermediate tempo saves one full transition. On this track,
that saving is larger than the local evidence penalty accumulated by fitting
the middle 150-BPM region at 100 BPM. Frontier class compression can later
drop the correct history when it shares the same current 100-BPM class, but
that is secondary: removing the export cap does not change the winning
objective.

## What the result does and does not establish

The result rejects the frozen combination of local-frontier decomposition and
the inherited `CJ3` objective. It establishes that its change sparsity and
unconditioned right-tempo transition can collapse multiple real jumps into a
cheaper shortcut even when the BeatThis evidence and candidates are exact.
The same mechanism can affect any multi-jump track whose skipped middle-section
fit penalty is smaller than the saved transition penalty.

It does not show that fixed-lag decomposition is invalid, that a single
BeatThis cache lacks jump information, or that width 16 is necessarily too
small. The prototype reproduced constant and single-jump grids, and the
correct two-jump path remained available inside the bounded graph. It also
does not measure real-audio phase, drift, weak boundary accuracy, runtime, or
fallback rate because the protocol stopped before those stages.

## Verification and provenance

The expected failure is reproduced by:

```sh
.venv/bin/python -m pytest -q \
  tests/timing/test_timing_v3_local_frontier.py::test_two_jumps_over_min_section_duration_are_serialized_as_tight_sections \
  --tb=short
```

It reports one failure: expected BPMs `[120, 150, 100]`, observed
`[120, 100]`. The remaining focused checks pass with:

```sh
.venv/bin/python -m pytest -q \
  tests/timing/test_timing_v3_local_frontier.py \
  -k 'not two_jumps_over_min_section_duration' --tb=short
```

Result: `48 passed, 1 deselected`. Python compilation and `git diff --check`
also pass. The retained failing test is intentionally neither weakened nor
marked expected-failure: it is the executable negative result.

Relevant source snapshots at the decision point:

| Component | SHA-256 |
| --- | --- |
| Experiment 005 card | `898ecef9bd88c1878713b5e5ad9fbce595b6bca3b23c247d53220ad27b312aea` |
| local-frontier prototype | `b4b40871f730e9ab1a7e968f5ac65dda370dd95c14ea0849027e86ba556524db` |
| local-frontier synthetic tests | `a275e75052d5f0cfc0cbb9ebf2c3d67c7abb359a034a2a1d51878f3f8ab46f98` |
| inherited Experiment 004 core | `736e7c47e57d8567b47a56fa576f0187cf5b25f2dadd33a72ba59c278080528d` |

These files were uncommitted source snapshots in a dirty research worktree;
their byte hashes, rather than a commit hash, identify the evaluated state.

## Decision and next work

Experiment 005 ends as a negative synthetic result. Implementing the LF0-LF2
controls, selecting a block schedule, adding a real-cache runner, or evaluating
repair80 would not answer the failed question and would violate the card's
early-stop rule.

The next bounded question must change the objective, not the decomposition
budget. The smallest candidate is boundary-conditioned tempo-pair evidence:
test whether the observed left/right period pair can discriminate a legitimate
`120 -> 150` transition from a `120 -> 100` shortcut without using `.osu`,
metadata, a larger frontier, or a global replay. Evidence-calibrated change
sparsity and joint adjacent-boundary segment scoring remain separate mutation
families. They require a new Experiment Card and a broader synthetic matrix
before any exposed or fresh real-audio evaluation.
