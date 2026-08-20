# Timing v3 Experiment 004 Result: Global Constant/Jump Assembly

Date: 2026-08-11

Decision: reject the frozen global constant/jump search as a promotion
candidate. The repaired 80-audio execution still produced row-scoped timeouts
on `13/80` rows and a selected fallback rate of `16.25%`. Both are hard-stop
conditions under the pre-registered protocol. No Experiment 004 holdout,
broad500, or full5050 input was selected or evaluated, and the production
`TimingV3Fitter` was not switched.

The negative result is narrower than “one BeatThis cache cannot support timing
v3.” It shows that this particular whole-track candidate graph and exhaustive
beam replay do not fit the frozen `180 s` per-audio budget on the exposed
repair workload. Exact score-cache and lattice optimizations remain useful
implementation work, but they do not rescue the global search family. The next
bounded question is local phase-continuous frontier decomposition, not another
weight or timeout change inside Experiment 004.

## Scope and measurement boundary

Experiment 004 consumed exactly one declared, shift-zero BeatThis frame cache
per audio group. Candidate extraction and path assembly read only beat and
downbeat probabilities, frame rate, and cache provenance. `.osu` redlines,
objects, metadata, network sources, labels, and raw audio were excluded from
inference. The evaluator could use stored weak comparators only after a grid or
tagged fallback had been produced.

The original card was frozen at SHA-256
`0421de8abb1a016a215002ec08903282d7ed2500a9fb85a49ab0dc9fb4c1230e`.
Clarification 001 was frozen before any holdout at SHA-256
`b9dba17cc188c7f95be88b7d55e3fd7d7bc28c691dc43773c698adc9c88d058f`.
Together they fixed these runtime boundaries:

- one `180 s` timeout for cache load, current-v2 fitting, candidate extraction,
  and the ordered `CJ0` through `CJ3` loop;
- at most `120000` unique section-score cache misses per audio and variant,
  with miss `120001` rejected as a tagged fallback;
- an under-30-minute repair80 budget;
- completed variant prefixes retained when a later variant times out;
- no best-so-far success after an attempt cap or timeout.

`formal_execution_ready=true` in a projection summary means its inputs,
provenance, and runner contract were complete. It is not a quality or runtime
pass. The independent `hard_guards.ok` field and later weak-evidence stage make
that decision.

## First repair execution

The immutable v1 repair execution exposed two implementation problems before
it could test the hypothesis cleanly: equivalent geometric scores were split
by over-specific cache keys, and a row timeout replaced already completed
variant results. It nevertheless established that the original implementation
was far outside the intended operating range.

| Measurement | v1 result |
| --- | ---: |
| Selected fallback | `75/78 = 96.15%` projection-evaluable rows |
| Selected fallback reasons | 61 attempt-cap, 14 timeout |
| Rows with a completed projection result | 78/80 |
| `CJ3` accepted | 3/80 |
| Projection wall time, four workers | `2106.516 s` |
| Weak-comparator decision | `kill` |
| Pure-`CJ3` paired phase coverage | `3/76 = 3.95%` |

The v1 per-variant timeout attribution is not used for causal accounting
because the runner had not yet preserved completed prefixes. The row-level
timeouts, cap failures, and low accepted coverage remain valid observations.
The v1 projection and weak outputs are retained as immutable audit evidence,
not as the final repaired execution.

## Protocol-preserving repairs

Clarification 001 made the intended cache equivalence and timeout semantics
testable without changing candidates, weights, priors, beam width, attempt
cap, timeout, or gates. The implementation then received only
protocol-preserving repairs:

- geometry-only score keys for `CJ1`/`CJ2`, canonical modulo-4 phase classes
  for downbeat-aware scores, and mandatory terminal BPM identity;
- immediate serialization of each completed variant so a later timeout cannot
  erase the prefix;
- exact count, support, terminal, transition, and edge-bundle caches;
- deterministic bounded-beam insertion and replay improvements;
- vectorized regular-lattice peak matching with scalar fallbacks for widened,
  non-finite, and numerically extreme inputs.

The final lattice implementation is an exact-equivalence optimization, not a
new candidate. Random, ULP-boundary, duplicate-peak, half-open interval, and
large-index differential tests compare it with the preserved scalar helpers.
The large-index guard is material: at BPM 1000 with an origin beyond `2**53`,
the materialized shortcut and scalar reference differ, so the public path
deliberately retains the scalar result.

## Repaired projection execution

The immutable v2 projection was run after the cache-key and timeout-attribution
repairs, using four spawn workers on CPU NumPy. Its formal source snapshot
predates the later exact lattice vectorization; the distinction is recorded
below rather than treating diagnostic repairs as a formal rerun.

| Measurement | v2 result |
| --- | ---: |
| Stage/cache-valid/projection-evaluable | `80/80/80` |
| `CJ0` | 80 accepted |
| `CJ1` | 75 accepted, 5 timeout/not-run |
| `CJ2` | 71 accepted, 9 timeout/not-run |
| `CJ3` | 67 accepted, 13 timeout/not-run |
| Selected fallback | `13/80 = 16.25%` |
| Selected fallback reason | 13 timeout |
| Projection wall time, four workers | `1314.025 s` |
| Hard guard | fail |

The timeout sets were nested as required by ordered row termination: `CJ1`
timed out on 5 rows, `CJ2` on 9, and `CJ3` on 13. The run finished inside the
aggregate 30-minute repair budget, but that does not compensate for individual
rows failing the frozen `180 s` guard. A timeout is a row-level hard guard and
also makes the `16.25%` selected fallback rate worse than the card's `>10%`
kill band. Therefore the protocol forbade creating a holdout from this source.

The projection summary intentionally has no oracle phase ratios. Weak
comparators are a later layer, and the projection-only status is
`undefined_projection_only`.

## Weak-evidence evaluator failure

The v2 weak-evidence pass wrote 34 immutable rows and then stopped before row
34 could be serialized. That projection row had a tagged current-v2 fit
failure while `CJ3` produced an accepted grid. The evaluator independently set
`pure_cj3_phase_matched=true` and `current_v2_phase_matched=false`, then
violated its paired-denominator invariant:

```text
pure_cj3_phase_matched requires current_v2_phase_matched
```

The defect was in denominator construction, not in the grid search. The fix
computes current-v2 matched availability first and gates pure-`CJ3` and
selected-safety paired flags on it, while preserving method-level `CJ3`
metrics and the broader comparator-availability field. A synthetic
current-v2-unavailable/`CJ3`-accepted regression test fails on the old source
and passes on the repaired source; metrics tests also pin the forbidden
`pure=true/current=false` state.

No v2 weak summary exists, so no v2 phase or drift aggregate is reported. The
34-row file is an immutable failed-run artifact and must never be resumed or
overwritten. Because the weak evaluator is in behavior provenance, its source
change would require a fresh projection and weak run before any positive stage
decision. Such a rerun was not performed: the projection hard guard had
already failed, and the final runtime diagnostics below show that an exact
acceleration-only rerun could not pass.

## Why exact acceleration does not rescue the search

The final core source vectorizes the dominant regular-lattice statistics while
preserving scalar behavior at numerical edge cases. On the maximum synthetic
graph, the exact replay improved from roughly `96.7 s` to `55.59 s`; focused
microbenchmarks measured about `12.6x` for count support and `49.3x` for peak
recall. Those results show a real implementation improvement, but the real
repair graph remains structurally too large.

A single-process diagnostic on required repair row 15 used the final core
source. The cache covers `1255.420 s` with 62,771 frames, 16 origin candidates,
256 tempo candidates, 101 boundary candidates, 3,302 beat peaks, and 965
downbeat peaks. Current-v2 fitting took `12.854 s`, candidate extraction
`3.850 s`, and `CJ0` took `0.022 s`. `CJ1` then consumed `120.653 s` before
returning `edge_attempt_cap_exceeded` at exactly 120,000 attempts. The row
reached `137.388 s` before later variants and did not complete under a
`175 s` diagnostic alarm.

An independent `CJ2`-only diagnostic on the same row removed the earlier
variant cost. Candidate extraction took `3.877 s`; the process still did not
complete before its `165.010 s` alarm. Consequently, even a hypothetical
zero-cost `CJ1` would not make the frozen ordered row fit under `180 s` once
current-v2, extraction, `CJ0`, `CJ2`, and `CJ3` are included.

This isolates the remaining cause: dense whole-track replay and scoring scale
with the global boundary/tempo/count graph. It is not sufficient to attribute
the failure to redundant score-cache misses, multiprocessing contention, or
one slow earlier variant. Changing the candidate graph, beam semantics,
timeout, or section-search budget would be a new experiment.

## Reproducibility record

Formal artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `timing_v3_exp004_repair80_projection_v1.jsonl` | `948837e36e4cdfc0614a1a45082fc6b2ae3da407efaf34e6bcc8586b9bf75e0d` |
| `timing_v3_exp004_repair80_projection_v1_summary.json` | `5862990066ac422c4c10bd32c9f74bf8d602d6c87de897fdee8b9f26d497d9af` |
| `timing_v3_exp004_repair80_weak_v1.jsonl` | `6f0a07f75b0676bca1b8b7dae4f1ca4435d6d8c4294f926c2a4545c01a3fe294` |
| `timing_v3_exp004_repair80_weak_v1_summary.json` | `33d21a2dce278d37b5e0e216e83848f9cfccacefe7764c82e98cd0d8deb0d1a8` |
| `timing_v3_exp004_repair80_projection_v2.jsonl` | `39e1329344c5da5be6d655d6ed227526fbf904cd48f91048447fac29f5fd6063` |
| `timing_v3_exp004_repair80_projection_v2_summary.json` | `b6ee9e26b62a6e5ac1816aa9f796341fcd81fbe8c1de51a9188c19696bb04efb` |
| failed `timing_v3_exp004_repair80_weak_v2.jsonl`, 34 rows | `ac006d0d5ba8803dece494c86b39cacd5f4edd903a01484c796cbc48a712a52b` |

The formal v2 projection used commit
`be8993b7fe8325a98d4d8d3b80138b1bd8ffe1b7`, Python 3.10.20, NumPy 1.26.4,
macOS arm64, four spawn workers, behavior fingerprint
`ccb2b8304110a9a8b909bc5da569f2bed9c58fd7cd04217d1c7334e02ca404b3`,
and config fingerprint
`8f8beca14471ea0c3af4d362e1de50c2225fe21eb9509aee0a8fcca3eff4cab0`.
The worktree was dirty, and the summary records the complete dirty-file list.

Relevant source snapshots:

| Component | SHA-256 | Meaning |
| --- | --- | --- |
| formal v2 global core | `ad4fe9ca05ffc25ba2fa9b754cc4600338fab3e8f2cdc3a371b436cbe8bdc50c` | source used by projection v2 |
| final exact-accelerated global core | `736e7c47e57d8567b47a56fa576f0187cf5b25f2dadd33a72ba59c278080528d` | diagnostic and retained implementation |
| runner | `fed1ae54123a815497c01489cffc286a764bf4edd0a0b89fcb77d114b55850d3` | ordered variants and timeout attribution |
| formal v2 weak evaluator | `816949012a86edb7f92427a8b2f3002e88769448a19f67db1097802701e4a6b9` | source that stopped after 34 rows |
| repaired weak evaluator | `f7e5732d2ab1470699300aed6a5bb6c50c4d7183719b2334325a4088b1e884f7` | denominator fix, not formally rerun |
| final global-core tests | `6a40e9f46b4cbdd617407127f15186402d1f49e139d1857a57640f79177c71f1` | exact lattice and search coverage |
| repaired weak-evaluator tests | `3929a0939c0a984d027e3ecb7e39bc58e0503f6328d3c934fcd5cbf65ca848f1` | unavailable-v2 paired denominator coverage |
| metrics invariant tests | `0ebadb32629da44239aeceffc9a078206a1898fde7a560949374c1f376d49350` | paired-denominator invariant |

Focused verification passed for lattice differential cases, non-stress global
search, the maximum synthetic graph's exact two-run replay, runner timeout and
resume behavior, weak-evidence construction, metrics invariants, Python
compilation, and whitespace checks. The exact accelerated core was not used to
create a replacement formal repair artifact, so diagnostic timings and formal
v2 aggregate counts remain explicitly separate.

## Decision and next work

Experiment 004 ends before holdout. No `.osu` comparator row, holdout identity,
broad manifest, or full-corpus result was inspected to tune the fitter after
the repair evidence. The current production timing module and the completed
5,050-row v2 baseline remain the safety comparator.

The next experiment may reuse the single-cache candidate evidence and the
phase-continuous `TimingV3Grid` schema, but it must replace global graph replay
with bounded local inference. Its state handoff must preserve one absolute beat
axis and exact half-open seams, carry multiple plausible phase/tempo states
rather than locking one prefix, and test block duration/overlap on exposed
repair data before any new audio-disjoint holdout is selected. Ramp sections
remain out of scope until the jump-only decomposition passes its runtime,
fallback, phase, drift, and section-count gates.
