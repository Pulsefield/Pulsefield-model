# Timing v3 Experiment 006 Measurement Repair 002

Date: 2026-08-12

## Mode

- Mode: planner
- Route: `TEST`
- Parent card: `timing_v3_experiment_006_boundary_pair_transition.md`, frozen
  SHA-256 `789c1ffe08d2ed05c2268b6dc0a2fe4d0f4cd9bdf0b0d4a84dd57a62f9f895f7`.
- Parent measurement repair: `timing_v3_experiment_006_measurement_repair_001.md`,
  frozen SHA-256 `c1eda039a57c95931ca6d0cca6289e507a5a474a86d5ef339ee11004256b2280`.
- Source entering this repair: `local_frontier.py` SHA-256
  `97060ec638a6c3e57b84102c7b1eb3c09b0d8c1d78a7182f1c8c786d24d3c6a1`.
- Scope: synthetic diagnostics and verifier only. This card does not authorize
  scoring, search, candidate, fixture, expected-grid, cap, tie-break, real
  cache, `.osu`, audio, network, or production-runner changes.

## Hypothesis

Measurement Repair 001 exposes actual scored edges and occurrence-specific
membership flags, but terminal and provisional public ledgers do not expose
the edge-order references used to derive those flags. Adding Exp006-only
public occurrence-reference records will make the v2 diagnostics
self-auditable without changing any mathematical result or legacy behavior
projection.

## Root Objective

Remove the remaining self-proof gap in Exp006 v2 diagnostics: an external
verifier must be able to reconstruct retained-terminal, retained-provisional,
and selected-traceback membership flags from public occurrence references and
the public `actual_scored_edges` stream, rather than trusting flags alone.

## Goal Decomposition

1. Preserve the existing public `actual_scored_edges` stream, edge ordering,
   component cache, resource records, class coverage records, grids, and
   behavior projections.
2. Add public Exp006-only occurrence references for terminal paths,
   provisional paths, and the selected traceback.
3. Make the matrix verifier reconstruct membership flags from those references
   and fail if replay-key identity alone would over-select duplicate
   occurrences.
4. Keep explicit Exp005 ledgers on the v1 schema and default Exp005 payloads
   byte-identical to their frozen oracles.

## Evidence Entering the Card

Current focused verification on source SHA
`97060ec638a6c3e57b84102c7b1eb3c09b0d8c1d78a7182f1c8c786d24d3c6a1`
passes:

- `.venv/bin/python -m pytest -q tests/timing/test_timing_v3_local_frontier.py tests/timing/test_timing_v3_boundary_pair_transition.py tests/timing/test_timing_v3_boundary_pair_transition_matrix.py --tb=short`
  -> `108 passed in 13.50s`;
- `.venv/bin/python -m pytest -q tests/timing/test_timing_v3_*.py --tb=short`
  -> `504 passed, 2 skipped, 9 subtests passed in 157.28s`.

The remaining issue is observability, not behavior: `_Path.transition_edge_orders`
and `_ProvisionalTransitionOccurrenceRecord.edge_orders` exist internally, and
`actual_scored_edges` exposes the resulting membership flags, but public
terminal/provisional ledgers contain only replay-key transition entries.
Duplicate logical replay-key edges can exist, so replay-key identity is not a
sound public proof of occurrence membership.

## Candidate Variants

### MR2-A: keep flags only

Rely on `actual_scored_edges.retained_*` and `selected_traceback_path` flags.
Reject: this is compact but self-referential. A verifier cannot distinguish a
correct occurrence-specific implementation from one that marked every duplicate
logical edge.

### MR2-B: add edge orders to existing terminal/provisional ledger dataclasses

Add `edge_orders` fields to `TerminalObjectiveLedger` and
`ProvisionalTransitionLedgerRecord`. Reject: these dataclasses are shared with
explicit Exp005 ledgers, so this would perturb v1 legacy payloads and the
frozen `3911d91c...` explicit-Exp005 behavior oracle.

### MR2-C: add Exp006-only occurrence-reference records

Selected. Add v2-only records to `BoundaryPairObjectiveDiagnosticsV2`:

- one terminal-path occurrence record per terminal ledger rank;
- one provisional-path occurrence record per provisional ledger record;
- one selected-traceback occurrence tuple.

Each record references the existing public ledger by rank or record index and
contains the exact edge-order tuple. The matrix verifier reconstructs
membership flags solely from these public edge-order tuples.

## Selected Variant

Selected: `MR2-C`.

The change is schema-only and Exp006-only. It does not change scoring, search,
candidates, local evidence, transition components, block scheduling, export
selection, or any expected grid.

## Frozen Measurement Contract

Add three v2-only public fields:

```text
terminal_path_occurrence_records: tuple[
  {selection_rank, replay_key, selected, edge_orders}
]

provisional_path_occurrence_records: tuple[
  {record_index, block_index, committed_replay_key,
   ranked_lookahead_replay_key, edge_orders}
]

selected_traceback_edge_orders: tuple[int, ...]
```

Rules:

- `edge_orders` are exact zero-based occurrence IDs in
  `actual_scored_edges.edge_order`.
- Terminal occurrence records are sorted by terminal `selection_rank` and
  one-to-one with `terminal_path_ledgers`.
- Provisional occurrence records are sorted by append order and one-to-one
  with `provisional_path_ledgers`.
- `selected_traceback_edge_orders` equals the edge-order tuple for the single
  selected terminal path; it is empty for a selected path with no transitions.
- The public verifier reconstructs:
  - `retained_terminal_path` from the union of terminal edge-order tuples;
  - `retained_provisional_path` from the union of provisional edge-order
    tuples;
  - `selected_traceback_path` from `selected_traceback_edge_orders`.
- Every referenced edge order must exist exactly once in `actual_scored_edges`;
  every selected edge must also be terminal-retained.
- Replay-key identity may be used only as a negative regression guard proving
  that duplicate logical edges would have been over-selected by the old
  verifier. It must not determine membership.

`BOUNDARY_PAIR_TRANSITION_CONTRACT_VERSION` advances from v2 to v3. The
complete Exp006 objective diagnostic fingerprint intentionally changes.
The following must remain exactly unchanged:

- all 44 selected Exp006 grids;
- all candidate, replay, and grid fingerprints;
- default Exp005 legacy aggregate `a7401b73d9ed9dca9b3f7b65ae599e6715d9a9df2bf621cc063d1f54e88b57c6`;
- Exp006 behavior-projection aggregate `f72574d4af07d6b2b13699276ba9a60d68c915819459851edca7217780b26af8`;
- explicit Exp005 behavior-projection aggregate `3911d91c787e92af8b4e9554a20186b7f178f1a66e9d040f4c86f1963806a127`;
- unique transition-component trace `abf249fdff92c60712b2d05e02f736768079958e5fa7353a86a921e31f66e7a9`;
- scored-edge access trace `e1f8135afbe114aabeeb7162694fa84f8740c84242eb4313e6de44ca3b27e438`.

## Dataset Slice

The same source-owned 11-fixture by 4-schedule synthetic matrix. No file in
`artifacts/`, BeatThis cache, audio, `.osu`, manifest, schedule16, repair80,
holdout, broad500, full5050, API snapshot, or network source may be accessed.

## Baseline / Comparator

Measurement Repair 001 at source SHA
`97060ec638a6c3e57b84102c7b1eb3c09b0d8c1d78a7182f1c8c786d24d3c6a1`.
The comparator behavior is the existing green focused/full guard and the five
frozen aggregate oracles listed above.

## Primary Metric

The matrix verifier reconstructs every occurrence membership flag from public
occurrence-reference records and passes all 44 arms.

## Secondary Metrics and Guards

- the five frozen behavior/cache/access oracles remain exact;
- the four original-kill default Exp005 full-payload hashes remain exact;
- explicit Exp005 remains on v1 objective diagnostics with no v2/v3
  measurement fields;
- full Timing-v3 guard passes;
- `py_compile` and `git diff --check` pass;
- no source change outside `local_frontier.py` and the two Exp006 test files;
- complete focused suite remains under two minutes.

## Verify Commands

```sh
.venv/bin/python -m pytest -q \
  tests/timing/test_timing_v3_local_frontier.py \
  tests/timing/test_timing_v3_boundary_pair_transition.py \
  tests/timing/test_timing_v3_boundary_pair_transition_matrix.py --tb=short
.venv/bin/python -m pytest -q tests/timing/test_timing_v3_*.py --tb=short
.venv/bin/python -m py_compile \
  src/pulsefield_model/timing/v3/local_frontier.py \
  tests/timing/test_timing_v3_boundary_pair_transition.py \
  tests/timing/test_timing_v3_boundary_pair_transition_matrix.py
git diff --check
```

## Positive, Negative, and Ambiguous Signals

- Positive: every occurrence membership flag is reconstructable from public
  edge-order references, while all frozen behavior/cache/access oracles pass.
- Negative: self-contained occurrence proof requires changing search,
  scoring, candidates, fixtures, or Exp005 v1 diagnostics. Kill this repair.
- Ambiguous: public occurrence references cannot be made deterministic or
  cannot be kept one-to-one with existing public ledgers without changing
  terminal ordering. Stop and return to planner mode.

## Kill Criteria

Stop on any grid, candidate fingerprint, replay fingerprint, default payload,
explicit Exp005 behavior projection, Exp006 behavior projection, cache/access
trace, terminal ordering, occurrence count, or runtime regression; on any
non-synthetic input access; or if diagnostics are consumed by inference.

## Result Interpretation Plan

A positive result closes the Exp006 synthetic observability loop and permits a
result log for the parent Exp006 card. It does not authorize schedule16,
repair80, holdout, broad, full5050, production fitter, ramp support, or real
cache evaluation. Those require a separate real-data protocol card.

