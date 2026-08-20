# Timing v3 Experiment 006 Measurement Repair 001

Date: 2026-08-12

## Mode

- Mode: planner
- Route: `TEST`
- Parent card: `timing_v3_experiment_006_boundary_pair_transition.md`, frozen
  SHA-256 `789c1ffe08d2ed05c2268b6dc0a2fe4d0f4cd9bdf0b0d4a84dd57a62f9f895f7`.
- Parent implementation before this repair: `local_frontier.py` SHA-256
  `36e0662f4b205708ac527bda5a3838e58a159cfded3eeabc8ebb8c309fd54f36`.
- Scope: synthetic diagnostics and verifier only. This card does not authorize
  a scoring, search, candidate, fixture, expected-grid, cap, tie-break, or
  real-data change.

## Hypothesis

The eight red dense-control arms are a measurement failure, not evidence that
the selected Exp006 grids are wrong. Terminal and provisional ledgers describe
retained paths and therefore omit scored successors that were later pruned.
An Exp006-only, append-only scored-edge ledger plus explicit resource and class
coverage records can close the original card's observability requirements
without changing any mathematical result.

## Root Objective

Make the frozen 44-arm Exp006 result independently auditable while proving
that the repair is observational only.

## Goal Decomposition

1. Distinguish immutable boundary candidates, transition-component cache
   entries, actual scored successor edges, provisional paths, retained terminal
   paths, and selected traceback edges.
2. Expose every actual scored edge and every unique component-cache entry in a
   deterministic schema.
3. Expose the per-block resource counters and exported class keys required by
   the parent card.
4. Prove the default Exp005 payload and every paired grid/candidate/search
   fingerprint remain unchanged.

## Evidence Entering the Card

The stricter matrix verifier currently reports `36/44`; all eight failures are
the two dense fixtures under four schedules. All 44 selected grids are still
exact. Read-only tracing showed:

| Fixture | Schedule | scored-edge occurrences | unique component entries | scored anchors |
| --- | --- | ---: | ---: | --- |
| dense compatible | S30 | 1352 | 299 | 0,1,2,3 |
| dense compatible | S60/S90/S64 | 1170 each | 234 each | 0,1,2,3 |
| dense alias | all four | 11 each | 11 each | 0 |

For dense compatible, exact zero-pair floor components occur at anchors
`{0,1,3}`. For dense alias they occur only at anchor `{0}`. The remaining
injected candidates are merge-valid but are not necessarily phase-feasible
transitions. For example, the `240 BPM` false-island lattice after 12 seconds
misses the later candidates by 50--100 ms while its frozen tolerance is 37.5
ms. A boundary candidate that never becomes a feasible successor is not a
false transition edge.

The pre-repair aggregate hashes, used only as parity oracles, are:

- 44-arm default Exp005 legacy results:
  `a7401b73d9ed9dca9b3f7b65ae599e6715d9a9df2bf621cc063d1f54e88b57c6`;
- 44-arm Exp006 complete v1 results, retained as historical evidence but
  expected to change with the intentional v2 schema/fingerprint:
  `89eaea291ff8ace050b4de110752804c91f4e6747ed1f74e91c66d4021d4a866`;
- 44-arm Exp006 legacy-behavior projection, the formal v2 parity oracle:
  `f72574d4af07d6b2b13699276ba9a60d68c915819459851edca7217780b26af8`;
- 44-arm explicit Exp005-ledger behavior projection:
  `3911d91c787e92af8b4e9554a20186b7f178f1a66e9d040f4c86f1963806a127`;
- 44-arm unique transition-component trace:
  `abf249fdff92c60712b2d05e02f736768079958e5fa7353a86a921e31f66e7a9`;
- 44-arm scored-edge access trace:
  `e1f8135afbe114aabeeb7162694fa84f8740c84242eb4313e6de44ca3b27e438`.

The original-kill default Exp005 full-payload hashes are:

| Schedule | SHA-256 |
| --- | --- |
| S30 | `95fe5e5be4ffdc846b3f120d86011d8a24545e78ffe7601d4ed127f9c2c51d2a` |
| S60 | `5151384b5289cc1cb28baf1b679a3b21108bd3018f72141ecc864cde050c0eff` |
| S90 | `1e9f6f89522940e23cb0a237a393318d90f73afee788922dc4020ea54b7e9a7b` |
| S64 | `483259df753406b3410296374e5b3c3485ee5556fd432aaf6d32f92c4e95b229` |

Each full-payload hash is computed from
`{'reason': result.reason, 'grid': result.grid.to_dict(),
'diagnostics': asdict(result.diagnostics), 'objective':
result.objective_diagnostics}` using UTF-8 JSON with `sort_keys=True`,
`separators=(',', ':')`, and `allow_nan=False`.

All aggregate hashes use this exact helper:

```python
def stable_sha(payload):
    return hashlib.sha256(json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
```

Rows are generated in the exact tuple order of `MATRIX_CASES`, with inner
schedule order `(S30, S60, S90, S64)`. The exact schemas are:

```python
legacy(result) = {
    "reason": result.reason,
    "grid": None if result.grid is None else result.grid.to_dict(),
    "base": asdict(result.diagnostics),
    "objective": (
        None if result.objective_diagnostics is None
        else asdict(result.objective_diagnostics)
    ),
}

exp006_behavior_projection(result) = {
    "reason": result.reason,
    "grid": result.grid.to_dict(),
    "base": asdict(result.diagnostics),
    "objective_legacy_fields": {
        key: value
        for key, value in asdict(result.objective_diagnostics).items()
        if key not in {"contract_version", "deterministic_fingerprint"}
        # After the repair, also exclude every field introduced by the v2
        # measurement schema.  The remaining key set must equal the exact v1
        # BoundaryPairObjectiveDiagnostics field set minus the two keys above.
    },
}

row = {
    "case": case.case_id,
    "arm": arm.value,
    "baseline_sha": stable_sha(legacy(baseline)),
    "exp006_sha": stable_sha(legacy(exp006)),
    "cache_sha": stable_sha(sorted(
        cache_miss_component_asdicts,
        key=lambda value: json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    )),
    "cache_count": len(cache_miss_component_asdicts),
}

baseline_aggregate = stable_sha([
    {"case": row["case"], "arm": row["arm"], "sha": row["baseline_sha"]}
    for row in rows
])
exp006_aggregate = stable_sha([
    {"case": row["case"], "arm": row["arm"], "sha": row["exp006_sha"]}
    for row in rows
])
exp006_behavior_aggregate = stable_sha([
    {
        "case": case.case_id,
        "arm": arm.value,
        "sha": stable_sha(exp006_behavior_projection(exp006_result)),
    }
    for case in MATRIX_CASES
    for arm in (S30, S60, S90, S64)
])
cache_aggregate = stable_sha([
    {
        "case": row["case"], "arm": row["arm"],
        "sha": row["cache_sha"], "n": row["cache_count"],
    }
    for row in rows
])
```

The scored-edge access trace for one arm is the natural-order list of exact
tuples `(objective_variant, boundary_anchor_id, source_time_hex,
left_period_hex, right_period_hex, left_bpm_hex, right_bpm_hex)` derived from
the public scored-edge stream. Each aggregate scored row is exactly
`{"case": case_id, "arm": arm_value, "count": len(accesses),
"sha": stable_sha(accesses), "unique_count": len(set(accesses)),
"unique_sha": stable_sha(sorted(set(accesses)))}`; the access aggregate is
`stable_sha(scored_rows)`. Lists and tuples both serialize as JSON arrays; no
other enum, float, or path conversion is permitted. The formal post-repair
verifier must reproduce these oracles from public ledgers, not by monkeypatching.

## Candidate Variants

### MR-A: weaken the dense assertion

Delete the all-anchor assertion and retain only the final-grid checks. Reject:
this hides whether the `0.045` floor was ever applied.

### MR-B: expose only the component cache

Expose the already deterministic cache values. Reject as incomplete: a cache
entry proves a component was constructed, but cannot attribute repeated
scored successors, block ownership, or predecessor/successor paths.

### MR-C: Exp006-only complete measurement ledger

Selected. Expose both the sorted unique component cache and an append-only
actual-scored-edge stream, plus per-block resource and exported-class records.
All writes are conditional on the existing `record_objective_ledgers=True`;
the default Exp005 call executes the same branches and returns exactly the
same payload as before.

## Frozen Measurement Contract

An **actual scored edge** is exactly one successor in
`_jump_successors_for_closure` for which `transition_components(...)` has
returned and `components.normalized_increment` is added to the successor's
transition objective. It is recorded immediately after the successor replay
key and ledger entry are constructed and before the successor is appended.
It is not synonymous with a boundary candidate, cache miss, retained path, or
selected path.

Each scored-edge record contains:

- monotonic zero-based `edge_order` in natural generation order;
- current block index and stage (`core` or `lookahead`);
- predecessor and successor replay keys;
- boundary anchor, source time, snapped beat, and snapped time;
- exact `float.hex` left/right periods and BPMs;
- objective variant and the component-cache identity;
- normalized increment and later membership flags for retained terminal,
  provisional, and selected traceback paths.

The component-cache ledger is sorted by its existing exact cache key:
objective variant, boundary anchor, source time hex, left/right period hex,
and left/right BPM hex. It contains every component field and has exact length
`transition_cache_size`.

Each block resource record has the following exact fields and capture points:

- `block_index`, `core_start_ms`, `core_end_ms`, and `lookahead_end_ms` from the
  already frozen block schedule;
- `incoming_path_count = len(paths)` immediately on block entry, before the
  core `_advance_paths_interval` call;
- `raw_committed_path_count = len(committed_candidates)` immediately after
  that core call and before `_dominant_paths_by_future`;
- `dominant_committed_path_count = len(committed_candidates)` immediately
  after `_dominant_paths_by_future`;
- `lookahead_call_count`, one for each dominant committed path, and
  `lookahead_successor_count`, the sum of `len(lookahead_paths)` immediately
  returned by those calls before their per-call `min` selection;
- `pre_export_state_count = len(frontier_states)` immediately before
  `export_frontier`, and `exported_state_count = len(exported.states)`
  immediately after it;
- `block_score_miss_count_before = 0` immediately after the existing
  `context.start_block()`, and `block_score_miss_count_after` immediately
  after all core/lookahead scoring and before block diagnostics are appended;
- row score-miss, transition-component-cache, and scored-edge occurrence
  counts captured at the same block-entry and block-exit points using explicit
  `*_count_before` and `*_count_after` fields;
- the existing `dominance_pruned_state_count` and `width_pruned_state_count`
  from the returned `FrontierExport`, plus every frozen config cap copied by
  value.

Each class-coverage record is produced inside `export_frontier` without
changing its order keys and contains sorted exact unique
`(alias_family, global_downbeat_phase)` keys and raw state counts at four
precise points: `input` before future-equivalence dominance;
`post_future_equivalence` from `best_by_equivalence.values()`;
`reserved` after `best_by_class` representatives are ordered and truncated to
`max_states`; and `final` after the existing width-fill loop and final sort.
It also contains the four corresponding state counts, where unique-key count
and raw-state count remain separate. The main loop binds this record to its
block index. This record and every other v2 measurement field are produced
only when both `record_objective_ledgers is True` and `objective_variant is
EXP006_PAIR_CONDITIONED_CHANGE_FLOOR_1_4`. The explicit
`EXP005_CONSTANT_CHANGE` comparator also records legacy objective ledgers, so
the ledger flag alone is insufficient. Explicit Exp005 must retain the v1
legacy field set and its behavior-projection aggregate must remain exactly
`3911d91c...`. Measurement must not change export selection.

`BOUNDARY_PAIR_TRANSITION_CONTRACT_VERSION` advances to v2 and only the
Exp006 objective-diagnostics deterministic fingerprint intentionally changes.
`LOCAL_FRONTIER_CONTRACT_VERSION`, `LocalFrontierDiagnostics`, default Exp005
diagnostics, grids, replay fingerprints, candidate fingerprints, and all
search behavior remain unchanged.

## Dataset Slice

The identical frozen 11 source-owned fixtures and four schedules. No file in
`artifacts/`, BeatThis cache, audio, `.osu`, manifest, schedule16, repair80,
holdout, broad500, full5050, API snapshot, or network source may be accessed.

## Primary Metric

`44/44` exact grids with the reviewed verifier, where dense floor coverage is
evaluated over the public actual-scored-edge stream. Every scored Exp006 edge
must satisfy `0.18 * change_cost >= 0.045`; every scored edge with
`pair_cost == 0` must satisfy binary64 `change_cost == 0.25` and sparsity
component `0.045`.

The frozen dense occurrence, unique-cache, anchor, and exact-zero-floor sets
above must match. A candidate absent from the scored-edge stream is reported
as not phase-feasible; it is not silently counted as a transition.

## Secondary Metrics and Guards

- exact pre/post equality of all 44 default Exp005 legacy payloads;
- exact pre/post equality of the 44 explicit Exp005-ledger behavior
  projections under aggregate `3911d91c...`;
- exact equality of the four original-kill full-payload golden hashes;
- exact pre/post equality of all 44 Exp006 grids, base diagnostics, replay and
  grid fingerprints, candidate fingerprints, terminal ordering, and legacy
  objective fields, certified by the formal behavior-projection aggregate
  `f72574d4...`; the complete v1 aggregate is not a parity oracle because the
  v2 contract and objective fingerprint intentionally change;
- exact reproduction of both aggregate trace hashes from public ledgers;
- scored-edge order, cache order, block resources, class coverage, and new
  objective fingerprint are deterministic across two runs;
- every ledger is finite JSON and internally reconstructable;
- no source change outside `local_frontier.py` and the two Exp006 test files;
- complete focused suite under two minutes and every arm under ten seconds.

## Local Verification

1. First run the current reviewed matrix and reproduce exactly eight dense
   observability failures with all 44 expected grids correct.
2. Add failing tests for the frozen v2 ledger schema and the four default
   Exp005 golden payloads.
3. Add only measurement writes, then require the public ledgers to reproduce
   pre-repair hashes and dense traces.
4. Run every arm twice and compare all mathematical and diagnostic payloads.
5. Run the full Timing-v3 guard.

## Verify Commands

```sh
.venv/bin/python -m pytest -q \
  tests/timing/test_timing_v3_local_frontier.py \
  tests/timing/test_timing_v3_boundary_pair_transition.py \
  tests/timing/test_timing_v3_boundary_pair_transition_matrix.py
.venv/bin/python -m pytest -q tests/timing/test_timing_v3_*.py
.venv/bin/python -m py_compile \
  src/pulsefield_model/timing/v3/local_frontier.py \
  tests/timing/test_timing_v3_boundary_pair_transition.py \
  tests/timing/test_timing_v3_boundary_pair_transition_matrix.py
git diff --check
```

## Positive, Negative, and Ambiguous Signals

- Positive: all measurement oracles and `44/44` pass, with no legacy behavior
  change and all required public ledgers complete.
- Negative: complete observability requires changing candidates, phase
  feasibility, scoring, search, fixture times, caps, or expected grids. Kill
  this repair and leave Exp006 ambiguous.
- Ambiguous: a pre-repair oracle cannot be reproduced from the frozen source
  and fixture bytes, or diagnostics perturb ordering/resource behavior. Stop;
  do not reinterpret the parent experiment.

## Kill Criteria

Stop immediately on any grid, candidate fingerprint, default payload, replay
fingerprint, terminal ordering, occurrence count, cache count, resource cap,
or runtime regression; on any non-synthetic input access; or if a proposed
diagnostic value is consumed by inference.

## Files Likely to Change

- `src/pulsefield_model/timing/v3/local_frontier.py`
- `tests/timing/test_timing_v3_boundary_pair_transition.py`
- `tests/timing/test_timing_v3_boundary_pair_transition_matrix.py`
- this card and a result note after execution

Production fitters, providers, runners, evaluators, split selectors, packaged
configuration, and every real-data surface remain out of scope.

## Result Log Template

- Parent card/source SHA:
- Repair card/source/test SHA:
- Pre-repair eight failures reproduced:
- Default Exp005 golden parity:
- 44-arm grid/base-diagnostic parity:
- Scored-edge and cache aggregate parity:
- Dense occurrence/cache/anchor/floor results:
- Per-block resource/class ledger validation:
- Deterministic replay result:
- Runtime:
- Full guard result:
- Interpretation:
- Recommendation: `KILL | MUTATE | TEST`

## Pre-Execution Gate

- Card complete: yes.
- Code execution allowed: yes, only for the measurement fields and synthetic
  tests explicitly named here.
- Research behavior change allowed: no.
- Real-data execution allowed: no.
- Remaining ambiguity before execution: none; any mismatch returns to planner
  mode rather than weakening an oracle.

## Next-Loop Action

If positive, write the Exp006 synthetic result and return `TEST` to a separate
real-cache schedule16/repair80 card. If negative or ambiguous, keep Exp006
unresolved and `MUTATE` only the measurement contract in a new card.
