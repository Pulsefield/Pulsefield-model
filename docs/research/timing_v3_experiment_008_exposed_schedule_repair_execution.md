# Experiment Card: Timing v3 Experiment 008 — Exposed Schedule/Repair Execution

## Mode

- Mode: planner
- Route: TEST
- Card state: draft; drafting is owner-authorized, but these bytes are not yet
  accepted for code or real-data execution.
- Source idea: transport the unchanged Exp006 E6-D candidate and frozen Exp007
  protocol through the already-exposed schedule/repair gates with one
  source-owned, algorithm-neutral real-cache adapter.
- Acceptance source, if any:
  `/Users/l/.codex/attachments/97bd173a-3590-4524-8b1f-f7a90c5e0223/goal-objective.md`,
  `docs/research/timing_v3_experiment_007_real_cache_schedule_repair.md`, and
  `docs/research/timing_v3_experiment_007_result.md`.
- Goal-objective file SHA-256:
  `38c8afb2216d74103db438025dc9582db91f9648f117041ae63bfd29801c46eb`.
- Owner authorization recorded: exact 2026-08-13 UTF-8 message
  `授权你之后的一切起草行为, 继续`, SHA-256
  `333683287e72792df37a4f05979f744c4fa95fc033c2b9137e97549ef3a98652`,
  authorizes this and future drafting. It does not accept this card, authorize
  implementation, or authorize opening a real identity, cache, audio, `.osu`,
  metric, or result. The final archive must preserve this authority record
  together with the Exp007 card/result identities below so it is not dependent
  on the local attachment path alone.
- Source snapshot / evidence grade: Exp007 synthetic TEST pass; behavior
  source-closure fingerprint
  `c22730ed11b7b4a872a434bebea2bf57279ad328caa7ff0c82dfc96e12ac573b`;
  no Exp008 real-data evidence.

## Hypothesis

A minimal source-owned execution adapter can bind the exact current
shift-zero BeatThis cache, current-v2 comparator, Exp006 E6-D candidate, frozen
Exp007 runner/protocol, immutable artifacts, and exposure ledger without
changing candidate, objective, thresholds, reducers, fallback routing, or
truth policy. If that adapter passes no-data differential verification, then
the already-exposed staged sequence
`schedule16 -> source-only winner commit -> winner-only weak veto -> repair80`
can produce one unambiguous positive, negative, ambiguous, timeout, integrity,
or hard-failure result under the frozen gates.

This hypothesis does not assert that Exp006 will pass, that `.osu` is truth,
that the existing BeatThis caches share any particular chunk-seam geometry,
or that a passing repair80 result is production evidence.

## Root Objective

Answer whether the unchanged constant/jump Timing v3 candidate is safe and
promising enough on already-exposed diagnostic data to justify a later,
separately accepted no-new-data review and fresh holdout card, while preserving
one global absolute beat axis, exact phase continuity, explicit current-v2
fallback, and zero manufactured best-so-far outputs after failure.

## Goal Decomposition

- Subgoal 1: add and freeze only the missing real-cache execution adapter,
  proving on synthetic fixtures that it is a mechanical transport layer over
  the Exp007 APIs.
- Subgoal 2: after a second explicit owner acceptance of the final source/card
  hashes, execute the four schedule16 arms and commit a source-only winner
  without opening weak evidence.
- Subgoal 3: only after a positive source commit, apply the selected-winner
  weak veto; only after a weak pass, execute selected-arm repair80 and stop for
  a no-new-data result review.

## Candidate Variants

- Variant A — source-owned gated execution: implement one thin Exp008 adapter,
  freeze it with the current behavior closure and exact configs, then run the
  full ordered exposed sequence with a hard stop between every gate.
- Variant B — ad-hoc callback/script: pass a locally defined callback directly
  to `run_synthetic_exp007_*` and record only command text or a notebook hash.
- Variant C — schedule-only card: freeze and execute four schedule arms, then
  write another card before the already-frozen winner-only veto and repair80.
- Variant D — outcome-adaptive expansion: use weak evidence to choose/retry a
  runner-up, add BeatThis chunk-seam analyses, regenerate caches, or introduce
  raw-audio Family B when source or weak evidence is poor.

## Local Verification Matrix

- Variant A: potentially passes. It is replayable only if the adapter is a
  repo-owned top-level spawn-picklable callable, its exact source/test hashes
  are inserted into the final card revision, the detached tree manifest and
  composed execution identity are frozen after that card, and differential
  fixtures prove byte-identical candidate/current-v2/row construction against
  the frozen Exp007 builders.
- Variant B: reject. The real behavior would sit outside the frozen source
  closure and could not be reconstructed from immutable repository bytes.
- Variant C: reject for this loop. Exp007 already froze the three downstream
  transitions and their stop rules; splitting them adds an administrative
  card boundary without isolating an algorithmic change. The in-card stage
  markers already prevent later access after any non-pass result.
- Variant D: reject. It leaks outcome information into selection, changes the
  evidence family, or relies on cache-generator/chunk provenance that is not
  established. Each such change requires a new no-data card and fresh
  applicability analysis.

## Selected Variant

- Selected: Variant A.
- Rejected: Variants B, C, and D for the reasons above.
- Why this is the smallest useful test: the candidate, candidate set,
  schedules, comparator, metrics, thresholds, worker protocol, artifact
  schemas, selection order, weak veto, and repair80 reducer already exist and
  passed Exp007 synthetic verification. The only missing executable surface is
  a source-owned binding from authoritative exposed identities to those APIs.

## Selection Pressure

- Primary pressure: preserve exact behavior and make every real input/output
  identity reconstructible before drawing a data conclusion.
- Guard pressure: weak evidence must never enter schedule eligibility or
  ordering; fallback must remain a product degradation rather than candidate
  success; any hard/integrity failure invalidates the affected stage.
- Runtime pressure: four fixed-order schedule arms must finish in less than
  1,200 seconds total; selected repair80 must finish in less than 1,800 seconds;
  each audio/arm must finish in less than 180 seconds with four workers whose
  lifetime RSS is at most 4 GiB each.
- Kill pressure: kill before real data if the adapter needs a behavior change,
  an unbound script, network access, additional cache/audio features, changed
  schema/reducer/threshold, or an unarchived mutable source tree.

## Research Question

Can the unchanged Exp006 E6-D constant/jump candidate, transported only by a
hash-frozen source-owned adapter, complete the frozen Exp007 exposed sequence
with a positive source-only winner, a passing winner-only weak veto, and an
all-pass repair80 result, without any timeout, integrity failure, hard failure,
threshold relaxation, later-data rescue, or confusion of v2 fallback with v3
success?

## Closest Analogies / Novelty Layer

- Closest analogies: staged model-selection pipelines with a label-free tuning
  reducer, post-selection veto, immutable execution manifests, and sequential
  safety gates.
- Relevant taxonomy bucket: experiment execution/integrity and conservative
  structured-prediction evaluation, not a new timing algorithm.
- Novelty layer, if any: exact state-carrying constant/jump timing behavior is
  inherited from Exp006; Exp008 adds no representation novelty.
- Representation novelty vs engineering variation: no new representation;
  only a reproducible real-input adapter and staged execution plumbing.

## Minimal Change

The accepted implementation may add exactly one real execution module and its
targeted tests. Suggested paths are:

- `src/pulsefield_model/timing/evaluation/exp008_execution.py`;
- `tests/timing/test_timing_v3_exp008_execution.py`.

The adapter may only:

1. parse explicitly supplied authoritative repair80 identity/label artifacts;
2. replay the frozen schedule16 selector;
3. resolve and snapshot one shift-zero cache per identity;
4. call the existing Exp007 builders/validators/runner/artifact APIs in their
   frozen order;
5. call the unchanged current-v2 and bounded Exp006 APIs exactly once per
   required row/arm with the shared restricted prediction/candidate set;
6. publish only the frozen canonical artifacts, exposure delta, outcomes, and
   summaries; and
7. provide a CLI whose explicit paths and hashes are recorded in the run
   manifest.

The adapter also owns two missing integrity wrappers, in this same module:

1. an Exp008 non-empty exposure-ledger publisher, because the Exp007 publisher
   intentionally accepts only an empty synthetic delta; and
2. an Exp008 live execution session that requires an OS-released advisory lock
   plus the exact Exp007 audit RunLock binding for every publication.

These wrappers may call public Exp007 canonical validators/builders but may not
weaken or monkeypatch them. Exp007 artifacts retain their Exp007 schema and
experiment id. Exp008 wrapper artifacts use the exact new identities below and
do not claim to be Exp007 results.

It may not change any existing behavior source. If implementation requires an
edit to Exp006, Exp007 protocol/runner/metrics/artifacts/weak evidence,
current-v2, provider/cache semantics, schema, evaluator, thresholds, or
fallback routing, this card is killed or mutated before any real input opens.

The adapter must not be added to the Exp007 historical result. Exp007 remains
immutable evidence; the new adapter and new behavior closure belong to Exp008.

## Frozen Source and Config Identity

The current no-data baseline is:

| Identity | SHA-256 |
| --- | --- |
| Exp007 card | `fb8b13cef083006f04165ac0f3d691b3e3965f6342b4ed264e0c283a1dec59d7` |
| Exp007 result | `9698849d33c4187678974b46b4b16931dc4d9359bd0de28d5bd61d442da035e4` |
| Exp007 behavior source closure | `c22730ed11b7b4a872a434bebea2bf57279ad328caa7ff0c82dfc96e12ac573b` |
| Recursive schema registry | `aa82b4fc44438413997be10b95ad85becfbc48a0a5cbd38d6e543aecdd41b80d` |
| RunConfig schema descriptor | `465ce0e4d54f6c37ea38fe438f3dbd7b981223ca77f3c411451a604af8830896` |
| RowResult schema descriptor | `e50f97de5b14b43d1f7899f4e847ae4363ab59f55a7785c26a6f33040ec1bf78` |
| SourceArmSummary schema descriptor | `716fe96f7a7621231b3c691d1afe43035970cfc32bf2b1703f5ef811151374e3` |
| ConfigSelection schema descriptor | `7aa6774645e7bc943511001d4bb028b1640245f17c11eb15e33c32ff659dd535` |
| FourArmStageSummary schema descriptor | `2151ccb877fca1228bd6328daa98ff08503960651132910a1587521ca5b22882` |
| Repair80Summary schema descriptor | `83368350d6bedcbbd17c978946d5382f7952e133581c1ba6659af0c6bc93ffe1` |

Selected source snapshots that must remain byte-identical unless this card is
mutated and re-reviewed:

| Source | SHA-256 |
| --- | --- |
| `local_frontier.py` | `3f8f677ad82b16dc658b730132b350ea9ee319790394b1e3c66705328b2f0999` |
| `exp007_protocol.py` | `06e191ea62d7faa50fc2d31264b9e1d8c6ac3738ff361923675c4fbcec4236cc` |
| `exp007_selector.py` | `23d2f2f9f4119e275f9a0510617ee59c936a4728eb4d589b2a37362f14dd37b0` |
| `exp007_runner.py` | `6c65d3bf2a6e44f739e3d3b399b4887e6315629f1f33e577e99cc67be1481553` |
| `exp007_metrics.py` | `b7cb51ba2d91b3ae0799a2a8686ff3cb8feac82d4ecf2ec0be348c8852e640a3` |
| `exp007_weak_evidence.py` | `2bb43c6ad4a4cc7991da66966071c2ca7fd23a978f2f074525e97759115b511a` |
| `exp007_artifacts.py` | `18443a886580d0797bb7d028895c871dc0f9add41d7c4b1fc13eba0b7177aa9c` |
| `providers/__init__.py` | `e08e3598b0e7990ca09b8283ad2896d04e016f8891494e0edbb00419a890f49d` |
| `providers/beatthis.py` | `76f99667553cd22d9c9b34915c284a33c51cbec71c22fcc3c8129e5ff4e4ec6b` |
| `providers/beatthis_cache.py` | `7982bda6d973ded2433c1b15db46f3500f39d1e4a606f46a467d7998ec8b9891` |
| `providers/oracle.py` | `72d32cee653c258fbfa36ca85bfe61252e599aff2787db4123a36aecc9d1c6a8` |

This selected table is not the complete behavior surface. The authoritative
Exp007 closure fingerprint above binds 44 source files, including current-v2
grid-fitting and recursively imported package modules. The final Exp008
archive must contain the complete canonical
`SourceClosure.behavior.relative_source_files`, import-edge, and module-
identity manifests; the final card must bind that manifest by literal SHA-256.
No omission from this human-readable table removes a file from the closure.

The following values deliberately do not yet exist and therefore keep this
draft non-executable:

- final `exp008_execution.py` SHA-256;
- final Exp008 targeted-test SHA-256;
- authoritative repair80 identity artifact, label artifact, and ordered-row
  SHA-256 values;
- replayed schedule16 selector-manifest SHA-256;
- canonical cache, current-v2 GridFitter, and weak-comparator config SHA-256
  values;
- the four canonical schedule RunConfig fingerprints;
- the selected repair80 RunConfig fingerprint, which is outcome-dependent and
  can be created only after a positive source commit and weak pass; and
- post-card immutable archive/tree-manifest identity for the currently
  untracked Timing v3 source, tests, cards, and results; and
- post-card Exp008 composed execution-identity fingerprint, defined below.

The exact Exp007 `SourceClosure` schema and entry-module list remain immutable
and do not include the future Exp008 adapter. Exp008 therefore must not claim
that the adapter is inside that closure. Its composed execution identity is
instead:

```text
Exp008ExecutionIdentity := exact{
  schema="pulsefield_model.timing_v3_exp008_execution_identity_v1",
  exp007_source_closure_fingerprint_sha256,
  exp008_adapter_sha256,
  exp008_test_sha256,
  immutable_tree_manifest_sha256,
  exp008_card_sha256,
  python_behavior_version,
  numpy_behavior_version,
  canonical_json_contract_sha256,
  execution_identity_fingerprint_sha256
}

execution_identity_fingerprint_sha256 = sha256(canonical JSON of every prior
field in the exact order-independent object, omitting only the fingerprint)
```

This wrapper identity requires targeted mutation tests for every constituent
and is published before real preflight. It composes with, and never replaces
or rewrites, the frozen Exp007 behavior closure. Its hash is not written back
into this card: it contains `exp008_card_sha256`, so doing so would create an
uncomputable self-reference.

The Exp008 exposure update is:

```text
Exp008ExposureObservation := exact{
  cache_audio_key,audio_group_key,exposure_stage,exposure_reason,
  observed_at_or_run_id,observed_payload_kind,source_manifest_sha256,
  observation_fingerprint_sha256
}
  exposure_stage in {schedule16,repair80,accidental_batch}
  observed_payload_kind in
  {identity,identity_label,cache,prediction,grid,metric,diagnostic,runtime,
   failure,trace,osu,rendering,batch_aggregate}
  observation_fingerprint_sha256 = sha256(canonical JSON of every prior field)

Exp008ExposureLedgerUpdate := exact{
  schema="pulsefield_model.timing_v3_exp008_exposure_ledger_update_v1",
  experiment_id="timing_v3_experiment_008",
  generated_at_utc,
  owner_run_id,
  execution_identity_fingerprint_sha256,
  prior_exposure_manifest_sha256,
  prior_union_cache_audio_keys_sha256,
  previous_update_fingerprint_sha256,
  update_index,
  update_reason,
  observation_count,
  observations_sha256,
  observations,
  delta_unique_key_count,
  delta_cache_audio_keys_sha256,
  union_unique_key_count,
  union_cache_audio_keys_sha256,
  update_fingerprint_sha256
}
```

`observations` are sorted uniquely by
`(cache_audio_key,exposure_stage,observed_payload_kind,
source_manifest_sha256,exposure_reason,observed_at_or_run_id)` and may repeat a
cache key across distinct observation events. Identity-artifact observation
uses `observed_payload_kind="identity"`; label-artifact observation uses
`"identity_label"`; each observation binds its own exact source-manifest SHA.
Preflight observations use `exposure_stage="repair80"` because opening the
authoritative repair80 wrappers observes all 80 existing exposed identities
before schedule derivation. Every observation SHA, list SHA, count, delta
unique-key hash, and union unique-key hash is recomputed from canonical JSON.

Updates form an append-only hash chain: index zero has null previous-update
SHA and each later update binds the complete prior update. Each update is
published to a unique canonical path containing zero-padded `update_index` and
`update_fingerprint_sha256`; no prior update is replaced. A later schedule,
repair, label, or accidental observation publishes a new update containing
only observation tuples not already in the chain, even when their keys were
previously exposed, then recomputes the complete key union. Thus mixed-stage
exposure and multiple sources for one key are represented by separate ordered
updates, never forced through one Exp007 stage lock or one divergent
destination.

Exact update preimages are:

```text
observation_count = len(observations)
observations_sha256 = sha256(canonical JSON of the complete sorted observations)
delta_unique_key_count = len(sorted unique keys in observations)
delta_cache_audio_keys_sha256 = sha256(canonical JSON of those sorted keys)
union_unique_key_count = len(sorted union of prior-union and delta keys)
union_cache_audio_keys_sha256 = sha256(canonical JSON of that sorted union)
update_fingerprint_sha256 = sha256(
  canonical JSON of the complete update object omitting only
  update_fingerprint_sha256)
```

`update_index=0` requires `previous_update_fingerprint_sha256=null`; every
later index is exactly the prior index plus one and requires the complete prior
update fingerprint. A missing, forked, repeated, skipped, or reordered link is
fatal.

Publication requires a live Exp008 session, one canonical buffer, same-dir
exclusive temp, file fsync, atomic replace, directory fsync, divergent-existing
refusal, then read-back validation. The wrapper must never call the Exp007
empty-only publication function for a real delta.

The live execution session is exact:

- before any authoritative input byte opens, acquire a nonblocking exclusive
  POSIX advisory lock on a canonical output-root session-lock file and keep its
  file descriptor open for the entire process lifetime. The root session binds
  owner id, resume-token SHA, composed execution identity, final accepted card,
  immutable tree manifest, owner-acceptance record, output root, and canonical
  invocation-manifest SHA;
- this root session alone may publish preflight exposure before data-derived
  selector/config hashes exist. After those hashes are frozen and before each
  schedule/repair stage publishes anything, create or validate that stage/
  arm's frozen Exp007 RunLock whose selector/source/config/output-root/owner
  fields exactly match the invocation;
- every Exp008 wrapper publication validates the advisory-lock descriptor and
  root session snapshot; every call into an Exp007 stage publication API also
  validates the exact live stage/arm RunLock snapshot is unchanged;
- normal stage completion releases its Exp007 audit lock with the explicit
  owner token and fsyncs the directory. Overall completion then releases the
  advisory lock;
- after process loss, the kernel releases the advisory lock. A restart may
  resume in place only when it reacquires that lock, presents the same explicit
  `owner_run_id` plus a 256-bit resume token whose SHA (not plaintext) was
  frozen in the invocation manifest, validates the unchanged Exp007 RunLock
  and every source/config/output/prefix binding, and publishes a canonical
  takeover record before constructing a live session handle;
- a live holder, missing/wrong token, changed lock, ambiguous process state,
  stale/gapped/swapped prefix, or takeover-publication failure forbids in-place
  continuation. The only allowed alternative is a complete rerun from row zero
  in a new fingerprinted output root; no immutable artifact is overwritten.

The resume token is local integrity authority, never a data or result key. It
must not appear in a result, log, command line, or artifact; only its SHA-256 is
stored. Tests use synthetic tokens and verify redaction.

Workers never receive an output-root path, root-session handle, run-lock
handle, resume token, or publication callable. They return one canonical
envelope through the frozen runner only. All filesystem publication is
parent-owned after the parent revalidates the live root session and applicable
stage RunLock. Parent loss therefore prevents any worker from publishing after
the advisory lock is kernel-released.

The detached immutable tree manifest is exact:

```text
Exp008TreeFile := exact{
  archive_path,source_kind,source_path,size_bytes,sha256,executable
}
  source_kind in {repo,authority,generated_audit}
  archive_path/source_path are normalized POSIX relative paths;
  size_bytes is nonnegative; executable is bool

Exp008TreeManifest := exact{
  schema="pulsefield_model.timing_v3_exp008_tree_manifest_v1",
  base_git_commit,include_policy_sha256,file_count,files_sha256,files,
  manifest_fingerprint_sha256
}
```

Exact tree preimages are:

```text
file_count = len(files)
files_sha256 = sha256(canonical JSON of the complete sorted Exp008TreeFile list)
manifest_fingerprint_sha256 = sha256(
  canonical JSON of the complete manifest object omitting only
  manifest_fingerprint_sha256)
```

`base_git_commit` is the exact lower-case 40-hex commit or null. Every count,
SHA, include rule, and file record is validated from bytes rather than trusted
from the producer.

Files are regular-file byte snapshots sorted uniquely by `archive_path`.
Directories, symlinks, hard-link aliases, sockets, devices, traversal,
absolute paths, Unicode-normalization collisions, case-fold collisions, and a
file changing between stat/read/stat are fatal. `executable` is derived only
from whether any executable bit is set; owner/group numbers, timestamps, and
other mode bits are excluded. File blobs are stored read-only at
`blobs/sha256/<sha256>` and verified before and after an isolated restore.
Canonical manifest JSON is stored beside them. A Git commit may transport
these exact bytes but never substitutes for the manifest identity.

The include policy is the exact union of:

1. every path in the validated Exp007
   `SourceClosure.behavior.relative_source_files` and
   `required_non_import_files` lists;
2. `AGENTS.md`, `README.md`, `pyproject.toml`, and `uv.lock`;
3. every repo file matching `docs/research/timing_v3_*.md` in the independently
   reviewed final tree, including all preserved Experiment 001–008 evidence,
   the problem log, task definition, and completion audit;
4. every repo file matching `tests/timing/test_timing_v3_*.py` in that same
   tree, plus the final adapter and targeted Exp008 test even if a naming rule
   changes;
5. byte-for-byte authority copies at
   `authority/goal-objective.md` and
   `authority/owner-drafting-authorization.utf8`, whose hashes must equal the
   values frozen at the top of this card; and
6. a canonical NUL-delimited Git porcelain-v1 status snapshot for audit only.

`include_policy_sha256` hashes canonical JSON containing these six literal
rules and the final enumerated archive paths. No other dirty/untracked file is
implicitly accepted by omission or inclusion. Before owner acceptance, an
isolated temporary restore must recompute every blob/file/manifest hash, the
Exp007 closure, adapter/test/card hashes, and run all no-data guards without
reading the original worktree or any real artifact.

Before any real artifact opens, a final revision of this same card must replace
the adapter and targeted-test identities with literal hashes and retain the
exact semantic config fields below. Then, without modifying the card again:

1. compute the external full-file card SHA-256;
2. build an immutable content-addressed tree manifest/archive that includes the
   final card and all required source/test/history bytes, and compute its SHA;
3. build and validate `Exp008ExecutionIdentity` from those external hashes; and
4. receive independent review plus explicit owner acceptance naming the exact
   card, tree-manifest, and execution-identity SHA-256 values.

This order has no hash cycle. The data-dependent manifest and RunConfig hashes
are then generated as the first authorized preflight outputs and must be
atomically frozen before any cache array opens. A descriptor hash is never
substituted for an instance-config hash.

The second owner decision is preserved as a detached artifact created only
after the owner supplies a new verbatim acceptance message:

```text
Exp008OwnerAcceptanceRecord := exact{
  schema="pulsefield_model.timing_v3_exp008_owner_acceptance_v1",
  accepted_scope="exposed_schedule16_source_weak_repair80_only",
  final_card_sha256,tree_manifest_sha256,
  execution_identity_fingerprint_sha256,
  owner_message_utf8,owner_message_sha256,recorded_at_utc,
  acceptance_record_fingerprint_sha256
}
```

The message must explicitly accept all three named hashes and the exact scope;
silence, a generic “continue,” this drafting authorization, or acceptance of
only one constituent is invalid. The record fingerprint hashes canonical JSON
of every prior field. The root live session and invocation manifest bind this
record fingerprint, while the result preserves the verbatim bytes and hash.

Frozen semantic config:

- stages: `schedule16`, then `repair80`;
- selector seed: `timing-v3-exp005-schedule16-v1`;
- schedule order: `S30,S60,S90,S64`;
- source tie rank: `S64=0,S90=1,S60=2,S30=3`;
- candidate method: `exp006_pair_conditioned_change_floor_1_4`;
- current-v2 baseline: `current_v2_grid_fitter`;
- selected product: `exp006_or_current_v2_fallback`;
- weak method: `weak_osu_redline_object_grid_v1`;
- local-frontier caps: exported frontier 16, beam 64, boundary candidates 32,
  tempo candidates 64, blocks 192, sections 20, section-score misses 30,000
  per block and 500,000 per audio;
- runner: four fresh `spawn` workers per arm/stage, ordered
  `imap(chunksize=1)`, persistent workers, `maxtasksperchild=None`;
- parent poll at most 0.25 seconds; finish/result guard 5 seconds; terminate
  grace 5 seconds; kill grace 5 seconds;
- row artifact strictly below 1 MiB, candidate payload strictly below 64 MiB,
  and reference bundle strictly below 66 MiB (`69,206,016` bytes); equality or
  excess maps to `artifact_resource_cap`. Every other frozen manifest cap is
  also exclusive.

## Files Likely to Change

- `docs/research/timing_v3_experiment_008_exposed_schedule_repair_execution.md`
  while it remains a draft only;
- `src/pulsefield_model/timing/evaluation/exp008_execution.py` after a separate
  explicit acceptance authorizes no-data implementation;
- `tests/timing/test_timing_v3_exp008_execution.py` under the same boundary;
- after an actual completed run, one immutable Exp008 result and append-only
  problem/decision-log entry.

No existing Experiment 001–007 card/result may be edited to absorb Exp008.

## Read-Only Context Files

- `/Users/l/.codex/attachments/97bd173a-3590-4524-8b1f-f7a90c5e0223/goal-objective.md`;
- `AGENTS.md`, `README.md`;
- `docs/research/timing_v3_task_definition.md`;
- `docs/research/timing_v3_problem_log.md`;
- Experiments 001, 004, 005, 006, and 007 cards/results;
- the exact Exp007 modules/tests and closure-bound current-v2/provider/cache
  files listed above.

`artifacts/` must not be searched, listed, or scanned broadly. After final
card acceptance, the adapter may open only explicit authoritative paths named
by the accepted invocation and their exact schema-declared dependencies.

## Dataset Slice

- Source identity unit: unique `cache_audio_key`, with `audio_group_key` used
  only where the frozen schedule source reducer requires one-to-one grouping.
- Source pool: exactly the existing exposed repair80 identity artifact and its
  exact 80-row label artifact. They are inputs, not newly selected data.
- Schedule16: derive exactly 16 identities from the complete validated 80-row
  identity/label join using seed `timing-v3-exp005-schedule16-v1`, exclusive
  priority `long,dense,jump,stable`, quota four each, the frozen later-class
  deficit donor order, then `deficit_remaining`. Rank by
  `sha256(seed + "\0" + cache_audio_key)`, then key.
- Repair80: exactly all 80 validated identity rows in authoritative order;
  `cache_audio_key` must be unique. Duplicate `audio_group_key` is allowed and
  must not reduce the 80-row denominator.
- Schedule pre-cache integrity: after exact selector replay, the 16 selected
  rows must also have unique `audio_group_key`, as required by the frozen
  source reducer. Do not de-duplicate, substitute, rerank, or select another
  row. A duplicate publishes the already-required exposure update and a hard
  preflight integrity outcome, then stops before any cache opens.
- Inference input: one shift-zero BeatThis `final0` cache per identity. No raw
  audio, extra shift, `.osu`, metadata, catalog/API, or network information may
  enter candidate inference or schedule selection.
- Weak veto/evaluation: `.osu` redline/object evidence only for the already
  committed selected winner and later selected repair rows; never a candidate,
  selector, schedule order, threshold, or fallback input.
- Explicitly forbidden: holdout100, broad500, full5050, fresh identities,
  rendering/listening, raw-audio Family B, cache regeneration/migration, and
  BeatThis chunk-seam analysis.

Every observed schedule16 key and every opened repair80 key must enter the
Exp008 exposure delta. Identity, cache, prediction, grid, metric, diagnostic,
runtime, failure, trace, `.osu`, rendering, or batch-aggregate observation all
count as exposure. An accidental batch exposes every key in that batch;
uncertain means exposed. The ledger update must be lock-protected,
file-fsynced, atomically replaced, directory-fsynced, and unioned with all prior
exposure before any future held-out selection.

## Baseline / Comparator

- Pure candidate: bounded Exp006 E6-D on the exact shared restricted
  prediction and candidate set.
- Primary safety baseline: unchanged current-v2 `GridFitter` on the same
  restricted shift-zero cache activation.
- Selected product: accepted pure candidate, otherwise an explicitly tagged
  current-v2 fallback only when current v2 is accepted; fallback never enters
  pure-candidate success denominators.
- Hard failure: input/schema/integrity/execution/publication failure, with no
  product or best-so-far grid.
- Weak comparator: selected-winner `.osu` redline/object-derived evidence,
  treated as correlated weak annotations and a veto only.

## Primary Metric

The primary outcome is the ordered stage decision, not a single headline
number:

1. schedule16 must produce four successful immutable arm outcomes and a
   positive source-only winner commit;
2. the selected winner must pass every weak-veto gate; and
3. selected-arm repair80 must pass every denominator, quality, runtime,
   fallback, overlap, continuity, replay, schema, source, cache, and hard guard.

The experiment is positive only if all three are pass. A negative or ambiguous
decision at any stage is the experiment result and blocks the next stage. A
timeout, integrity failure, hard failure, or inability to publish its durable
outcome is a hard stop and never becomes an ordinary metric row.

## Secondary Metric

Report all frozen schedule source denominators and exact audio-set hashes;
pure and selected-product phase/drift/coverage separately; fallback and
no-origin/path reasons; overlap and section excess; boundary diagnostics;
comparator availability/conflict; p50/p90/max row runtime; four worker lifetime
RSS values; artifact sizes; zero-seam and serialization guards; and exact
source/config/cache/exposure identities.

Comparator-unavailable rows remain in cache validity, execution, fallback,
runtime, and product coverage denominators but not oracle phase/BPM aggregates.

## Verify Command / Evaluation Procedure

### Phase 0 — no-data adapter and final freeze

This draft authorizes no command. After an explicit owner acceptance limited
to no-data implementation:

1. add only the adapter and targeted synthetic tests described above;
2. use injected fake paths/loaders and temporary synthetic caches; tests must
   fail if a path resolves into `artifacts/` or a real asset root;
3. prove one call each to restricted prediction, candidate extraction,
   current-v2, and bounded Exp006 per `(audio,arm)`; prove the identical
   candidate payload/current-v2 result across all four schedule arms;
4. prove explicit-path-only input, no network import/call, no broad directory
   traversal, strict source/cache rechecks, exact failure attribution, atomic
   publication, resume, and exposure behavior;
5. prove composed execution-identity mutation rejection, exclusive artifact
   caps at limit-minus-one/limit/limit-plus-one, and exposure publication after
   identity-byte observation even when later preflight validation fails;
6. prove a spawn-picklable top-level row callable; non-empty exposure union and
   divergent-destination rejection; live-lock enforcement on every wrapper;
   process-loss advisory-lock release; same-owner/token takeover; wrong-token,
   live-holder, changed-lock, and stale-prefix rejection; and token redaction;
7. rerun the frozen Exp005/006 guard, all Exp007 tests, targeted Exp008 tests,
   and full Timing-v3 suite;
8. run compilation, `git diff --check`, source-closure recomputation, and a
   no-real-access audit;
9. insert only the literal adapter and targeted-test hashes into this card,
   independently review those final bytes, and then make no further card edit;
10. compute the external final-card SHA-256;
11. immutably archive or commit the complete source/test/final-card/history tree
   with a content-addressed manifest, compute its SHA, then build and validate
   the composed execution identity. Hashes alone in a mutable untracked
   worktree are insufficient for real execution;
12. independently verify the acyclic preimages and obtain explicit owner
   acceptance naming the exact final-card, tree-manifest, and composed-
   execution-identity SHA-256 values. Neither post-card hash is written back
   into the card.

Expected no-data guards after implementation:

```sh
.venv/bin/python -m pytest -q \
  tests/timing/test_timing_v3_local_frontier.py \
  tests/timing/test_timing_v3_boundary_pair_transition.py \
  tests/timing/test_timing_v3_boundary_pair_transition_matrix.py --tb=short
.venv/bin/python -m pytest -q tests/timing/test_timing_v3_exp007_*.py --tb=short
.venv/bin/python -m pytest -q tests/timing/test_timing_v3_exp008_execution.py --tb=short
.venv/bin/python -m pytest -q tests/timing/test_timing_v3_*.py --tb=short
.venv/bin/python -m py_compile \
  src/pulsefield_model/timing/v3/local_frontier.py \
  src/pulsefield_model/timing/evaluation/exp007_protocol.py \
  src/pulsefield_model/timing/evaluation/exp007_selector.py \
  src/pulsefield_model/timing/evaluation/exp007_runner.py \
  src/pulsefield_model/timing/evaluation/exp007_metrics.py \
  src/pulsefield_model/timing/evaluation/exp007_weak_evidence.py \
  src/pulsefield_model/timing/evaluation/exp007_artifacts.py \
  src/pulsefield_model/timing/evaluation/exp008_execution.py
git diff --check
```

### Phase 1 — accepted-card preflight

Only after final-byte owner acceptance:

1. verify the immutable archive/tree manifest and every frozen source/test/card
   hash plus the detached owner-acceptance record before importing the adapter;
2. verify supported CPython/POSIX interval timers, macOS/Linux RSS semantics,
   four-worker spawn support, free-space limits, and output-root containment;
3. acquire the Exp008 root live execution session before opening any
   authoritative input; refuse a live/divergent session and apply the exact
   takeover protocol on process-loss resume;
4. open and validate only the explicit prior exposure manifest; its keys are
   already exposed and seed the union but are not new observations;
5. snapshot the explicit authoritative 80-row identity artifact once; as soon
   as keys are parseable, publish and read back one non-empty identity-
   observation update for conservatively all 80 keys before opening labels;
6. snapshot the explicit label artifact once, then immediately append and read
   back a distinct `identity_label` observation update for all parseable keys;
   only after both durable updates may exact schema/count/order/uniqueness and
   cross-source refs be called validated;
7. replay and atomically freeze schedule16 selector manifest before any cache
   opens. It may depend only on the complete identity/label join; metric,
   prediction, cache array, `.osu`, runtime, failure, result, grid, phase,
   drift, and boundary fields are forbidden;
   validate that the selected 16 have unique `audio_group_key`; otherwise stop
   as the frozen pre-cache integrity failure above without altering selection;
8. materialize and atomically freeze canonical cache/current-v2/weak configs,
   the four schedule RunConfigs and Exp007 SourceClosure; validate all
   fingerprints authoritatively under the already-live root session, then
   acquire the exact S30 Exp007 audit RunLock before the first stage artifact;
9. if any preflight identity, source, config, provenance, platform, archive,
   output, or exposure check fails, publish/quarantine the allowed failure
   record and stop before a cache array opens.

The weak config in this preflight is canonical static policy JSON only. Hashing
it must not resolve, list, stat, parse, or otherwise observe an `.osu` path,
comparator manifest, redline, hitobject, weak row, or weak aggregate. Those
inputs remain unopened until the committed positive source winner reaches
Phase 3.

Opening identity or label bytes is itself exposure. From the first such byte
snapshot onward, every parseable key must be published in the Exp008 exposure
delta before a preflight failure may be called durably quarantined. If exact
key recovery is uncertain or the source proves to be a larger accidental
batch, publish every recoverable key as `accidental_batch`, record the source
artifact SHA and uncertainty, conservatively exclude the entire implicated
source batch from future held-out selection, and stop. Failure to durably
publish that exposure state is itself fatal and leaves no claim of a safely
closed preflight.

The current caches may be used only as chunk-conditioned activation/ranking
scores. This card makes no BeatThis chunk-seam claim. If the execution would
need a generator/checkpoint/chunk/border/aggregation/shift assertion beyond
the exact cache identity/config bytes available, stop and write an auditable
regeneration/migration plan under a new card; do not infer provenance.

### Phase 2 — schedule16 and source-only commit

1. Execute `S30`, `S60`, `S90`, then `S64` as four separate, non-concurrent
   arms. Each arm uses exactly four fresh spawn workers and finishes cleanup,
   outcome publication, directory fsync, and process exit before the next.
2. Per `(audio,arm)`, enforce the frozen order: source/cache check; stable
   shift-zero snapshot; restricted prediction; little-endian float32 hashes;
   one candidate extraction; one current-v2 fit; one bounded Exp006 call with
   explicit identical `candidate_set=`; product/grid/projection/diagnostics;
   source/cache recheck; canonical envelope and atomic row publication.
3. Any row at or above 180 seconds, worker death/replacement, broken stream,
   RSS breach, byte-cap breach, schema/source/cache/config/replay mismatch,
   stage deadline, or publication failure hard-stops the schedule. Later arms
   receive immutable not-run outcomes.
4. After all four arms succeed, require exact cross-arm equality for identity
   order, cache identity/content/config/audio key, source closure, non-arm
   config, restricted input arrays, full candidate schema/fields/payload/hash,
   and current-v2 result. No normalization, majority, or partial selection.
5. Compute source selection without importing/opening weak evidence. Atomically
   publish candidate-global manifest, pending ConfigSelection, then the unique
   successful FourArmStageSummary. Only the directory-fsynced final summary
   commits the winner.

Source E0 eligibility requires for each arm: success; 16 finite runtimes and
RSS values; candidate tagged fallback at most 1/16; no-origin/no-path count
zero; p90 row runtime at most 60 seconds; every row below 180 seconds; max
worker RSS at most 4 GiB; accepted candidate seam exactly 0.0 ms; accepted
section count at most 20; and all source/cache/candidate/current-v2/replay/
schema guards. Fewer than two E0 arms is ambiguous.

The exact source common sets require overlap `n>=5` and section comparison
`n>=8`; otherwise ambiguous. E1 removes any E0 arm with p90 overlap above
90.0 ms. An empty E1 is negative. Otherwise choose the first arm by:

```text
(candidate_fallback_count,
 no_origin_or_path_count,
 p90_overlap_ms,
 section_inflation_violation_count,
 p90_section_excess,
 tie_rank)
```

Runtime/RSS are gates only and never enter this order tuple. A source
ambiguous/negative/hard outcome stops before weak access. There is no retry or
runner-up promotion.

### Phase 3 — winner-only weak veto

Only a validated successful FourArm marker with positive source selection may
open weak inputs. Read only the selected winner's 16 immutable rows and their
exact weak dependencies. Losing-arm grids remain unopened by the weak
evaluator.

Required minima:

| Denominator | Minimum |
| --- | ---: |
| current-v2 phase matched | 8 |
| pure Exp006 phase matched | 8 |
| selected-safety phase matched | 8 |
| alias-drift common | 8 |
| weak-change boundary audio | 5 |

Exact weak-veto gates:

| Metric | Pass | Ambiguous | Negative |
| --- | --- | --- | --- |
| pure mean phase ratio | `<=1.05` | `(1.05,1.10]` | `>1.10` or positive infinity |
| pure p90 phase ratio | `<=1.10` | `(1.10,1.15]` | `>1.15` or positive infinity |
| pure phase coverage | `>=95%` | `[90%,95%)` | `<90%` |
| max alias max-prefix drift mean/p90 ratio | `<=1.15` | `(1.15,1.30]` | `>1.30` or positive infinity |
| pure-minus-v2 mean boundary F1 delta | `>=-0.05` | `[-0.10,-0.05)` | `<-0.10` |

Undefined, insufficient, or comparator-conflicting evidence is ambiguous. At
exactly `-0.10`, boundary delta is ambiguous; at exactly `-0.05`, it passes.
Any negative band is negative; otherwise any ambiguous band is ambiguous; only
all-pass authorizes repair80. An unexpected exception, mutation, malformed
metric, schema failure, or publication failure is a fatal weak protocol error.
It retains the recorded source winner for audit but forbids repair and never
evaluates a runner-up.

### Phase 4 — selected-arm repair80

Only an atomically published weak success outcome with decision `pass` and
action `authorize_repair80` may construct the repair input binding and selected
arm RunConfig. Execute exactly the committed winner over all 80 authoritative
identity rows, with a new independent four-worker stage, no reselection, and
the same candidate/comparator/config/source closure.

Required minima:

| Denominator | Minimum |
| --- | ---: |
| pure Exp006 phase matched | 40 |
| selected-safety phase matched | 40 |
| stable pure paired | 5 |
| jump pure paired | 15 |
| long pure paired | 5 |
| overlap-available accepted audio | 20 |
| jump weak-change boundary common | 15 |

Exact repair80 gates:

| Metric | Pass | Ambiguous | Negative |
| --- | --- | --- | --- |
| pure mean phase ratio | `<=1.05` | `(1.05,1.10]` | `>1.10` |
| pure p90 phase ratio | `<=1.10` | `(1.10,1.15]` | `>1.15` |
| pure phase coverage | `>=95%` | `[90%,95%)` | `<90%` |
| max stable mean/p90 phase ratio | `<=1.10` | `(1.10,1.20]` | `>1.20` |
| jump mean phase ratio | `<=1.05` | `(1.05,1.15]` | `>1.15` |
| jump alias max-prefix drift mean ratio | `<=0.90` | `(0.90,1.15]` | `>1.15` |
| pure-minus-v2 jump boundary F1 delta | `>=-0.05` | `[-0.10,-0.05)` | `<-0.10` |
| max long alias max-prefix drift mean/p90 ratio | `<=1.15` | `(1.15,1.30]` | `>1.30` |
| candidate tagged fallback / 80 | `<=5%` | `(5%,10%]` | `>10%` |
| no-origin/no-path / 80 | `<=3%` | `(3%,5%]` | `>5%` |
| p90 row runtime / 80 | `<=30 s` | `(30,60] s` | `>60 s` |
| accepted+available overlap p90 | `<=45 ms` | `(45,90] ms` | `>90 ms` |

Any missing denominator is ambiguous. Structured positive infinity is
negative where applicable; undefined is ambiguous. Stable pure rows with
candidate section excess over current v2 greater than one are negative.

Hard guards are zero hard failures, exact accepted candidate seam 0.0 ms,
accepted section count at most 20, every row below 180 seconds, each worker RSS
at most 4 GiB, and strict deterministic replay/schema/source/cache/config/
exposure integrity. A hard, negative, or ambiguous repair result stops. A pass
authorizes only an immutable Exp008 result and a no-new-data acceptance review;
it does not authorize holdout selection/execution or production acceptance.

### Phase 5 — no-new-data result and replay

1. Atomically publish the final stage outcome, Exp008 exposure delta, command/
   environment manifest, and result references.
2. Close all data handles and run zero-compute resume/identity replay from
   immutable bytes. No cache array, `.osu`, metric recomputation, or worker
   pool is allowed during zero-compute replay.
3. Record positive, negative, ambiguous, timeout, integrity, or hard result
   without reinterpretation. Include every command/result, source/config/cache/
   card/exposure hash, denominator set hash, runtime/RSS value, fallback reason,
   failure attribution, and unopened later stage.
4. Stop. Do not select holdout100, broad500, full5050, change a production
   default, or write integration/rollback claims under Exp008.

## Guard Check

- Experiments 001–007 remain immutable evidence.
- The adapter has no candidate, metric, reducer, threshold, objective,
  selector, fallback, or truth-policy choice.
- Schedule source selection has no weak-evidence import or input dependency.
- The final successful FourArm marker, not a standalone ConfigSelection,
  commits the winner.
- Weak evidence is winner-only and veto-only; no runner-up branch exists.
- Repair80 is selected-winner-only and exactly 80 identity-level rows.
- Pure candidate and selected-product safety denominators stay separate.
- A current-v2 fallback is explicitly tagged and never counted as v3 success.
- Any hard failure yields no product, projection, weak pair, or best-so-far
  grid for the failed row/stage.
- Source/cache identity is checked before load, after candidate extraction,
  after current-v2, after Exp006, before row publication, and before summaries.
- Every immutable publication uses one canonical buffer, same-directory unique
  temp, file fsync, atomic replace, directory fsync, and divergent-destination
  refusal.
- Resume validates final outcomes/dependencies first and only then exact
  contiguous prefixes; stale, gapped, swapped, or orphaned state fails closed.
- No network, API, catalog BPM, raw audio, extra shift, cache regeneration,
  rendering, listening, holdout, broad500, full5050, ramp, or production path
  is authorized.
- Existing cache provenance cannot support BeatThis chunk-seam claims. This
  card uses no such variable or conclusion.
- The current v2 implementation remains untouched and is the full rollback.

## Qualitative Check

Inspect only canonical schemas, source/config/identity manifests, failure
records, exposure entries, immutable artifact references, and aggregate
diagnostic tables needed to validate the frozen protocol. Do not listen to or
render audio, inspect waveform/spectrogram content, browse arbitrary caches,
or use subjective quality to alter a prediction, gate, or interpretation.

## Positive Signal

- The adapter is proven algorithm-neutral and reproducible under the final
  frozen source/config/card hashes.
- All four schedule arms complete with exact cross-arm identity/candidate/v2
  equality and at least two E0 arms.
- Source selection is positive and commits exactly one winner mechanically.
- The winner-only weak veto has sufficient denominators and all-pass gates.
- Repair80 has all required denominators, all-pass quality/value/safety/runtime
  gates, zero hard failures, exact continuity, bounded sections/resources, and
  successful zero-compute replay.

This signal only supports drafting a later no-new-data acceptance-review card.

## Negative Signal

- The adapter cannot remain mechanical or source-owned.
- Source selection is negative, weak evidence has any negative gate, or
  repair80 has any negative gate.
- Passing would require changing a threshold, evaluator, denominator,
  candidate, objective, feature, schedule, tie-break, fallback route, or data
  set.
- Safety passes but the frozen value gate is ambiguous; “not worse” alone is
  not a positive result or promotion evidence. Preserve the table's ambiguous
  classification and stop for a narrower no-data card rather than
  reclassifying it as negative.

## Kill Criteria

Kill immediately and do not open the next stage if:

- final adapter/source/test/archive/card hashes are absent or differ;
- an authoritative input/config/source/cache/exposure check fails;
- any row reaches 180 seconds, schedule reaches 1,200 seconds, repair reaches
  1,800 seconds, a worker exceeds 4 GiB, or a byte cap is reached/exceeded
  under its frozen exclusive semantics;
- any worker dies, is replaced, sends a mismatched envelope, or cannot be
  attributed fail-closed;
- any schema, canonical JSON, replay, resume, source, cache, cross-arm,
  artifact, lock, fsync, or publication guard fails;
- source selection, weak veto, or repair80 is negative or ambiguous;
- an accidental fresh/sealed identity exposure occurs; or
- continuation would need Family B, cache regeneration, BeatThis chunk-seam
  assumptions, holdout/broad/full data, ramp support, or production changes.

No later data or runner-up may rescue a killed stage. A behavior-affecting fix
requires a new or explicitly mutated no-data card and restarts synthetic
verification plus all four schedule arms on new immutable paths.

## Expected Failure Modes

- an adapter may accidentally rely on private synthetic types without binding
  real cache/source semantics;
- a callback or config object may not be spawn-picklable;
- identity/label wrappers may hash parsed JSON instead of the exact source
  bytes or may reorder rows;
- cache configuration may bind absolute/volatile paths rather than deterministic
  content/config identity;
- schedule arms may recompute or drift in candidate/current-v2 payloads;
- a standalone pending selection may be mistaken for a committed winner;
- weak code may be imported or open inputs before the source marker validates;
- duplicate selected-schedule `audio_group_key` may reach source reduction
  instead of failing before cache access, or repair80 may be wrongly
  de-duplicated by audio group instead of retaining 80 identity denominators;
- fallback rows may contaminate pure-candidate metrics;
- runner timing may attribute queue wait or ordered-imap blocking to a row;
- a summary/publication deadline after row 80 may be tagged as a row failure;
- exposure may be published too late or omit a failed/accidental key;
- an untracked source tree may change after its hashes are reported;
- a human-readable report may imply cache chunk geometry or boundary adequacy
  beyond the weak evidence.

## Confounders

- schedule16 and repair80 are already exposed diagnostic data, not fresh
  acceptance evidence.
- `.osu` maps are correlated mapper annotations and can be unavailable,
  conflicting, or systematically biased.
- current BeatThis values are chunk-conditioned activations, not independent
  beat posteriors or truth.
- cache-generator/checkpoint/chunk/border/aggregation provenance is not proven
  strongly enough for chunk-seam analysis.
- macOS/Linux worker RSS is a lifetime high-water mark and runtime is affected
  by cache warmth; runtime/RSS therefore gate but do not order schedule arms.
- source-only overlap measures retained-lineage recomputation, not all pruned
  counterfactual states.
- repair80 can diagnose regressions but cannot establish production value or
  fresh generalization.

## Expected Runtime / Runtime Budget

- No-data adapter verification: target under the existing Timing-v3 suite
  budget; the full current suite previously completed in about 365 seconds.
- Schedule16: four arms, exact fixed order, less than 20 minutes total.
- Per schedule or repair audio/arm: strictly less than 180 seconds.
- Repair80: one selected arm, strictly less than 30 minutes total.
- Workers: exactly four fresh spawn workers per independent arm/stage; each
  lifetime peak RSS at most 4,294,967,296 bytes.
- No automatic retry after a timeout, integrity failure, hard failure, weak
  veto, or failed final publication.

## Result Interpretation Plan

- Positive result would suggest: the unchanged Exp006/Exp007 candidate and
  execution contract deserve a separate no-new-data acceptance review before
  any fresh holdout is selected. It would not establish production quality.
- Negative result would suggest: retain current v2, record the failing frozen
  value/safety/source gate, and KILL or design one new no-data hypothesis. Do
  not relax gates or inspect later data.
- Ambiguous result would require: retain current v2 and write one narrower
  no-data TEST card addressing the exact missing denominator/uncertainty; do
  not treat ambiguity as a pass.
- Human owner decides: first, whether to authorize no-data adapter
  implementation from this draft; second, whether the final hash-complete card
  authorizes real exposed execution; and later, after a positive Exp008 result,
  whether to accept a separate fresh-holdout card.
- Next-loop action if positive: write an immutable Exp008 result, perform only
  no-new-data acceptance review, and draft—but do not execute—a fresh
  audio-disjoint holdout100 card.
- Next-loop action if negative: KILL the current candidate for later gates or
  MUTATE exactly the attributed mechanism under a new no-data card.
- Next-loop action if ambiguous: stop and create one decision-complete no-data
  card; preserve all exposure and exclude any accidentally exposed identity
  from future held-out selection.

## Result Log Template

- Experiment: Timing v3 Experiment 008
- Date:
- Final accepted card SHA-256:
- Owner acceptance reference:
- Immutable archive/tree-manifest SHA-256:
- Commit / run id:
- Exp007 source-closure fingerprint:
- Exp008 composed execution-identity fingerprint:
- Adapter/test SHA-256:
- Dataset slice and authoritative input hashes:
- Prior exposure and Exp008 delta hashes:
- Cache/current-v2/weak config hashes:
- Selector and four schedule RunConfig hashes:
- Selected repair RunConfig/input-binding hashes, if authorized:
- Baseline / comparator: current-v2 on identical restricted cache activation
- Runtime: schedule aggregate, repair aggregate, row p50/p90/max
- Worker RSS: four lifetime values per arm/stage
- Primary metric value: ordered source/weak/repair decisions
- Secondary metric value: all denominator/set/gate/fallback/overlap values
- Verify command / result:
- Guard command / result:
- Qualitative observations: schema/artifact audit only
- Positive signal observed:
- Negative signal observed:
- Kill criteria triggered:
- Checks performed:
- Failed checks:
- Suspected confounders:
- Selected variant: A
- Candidate variants rejected before execution: B, C, D
- Local verification outcomes:
- Selection pressure observed:
- Current product status counts: v3 accepted / v2 fallback / hard failure
- Pure candidate fallback count and reasons:
- Unopened later stages:
- Zero-compute replay result:
- Interpretation:
- Recommended next step:
- Human owner decision:

## Pre-Execution Gate

- Card complete: no. The source-owned adapter does not yet exist, its
  source/test hashes and the post-card composed-identity/archive hashes are
  absent, and these draft bytes have not received independent final review or
  owner acceptance.
- Code execution allowed after this card: no. A separate explicit owner
  acceptance may authorize Phase 0 no-data implementation only. Real-data
  execution remains forbidden until the final revision and its acyclic
  post-card artifacts receive a second explicit acceptance naming all three
  external SHA-256 values.
- Closed loop complete: no.
- Remaining ambiguity: whether the adapter can be implemented without changing
  frozen behavior; exact authoritative input/config hashes; immutable archive
  mechanism for the untracked Timing v3 tree; and the empirical staged result.

## Next-Loop Action

- If positive: after owner authorization, implement and verify Phase 0 with no
  real data, finalize hashes, review, and request explicit data acceptance.
- If negative: record the adapter/reproducibility blocker and KILL or MUTATE
  before any real artifact opens.
- If ambiguous: narrow the adapter/integrity question under a new no-data card;
  do not broaden data or evidence permissions.

## Novelty Notes

- Closest analogies: preregistered staged evaluation and immutable
  content-addressed execution pipelines.
- Novelty layer, if any: no algorithm novelty in Exp008.
- Representation novelty vs engineering variation: entirely engineering and
  evidence-integrity variation around the inherited absolute-beat constant/jump
  representation.
