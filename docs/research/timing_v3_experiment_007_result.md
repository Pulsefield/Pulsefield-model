# Timing v3 Experiment 007 Result: Real-Cache Protocol Freeze

Date: 2026-08-13

Decision: synthetic `TEST` pass. The source-owned Exp007 protocol, bounded
diagnostics API, schedule selector, four-arm runner contract, source-only
reducers, winner-only weak veto, artifact/resume layer, and repair80 summary
contract pass the frozen synthetic verification. This result satisfies the
evidence precondition for asking the human owner whether to authorize drafting
a separate Experiment 008 card. It does not itself authorize drafting Exp008
or opening a real identity, BeatThis cache, audio file, `.osu` file,
label/metric artifact, schedule16, repair80, holdout, broad500, full5050, API
snapshot, or network source.

The Exp007-authorized implementation did not edit the production fitter,
provider/cache implementation, Hydra config, mapper path, ramp primitive,
promotion threshold, or data split. This is not a claim that the dirty
worktree equals `HEAD`: relevant pre-existing provider/cache/network-adjacent
changes retained in the worktree are disclosed below, with closure-bound and
out-of-closure files distinguished. None is accepted as a production change
by this result.

## Frozen card and source identity

- Exp007 card SHA-256:
  `fb8b13cef083006f04165ac0f3d691b3e3965f6342b4ed264e0c283a1dec59d7`.
- Behavior source-closure SHA-256:
  `c22730ed11b7b4a872a434bebea2bf57279ad328caa7ff0c82dfc96e12ac573b`.
- Recursive schema-descriptor registry SHA-256:
  `aa82b4fc44438413997be10b95ad85becfbc48a0a5cbd38d6e543aecdd41b80d`.
- Repository commit recorded by the source audit:
  `be8993b7fe8325a98d4d8d3b80138b1bd8ffe1b7` with an explicitly dirty
  research worktree. Byte hashes, not the commit alone, identify this result.

Selected descriptor hashes:

| Schema | SHA-256 |
| --- | --- |
| Source closure | `0f77f5d230f6e9039650b86c39ef90608cccf7c71111bcaaaae46a35e2d75799` |
| Run config | `465ce0e4d54f6c37ea38fe438f3dbd7b981223ca77f3c411451a604af8830896` |
| Row result | `e50f97de5b14b43d1f7899f4e847ae4363ab59f55a7785c26a6f33040ec1bf78` |
| Source-arm summary | `716fe96f7a7621231b3c691d1afe43035970cfc32bf2b1703f5ef811151374e3` |
| Config selection | `7aa6774645e7bc943511001d4bb028b1640245f17c11eb15e33c32ff659dd535` |
| Four-arm summary | `2151ccb877fca1228bd6328daa98ff08503960651132910a1587521ca5b22882` |
| Repair80 summary | `83368350d6bedcbbd17c978946d5382f7952e133581c1ba6659af0c6bc93ffe1` |

Final source snapshots:

| Component | SHA-256 |
| --- | --- |
| `local_frontier.py` | `3f8f677ad82b16dc658b730132b350ea9ee319790394b1e3c66705328b2f0999` |
| `exp007_protocol.py` | `06e191ea62d7faa50fc2d31264b9e1d8c6ac3738ff361923675c4fbcec4236cc` |
| `exp007_selector.py` | `23d2f2f9f4119e275f9a0510617ee59c936a4728eb4d589b2a37362f14dd37b0` |
| `exp007_runner.py` | `6c65d3bf2a6e44f739e3d3b399b4887e6315629f1f33e577e99cc67be1481553` |
| `exp007_metrics.py` | `b7cb51ba2d91b3ae0799a2a8686ff3cb8feac82d4ecf2ec0be348c8852e640a3` |
| `exp007_weak_evidence.py` | `2bb43c6ad4a4cc7991da66966071c2ca7fd23a978f2f074525e97759115b511a` |
| `exp007_artifacts.py` | `18443a886580d0797bb7d028895c871dc0f9add41d7c4b1fc13eba0b7177aa9c` |

### Dirty-worktree scope disclosure

The repository was already dirty when Exp007 implementation resumed. In
particular, the following production- or network-adjacent paths are relevant
to preventing an over-broad reading of this result. This is a scope disclosure,
not a worktree-wide production acceptance audit.

| Path | Git state | Closure status | SHA-256 |
| --- | --- | --- | --- |
| `src/pulsefield_model/timing/providers/__init__.py` | modified | bound | `e08e3598b0e7990ca09b8283ad2896d04e016f8891494e0edbb00419a890f49d` |
| `src/pulsefield_model/timing/providers/beatthis.py` | modified | bound | `76f99667553cd22d9c9b34915c284a33c51cbec71c22fcc3c8129e5ff4e4ec6b` |
| `src/pulsefield_model/timing/providers/beatthis_cache.py` | untracked | bound | `7982bda6d973ded2433c1b15db46f3500f39d1e4a606f46a467d7998ec8b9891` |
| `src/pulsefield_model/timing/providers/oracle.py` | clean | bound | `72d32cee653c258fbfa36ca85bfe61252e599aff2787db4123a36aecc9d1c6a8` |
| `src/pulsefield_model/timing/build_beatthis_cache.py` | untracked | outside closure | `a004cc1815da980ae4ab2f87758549706a65299edd8cd1a47dcf108075c805aa` |
| `src/pulsefield_model/data/fetch_osu_beatmapset_metadata.py` | untracked | outside closure | `1040e4dff3259436b8a8753deb8c584c6f006fcc1c25dffd28d364f2216fef0d` |

`beatthis_cache` is a frozen source-closure entry module. Recursive import
closure also pulls in the provider package initializer, `beatthis.py`, and
`oracle.py`. Consequently, the behavior-closure fingerprint above binds their
exact current bytes; it does not isolate them away. That binding proves source
identity and import closure only. Exp007 did not mutate these files, execute a
real provider/cache path, test their production acceptance, or approve their
diff from `HEAD`. They remain outside the interpretation of the Exp007
synthetic protocol pass and require their own review before any production
use.

The cache builder is not in the closure and can materialize/migrate BeatThis
caches. The metadata fetcher is also outside the closure and contains an
osu! API `urlopen` path. Exp007 did not import or execute either file; neither
its bytes nor its behavior is accepted by this result. Their disclosure does
not assert that the rest of the dirty worktree has been production-reviewed.

Final Exp007 test snapshots:

| Component | SHA-256 |
| --- | --- |
| `test_timing_v3_exp007_protocol.py` | `714e4ba6a073fce390fdd8bb8551d2a6d7725dad0babc64abe1a896bc425cddb` |
| `test_timing_v3_exp007_selector.py` | `a6fc1c101f4c35a24874ea101cf7d72915c564500329af78791e07243b8601e2` |
| `test_timing_v3_exp007_runner.py` | `8c4f3d42a8a94686716ef13cbc970584fa6e810ad49aad6797138c11b7a0a844` |
| `test_timing_v3_exp007_metrics.py` | `b68d95be957ea51dac073816eeb7f230ce5344d597c0b6849236eed5be6d31e5` |
| `test_timing_v3_exp007_weak_evidence.py` | `23374df76d7ab0cb2dc7c282fbae926f8ce3168efb2346850b26f29c8f2d12f9` |
| `test_timing_v3_exp007_artifacts.py` | `980d289b48206d5c5efd0d1b2c986f1b9781397fd8d3cc28cd4a7a71bb9a07b7` |
| `test_timing_v3_exp007_overlap.py` | `8eaed1e367ec67ea12840bdbd7a5bac567316bf6bb7de0b7ebd2dfc4417ba54c` |

Test bytes are deliberately outside the behavior-source closure, so both the
behavior closure and the independent test hashes are retained.

## Verification result

The frozen Exp005/Exp006 behavior guard passed first:

```text
109 passed in 14.08s
```

This pins the 44-arm Exp006 matrix, FULL behavior, the five aggregate oracles,
and the original Exp005 payload hashes before interpreting Exp007.

The first complete Exp007 run was not hidden or reclassified as a pass:

```text
244 passed, 8 failed in 190.35s
```

All eight failures were in the synthetic runner surface. They exposed three
protocol/test implementation defects, not a candidate, objective, threshold,
or real-data result:

1. schedule fixtures used different selector and input-manifest SHAs;
2. authoritative source-summary validation constructed each expected row ref
   but omitted the append before hashing the complete list;
3. a repair-summary timeout fixture could consume its 10 ms budget in the
   synthetic row loop before entering the intended summary phase.

After fixing those attributed layers, independent review found one additional
fail-closed attribution gap: a repair80 deadline crossing after row 80 had
committed but before summary handling was tagged `pool_stream`. The runner now
uses `repair_summary` only when the complete 80-row ordered prefix is already
committed; mid-row and incomplete-prefix deadlines remain `pool_stream`.
Red/green coverage pins that boundary. A final review also identified and then
cleared a synthetic helper that had copied default config fingerprints instead
of carrying the resolved config.

A later source-only completion audit reopened the checkpoint after reproducing
one integrity-path failure on the repository's Python 3.10 runtime:
`make_source_closure(repo_root)` used `datetime.UTC`, which exists only on
Python 3.11 and later. Existing tests had hidden the default branch by always
injecting `generated_at_utc`. A regression test was first observed red under
Python 3.10.20, the implementation was changed to
`datetime.timezone.utc`, and the focused, Exp007, and full Timing-v3 guards
were rerun before replacing the source/test hashes above. No candidate,
metric, threshold, selector, data permission, or real-data result changed.

Final commands and results from the final source/test snapshot:

```text
.venv/bin/python -m pytest -q \
  tests/timing/test_timing_v3_local_frontier.py \
  tests/timing/test_timing_v3_boundary_pair_transition.py \
  tests/timing/test_timing_v3_boundary_pair_transition_matrix.py --tb=short
109 passed in 14.08s

.venv/bin/python -m pytest -q tests/timing/test_timing_v3_exp007_*.py --tb=short
255 passed in 199.21s

.venv/bin/python -m pytest -q tests/timing/test_timing_v3_*.py --tb=short
760 passed, 2 skipped, 9 subtests passed in 364.54s

.venv/bin/python -m py_compile \
  src/pulsefield_model/timing/v3/local_frontier.py \
  src/pulsefield_model/timing/evaluation/exp007_protocol.py \
  src/pulsefield_model/timing/evaluation/exp007_selector.py \
  src/pulsefield_model/timing/evaluation/exp007_runner.py \
  src/pulsefield_model/timing/evaluation/exp007_metrics.py \
  src/pulsefield_model/timing/evaluation/exp007_weak_evidence.py \
  src/pulsefield_model/timing/evaluation/exp007_artifacts.py
passed

git diff --check
passed
```

Independent blocker/scientific review accepted the frozen card before source
execution. Independent post-implementation review found no remaining
actionable finding after the deadline-attribution and non-default-fingerprint
regressions were added.

## Contract coverage

- FULL and BOUNDED return identical reason, grid, and base diagnostics on all
  44 Exp006 matrix arms; bounded mode does not allocate full ledgers.
- Exact overlap lineage, half-open domains, absolute integer-beat
  intersection, p90 calculation, unavailable reasons, and all frozen caps are
  covered.
- Recursive schemas reject nested extras, duplicate JSON keys, nonfinite
  values, bool-as-int, wrong unions, stale dependencies, and invalid
  decision/action pairs.
- Selector replay preserves the Exp005 seed, exclusive priority, deficit
  order, and identity-only dependency.
- The runner covers fixed arm order, four fresh spawn workers, control-channel
  handshake, deadlines, worker death/replacement, teardown, RSS normalization,
  candidate/current-v2 equality, and exact failure attribution.
- Source ordering does not import or consume weak rows. Only a committed source
  winner can enter winner-only weak veto; no runner-up can be promoted.
- Product, fallback, hard-failure, comparator-unavailable, and repair80
  identity denominators are mechanically rebuilt from immutable rows.
- Atomic publication, content-addressed candidate bundles, resume, source/cache
  closure, path containment, locks, and exposure deltas are covered with
  synthetic temporary roots.

## Real-data and production audit

No real identity, cache, prediction, audio, `.osu`, label, metric, exposure,
schedule16, repair80, holdout, broad500, full5050, API snapshot, or network
source was opened or written by the Exp007 source/test execution. Static
inspection of the directly executed Exp007 protocol/test paths found only:

- the allowed local loopback socket used by the frozen multiprocessing control
  channel;
- temporary synthetic artifact-relative paths exercised by containment and
  atomicity tests;
- documentation mentioning `.osu`; and
- tests that explicitly reject network imports.

The wider behavior closure also byte-binds provider/cache/oracle modules that
contain real cache, audio, and `.osu` accessors. Those modules were closure
inputs only: Exp007 did not call their real-data accessors, and this result does
not accept their production behavior. Their exact Git state and hashes are
disclosed above.

There is no Exp008 document or real exposure delta in this result.

The dirty worktree does contain the out-of-closure cache builder and network-
capable metadata fetcher disclosed above. They were source-inspected only to
bound this statement, not executed. Accordingly, this is not a claim that the
entire dirty worktree contains no cache-writing or network-capable code.

The production-path audit is therefore scoped as follows: the Exp007 changes
did not introduce a fitter/provider/cache-format/config mutation. The source
closure contains all four provider-package paths shown above: three are dirty
and one (`oracle.py`) is clean. Those paths and the two out-of-closure
cache/network scripts are explicitly not cleared by this result. The full
Timing-v3 verifier ran against the exact dirty-worktree bytes named by the
source closure; it did not bind or execute the two out-of-closure scripts, and
it does not convert any unrelated production-path change into Exp007 evidence.

## Cache-provenance limitation retained for the next loop

Source review can prove the cache schema/config fields for provider,
checkpoint, shift, frame rate, dtype/cache version, audio key, and content
identity. It cannot prove from each existing NPZ the BeatThis package version,
internal chunk size, border/padding, or internal aggregation mode used when the
cache was generated. Therefore no existing-cache claim may treat an assumed
BeatThis 1.1.0 chunk seam as a musical or analytical boundary. A later no-data
card must either bind an auditable generator/build manifest or define a
versioned regeneration/migration plan before any seam-specific claim.

This limitation does not invalidate Exp007, which opened no real cache and made
no cache-seam claim.

## Interpretation and next action

Exp007 is positive only for protocol readiness. It does not establish real
BeatThis period reliability, schedule quality, phase/drift safety, boundary
localization, fallback rate, runtime on real rows, product acceptance, or
production readiness.

The next permissible action is to ask the human owner whether the reviewed
Exp007 source/test/card hashes authorize drafting a separate Experiment 008
card. Only after that explicit decision may the no-data draft be written and
independently reviewed. If the owner then accepts the frozen card, it may
authorize exactly:

```text
exposed schedule16
-> source-only winner commit
-> winner-only weak-evidence veto
-> selected-arm exposed repair80
```

It may not directly authorize fresh holdout, broad500, full5050, ramp support,
or a production-default change. An ambiguous, negative, timeout, integrity, or
hard-failure result at any Exp008 gate must stop the loop without opening a
later data layer.
