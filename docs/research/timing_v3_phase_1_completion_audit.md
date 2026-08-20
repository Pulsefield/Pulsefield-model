# Timing v3 Phase 1 Completion Audit

Date: 2026-08-13

Decision: **incomplete; pause at the no-data boundary**. The frozen Exp007
protocol is a synthetic `TEST` pass after the Python 3.10 source-closure repair.
It does not prove any real-cache, product, holdout, broad, full-corpus, or
production-integration gate.

This audit inspected source, tests, Git state, and durable documentation only.
It did not open `artifacts/`, a real identity, cache, prediction, audio, `.osu`,
label/metric output, API snapshot, or network source, and it created no Exp008
document or exposure delta.

## Current checkpoint

- Frozen Exp007 card SHA-256:
  `fb8b13cef083006f04165ac0f3d691b3e3965f6342b4ed264e0c283a1dec59d7`.
- Refrozen behavior source-closure SHA-256:
  `c22730ed11b7b4a872a434bebea2bf57279ad328caa7ff0c82dfc96e12ac573b`.
- Schema-descriptor registry SHA-256:
  `aa82b4fc44438413997be10b95ad85becfbc48a0a5cbd38d6e543aecdd41b80d`.
- Focused Exp005/006 guard: `109 passed in 14.29s`.
- Exp007 suite: `255 passed in 199.21s`.
- Full Timing-v3 guard:
  `760 passed, 2 skipped, 9 subtests passed in 364.54s`.
- Independent review of the final Python 3.10 timestamp repair found no
  implementation blocker after stale result hashes were replaced.

The result and problem log retain the earlier red runs, defect attribution,
dirty-worktree boundary, and exact source/test hashes. Exp001-006 remain
immutable historical evidence.

## Section 13 completion matrix

| Requirement | Status | Current evidence | Missing proof |
| --- | --- | --- | --- |
| 1. Complete constant/jump source, schema, runner, evaluator, and tests | Incomplete | Source-owned absolute-beat schema, constant/jump/local-frontier implementation, Exp007 protocol/runner/reducers/artifact layer, and synthetic tests pass. | The final product path has not passed real schedule/repair or later integration gates. |
| 2. Freeze final algorithm/config/feature/comparator policy before holdout | Incomplete | Exp007 freezes protocol readiness and exact source bytes. | No real schedule winner, repair result, or later no-new-data acceptance freeze exists. |
| 3. Frozen holdout100 and broad500 pass | Missing | Earlier negative experiments are preserved; Exp007 opened neither stage. | Both gates remain unauthorized and unexecuted. |
| 4. Identical-source/config full5050 replay | Missing | Exp001's full result is a current-v2 baseline, not the final v3 product replay. | Final frozen v3 full5050 has not run. |
| 5. Immutable product status for all 5,050 rows | Missing | Exp007 defines and tests synthetic row/status schemas. | No final 5,050-row product manifest exists. |
| 6. Promotion, safety, runtime, and long-stratum gates pass | Missing | Synthetic correctness/resource contracts pass. | Real paired quality, value improvement, fallback, runtime, memory, and hard-stratum evidence is absent. |
| 7. Zero-compute resume/identity replay | Incomplete | Synthetic atomic/resume/source/cache-closure tests pass. | No final 5,050-result zero-compute replay exists. |
| 8. Complete delivery bundle | Incomplete | Updated task definition; Experiment 001-007 cards/results; source/tests; Exp007 hashes; and problem log exist in the workspace. | Later cards/results, accepted manifests/reports/diagnostics, rollback runbook, and explicit disabled-default integration proof are absent. Most Timing-v3 files are also untracked and are not yet durable in a clean checkout. |
| 9. Current v2 remains complete rollback | Proven so far, not final | Production inference still calls current `GridFitter`/dense-v2; no accepted default switch exists. | Disabled-by-default v3 integration and an operator rollback document remain final-handoff work. Any default switch requires owner approval. |

## Ordered dependency chain

The only completed stage in the current Phase 1 chain is source-owned
protocol/schema/unit/synthetic verification. The remaining chain is:

```text
owner authorizes drafting Exp008
-> draft, independently review, and freeze Exp008 without real data
-> exposed schedule16
-> source-only winner commit
-> winner-only weak-evidence veto
-> selected-arm exposed repair80
-> no-new-data acceptance review
-> fresh audio-disjoint holdout100
-> broad500
-> frozen full5050 replay
-> zero-compute replay and delivery/integration handoff
```

Each arrow is conditional on a complete pass of the preceding gate. A
negative, ambiguous, timeout, integrity failure, or hard failure stops the
chain before the next data layer. Later data cannot rescue an earlier result.

## Active limitations and ruled-out actions

1. **Owner decision:** the Exp007 card assigns the decision to write Exp008 to
   the human owner. No such authorization is recorded. Drafting Exp008 or
   opening schedule16 would therefore exceed the current checkpoint.
2. **Cache provenance:** current NPZ metadata does not prove historical
   BeatThis package/chunk/border/padding/aggregation/build-environment details.
   Exp008 can remain non-seam-specific, but no seam-specific claim is allowed
   until a separate no-data provenance or regeneration/migration plan closes
   this gap.
3. **Repository durability:** the Timing-v3 source/docs/tests are largely
   untracked. Their hashes detect mutation but do not make their bytes
   recoverable from a clean checkout. Selective commit or another immutable
   archive requires an owner-approved scope because the worktree contains
   unrelated changes.
4. **Integration:** v3 is disabled today by omission—the production mapper
   runtime still uses v2—but there is no explicit v3 feature selector,
   product-status boundary, or rollback runbook. Implementing that behavior
   before empirical acceptance would be premature and is not authorized by
   Exp007.
5. **No cache-builder repair in this checkpoint:** a read-only audit noticed a
   deterministic sibling `.tmp` report path in the unrelated, out-of-closure
   cache builder. It was not changed because Exp007 neither owns nor accepts
   that production-adjacent path.

## Recommendation

Recommended next research action: `TEST`, contingent on explicit owner
authorization to draft one bounded Exp008 card. The draft itself remains a
no-data action. Its maximum execution scope is exposed
`schedule16 -> source-only winner -> winner-only veto -> repair80`; it must not
authorize holdout100, broad500, full5050, ramp output, Family B audio features,
or production-default changes.

