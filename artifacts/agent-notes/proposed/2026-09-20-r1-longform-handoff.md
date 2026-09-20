# Agent Note: Resume playable R1 long-form generation

Note ID: 2026-09-20-r1-longform-handoff
Status: proposed
Kind: process
Created: 2026-09-20
Updated: 2026-09-20
Product revision: 8cf31e8d177fab28060ce92be4f5e92f8f8585f3
Scope: Pause handoff; human priorities, completed routing evaluation, remaining quality gaps, and resumption
Related: 2026-09-20-r1-longform-structure; 2026-09-20-r1-training-distribution; 2026-09-20-r1-persistent-seed

## Question or Decision

Build a highly playable train/inference setup on the available M5 Air, 24 GiB, corpus and annotations, toward Pulsefield. This phase remains audio-free with a supplied timing skeleton and original seed. The goal is **not complete**; no candidate has final quality acceptance.

Human instructions, in current priority order:

- Focus on long-form structural stability and preventing pattern-diversity collapse; retain diverse LN organization organically combined with TAP.
- Cross-difficulty stability across 2★–6★ remains a secondary goal, explicitly deferred from immediate optimization.
- Revise the plan or architecture substantially when justified. Add modules or scale parameters when needed; playability takes priority over speed and parameter economy.
- Quota/time is limited: avoid broad controls and ablation suites. Prefer one targeted intervention, complete native generation, and decisive failure inspection.
- Use `../beatmap-lens` judgment scales and human gold before declaring success. Commit meaningful progress.
- Immediate request: persist a concise handoff before the user pauses. No new experiment was started for this handoff.

## Repository State and Evidence

Continue in sibling worktree `Pulsefield-model-longform-memory`, branch `codex/r1-longform-memory`, clean at the product revision above. The original `Pulsefield-model` worktree on `witness-style-probe` retains pre-existing dirty M0–M4 work; leave it untouched. Notes live in the registered `Pulsefield-model-agent-notes` worktree, orphan branch `agent-notes`; never merge it into product history. No training, generation, or rendering job remains running.

Primary research owner: Note ID `2026-09-20-r1-longform-structure`. At notes revision `e7df877733f66d9569522f0d8f0f97780c66bf9c`, it ends with routing training completed and evaluation frozen. **The completed evaluation and subsequent visual observations below extend that stopping point.** This handoff does not supersede its experiment history or change note acceptance.

Product owners are `docs/research/bounded_typed_continuation.md`, `docs/research/r1_training_distribution.md`, and `src/pulsefield_model/research/bounded_typed_continuation/`. Routing implementation, packaged configuration and tests are committed in the product revision above; latest routing quality results are not yet in curated product documentation.

### Completed interventions

On the same 8 VAL groups, indices 3/9/15/19/21/23/26/27, seeds 17/23 (16 complete outputs), H denotes a mandatory-head row:

| Candidate | Maximum consecutive H containing one fixed lane | Pooled below-40 ms diagnostic events | Interpretation |
| --- | ---: | ---: | --- |
| Persistent-seed 5M | 77 | 152 | Persistent seed alone did not resolve long-form drift. |
| Full-history memory 6M | 152 | 69 | Memory alone failed: 18.2 s quadruple wall and held-lane lock. |
| Full-parameter recovery 6.25M | 11 | 3 | Repetition improved, but substantial independent LN organization disappeared; not adopted. |
| Frozen-base routing 6.25M | 22 | 68 | Historic collapse improved while retaining meaningful LN structure in some guards; mixed, not accepted. |

Routing adds a mirror-equivariant 512-wide MLP over 16 head masks. For a fixed state and head mask, its common score shift preserves the base conditional TAP/LN and simultaneous-release distribution; release-only candidates receive exactly zero correction. All 2,777,232 inherited parameters and Adam states remain bit-identical. Only 139,776 new parameters train, for 2,917,008 total. Ordinary CE plus native complement loss used the existing TRAIN-only 80-query/8-group pool; all queries are H states, with no blocking-hold release examples. Sampling remains native at temperature 1, without added decoding bans.

Training added 250k source onsets in 333 updates/300 s on CPU1, sampled peak RSS 956 MB, swap growth zero. The 16 complete generations took 329 s; mechanics/export/reparse passed, with no below-10 ms diagnostic event. The 68 versus 69 short-gap count is not evidence of a meaningful improvement over memory 6M. Generated stars 2.39–5.76 on this reused slice do not establish difficulty calibration.

### Exact resumption artifacts

Paths below are relative to `Pulsefield-model-longform-memory`. Let `R` denote `artifacts/bounded-typed-continuation/routing-recovery-20260920-v1/`.

- Candidate: `R/routing-6250k/checkpoint.pt`, SHA-256 `dff1727bcb634666992dc4c836ee507c02f31917b1a32f2f1f1d5329eaf5f080`.
- Complete comparison: `R/evaluation-v1/readout.json`, SHA-256 `ddae474b166e426dcd4c29943e0f3b2e97da95e19f347a5e3589bcd0e20f574d`.
- Exact LN guard recount: `R/retention-readout.json`, SHA-256 `309e78601bc505a522ebf0aa1cbcf8e3b03f06dc7ef782cf3ffb2b5137149a5f`.
- Parent: `artifacts/bounded-typed-continuation/longform-memory-20260920-v1/memory-6000k/checkpoint.pt`, SHA-256 `b44c9c83df84f2d9acd039edf64c1fc40377d2fc9ae0d25e9997ef548ae8876f`.
- Native pool: `artifacts/bounded-typed-continuation/native-recovery-20260920-v1/pool-v2/manifest.json`, SHA-256 `f712c597eefb8269f61a540c3459aad284c247b0ce3e61d9d8cc6fd5af4abbc3`.

Generated files are local evidence and may be absent in a fresh clone. Do not broadly scan artifacts or rerun completed experiments to rediscover their results.

## Risks and Falsifying Evidence

The following canonical Beatmap Lens montages were inspected after the machine readout, whose semantic status still says unreviewed. These are agent observations, not new human labels or complete acceptance:

- Source 21/seed 17, 220097–228097 ms: independent LN/TAP organization returns. Counts `(LN heads, H with >=2 held lanes, independent close rows)` are memory `(73,19,36)`, full recovery `(36,0,3)`, routing `(72,20,34)`. Seed 23 routing counts `(79,19,31)` are available but that montage remains unviewed.
- Source 09, 97684.5–105684.5 ms: routing seed 17 remains mostly TAP, `(18,1,1)`, versus memory `(79,35,22)`; seed 23 has meaningful brief LN passages, `(30,10,7)`. This is mixed retention, not a clean guard pass.
- Historic 19/seed 17 and 23/seed 23 collapse contexts now show changing singles/chords and LN/TAP relations. Late 27/seed 17 has independent LN flow and varied lengths, but is much more LN-heavy than its source; global/phase type drift is unresolved.
- Remaining static allocation: 23/seed 17 has 22 H over 4.17 s while two lanes remain held; 26/seed 23 has 18 H over 9.27 s with three held lanes and roughly 545 ms pulses. Inspect musical/action context before calling these failures. A 45-head/3.14 s inner/outer-pair cycle is recognizable Trill, not automatically collapse.

Viewed artifacts are under `R/review-ln/`, `review-ln-other-seed/`, `review-worst/`, `review-historic/`, and `review-cycle/`, each with scoped source/candidate montages and manifests. Pointwise conditional preservation cannot guarantee whole-trajectory LN preservation. Routing cannot directly repair release-only decisions or change a distribution when every legal action shares the sole available head mask.

## Alternatives or Hypothesis Branches

Do not repeat failed memory-only, persistent-seed-only, action-consequence/access-path, or full-parameter-recovery experiments without new evidence. TRAIN census already found 80.08% of exposure at 2–6★ and only 1.39% above 6★; a mostly-too-hard corpus is not the simple explanation. LN-heavy exposure is sparse; census details belong to the related distribution note/doc.

A release-policy adapter for native held states, or a section/planning module, remains a possible next intervention. Neither is selected or implemented. Do not introduce unconditional anti-Jack rules, LN-duration caps, or a release penalty merely from the remaining counters.

## Selected Direction

1. Preserve this candidate and append the completed routing result to the research owner; promote durable conclusions to curated docs when resuming product work.
2. Before further tuning on the reused 16 cases, freeze a small fresh VAL musical-group cohort for this exact checkpoint. Proposed scope: 8 groups, two per source band 2–3/3–4/4–5/5–6, balancing long charts and independent LN structure, duration >=180 s. This samples coverage without restoring difficulty optimization as the main task.
3. Source-only metadata already exists in sibling `Pulsefield-model-bounded-typed`, `artifacts/bounded-typed-continuation/difficulty-ln-longform-20260920-v1/{census.json,screen-conditions.json}`. Exclude the census's 48 prior groups and screen's 28 groups. Catalog and allocation group IDs can differ: validate both groupings. No fresh selections, Card, or run exists yet; freeze exact selection/seeds/bounds before generation.
4. Generate whole charts and inspect early/middle/late organization, worst repeat/held/cycle contexts and independent LN/TAP relations against gold. Decide whether release control, planning, or capacity actually needs intervention from those failures. Avoid a new broad ablation suite.

## Verification or Evaluation

Use `uv run --offline --python 3.10 --extra mps python ...` from the longform worktree; tests additionally use `--group dev`. Actual training used CPU1, not MPS. Prior targeted checks cover CPU/MPS routing equivalence, conditional/R-only preservation, mirror symmetry, frozen weights/Adam, native/dense agreement, strict resume and packaged export. Routing/memory/seed selection passed 23 tests; final affected seed/export/smoke selection passed 36, with overlapping coverage. No tests or training were rerun for this notes-only handoff.

Preserve supplied external timing when replaying native prefixes; never rebuild timing from emitted rows. Do not expose suffix lanes, head counts, TAP/LN types, or pairings as inputs. Keep TEST payloads untouched and machine negatives separate from human annotations.

For final judgment, use the installed `mania-pattern-judgment` skill and `../beatmap-lens/annotation/foundations/15fa68913bdb2bf395a189df7ab433f6d5b126fc35c46c1dbc8e607ce2182e97.json`. Existing gold refresh is in sibling `Pulsefield-model-bounded-typed`, `artifacts/bounded-typed-continuation/r1-exposure-20260919-v1/masked-inspection-v1/calibration-refresh-v1/`. Human IDs `human-03f7e300cf02f58f3dcbba66` and `human-2af74649b91a0eb016f7c6d2` calibrate independent LN and LN/TAP Stream. Slow-Jack gold is rendered under `native-recovery-20260920-v1/jack-gold/` in the longform artifact root. Pattern presence, short-gap thresholds, entropy, LN fraction, and source-style similarity alone do not establish playability.

## Next Lifecycle Condition

Resume the active goal from the fixed routing checkpoint and fresh-cohort verification plan. This proposed process note records state only; it neither accepts the research Cards nor declares the model ready. The handoff does not change the goal's status or authorize a remote push.
