# Agent Note: Test source-action representation access on an expanded local corpus

Note ID: 2026-09-14-source-action-stage3-pilot
Status: proposed
Kind: research
Created: 2026-09-14
Updated: 2026-09-14
Product revision: 790add7b07ecf08e1b277797a39ce28e9683a5b8
Scope: One MPS pilot comparing contextual, composed H-only and composed all-level predictors, with local-corpus allocation, frozen human probes and actual-update diagnostics.
Related: 2026-09-14-source-action-stage2-access-comparison

## Question or Decision

Does directly reading retained source/local/relation states improve held-out
action-block prediction and reusable human assessment beyond the same composed
encoder read through H alone? A separately trained contextual reference measures
the value of the larger composed backbone as a package.

The repository owner on 2026-09-14 explicitly authorized preparation and execution
on the current device, delegated dataset-split decisions, added the local beatmap
corpus, set a 2.5-hour budget with regular checkpoints, and authorized reusable
Parquet allocations. This is execution authority for the bounded pilot. The exact
new Card has not received separate revision-specific acceptance; its Note remains
proposed and the execution is exploratory. No acceptance or implemented lifecycle
transition is inferred.

## Repository State and Evidence

Product commit `3847c1fb1b7ec0a236ca1678f66437f605cbe5ef` implements the three
predictors, paired views, semantic readers and response diagnostics. Clean
descendant `790add7b07ecf08e1b277797a39ce28e9683a5b8` adds the runner, raw-corpus
allocation and an optional post-update probe callback. It changes no predictor,
loss, optimizer implementation or existing default probe behavior.

The source-action and affected replay/model checks passed 82 tests in 15.57s.
After adding the Parquet round-trip check, the five owning pilot tests passed in
1.39s. The added callback is covered by the semantic tests and a stop/mode-restore
test. A four-source local materialization check read two novel training and two
novel validation groups without an optimizer update; all four parsed successfully.

The local index contains 14,689 four-key records in 3,988 connected metadata
components. Removing components touching any annotation source leaves 2,753
training, 372 validation and 335 reserved-test candidate components under the
new hash split. Six published training groups connect to held-out annotations.
Source/set/song metadata grouping does not establish comprehensive audio
deduplication.

## Alternatives or Hypothesis Branches

- H-only and all-level composition improve equally: the backbone/capacity package
  may help; direct access has no demonstrated benefit.
- All-level reads improve beyond composed H-only, with informative context gains
  and frozen reuse: the access hypothesis receives preliminary evidence.
- The smaller reference matches the useful gains: retain it as the simpler option.
- No arm uses informative context or transfers: inspect objective shortcuts,
  decoder-prefix dependence and probe fitting before expanding the backbone.

## Research Record

### Experiment Card: source-action-access-local-pilot

#### Identity and Authority

- Owning Agent Note ID: 2026-09-14-source-action-stage3-pilot.
- Card ID: source-action-access-local-pilot.
- Revision: 1.
- Exact Card acceptance: none; exploratory execution under the explicit owner
  authorization recorded above.
- This Card fixes the following choices before model training.

#### Question and Hypothesis

- Question: does full-bank access improve held-out prediction over H-only access
  to the same composed backbone, and does pretraining produce frozen semantic reuse?
- Mechanism: queries can directly retrieve source and composed local patterns
  without requiring all useful information to pass through the final contextual
  state.
- Decision: whether to confirm direct access, prefer a smaller alternative, or
  refine the learning objective and readout.

#### Analogues and Branch Selection

- Closest carried-forward analogues: [MusicBERT](https://aclanthology.org/2021.findings-acl.70/)
  for structured masked symbolic prediction and
  [MAE](https://arxiv.org/abs/2111.06377) for visible-input encoding separated from
  reconstruction. The owning comparison rationale is
  `docs/research/source_action_representation_directions.md`.
- Shared mechanism: a broader prediction objective trains reusable representations
  from partially observed source structure.
- Difference: exact four-lane action rows, explicit LN replay, supplied event times
  and query-dependent reads of retained composition levels.
- Novelty remains a provisional domain adaptation/combination. This pilot does not
  establish a new general learning principle.
- Persistent slots, orthogonality penalties, optimizer search, audio conditioning
  and causal timing generation are outside the comparison.

#### Fixed Comparison

- Baseline and intervention executable source: clean
  `790add7b07ecf08e1b277797a39ce28e9683a5b8`.
- Architectural parent: `3847c1fb1b7ec0a236ca1678f66437f605cbe5ef`.
- Primary baseline: `composed_h`, 212,601 predictor parameters, random seed-17
  initialization. No pretrained checkpoint.
- Intervention: `composed_all`, the same 212,601-parameter backbone/reader/decoder
  initialization with all six levels readable instead of H alone.
- Contextual reference: `reference_h`, 102,561 parameters, trained with the same
  objective, exposure, dimensions and optimizer settings.
- Baseline run identity: `artifacts/source-action-modeling/stage3-pilot/20260914T081051Z`, `composed_h-endpoint.pt`.
  Its trained baseline value is not yet measured; the paired control is trained
  concurrently. Stage 1 training NLL is not used as a structural baseline.
- All predictors use default width-64 `ModelConfig`, zero dropout, common
  shared-component initialization and float32 parameters.
- Behavior-neutral instrumentation: manifests, checkpoint callbacks, timing,
  observed device allocation, process peak RSS and disjoint module decomposition
  of the existing actual-update response.
- Source owners: `source_action_modeling/pilot.py`, `local_corpus.py`,
  `comparison.py`, `semantic_probe.py`, `diagnostics.py`, `checkpoint.py`.
  No further model edits are part of this Card.

#### Dataset Slice and Split

- Annotation publication:
  `b22a7a443783e05fee4db4b1d22b8e573ad448ae`.
- Pinned annotation split:
  `15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a`.
- Prepared cohort:
  `252fe593514adc5498011b55dbf62224cba55651025233fc1ed9235078bb95b8`.
- Local source index: `artifacts/indexes/beatmap_index_4k.parquet`, SHA-256
  `2d2814c3b3ec47cd1247e8d50cef555e12ace7eae60a79944102926c142c2d8e`;
  original files are under `dataset/0/`.
- The index supplies paths and grouping metadata only. Legacy difficulty, timing
  filters, tensors and training code are not V3 or source-action authorities.
- Join positive beatmap IDs, positive set IDs, published source groups and
  NFKC/casefold/whitespace-normalized artist/title pairs transitively.
- Preserve published annotation split assignments. Exclude the six training
  groups connected to any annotation validation/test source from both action
  pretraining and human probe fitting.
- Added raw-corpus components must touch no annotation source. Stable component
  hashes assign buckets 0..7 to train, 8 to validation and 9 to reserved test.
- In component-hash order, select 1,024 valid novel training groups and 128 valid
  novel validation groups. Inspect at most twice each requested count. Select one
  chart per group by a fixed path hash. Verify original metadata, source bytes,
  four-key parsing and unambiguous source replay. Record rejected files/reasons;
  do not repair or retime invalid source actions. Reject repeated exact
  arrangements among selected local sources, including across train/validation.
- Each selected raw source supplies two deterministic 128-event windows, selected
  from event positions. A source with exactly 128 events supplies one unique
  window. Scope equals context for these unlabeled windows; entering occupation
  is explicitly conditioned. No semantic labels are transferred to these windows.
- Annotation action pretraining uses every remaining deduplicated original
  training scope/context/rate input with 4..256 source events. Annotation
  structural validation uses the corresponding original validation inputs.
- Human probes use all eligible original 1x human train/validation judgments under
  the existing conflict/confidence policy and High-confidence Tech requirement;
  semantic scopes are not truncated to the prediction cap. Machine fallback is
  excluded.
- No annotation test chart or newly assigned local test source payload is opened.
  Validation remains development evidence; metadata and existing source-group
  links do not prove complete song/audio independence.
- Persist complete source/component/split membership and selected windows as
  `dataset/source-assignments.parquet` and `dataset/windows.parquet` under the
  run directory, with schema, protocol, counts and file hashes in
  `dataset/manifest.json`. `population.json` additionally records eligibility,
  exclusions, source/window provenance and semantic indices.

#### Evidence and Decision Rule

- Primary: mean within-block per-target-row detailed-view NLL, averaged within
  validation groups and then equally across groups. Compare
  `NLL(composed_h) - NLL(composed_all)`; larger is better.
- Validation combines eligible original annotation groups and 128 novel local
  song components. Two distinct starts per feasible 4/16/64-event scale/group
  use a hash-selected eligible context. No action value or model output selects
  validation blocks.
- Report all raw losses, first-row and later-row losses, near-minus-detailed and
  coarse-minus-detailed gains by event scale and duration. The primary uses the
  fixed full manifest and 2,000 paired source-group bootstrap draws, seed 17.
- Practical primary threshold: gain at least 0.02 nats/target row and a strictly
  positive lower 95% paired bootstrap bound.
- Semantic regression guards: all-level trained frozen probe NLL may exceed the
  composed-H trained probe by at most 0.05 nats in concept/group-macro NLL and
  0.10 nats for any individual concept.
- Reuse criterion: all-level pretrained-versus-matched-untrained probe
  concept/group-macro NLL improves by at least 0.02 nats. Report each arm's matched
  contrast with identical head initialization and sampled human cells.
- Qualitative checks: detailed/coarse gains must accompany improved raw detailed
  NLL; inspect first-row dependence, Trill training/validation separation and
  false positives, selective Jack/Stream pairs, LN presence and conditional
  strength, and High-confidence Tech support. A uniform presence-prior shift
  does not establish a semantic distinction.
- Positive: primary threshold and semantic guards hold, with informative context
  and reuse evidence; recommend a fixed-comparison confirmation.
- Negative: access gain is below zero with a nonpositive upper confidence bound,
  or there is clear semantic regression; do not credit direct access.
- Ambiguous: intervals cross zero, practical threshold is missed, fit remains
  inadequate, guards lack support, or runtime prevents required measurements.
  Report that limitation and refine the comparison.
- No formal SUPPORTED/adoption claim follows from this exploratory Card or one
  seed, even when numeric thresholds are met.

#### Jacobian and Update Diagnostics

- Use actual parameter displacement, including AdamW history, clipping and weight
  decay. Compute scalar gradient-dot-displacement with the existing reverse-mode
  implementation; no full Jacobian matrix is required.
- At paired updates 1, 100, 1,000 and 3,000, when reached, measure a fixed training
  block log-likelihood response. Record encoder/reader/decoder contributions,
  finer disjoint parameter-module sums, actual response change and residual.
- After the selected endpoint and frozen-probe evaluation, keep each trained
  concept head fixed. Select up to two distinct-group positive and two negative
  human training cells per concept, sorted by cell ID.
- Continue the same paired optimizers for exactly three separately identified
  diagnostic suffix updates. Measure endpoint-to-suffix-1 and endpoint-to-suffix-3
  block likelihood and positive-minus-negative presence margins. The suffix is
  not the endpoint used for primary or semantic comparison.
- Report individual semantic probabilities beside contrast responses. A residual
  greater than 25% of max(abs(actual change), 1e-6) limits module interpretation.
  These are local update effects, not training-example, action or slot attribution.

#### Reproduction and Bounds

Run from the product repository root at the clean executable revision:

```sh
uv run --offline --extra mps --group dev python -u - <<'PY'
from pathlib import Path
from pulsefield_model.research.source_action_modeling.pilot import run_pilot
run_pilot(Path('artifacts/source-action-modeling/stage3-pilot/20260914T081051Z'), device='mps')
PY
```

- Environment: Apple M5 MacBook Air, 24 GB unified memory; Python 3.10.20,
  PyTorch 2.11.0, macOS arm64, MPS, one Torch CPU thread. CUDA is unavailable.
- Seed: 17 for models, sampling and probes; fixed likelihood diagnostic sampler
  seed 17017. Seeds 29/43 are reserved for a separate confirmation.
- AdamW: learning rate 0.0003, weight decay 0.0001, gradient norm cap 1.0,
  no scheduler. Each update contains four blocks, each exposed through all three
  views to every model. All views retain the same skeleton and decoder conditions.
- Sampling: uniform group, context, feasible 4/16/64-event size and event start.
  Loss averages row NLL within blocks and then blocks; each view has equal weight.
- Near radius: eight timeline rows. Detailed/coarse share the implemented
  permitted far count/occupation summaries and common near-derived entry state.
- Pretraining stops after 3,000 complete paired updates, 6,000 pretraining seconds,
  or total elapsed time reaches 6,600 seconds, whichever comes first. All arms
  share the same complete update count. Fewer than 100 updates is incomplete.
- Selection: the last complete common update under those declared bounds,
  independent of validation. Initial and endpoint structural evaluations only.
- Frozen probes: 400 steps, batch eight, AdamW learning rate 0.001 with the existing
  default weight decay; one matched trained/untrained pair for each configuration.
  No probe hyperparameter search or validation stopping.
- Checkpoints: all three models, optimizer states, sampler position/policy and RNG
  every 100 paired updates or 300 seconds, whichever occurs first, at complete
  update boundaries; endpoint and three-step suffix snapshots are additional.
  Save frozen-head weights, optimizer and fit-position metadata every 100 steps.
- Total runtime ceiling: 9,000 seconds (2.5 hours), including data preparation,
  structural evaluation, probes, diagnostics and checkpoint writing. Reserve at
  least 2,400 seconds from the training admission boundary for finalization.
- Sample MPS driver allocation between steps and abort above 8 GiB; report the
  maximum observed allocation and process peak RSS, not an allocator peak claim.
  Output storage ceiling: 1 GiB. No network downloads or new dependency installs.
- Fresh output: `artifacts/source-action-modeling/stage3-pilot/20260914T081051Z`. Exclusive creation; no overwrite or resume.
  Checkpoints preserve recovery state, but resumption is a separately recorded run.
- Stop on source/cohort/split mismatch, insufficient valid groups, nonfinite
  training/semantic loss, invalid paired conditions, memory/storage bound or
  total wall-clock exhaustion. Preserve partial reports and prior checkpoints.
- Confounders: unequal reference capacity; finite training horizon; decoder-prefix
  shortcuts; raw-fact access in the concept head; limited human positive support;
  metadata-only grouping; annotation-selected and raw event-window populations;
  one exploratory seed; possible MPS roundoff.
- Conclusion scope: conditional action arrangement at supplied times and original
  human assessment inputs. No timing generation, causal V3 transfer, player
  response or demand-state conclusion.

## Next Lifecycle Condition

Append a Result Log with the actual source revision, population and allocation
hashes, common update count, checkpoints, runtime, metrics, guards, diagnostics,
failures and deviations. The proposed lifecycle status remains unchanged unless
the owner explicitly requests a transition.

