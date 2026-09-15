# Agent Note: Test relation composition without assuming semantic factorization

Note ID: 2026-09-14-source-action-relation-composition
Status: proposed
Kind: research
Created: 2026-09-14
Updated: 2026-09-15
Product revision: 7adc7986379c913bd48be8afb8888024d008b852 with uncommitted composition, runner, configuration, diagnostics, tests and documentation changes
Scope: Equal-parameter source-action local/relation order comparison and a complete bounded MPS experiment
Related: 2026-09-14-source-action-time-local-representation, 2026-09-14-source-action-stage3-pilot

## Question and authority

The owner requested a genuinely executable experiment configuration after the
decoder/objective and time/local implementations, with careful consideration of
relation composition and the missing justification for semantic factorization.
The owner selected an overnight 8–12-hour budget on the current M5/24 GB machine.
The owner then explicitly requested full song/beatmapset coverage and a revised
experiment setting. The revision expands training to all eligible groups and
their beatmaps, with dynamic bounded windows. Original annotation sections remain
the human semantic fitting/evaluation inputs; they add no pretraining weight.
This task authorizes the scoped implementation and short software/resource
verification needed to supply that configuration. The owner subsequently
authorized stopping the already-running revision-1 overnight configuration and
preserving its directory/checkpoints. The owner then clarified that the full
overnight command would be executed manually; the agent must not start that run.
Execution is exploratory while product
changes remain dirty. This note remains proposed; no exact-revision acceptance,
product commit, lifecycle transition or remote publication is implied.

## Evidence, alternatives and direction

The baseline includes the context-dependent bilateral decoder, probability and
prefix-route contracts, smooth event-time coordinates, row MLP, conditioned local
kernels and direct source access. Its `combined` representation is the control.

The selected intervention reorders the same operators from `L1,L2,L3,R` to
`L1,L2,R,L3`. It changes no parameter tensor, initializer, source facts, number of
storage positions, loss, sampled exposure or reader. Post-retrieval local
composition can assemble recurrence evidence at successive anchors. Support and
optimization geometry change with order and may explain a gain without semantic
factorization. Both arms still have a contextual GRU and a prefix-aware decoder.

The alternative that the existing relation block and GRU already do the required
work remains live. Adding retrieval rounds, learned routing, semantic losses or
persistent slots would combine separate interventions and is deferred. A wider
backbone would require changing the fixed 64-channel interfaces, so this preset
increases only the supported feedforward/decoder dimensions, equally in both arms.

The information-path question has no automatic positive answer. Prediction loss
does not identify names or coordinates of latent factors, and the human concepts
overlap. The scoped operational target is reusable structural evidence under
held-out combinations and controlled changes. Player-facing demand requires
player-conditioned outcomes absent from these targets.

Closest analogues are [graph-network relation composition](https://arxiv.org/abs/1806.01261),
the [assumptions needed for unsupervised disentanglement](https://proceedings.mlr.press/v97/locatello19a.html),
and [causal abstraction through interchange interventions](https://proceedings.mlr.press/v162/geiger22a.html)
when a causal specification exists. This is a task-specific
adaptation and controlled comparison, not a new general architecture. No paper
establishes the desired rhythm-game semantic alignment.

The reusable computation, measurement, input and execution contracts live in
`docs/research/source_action_relation_composition.md`. The note owns the proposed
research decision and exact execution evidence, not product behavior.

## Experiment Card: source-action-relation-order-overnight

### Identity and Authority

- Owning Agent Note ID: 2026-09-14-source-action-relation-composition.
- Card ID: source-action-relation-order-overnight.
- Revision: 2.
- Authority: proposed; accepted revision none.

### Question and Hypothesis

- Question: Does composing locally after relation retrieval improve held-out
  source-action prediction and reusable structural evidence?
- Selected mechanism: `L1,L2,R,L3` gives an event-local operator access to
  relations already retrieved at neighboring anchors.
- Decision: whether this reorder merits a further semantic causal experiment,
  rather than adding storage slots or claiming demand identification.

### Analogues and Branch Selection

- Closest analogues and distinctions are listed above. Shared relation
  computation transfers; human semantic identification does not follow.
- Alternative: `L1,L2,L3,R` plus the existing contextual GRU suffices.
- Selection criterion: two trained arms with exactly the same parameter family
  and one order change provide a bounded falsifiable comparison.

### Fixed Comparison

- Clean baseline product OID: 7adc7986379c913bd48be8afb8888024d008b852.
- Baseline config: `combined` representation with feedforward 256, decoder hidden
  64, action embedding 16, interaction rank 16, bank width 64 and dropout zero.
  The serial chronological bank is numerically equivalent to that control.
- Baseline checkpoint/run/value: none; matched fresh initialization. Held-out
  baseline values and group uncertainty will be measured by the run. Software
  rehearsal scores are not baseline research evidence.
- Dataset: pinned annotation revision and split consumed by PreparedCorpus;
  local index `artifacts/indexes/beatmap_index_4k.parquet`, SHA256
  `2d2814c3b3ec47cd1247e8d50cef555e12ace7eae60a79944102926c142c2d8e`.
  All eligible training components and every valid distinct beatmap participate,
  including annotation-connected training sources and source-cache-only charts.
  `train_group_limit=null`; metadata-connected validation/test components remain
  reserved. Validation arrangements are processed first; exact source/arrangement
  duplicates and invalid source replay are excluded with explicit reasons.
  Original human sections do not create extra pretraining units. The six known
  held-out-connected annotation training groups are excluded from human fitting.
- Sampling risk: uniform song, beatmap within song, full-source window start,
  feasible target size in {4,16,64}, then target start. Contexts have min(N,256)
  source events, so charts with 4–255 events remain eligible. There is no fixed
  pair of training windows. Every arm shares the draw, exact original LN entry
  facts and all three views. Log the joint draw probability without importance
  weighting; longer charts and overlapping window positions have different
  marginal per-event exposure. A 16-source LRU bounds resident parsed objects.
- Evaluation slice: 64 hash-selected eligible validation groups, one beatmap
  and up to two fixed windows per group, then up to two targets per feasible
  scale/group. Exact source hashes, allocation dispositions, eligible window-start
  ranges, population identity and block manifest are written before updates.
  All eligible original human train/validation cells fit/evaluate frozen readers.
- Intervention: exchange R and L3, preserving every initial tensor and all six
  chronological reader positions. Both arms have 341,233 parameters.
- Instrumentation: typed Hydra projection, guards, snapshots, source identities,
  existing matched structure/path/human metrics and separate frozen structural
  ridge diagnostics. Synthetic inputs never update the action model.
- Files: source_action_modeling composition/experiment/structural_probe modules,
  shared encoder helper extraction, evaluation callbacks, full source catalog
  and sampler, packaged preset and focused tests. Read-only context: source-action objective
  and time/local contracts; the frozen annotation adapter and human targets.

### Evidence and Decision Rule

- Primary: serial minus interleaved detailed mean-row NLL, averaged within each
  represented source group and then across groups. Positive favors interleaving.
  Use the paired group bootstrap, 1,000 samples, and keep seeds separate.
- Practical threshold: at least 0.02 nats/row and a positive bootstrap lower
  bound in each seed. This is an initial decision rule, not a convergence claim.
- Human guard: interleaved versus serial trained-readout macro NLL regression
  no more than 0.05; no individual concept regression more than 0.10 nats.
- Reuse evidence: interleaved trained versus untrained human macro NLL gain at
  least 0.02; at S4, average MSE across the four exact structural statistics on
  combination holdout at least 10% below both serial-trained and
  interleaved-untrained. Report orderless control and tempo/delta diagnostics.
- Qualitative check: preserve first-position NLL, detailed-context gains,
  per-concept and same-input Jack/Stream distinctions, and actual coupled changes
  in the factorial diagnostic. Inspect raw paired cases before explaining them.
- Positive: the primary, guard and reuse pattern all hold; supports a further
  causal test of the order mechanism on these inputs only.
- Negative: no repeatable practical prediction benefit, or harmful human reuse;
  prefer retaining the serial control for this budget.
- Ambiguous: prediction gain without reuse, synthetic-only gains, inconsistent
  seeds, exhausted bounds, or a weakly fitted readout. Do not name latent slots
  as semantics. Small path divergence alone is never sufficient evidence.

### Reproduction and Bounds

- Entry: `uv run --offline --extra mps python -m
  pulsefield_model.research.source_action_modeling.experiment_hydra
  output_dir=artifacts/source-action-composition/overnight-01`.
  Run from the product root; the output must not exist.
- Seeds: 17 and 29; identical initial tensors and sampled exposures across arms
  within each seed, shared validation manifest across seeds.
- Updates: 8 blocks times three views, AdamW 3e-4, weight decay 1e-4, gradient
  cap 1; at most 20,000 updates or 14,400 training seconds per seed, minimum 1,000.
  Both arms stop on the same update; no validation checkpoint selection.
- Frozen human readers: 600 steps, batch 4, learning rate 1e-3, matched sampling
  per seed. Structural readouts use fixed ridge 0.1 and per-stage moments.
- Environment: Apple M5, 24 GB unified memory, macOS 26.6.2, Python 3.10.20,
  PyTorch 2.11.0, MPS, one PyTorch CPU thread; local uv dependencies, offline.
- Bounds: 39,600 seconds overall; reserve 5,400 seconds finalization per remaining
  seed; 8 GiB MPS driver allocation, 12 GiB process peak RSS, 4 GiB outputs.
  RSS and MPS measures overlap; their sum is not a physical memory ledger.
- Snapshots: every 500 updates or 300 seconds, atomically replacing one latest
  snapshot per arm, plus common endpoints and readouts. Existing output paths
  are rejected. No automatic whole-workflow resume or network access.
  Sampler RNG, population identity and cumulative source coverage travel with
  checkpoints. Per-seed updates record realized windows; coverage.json reports
  actual unique songs/beatmaps. Membership is not a guarantee of exposure before
  the time endpoint. Cache eviction changes neither draws nor checkpoint replay.
- Kill: nonfinite gradients/loss, invalid data/identities, resource/global time
  violation, or fewer than minimum paired updates. Report incomplete and raise;
  partial runs do not satisfy the card.
- Confounders: support/optimization changes, decoder prefix use, imperfect song
  deduplication, two seeds, synthetic grammar assumptions, readout learning and
  correlated structural statistics, unequal marginal event exposure under
  hierarchical sampling, and annotation coverage/label uncertainty. Human labels
  fit only frozen readers; this comparison cannot establish a benefit from
  annotation-guided sampling or supervised encoder training. Factor holdouts apply to the diagnostic
  fitting set, not guaranteed absence from corpus pretraining.
- Conclusion scope: complete action rows with a supplied time skeleton and
  original human review inputs. No player-demand identification, internally
  causal semantic factorization, generation quality or timing generation claim.

## Result Log: composition-software-verification-a

- Accepted revision: none. Card association: source-action-relation-order-overnight
  revision 1; exploratory software check with reduced exposure and population.
- Source: baseline OID above plus dirty implementation, module snapshot digest
  `5dc0d9e6acf8adf710cf107e7e5dbb519f7d3619d1a7924a5778c0aac2df0a8d`.
- Run: `artifacts/source-action-composition/verify-20260914-a`.
- Command: the Hydra entry with `seeds=[17] local_train_groups=32
  local_validation_groups=4 validation_groups=4 max_updates=6 min_updates=1
  readout_steps=3 bootstrap_samples=20 path_pairs=3 checkpoint_every=3 log_every=1
  training_seconds_per_seed=300 finalization_seconds_per_seed=900 max_seconds=1800`
  and the run's fresh output path.
- Population SHA256: 7881b1471e24260fd830786f77bcf232737d7260aa0d8ace865ef4509b4030f5;
  983 training contexts in 459 groups, 145 validation contexts in 67 groups before
  selecting four prediction-evaluation groups. Human cells retain original scopes.
- Validation block manifest SHA256:
  d0b8462eaf8718c1ab50ea2215f0f737eed67dc79a9ef78785da39ba8dc2e995.
- Outcome: all training, latest/endpoint snapshots, structure, path consistency,
  factorial probes and trained/untrained human reads completed in 143.89 seconds.
  Six paired updates consumed 14.01 seconds including compilation and checkpoints.
  Later paired updates took approximately two seconds each.
- Peak observed MPS driver allocation: 1,918,287,872 bytes. Process peak RSS:
  2,314,862,592 bytes. Both are below the configured separate guards.
- Baseline research values, decision thresholds, human regression guards and
  architectural recommendation: not evaluated. This exposure is a software check.
- Plan deviations: materially smaller population, one seed and six/three update
  horizons; bootstrap 20. Source remains dirty. Subsequent guard callbacks and
  report additions require verification against their final source identity.
- Evaluation: executable stages and preliminary resource feasibility only.
  Apparent loss differences at six updates do not test the research hypothesis.
- Decision: pending research evaluation; no SUPPORTED recommendation.

## Result Log: composition-software-verification-b

- Accepted revision: none. Card association: revision 1, exploratory software
  verification with the full intended population and both seeds.
- Run: `artifacts/source-action-composition/verify-20260914-b`; baseline OID above
  plus dirty implementation. Exact module snapshot SHA256:
  `91fdbe822eeb6420e762b40d0924baeb0934f7b21eac48f8f6705f2a249a290b`.
- Command: canonical Hydra entry and default population/model dimensions with
  `seeds=[17,29] max_updates=2 min_updates=1 readout_steps=3 bootstrap_samples=20
  path_pairs=3 checkpoint_every=1 log_every=1 training_seconds_per_seed=300
  finalization_seconds_per_seed=900 max_seconds=2400` and the fresh run path.
- Population SHA256:
  `1ff7ba35aa8f2c49c69cca1f962aaac808a46b0a99daa98934f8d31c6a2c9507`.
  All 2,048 local training and 128 local validation groups loaded. Combined
  population: 5,015 training contexts/2,475 groups and 393 validation
  contexts/191 groups. Human fitting/evaluation retained 382/61 original cells.
- Fixed 64-group validation manifest SHA256:
  `8f1f57abd1001239b441456406fc4e107e1e025c3cb69c04f312673f17e6f179`.
- Outcome: both seeds completed paired training, latest/endpoint checkpoints,
  initial/final structure, path diagnostics, all trained/untrained structural
  and human probes, and cross-arm human comparison. Total 635.51 seconds;
  stop reason update-bound at two updates per arm/seed. No resume or overwrite.
- Peak observed MPS driver allocation: 1,547,386,880 bytes. Process peak RSS:
  3,914,743,808 bytes. These are separate process observations, not additive
  physical memory ownership. Other short MPS checks ran concurrently, so timing
  is not an isolated throughput benchmark.
- Verification: identical trained/untrained human sampled indices, finite
  structural probe MSE, all six stages plus the orderless control, 64 evaluation
  groups and required checkpoint/output files were checked directly.
- Plan conformance: full intended corpus and seeds, materially shortened
  training/readout exposure and bootstrap count. Later edits disable Hydra's
  incidental log file and clarify comments/docstrings; targeted configuration
  checks cover these changes. No research acceptance or superiority is inferred.
- Baseline/intervention research values and guard decision: not evaluated at
  this software-only exposure. Evaluation/Decision for the overnight hypothesis
  remain pending.

## Result Log: maximum-shape-resource-check

- Accepted revision and Card execution: none; additional exploratory software
  resource check on the same baseline plus dirty composition/model code.
- Output: `artifacts/source-action-composition/resource-20260914.json`.
- Procedure: synthesize a 256-row tap cycle at 100 ms, mask a 64-row block at
  event start 96, duplicate eight times with all three paired views, and execute
  one forward/backward/AdamW update per arm on MPS using the canonical model
  dimensions. Default initializer seed 17; no real-corpus semantic evaluation.
- Observations: finite losses and completed updates; driver allocations
  1,814,069,248/1,839,235,072 bytes for serial/interleaved and peak process RSS
  529,678,336 bytes. First compilation took 5.42 seconds versus 0.63 seconds for
  the subsequent arm; these are not comparative architecture timings.
- Interpretation: configured maximum prediction tensor shape executes below
  the declared memory guard. It does not measure all possible intermediate
  allocation peaks or sustained training behavior. No research recommendation.

## Software checks for Card revision 1

`uv run --offline --extra mps --group dev pytest -q
tests/research/source_action_modeling
tests/research/scoped_style_modeling/test_replay.py
tests/research/scoped_style_modeling/test_model.py tests/test_package_layout.py`
passed 138 tests and 20 subtests in 37.44 seconds. Coverage includes exact CPU
baseline behavior, equal parameter initialization, actual operator order,
mirror/padding/gradient contracts, checkpoint continuation, typed packaged config,
factorial holdouts, frozen parameters, failure reporting and output collisions.
Repeated identical MPS forwards differed by up to 9.54e-7, so MPS equality checks
allow 2e-6 absolute reduction roundoff while tensor/input and CPU checks remain
exact. `--help`, `--cfg job`, local documentation links and `git diff --check`
also completed. Product changes remain uncommitted; no push was performed.

## Result Log: full-corpus-software-verification-c

- Accepted revision: none. Card association: revision 2; full training population
  and both seeds, with short software-only training/readout exposure.
- Run: `artifacts/source-action-composition/verify-all-songs-20260914-c`.
  Baseline product OID above plus dirty implementation; saved module SHA256
  `48e4c9235d899aab97148bd390dafd3a3ddaca5e6c4231f67e3f2fc80e43a779`.
- Command: canonical Hydra entry with `max_updates=2 min_updates=1
  readout_steps=3 bootstrap_samples=20 path_pairs=3 checkpoint_every=1 log_every=1
  training_seconds_per_seed=300 finalization_seconds_per_seed=900 max_seconds=3600`
  and the fresh output path. Default seeds 17/29, full candidate population,
  64 validation groups, batch sizes and model dimensions were retained.
- Population: 3,169 training groups, 11,564 distinct eligible training beatmaps,
  8,618,695 possible window starts; 435 validation candidate groups/1,652 beatmaps.
  The fixed 64 groups produced 124 source windows and 384 validation blocks.
  Human fitting/evaluation retained 382/61 original cells. All training candidate
  groups retained at least one eligible beatmap; six annotation training groups
  connected to held-out components were excluded from human fitting.
- Allocation dispositions across index and annotated source entries: 13,216
  eligible, 512 duplicate, 58 rejected, 1,448 held-out entries with no payload
  read. Rejections comprise 45 sources with fewer than four events, six original
  published-training sources in reserved validation components, and seven
  ambiguous/invalid original objects. No simultaneous close/head source was
  rejected in this corpus.
- Population SHA256:
  `8d84be31e416d35eabf0a83d04d642d2be32350a9f32df8b762155bd0918a62a`.
  Catalog SHA256:
  `7f61e06c5ee1be3838663ff97da77b3bcd55f9958a5f12cb7a780fbd05abf51f`.
  Validation manifest SHA256:
  `ab1fbc29facd651de714a94abb69b7734dc73bc24cf9861eb02f9d3f885d3cb4`.
- Outcome: completed all declared seeds, paired updates, latest/endpoints,
  initial/final action evaluation, path diagnostics, trained/untrained factorial
  and human readouts, and cross-arm human comparisons in 1,226.27 seconds.
  Required files, paired draw policy and identical human fitting indices between
  trained/untrained conditions were checked directly. Each seed drew only 16
  training blocks, covering 16 groups and 16 beatmaps. This is full-population
  loading and end-to-end execution, not an epoch over all beatmaps.
- Peak observed MPS driver allocation: 2,553,790,464 bytes; process peak RSS:
  1,074,708,480 bytes; outputs: 73,776,930 bytes. The old overnight run and some
  focused checks ran concurrently; elapsed time is not isolated throughput.
- Subsequent scoped changes restore unused historical pilot APIs, remove a
  sampler-specific type hint, record KeyboardInterrupt as incomplete, raise the
  default training cap to four hours per seed, and preserve supported simultaneous
  close/head lane actions in catalog ingestion. The last change does not change
  this measured population; a dedicated fixture verifies combined action 5.
  Configuration, interruption and corpus tests cover these final changes.
- Research evaluation/decision: pending. Two action updates and three human-head
  updates do not establish convergence, semantic benefit, order superiority or
  sustained overnight stability. No SUPPORTED conclusion or lifecycle change.

## Result Log: original-human-context-resource-check

- Accepted revision: none; software resource check for original annotation inputs.
- Output: `artifacts/source-action-composition/human-resource-all-songs-20260914.json`.
- Procedure: inspect every eligible original human input, select the four largest
  distinct training contexts (human cohort indices 452, 321, 158, 493), and execute
  a frozen encoder forward plus concept-reader backward/AdamW update in each
  arm on MPS, batch four, canonical dimensions and initializer seed 17.
- Largest padded timeline: 736 positions, including boundary markers. No human
  scope or context was truncated to the pretraining 256-source-event cap.
- Both arms completed with finite loss and no encoder gradients. Observed driver
  allocations were 152,895,488/148,701,184 bytes; peak RSS 1,183,776,768 bytes.
  This supports the declared human batch shape, not semantic learning benefit.

## Execution handoff: manual full-corpus overnight run

- The owner explicitly retained manual execution of the full caffeinate command.
  No revision-2 overnight run was launched by the agent.
- The prior `artifacts/source-action-composition/overnight-01` process was stopped
  at 2026-09-14 13:26:25 UTC after preserving an aligned seed-17 checkpoint pair
  at update 936. The last fully logged paired update was 999. Its report is
  marked incomplete with a separate `interruption.json`; the original directory,
  logs, latest snapshots and retained checkpoint copies remain available.
- Manual command, from the product root: `caffeinate -i uv run --offline --extra
  mps python -m pulsefield_model.research.source_action_modeling.experiment_hydra
  train_group_limit=null
  output_dir=artifacts/source-action-composition/overnight-all-songs-01`.
  The destination must not exist. Both seeds start from their paired fresh
  initialization; the old population's checkpoints are not resumed into this risk.

## Software checks for Card revision 2

`uv run --offline --extra mps --group dev python -m pytest -q
tests/research/source_action_modeling
tests/research/scoped_style_modeling/test_replay.py
tests/research/scoped_style_modeling/test_model.py tests/test_package_layout.py`
passed 144 tests and 20 subtests in 26.37 seconds on the final product worktree.
This includes full-corpus split isolation, source-cache-only training charts,
multiple charts per song, dynamic windows, exact draw probabilities, cache
eviction/replay, combined close/head actions, snapshot continuation, MPS model
contracts, frozen probes, Hydra projection and interruption reporting. The final
`--cfg job`, local documentation links and `git diff --check` also passed.
Product changes remain uncommitted; the scoped note update is committed locally
on agent-notes. No product commit, remote push or formal acceptance is implied.

## Next lifecycle condition

Commit and review the recoverable product implementation and exact proposed card
before revision-specific acceptance. Overnight results must retain source/input
identities, paired endpoints, per-seed metrics and deviations. A short software
check does not complete the proposed research experiment or accept its direction.


## Result Log: overnight-all-songs-01-evaluation-20260915

### Experiment and reproduction

- Owning Note: 2026-09-14-source-action-relation-composition. Accepted revision:
  none. Associated Card: source-action-relation-order-overnight revision 2,
  still proposed. The owner requested interpretation of the completed run and
  priorities for investigating networks and sampling.
- Run: `artifacts/source-action-composition/overnight-all-songs-01`; both arms
  started fresh under the manual full-corpus command recorded above. Saved
  configuration matches the full population, seeds 17/29, eight blocks per
  update, three views, 600 frozen-human-reader steps and 1,000 bootstrap draws.
- Recorded launch revision: `7adc7986379c913bd48be8afb8888024d008b852`, dirty.
  All saved source-action Python modules pass their saved SHA256 checks and
  exactly match the corresponding files in clean product commit
  `e8226324c6f05ffa28dc4ee435e98935e799496c`. This recovers those modules; it
  does not retroactively establish a clean launch or Card acceptance. Saved
  source aggregate SHA256:
  `637540f51abc3fc7e50b4a92e9a1811e45c76b2f1a182d39ff0b0f31ec38f4f7`.
- Population SHA256:
  `8d84be31e416d35eabf0a83d04d642d2be32350a9f32df8b762155bd0918a62a`.
  Training: 3,169 groups, 11,564 beatmaps, 11,522,113 source events. Action
  evaluation: 384 paired blocks from 64 groups. Human fitting/validation:
  382/61 cells; validation contains 18 groups. Tech has six validation cells,
  including only one positive and no supporting example.
- Environment: saved report records MPS, macOS 26.6.2 arm64, Python 3.10.20,
  PyTorch 2.11.0. The proposed environment identifies the owner's M5/24 GB host;
  the report does not independently identify its processor model.
- Status completed. Total 31,277.49 seconds (8.69 hours), below 39,600 seconds.
  Seeds 17/29 completed 4,467/4,196 paired updates in 14,297.03/14,400.01
  training seconds. Both stopped at their common time endpoint, without
  validation checkpoint selection. The dynamic reservation set seed 17's
  training budget to 14,295.26 seconds; checking between updates permits the
  small overshoot. Update counts differ across seeds, not within a pair.
- Maximum observed MPS driver allocation 2,667,495,424 bytes; process peak RSS
  5,913,264,128 bytes. Both are below their separate guards. They overlap in
  unified memory and must not be added. The runner reports completed rather
  than a resource stop. Outputs were retained; no resume or overwrite was used.
- Evidence: report/configuration/source snapshot, allocation, update logs,
  initial/final structure, four endpoint checkpoints, trained/untrained human
  and relation reports, fitted readouts, path reports and human comparisons.
  `analysis-20260915/summary.json` under the run records their hashes and compact
  derived statistics; `audit.py` reproduces the summary using the repository
  Python environment, NumPy and PyArrow. No neural forward or training is part
  of this post-hoc calculation.

### Results

NLL is lower-is-better. Action values are detailed-view mean-row NLL, averaged
within song group then across groups. Human values are the frozen reader's
human concept/group macro three-class NLL.

| Seed | Serial action NLL | Interleaved action NLL | Serial minus interleaved, paired group 95% interval | Serial human NLL | Interleaved human NLL |
| --- | ---: | ---: | --- | ---: | ---: |
| 17 | 2.047442 | 2.064200 | -0.016757 [-0.033748, 0.000332] | 0.964306 | 0.878661 |
| 29 | 1.984148 | 1.998581 | -0.014433 [-0.029268, 0.000251] | 0.893605 | 0.942422 |

Initial action NLL was about 4.50–4.56. Both arms learn prediction; neither seed
meets the proposed interleaved gain of 0.02 with positive interval lower bound.
Both intervals include zero. They measure source-group uncertainty conditional
on these checkpoints and do not measure training-seed uncertainty.

The human order advantage changes sign. Interleaved trained versus untrained
macro gain is +0.040699 at seed 17 and -0.038117 at seed 29. Interleaved Tech
NLL exceeds serial by 0.287494/0.469024, violating the proposed 0.10 per-concept
guard in both seeds; seed 29 Trill also regresses by 0.167735. Macro regression
at seed 29 is 0.048817, narrowly within the separate 0.05 macro bound. The
single Tech positive prevents a broad conclusion about Tech recognition.
Same-input Jack/Stream joint presence accuracy is 7/12 versus 7/12 at seed 17,
and 6/12 versus 7/12 at seed 29 (serial versus interleaved); matched untrained
readers score 6/12 by predicting both present. Preserve the individual
combinations rather than interpreting overall joint accuracy as selectivity.

The structural aggregate conceals a specific contrast. At S4, combination
holdout mean MSE across four statistics is 0.068927/0.063586 at seed 17 and
0.063207/0.058044 at seed 29 (serial/interleaved). The improvements are only
7.75%/8.17%, below the proposed 10%. Interleaved untrained S4 MSE is
0.033981/0.042807, better than the trained aggregate in both seeds.
However, three local statistics improve strongly:

| S4 combination-holdout MSE | Serial 17 | Interleaved 17 | Serial 29 | Interleaved 29 |
| --- | ---: | ---: | ---: | ---: |
| Adjacent same-lane recurrence | 0.003094 | 0.000924 | 0.009476 | 0.000441 |
| Two-step return without repeat | 0.002965 | 0.001078 | 0.011830 | 0.000607 |
| Adjacent hand transition | 0.004352 | 0.000392 | 0.002930 | 0.001684 |
| Lag-16 lane match | 0.265297 | 0.251952 | 0.228594 | 0.229442 |

Lag-16 alone accounts for 90.4%–99.1% of these models' aggregate S4 squared
error. Its training-mean prior MSE on the same holdout is 0.046875, substantially
better than either trained probe. On held-out paces, changing lane-to-hand
assignment leaves true lag-16 agreement unchanged but changes the trained S4
prediction by mean absolute 0.3593/0.2935 at seed 17 and 0.3290/0.2867 at seed
29 (serial/interleaved). The readout responds to irrelevant changes for this
statistic. This is a concrete failure of probe transfer, not proof that the
encoder has erased all long-range information. The three local statistics and
the untrained controls rule out describing the entire representation as failed.

Detailed context gives small, inconsistent gains over near context. The
near-minus-detailed mean-row NLL is -0.009723/-0.020720 at seed 17 and
+0.016928/+0.006803 at seed 29 (serial/interleaved). First-position detailed
NLL is 2.15–2.21 versus later-position 1.95–2.04; this is descriptive evidence,
not a causal estimate of decoder-prefix reliance.

The 24 fixed-information path pairs give equal-pair mean TV 0.0956–0.1065
and JS 0.0114–0.0163 nats. Mean true-target NLL changes under moving the same
prefix between encoder observation and decoder history vary in sign/magnitude.
These are routing comparisons on the fixed manifest prefix, not rollout quality
or broad evidence of route independence. They cover the first four validation
groups because the manifest has six blocks per group.

### Realized sampling and optimization

| Exposure per seed | 17 | 29 |
| --- | ---: | ---: |
| Block draws | 35,736 | 33,568 |
| Unique groups | 3,169 (100%) | 3,169 (100%) |
| Unique beatmaps | 10,267 (88.78%) | 10,131 (87.61%) |
| Target-event exposures before three-view replication | 994,116 | 940,468 |
| Unique target events | 909,641 (7.89% of population events) | 862,834 (7.49%) |
| Unique events inside sampled context windows | 5,092,970 (44.2%) | 4,918,493 (42.7%) |

Event coverage merges half-open intervals within each source SHA, using original
window offset plus target offset. Context-window coverage includes hidden
positions and must not be presented as visible-action coverage. Training labels
and gradients can be reused across overlapping draws; raw draws are not
independent examples or corpus epochs.

Each scale receives about one third of block draws. Median target duration is
2.30/2.32 seconds; the 90th percentile is 12.39/12.86 seconds; the full observed
range is 58 ms–66 seconds. A fixed event-count scale is not a fixed physical-time
or musical-phrase scale. Across all beatmaps, median draw count is two, 90th
percentile six, maximum 24; more than 10% are never drawn. Uniform song then
beatmap gives long charts and charts in large sets lower per-event exposure.
Uniform overlapping window/target starts also give edge events a different
marginal probability. Annotation sections receive no special pretraining weight.

Almost every update clips the global gradient norm at 1: 99.98% at seed 17,
100% at seed 29. Last-500 median pre-clip norms are about 3.63/3.66 and
2.99/3.05 (serial/interleaved). This does not establish an optimization defect;
module norms, actual Adam displacement and loss response are still absent.
Last-500 versus preceding-500 training losses fall by about 0.008 at seed 17
and 0.062–0.064 at seed 29. Random-minibatch averages do not establish held-out
convergence. Only initial/final action evaluations are available in this run;
latest checkpoints replace earlier snapshots.

### Plan conformance and evaluation

The full population, paired settings, evaluation families and bounded time
endpoints conform to the proposed revision-2 procedure. Sequential update
numbers, target bounds, reported block/source coverage, matching human fitting
indices across all four conditions per seed, and saved source hashes were
recomputed successfully. The formal source/acceptance conditions do not conform:
the launch was dirty and no accepted Card revision exists. The log remains
exploratory, with unchanged Card and Note status.

Observation: prediction learns, the proposed reorder fails the joint practical
criterion, three local structural probes benefit, and lag-16 transfer and human
reuse remain weak or unstable. Strong alternatives are a restrictive stage-moment
ridge readout, specialization to local conditional prediction, limited target
exposure, source-to-synthetic distribution shift, and human readout support or
optimization. None is isolated by the present order comparison.

### Decision and next discriminating work

Recommended outcome: **REFINE**. Keep serial as the reference and retain
interleaved as a candidate for its local-structure benefits. These are research
recommendations, not adoption or a lifecycle change. Prioritize:

1. **Separate accessible information from actual network use.** On saved
   checkpoints, test U/S1/S2/S3/S4/H individually and all-except-one with matched
   frozen-reader fitting. Fit linear and small nonlinear readouts with the same
   training-only selection rule and untrained/orderless controls. Analyze each
   exact relation separately, especially lag 16, and original human cases.
   Do not select probe capacity on the existing held-out outcomes. The closest
   methodological analogue is [Hewitt and Liang's probe controls](https://aclanthology.org/D19-1275/);
   task-specific controls must distinguish what the readout learns itself.
2. **Intervene on the paths used by the trained predictor.** Compare source
   projection/time/row interaction, conditioned local kernels and raw-source
   skips, relation attention R, contextual BiGRU H, ActionReader, and decoder
   GRU/bilateral interaction. Start with L3/R/H and reader access because the
   order and context results motivate them. Bypass or replace one computation
   while recomputing every dependent later state; separately mask only its
   direct bank-read route. Otherwise retained early states and source skips
   can confound the apparent removal. Use paired examples, legal occupancy,
   first/later/scale NLL and held-out semantic outputs. Distribution-matched
   replacements and readout refitting qualify zero-ablation artifacts.
   No single intervention ranks a module's universal necessity.
3. **Audit both sampling processes.** For training, cross source/set size,
   event count, physical duration, chart position, chord/LN mix and annotated
   section overlap against realized exposure and error. Evaluate on both the
   declared group-macro risk and a clearly separate event/duration-weighted
   population. A subsequent controlled training comparison could alter just
   one sampling distribution while preserving paired initialization, effective
   update exposure and evaluation; do not silently importance-weight the
   original estimand into another target population. For decoding, compare
   teacher-forced and self-fed continuation over 4/16/64 rows on the same
   supplied time skeleton, with repeated draws and a small fixed temperature
   set. Track error growth, lane/chord/LN occupancy and release behavior,
   recurrence, diversity, and joins to visible continuation. Legal masking
   can guarantee row legality without guaranteeing musical coherence or
   boundary compatibility. Timing generation is outside this model's output.
4. **Resolve local-versus-long-range dependence.** Keep local rows and supplied
   timing fixed while perturbing distant recurrence, then inspect output and
   stage response by decoder position and scale. Select representative path
   cases across groups and durations instead of only the first 24 manifest
   blocks. Internal swaps require matched source facts and an explicit
   hypothesized relation; [Geiger et al.](https://proceedings.mlr.press/v162/geiger22a.html)
   provide a causal-interchange analogue, but their training intervention is
   not part of the current model or this proposed read-only diagnosis.
5. **Check optimization after localizing the failure.** Record per-module
   pre-clip norms, parameter-relative Adam displacement, fixed-batch response,
   and equal-update validation curves before widening networks or lengthening
   runs. The existing `diagnostics.py` gradient-dot-displacement helpers can
   guide instrumentation; they are a local linearization, not causal importance.
   Attention maps likewise require independent intervention evidence, following
   the caution in [Jain and Wallace](https://aclanthology.org/N19-1357/).

Begin with saved-output error slicing, then a bounded frozen-checkpoint module
and rollout diagnostic. Design any new training comparison separately after
these observations distinguish the hypotheses. New run execution, acceptance,
and product adoption remain pending human research direction.
