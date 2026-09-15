# Agent Note: Locate source-action information and prediction dependence

Note ID: 2026-09-15-source-action-network-diagnostics
Status: proposed
Kind: research
Created: 2026-09-15
Updated: 2026-09-15
Product revision: e8226324c6f05ffa28dc4ee435e98935e799496c plus scoped diagnostic implementation
Scope: Frozen local/relation stages, contextual H and ActionReader access in the completed composition run
Related: 2026-09-14-source-action-relation-composition

## Question and authority

The owner requested investigation of the first three network priorities and
explicitly requested experiment code and human-facing visualization. This
covers the scoped implementation and bounded frozen-checkpoint investigation
needed to make those diagnostics inspectable. It does not accept a Note
revision, authorize action-model pretraining, or select a product architecture.
The record remains proposed and execution from an uncommitted diagnostic
worktree remains exploratory. The original order-comparison Card and evidence
remain in their existing owner; this note owns a different measurement task.

The completed run is
`artifacts/source-action-composition/overnight-all-songs-01`. Its action NLL
falls from approximately 4.5 to 2.0 without a reliable order benefit. S4 local
structural readouts improve under interleaving, while lag-16 agreement dominates
the aggregate failure. Human reuse changes sign across seeds. The open branches
are information loss, loss through orderless pooling, ineffective downstream
access, and genuinely weak dependence on distant information.

## Experiment Card: source-action-network-paths

### Identity and authority

- Card ID: source-action-network-paths. Revision: 1.
- Owning Note: 2026-09-15-source-action-network-diagnostics.
- Accepted revision: none. The owner's implementation/investigation request
  supplies task authority, not a formal acceptance transition.

### Hypothesis and alternatives

The stage bank may preserve temporal relationships that pooled readouts lose;
the trained predictor may rely strongly on H through routes omitted by a
memory-only ablation. The direct alternative is that richer temporal summaries
do not rescue lag-16 and that removing the suspected routes has little effect.
These branches determine whether subsequent work should target representation,
readout/access, or the learning objective/exposure.

Closest analogues: [controlled probes](https://aclanthology.org/D19-1275/),
[interchange interventions](https://proceedings.mlr.press/v162/geiger22a.html),
and [attention interpretation limits](https://aclanthology.org/N19-1357/).
This is a diagnostic adaptation to existing action networks, not a novel
architecture or a claim that internal states identify human concepts.

### Fixed comparison

- Clean implementation baseline: e8226324c6f05ffa28dc4ee435e98935e799496c.
- Checkpoints: serial/interleaved endpoints at seeds 17/29 from the completed
  run, at updates 4,467/4,196. Their per-file hashes and all input identities
  are recorded by the diagnostic runner. The original launch was dirty;
  source-action module snapshots match the clean baseline's corresponding files.
- Baseline evidence: original 64-group/384-block detailed mean-row NLL is
  2.047442/2.064200 at seed 17 and 1.984148/1.998581 at seed 29
  (serial/interleaved). Smaller diagnostic-slice baselines are recomputed and
  must match their original saved block NLL within 1e-4 before intervention.
- Real slice: first 16 groups under the fixed mechanism-v1 group hash order,
  one identity-selected original block at each available 4/16/64 scale. Rebuild
  original recorded validation windows and detailed views, including summaries
  and decoder entry; no test sources or corpus training pass.
- Structural stimulus: all 320 original factorial tap examples. Fit 96
  even-parity cases at 72/120/200 ms; assess 96 odd-parity cases and 128 fresh-pace
  cases at 90/160 ms. Four exact statistics, independently reported.
- Frozen readout comparison: 128 mean/second-moment features versus 320 features
  additionally containing same-channel products at lags 1/2/16. Standardization
  and ridge 0.1 use only fitting cases. Repeat with matched untrained encoders
  and the orderless fitting-mean control. No hyperparameter selection.
- Input stimulus control: 24 identity-selected distant pairs. Shuffle only one
  16-event region at positions 0–15 or 48–63. Target source positions 28–31,
  their seven-event halo, all timestamps, lane counts, and target actions stay
  fixed. Taps avoid LN legality changes. Derived recurrence and attack-history
  metadata are recomputed. Input perturbations are shared by all model variants.
- Internal interventions: each contrast removes/replaces one declared route or
  operator; retain a separate baseline per input condition. Memory-only singles
  and leave-one-out; H query, residual and the combined direct H context;
  projected read removal; identity bypasses L1/L2/L3/R/H. Normal forward
  recomputes descendants. For distant pairs, replace H memory/query/residual
  independently or together, and repeat edits with R/H bypass in both members.
- Important confound: ActionReader always uses H for its query and residual.
  A memory-only U or without-H condition retains both. H itself also receives
  observation metadata directly. Captions must identify which route changed.
- Instrumentation: checkpoint-only loading without optimizer/RNG restoration,
  state fingerprints, paired normalized-distribution metrics, case manifests,
  resource guards, offline report. Reader defaults and checkpoint policy remain
  unchanged; original NLL reproduction guards this assumption on real data.

### Measurements and decision rule

- Primary screening metric: changed-minus-baseline mean-row NLL, within group
  then across groups, for each isolated real-input intervention. Bootstrap 500
  paired groups. Positive means impaired prediction under intervention.
- Priority signal: at least 0.02 nats/row and positive interval lower bound in
  both seeds for the same route. This is an exploratory screening threshold,
  not a familywise-corrected significance declaration or universal necessity.
- Structural companion: per-statistic held-out MSE and prior-relative changes
  by stage and feature family. A lagged-summary benefit cannot separate its
  explicit lag bias from added feature capacity. Preserve fitting error and
  untrained controls rather than requiring only a favorable aggregate.
- Remote companion: eventwise RMS and layer-RMS-normalized change, query-region
  response, mean TV/JS and true-target NLL for each path replacement. Remote
  edits have no prescribed target improvement, so NLL sign is not semantic
  correctness. Hybrid activations may be off distribution.
- Guards: unchanged model weights/buffers and absent gradients; original baseline
  block NLL within 1e-4; identical target/supplied skeleton/legality in paired
  effects; finite normalized distributions; explicit fresh output and budgets.
- Qualitative check: inspect original/edited four-lane timelines, fixed target
  halo, activation traces, and original real target blocks with per-row losses.
- Negative result: targeted route has negligible effect and richer readout does
  not rescue the failed relation. Refine toward objective/exposure controls.
- Ambiguous: seed disagreement, only off-distribution changes, readout-family
  dependence, or an unchanged control responding. Preserve these distinctions.

### Reproduction and bounds

- Command from product root: `uv run --offline --extra mps python -m
  pulsefield_model.research.source_action_modeling.mechanism_hydra
  output_dir=artifacts/source-action-composition/mechanism-20260915-a`.
- Seeds 17/29, both orders. Batch eight, 16 validation groups, 24 remote pairs,
  ridge 0.1, 500 bootstrap draws, one CPU thread.
- Environment: existing Python 3.10.20/PyTorch 2.11.0 MPS environment on the
  owner's Apple Silicon machine; record actual versions/platform in outputs.
- Limits: 1,800 wall seconds, 6 GiB MPS driver, 10 GiB peak RSS, 256 MiB output.
  Memory guards overlap; checks cannot preempt a kernel. No network or training.
- Destination must be fresh; no resume or overwrite. Stop on guard failure,
  nonfinite predictions, identity/parity failure or parameter mutation; retain
  an incomplete report and raise.
- Outputs: configuration, source copies/patch/hash manifest, report.json and
  self-contained offline index.html. Renderer uses no CDN or external fetch.
- Conclusion scope: frozen action networks with supplied timing, exact synthetic
  statistics and a small original validation slice. Human concepts, generative
  sampling, player demand and architecture adoption remain outside this Card.

## Implementation and software verification

Implementation owners are mechanism.py, mechanism_cases.py, mechanism_experiment.py,
mechanism_config.py, mechanism_hydra.py and mechanism_report.py in source_action_modeling,
the packaged source_action_mechanism.yaml, ActionReader's optional diagnostic
arguments in representation.py, and focused mechanism tests. The operator and
measurement contract is documented in docs/research/source_action_mechanism_diagnostics.md.

The first real-checkpoint software run is
`artifacts/source-action-composition/mechanism-verify-20260915-a` with seed 17,
two real validation groups, four distant pairs, the full structural stimulus and
600-second cap. It completed in 42.37 seconds. All baseline-score parity and
frozen-state guards passed. This small run verifies execution and report data;
it is not the declared 16-group/two-seed investigation.

The initial focused selection (mechanism, representation and composition tests)
passed 21 tests in 12.68 seconds. Configuration composition and --help worked.
Final outgoing checks and visual inspection remain to be recorded after the
report is inspected. No lifecycle change or remote push is authorized.
