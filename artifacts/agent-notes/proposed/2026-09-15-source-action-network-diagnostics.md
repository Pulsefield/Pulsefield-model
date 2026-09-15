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

- Card ID: source-action-network-paths. Revision: 2.
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
- Numerical/graph controls: repeat the unchanged input and report its query
  RMS and prediction changes; record R's direct query-neighbor source positions
  on both observed graphs. Direct endpoints do not bound all information that
  local operators or H metadata can transmit.
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
  output_dir=artifacts/source-action-composition/mechanism-20260915-b`.
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

### Result Log: mechanism-20260915-a

#### Experiment and Reproduction

- Owning Note: 2026-09-15-source-action-network-diagnostics; accepted revision:
  none. Card: source-action-network-paths revision 1. Exploratory execution is
  within the owner's request to investigate these three paths and write code.
- Baseline source: e8226324c6f05ffa28dc4ee435e98935e799496c. Intervention:
  that OID plus uncommitted diagnostic files; source manifest SHA-256
  1d42f0e8c93b15fa59299895a73df6840894b91b9fe2a6b1197939c60c359a11.
- Baseline run: overnight-all-songs-01; checkpoints, input identities and
  original per-block scores verified by the runner. Seeds 17/29, both orders;
  four trained endpoints and four matched untrained controls.
- Procedure: the Card's packaged Hydra command with output_dir changed to
  artifacts/source-action-composition/mechanism-20260915-a; fresh destination,
  no overwrite or resume. Exact resolved config, environment, checkpoint hashes,
  copied sources, report.json and index.html are in that directory.
- Slice: 48 blocks from 16 original validation groups, all 320 structural cases,
  24 distant pairs. Case manifest SHA-256
  e616e2c2916b6035538574555ee2d33fba54204033ca5a845c02756eb7e7008f.
- Budget: 194.446 seconds; sampled peak MPS driver 1,205,764,096 bytes;
  peak RSS 586,498,048 bytes. All work completed within declared bounds.

#### Results

Mean-row NLL changes use equal group means and 500 paired group resamples.
H residual removal increased NLL by 1.036–1.652 nats/row; H bypass by
0.605–0.858; R bypass by 0.135–0.202. All four checkpoint intervals had positive
lower bounds for each of these interventions. Removing H from memory alone
changed NLL by -0.0011 to +0.0204 with every interval crossing zero. The complete
baseline and intervention values and intervals are preserved in report.json;
the final result log will use the report with numerical controls.

The original-score parity bound (1e-4), state fingerprint, absent-gradient and
common-legal-support guards passed. Visual inspection exposed a report-only
defect: changing aggregate target scale left an unrelated-scale case selected.
The corrected renderer filters case choices and keeps the selected intervention
inside the displayed category. No action-model weights changed.

#### Plan Conformance

Revision 1's populations, interventions, ridge penalty and bounds were followed.
R target responses near 1e-8 motivate an unchanged-input numerical control;
graph inspection shows hidden target rows reset action-dependent relation chains.
Revision 2 adds these diagnostics and a fresh output destination. The added
procedure/measurements require the revision increment, while the existing
causal comparisons and screening threshold remain unchanged. Rendering also
adds pair averages, small-number display and separate LN event glyphs.

#### Evaluation and Decision

The strongest current dependence is through H residual/context and R
computation. Off-distribution zero/identity interventions limit architectural
interpretation. Weak response to this particular distant stimulus does not
exclude long-range graph use on other inputs. Recommend REFINE: complete the
same bounded investigation with the numerical and graph controls, then interpret
stage readout, path dependence and remote sensitivity separately. This is an
evidence append and proposed Card revision, with no acceptance or lifecycle
transition. Subsequent training or a different experiment requires its own scope.

### Result Log: mechanism-20260915-b

#### Experiment and Reproduction

- Owning Note: 2026-09-15-source-action-network-diagnostics; accepted revision:
  none. Card: source-action-network-paths revision 2. Status remains proposed.
- Source baseline: e8226324c6f05ffa28dc4ee435e98935e799496c. Diagnostic source:
  that OID plus the uncommitted scoped implementation; copied-source manifest
  SHA-256 839190bcd6759d32e0e0b37c3b58c1a40ba9b7cf65a7838ac18e4773062572a2.
- Baseline: original overnight-all-songs-01 endpoints and detailed validation
  scores. The run uses the revision 2 command above and records each checkpoint
  hash and update, all configuration fields, environment and source copies.
- Outputs: artifacts/source-action-composition/mechanism-20260915-b/report.json
  and index.html. Fresh destination, no overwrite or resume; no optimizer/RNG
  restoration. Seeds 17/29, both orders, trained/untrained controls, 48 real
  blocks from 16 validation groups, 320 structural cases, 24 remote pairs
  (12 before, 12 after). Case manifest SHA-256
  e1f1aa89c2d38d9e75e84a5281b2f74561c52fd59f51d1fbe427377086cc6f19.
  The identity change from run A adds graph metadata; selected inputs are unchanged.
- Environment: Python 3.10.20, PyTorch 2.11.0, macOS 26.6.2 arm64, MPS.
- Budget: 213.638 seconds; peak RSS 586,612,736 bytes; sampled peak MPS driver
  1,205,764,096 bytes; output 20,381,622 bytes. Stop: completed all eight model
  conditions within bounds.

#### Results

Values below are nats per target row, macro-averaged within group then across
16 groups. Brackets are 95% paired group-bootstrap intervals from 500 draws.

| Contrast | 17 serial | 17 interleaved | 29 serial | 29 interleaved |
| --- | --- | --- | --- | --- |
| Original slice NLL | 2.178 [1.880, 2.474] | 2.177 [1.875, 2.471] | 2.188 [1.851, 2.516] | 2.227 [1.898, 2.554] |
| H residual zero: delta | 1.098 [0.821, 1.373] | 1.652 [1.214, 2.109] | 1.036 [0.799, 1.280] | 1.103 [0.834, 1.392] |
| H identity bypass: delta | 0.636 [0.442, 0.838] | 0.605 [0.411, 0.791] | 0.858 [0.597, 1.126] | 0.651 [0.427, 0.863] |
| R identity bypass: delta | 0.160 [0.089, 0.243] | 0.202 [0.113, 0.301] | 0.135 [0.080, 0.188] | 0.171 [0.101, 0.251] |
| H memory removal: delta | -0.001 [-0.016, 0.011] | 0.009 [-0.002, 0.020] | 0.008 [-0.0003, 0.018] | 0.020 [-0.002, 0.046] |

H residual removal, H bypass and R bypass meet the exploratory priority rule
in both seeds and orders. L1 and L3 bypass meet it in serial but not consistently
in interleaved; L2 does not. No individual memory-removal route meets the rule
consistently across seeds. Removing the complete projected attention read changes
NLL by 0.040–0.159; only 17/interleaved excludes zero. Redundant or distributed
access remains an alternative to a universally unnecessary reader.

R bypass increases first-position NLL by 0.356–0.564, versus 0.104–0.156 at later
positions. These are teacher-forced comparisons: later predictions receive true
action prefixes, so this is not free-running error accumulation.

On held-out factor combinations, adjacent recurrence MSE drops from approximately
0.049–0.058 at U to 0.00032–0.00211 at S1 with moment readouts. Local facts are
accessible early. H lag-16 moment MSE is 0.206–0.232; adding lagged features
reduces it to 0.060–0.084, still worse than the 0.046875 fitting-mean prior.
U with lagged features reaches 0.0081–0.0128, while untrained H lagged controls
reach 0.0073–0.0381. This readout helps but does not establish convincing lag-16
access from late trained states. Other facts, pace/fitting splits and individual
predictions are preserved; feature-family effects are not uniformly beneficial.

Across remote pairs, trained H query-relative RMS is 0.00292–0.00584, versus
approximately 3.7e-8–6.9e-8 in unchanged-input repeats. Full-edit TV averages
0.00207–0.01166, versus 1.7e-7–2.6e-7 for repeats. Replacing all H routes gives
TV approximately 0.00075–0.00118; retained-memory paths also transmit remote
changes. These are sensitivities, not additive path attributions or proof of
semantic use. Every observed R target graph directly reads events 27–32;
serial R's approximately 1e-8 target response must be interpreted with the
numerical control and missing direct remote edges. H's direct metadata input
remains a separate possible source of its response.

Every selected original block NLL score reproduced within 1e-4. State fingerprints,
absent gradients, fixed paired targets and common finite legal-distribution
support passed. Raw output also retains JS, sequence, first/later and per-row
effects; bootstrap intervals cover groups, not training-seed uncertainty.

#### Plan Conformance and Verification

The revision 2 comparisons and bounds were followed. After this completed run,
the product runner gained failure handling that removes index.html after a
post-render guard failure and announces completion only after rendering and
the final bounds check. This changes no completed comparison. Preserved copied
sources remain the execution authority for run B.

The mechanism/representation/composition selection passed 23 tests. After the
failure-path addition, all eight mechanism tests were rerun, giving 24 distinct
research tests in the final selection. Package layout passed one test with 20
subtests; --cfg job and --help succeeded. Relative documentation links resolve;
product/note whitespace checks pass. Coverage includes independent H routes,
downstream recomputation, controls and ridge isolation, configuration, embedded
HTML safety, and incomplete reports after time and post-render output limits.

Browser inspection verified three tabs, seed/order/case/level switches, scale
and intervention filtering, R-neighbor text, numerical notation, four-lane/LN
markers and per-position losses. The final report loads without console errors.
No product commit, remote push or CUDA execution belongs to this task.

#### Evaluation and Decision

Recommend REFINE. Early local readability, weak late lag-16 readout, and strong
dependence on H context/residual and R computation are distinct observations.
Zero/identity changes can disrupt trained activation distributions; the small
validation slice, two seeds, synthetic taps and unequal readout capacity limit
generalization. Screening intervals are not familywise corrected. No architecture
adoption or universal module necessity follows from these results.

Next discriminate H's recurrent predecessor input from direct metadata and
residual access, and compare temporal readouts with controlled capacity. A later
adaptation/ablation study is needed for architectural necessity. Subsequent
free-running sampling should measure prefix error accumulation and long-horizon
structure under the same supplied timing/entry conditions. The next experiment's
scope remains for human direction; this evidence append grants no acceptance,
training or lifecycle transition.

#### Interpretation clarification: lag and synthetic validity

Reinspection of mechanism-20260915-b confirms 320 structural cases but only
64 distinct lane sequences, each presented at five constant event gaps. These
are variants of eight generator-factor combinations with phase and mirror
changes, not 320 independent real-corpus patterns. Neither exact corpus matches
nor the frequency of this synthetic family in training have been established.
The remote-edit experiment uses this same synthetic tap family. Both omit
chords, LN and within-case gap changes; the structural probe also observes a
complete input, whereas the action task predicts hidden rows.

Lag is source-event position difference: lag 16 pairs event 1 with event 17,
not sixteen beats. At a 120-ms event gap this spans 1.92 seconds. The target
measures equal lanes across those pairs. A period-two alternation can have
perfect lag-16 agreement, so the statistic does not uniquely identify a
sixteen-event motif or phrase-level organization. Lagged probe features add
explicit separated-position comparisons to a separate fitted readout; they do
not modify or train the action encoder.

The trained H response exceeds numerical repeats, but untrained H also responds
(mean query-relative RMS approximately 0.00097–0.00149 across checkpoints).
The experiment therefore establishes a working remote influence path under
these stimuli, not learned musically appropriate long-range use. Full-edit TV
0.0021–0.0117 measures redistribution of prediction probability, not accuracy
improvement. Removing all H residual context is a different intervention from
changing the remote contribution to that context; its large effect cannot be
assigned wholly to distant information.

The structural probe remains useful as a controlled measurement check. Its
negative result means these fixed readouts fail on this synthetic family;
distribution mismatch, readout restrictions and irrelevant synthetic targets
remain alternatives to information loss. The real validation-block intervention
offers more direct evidence of fixed-model prediction dependence, while retaining
the activation-distribution and no-retraining caveats. Before architectural
decisions, prioritize checking these measurements on real held-out song groups,
initially with matching tap-only definitions and both visible/masked conditions,
then define suitable chord/LN measurements. This refines interpretation and the
next question only; no Card field, executed comparison or lifecycle state changes.
