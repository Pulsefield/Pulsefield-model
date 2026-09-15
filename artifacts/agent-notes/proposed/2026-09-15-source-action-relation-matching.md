# Agent Note: Query-conditioned relation matching at fixed exposure

Note ID: 2026-09-15-source-action-relation-matching
Status: proposed
Kind: research
Created: 2026-09-15
Updated: 2026-09-15
Product revision: 643f50519f6000e986cea6f40d36e93172de9159 (clean)
Scope: RelationAttention query matching, its source-action configuration/tests, and one paired exploratory pilot
Related: 2026-09-14-source-action-relation-composition; 2026-09-15-source-action-network-diagnostics

## Question and authority

Does adding q^T W_rel e / sqrt(head width) to the current additive score improve
held-out source-action prediction at fixed training exposure? The owner explicitly
requested implementation, actual module constraints, one bounded two-arm pilot,
evaluation and an adoption/defer decision on 2026-09-15. The learning-signal audit
is closed. Decoder-state zeroing is not a prerequisite or follow-up obligation.

This is an explicitly authorized exploratory run. Accepted Note revision: none.
Experiment Card ID/revision: none. The Note remains proposed; execution authority
does not imply revision-specific acceptance or a lifecycle transition. No remote
publication or additional seed, capacity, schedule or architecture search is
included. Prior notes retain their own scope and lifecycle.

## Baseline and intervention

- Clean pre-intervention baseline: a7ab19f26cd8dd05480c8376060aaced1af4f2a7.
  RelationAttention adds the edge projection and summed relation-MLP outputs,
  then uses that descriptor in score bias and value offset. Its optional extra
  descriptor is unused by the composition backbone. No query–relation term exists.
- Clean intervention: 643f50519f6000e986cea6f40d36e93172de9159. Both arms use this
  executable source. The `additive` arm reproduces the prior serial predictor.
  `query` alone adds a zero-initialized, bias-free 64×64 matrix. Descriptor MLP,
  bias/value paths, schedule, sampler, objective, optimizer and FP32 stay fixed.
- Shared parameters: 340,913; query total: 345,009. Added parameters are an
  intrinsic capacity/optimization difference, not an equal-total-count control.
- Closest analogue: Shaw et al., Self-Attention with Relative Position
  Representations, https://aclanthology.org/N18-2074/, equations 3–4. This adapts
  relation-aware query scoring to the existing beatmap descriptors; it is not
  a new representation or objective.
- Owning product contract: docs/research/source_action_relation_matching.md.

## Alternatives and falsification

The additional match could improve query-dependent retrieval under the existing
loss. Alternatively, additive bias/value and surrounding nonlinear layers may
already suffice, or 300 updates may be too short to establish an advantage.
Lower held-out NLL with regression guards intact favors the candidate within
this pilot. No reliable improvement, excessive regressions or incomplete execution
supports retaining additive. Module capacity alone cannot select an architecture
for real beatmaps; no synthetic-probe result substitutes for held-out prediction.

## Frozen exploratory procedure

- Both arms: four actions, `combined` representation, serial L1/L2/L3/R schedule,
  all-bank reader, fresh initialization, seed 17, no warm start.
- Use the full eligible training population (`train_group_limit: null`) through
  the existing song/beatmap/window/scale/start sampler. Fixed validation selects
  32 identity-hash song groups. Existing source, split and replay validation
  remain in force; test payloads are unopened.
- Inputs: artifacts/scoped-style-modeling/prepare-v1,
  artifacts/scoped-style-modeling/dataset-b22a7a4,
  artifacts/indexes/beatmap_index_4k.parquet, dataset, and
  artifacts/scoped-style-modeling/sources. The runner records verified population,
  validation and source hashes before training. Initial NLL is measured before
  the 300-update endpoint, not substituted from a six-action run.
- Exactly 300 paired updates, eight block draws and all three equally weighted
  views per update: 2,400 draws and 7,200 block/view exposures per arm. AdamW
  learning rate 3e-4, weight decay 1e-4, gradient cap 1, dropout 0, FP32.
- MPS, one CPU thread. Training limit 1,800 seconds; total limit 3,600 seconds;
  finalization reservation 1,200 seconds. Minimum updates equals maximum updates.
  Resource limits: MPS driver 8 GiB, peak process RSS 12 GiB, output 4 GiB.
  Guards run between updates/evaluation batches. Global/resource failure or fewer
  than 300 updates means incomplete; do not extend or resume automatically.
- Primary: detailed-view mean-row NLL, mean within song group then over groups.
  Gain = additive minus query. Screening threshold >=0.02 nats per row and the
  paired group 95% bootstrap lower bound >0 (500 draws). One seed supplies no
  uncertainty estimate across training seeds.
- Regression limits, query minus additive: <=0.05 nats for first-position NLL
  and each 4/16/64-row scale; <=0.05 human macro NLL and <=0.10 for each concept.
  Query model-update seconds summed over updates 11–300 must be <=1.15 times
  baseline. No checkpoint is selected by validation.
- Existing secondary stages: 12 fixed-information prefix pairs, the fixed
  structural factorial, and 100-step frozen human readers for trained/untrained
  arms. These bounded checks establish neither semantic factors nor generated
  chart quality. A fixed sequential arm order can confound small timing differences.
- Fresh output: artifacts/source-action-relation-matching/pilot-20260915-01.
  Collision is an error. No overwrite, resume or automatic extra run.

```sh
uv run --offline --extra mps python -m \
  pulsefield_model.research.source_action_modeling.experiment_hydra \
  --config-name source_action_relation_matching
```

## Constraint evidence before execution

The focused CPU/MPS selection passed 85 tests and 20 subtests in 17.78 seconds.
It includes actual-module zero recovery for outputs/common/input gradients,
nonzero new-matrix gradients, query-dependent relation-preference reversal with
equal neighbor keys, baseline initialization, hand exchange, padding, checkpoint
continuation, runner selection and packaged Hydra projection. The strict additive
probability comparison originally exposed 3e-8–6e-8 FP32 rounding from a common
score shift; its final 1e-7 bound preserves the exact preference requirement.

Command: uv run --offline --extra mps --group dev python -m pytest -q
tests/research/source_action_modeling/test_relation_matching.py
tests/research/source_action_modeling/test_composition_experiment.py
tests/research/source_action_modeling/test_time_local.py
tests/research/source_action_modeling/test_representation.py
tests/research/source_action_modeling/test_checkpoint_migration.py
tests/research/scoped_style_modeling/test_model.py tests/test_package_layout.py.
Hydra --cfg job/--help, four local documentation links and git diff --check
also passed. No CUDA coverage is claimed.

## Result Log: pilot-20260915-01

### Experiment and reproduction

- Owning Note: 2026-09-15-source-action-relation-matching. Accepted Note revision:
  none. Experiment Card ID/revision: none. The exploratory protocol was committed
  before launch as a997051fdf8bfcede71da709e719c1e9d34b66f2.
- Pre-intervention baseline: a7ab19f26cd8dd05480c8376060aaced1af4f2a7, clean.
  Both executable arms launched at 643f50519f6000e986cea6f40d36e93172de9159,
  clean. The command and configuration above were used without overrides.
- Run root: artifacts/source-action-relation-matching/pilot-20260915-01.
  Arm endpoints: seed-17/additive-endpoint.pt and seed-17/query-endpoint.pt.
  Common initialization SHA256:
  0f557d7ad0538a03d8f087d0ac4d47cbdee6a8f69224e16d59d3e84f8c9b9a78.
- Saved source aggregate SHA256:
  a2fdb902dc406a11e82db4fc7a10807a549756866af9dd5372f5e408f6091871.
  The snapshot includes the shared RelationAttention source and config as well
  as the source-action modules. All copied-module hashes were rechecked.
- Verified population SHA256:
  e56d8eaf3b40bbe9fe79bea56a8f4d67c4a775440358e049bc4c50235d2f5484.
  Training population: 3,169 groups, 11,564 beatmaps. Validation: 32 groups,
  61 source windows and 192 paired blocks. Human fitting/validation: 382/61
  cells, with 18 human validation groups.
- Fixed validation manifest SHA256:
  717e10ecbea9f7228f2bf7b444474f2513d8405a2e6fdafb6256fc5c11fe0254.
  Population/allocation/manifest hashes and identical evaluation batch identities
  across arms were verified. Test payloads remained unopened.
- Environment: MPS, macOS 26.6.2 arm64, Python 3.10.20, PyTorch 2.11.0, FP32,
  one CPU thread. Launch: 2026-09-15T12:20:47.028686+00:00.
- Fresh destination, no overwrite or resume. Both arms completed exactly 300
  paired updates. Stop reason: update-bound. Training 964.619 seconds; full run
  1,309.469 seconds (21.82 minutes). All declared auxiliary stages completed.
- Peak observed MPS driver allocation: 2,457,796,608 bytes (2.289 GiB).
  Peak process RSS: 2,991,325,184 bytes (2.786 GiB). These overlap on unified
  memory. Outputs occupy approximately 31.8 MiB. All guards remained within bounds.
- Each arm received 2,400 draws and 7,200 block/view exposures, containing
  190,836 target-row exposures across three views. Scale draw counts were
  835/831/734 for 4/16/64 rows. Training reached 1,690 groups and 2,061 beatmaps.
  Exposures include repeated views and overlapping windows, not independent rows.

### Results

Action NLL uses mean within group, then mean across groups. Positive paired gain
is additive minus query. Human NLL uses the frozen concept/group macro metric.

| Measurement | Additive | Query |
| --- | ---: | ---: |
| Initial detailed NLL | 4.335290690 | 4.335290693 |
| Final detailed NLL | 2.633393463 | 2.633629405 |
| First-position detailed NLL | 2.914092300 | 2.914598928 |
| Later-position detailed NLL | 2.604718710 | 2.604904800 |
| 4-row block detailed NLL | 2.729763863 | 2.730039398 |
| 16-row block detailed NLL | 2.543504278 | 2.543722485 |
| 64-row block detailed NLL | 2.626912249 | 2.627126331 |
| Near minus detailed mean-row NLL | 0.008303006 | 0.008364062 |
| Coarse minus detailed mean-row NLL | 0.000953691 | 0.001054649 |
| Frozen trained human macro NLL | 0.954594672 | 0.954999663 |
| Matched untrained human macro NLL | 0.943775508 | 0.943775513 |

Primary paired gain: -0.0002359414 nats per row; 95% paired group-bootstrap
interval [-0.0005167263, 0.00000712845], from 500 draws over 32 groups. This fails
both the >=0.02 practical gain and positive lower-bound criteria. The interval
does not establish a reliable difference in either direction.

First-position and scale regressions are at most 0.000506628, below 0.05. Human
macro regression is 0.000404991, below 0.05. Concept regressions, query minus
additive, are Jack +0.000391011, LN +0.000990658, Stream +0.000927727,
Tech -0.000785038 and Trill +0.000500600; all are below the 0.10 limit. The
short frozen readers do not outperform the matched untrained macro result.
Tech retains only one positive and no supporting validation cell.

At serial S4, combination-holdout MSE for the four structural facts is:

| Fact | Additive trained | Query trained | Matched untrained, approximately |
| --- | ---: | ---: | ---: |
| Adjacent same-lane recurrence | 0.045072 | 0.045088 | 0.006304 |
| Two-step return without repeat | 0.044743 | 0.044797 | 0.006520 |
| Lag-16 lane match | 0.041875 | 0.041920 | 0.062396 |
| Adjacent hand transition | 0.096646 | 0.097867 | 0.025319 |

All four query values are slightly higher at the relation output. The full
320-case factorial, all stages, pace holdout and counterfactual reports are
retained; these synthetic statistics are not human concepts. Both arms use the
same structural manifest. The 12 fixed-prefix route pairs yield equal-pair mean
TV 0.128974898/0.129360097 and JS 0.022244098/0.022315982 nats for additive/query.
Their history/observation batch hashes and row indices match across arms.

Measured model-update seconds for updates 11–300 are 178.187289/156.063702,
giving query overhead -12.42%, within the +15% bound. The runner always executes
additive first; cache/shape warmup can favor query. This measurement does not
isolate intrinsic operator cost, and no speedup claim or extra timing run follows.

The initial maximum per-row NLL difference was 9.536743e-7 on MPS. Every common
initial tensor matched exactly and W_rel was zero. Its final Frobenius norm is
0.960672319; the matrix was updated. This confirms training of the added matrix,
not beneficial use of the added capacity.

### Plan conformance and evidence audit

No protected setting changed. There was one seed, one schedule, one budget and
one run. Initialization/dtype, all 300 update indices, 2,400 draw indices, shared
views, finite update metrics, equal final sampler states, four-action checkpoint
contracts, common evaluation identities, matched frozen-reader fitting indices
and saved hashes passed the audit. Initial/final NLL and bootstrap estimates were
recomputed from saved per-block values. No additional model forward or training
was used for this audit. The script and compact result are analyze.py and
summary.json under the run root. Deviation disposition: none.

### Evaluation and decision

Observation: both arms learn prediction from the common initial NLL, but query
matching supplies no practical held-out gain under the fixed pilot exposure.
Regression limits pass; auxiliary measurements do not reverse that conclusion.
The capability constraints pass independently of these empirical results.

Interpretation: **retain additive as the reference and defer adopting query
matching**. Keep the candidate selectable. Recommended evidence classification:
REFINE, bounded to an exploratory pilot; no formal SUPPORTED recommendation.
This does not reject the mechanism at every training horizon or establish that
it is universally redundant. The strongest remaining alternative is insufficient
training exposure for useful query-dependent retrieval to develop. One seed,
partial corpus exposure, 32 validation groups, short human fitting, synthetic
transfer and teacher-forced prediction limit the claim. None authorizes another
run, added search or an unbounded diagnostic investigation.

All requested deliverables are complete: optional implementation, module
constraints, paired results and a defer decision. This milestone has no active
follow-up. The learning-signal audit remains closed; decoder-state zeroing is
neither a prerequisite nor a deferred obligation. Any further research needs
new human direction.

## Next lifecycle condition

Results and the operational defer decision are recorded. The Note stays proposed
because no exact revision was formally accepted; there is no automatic lifecycle
transition or publication. Further research requires new direction, not completion
of any residual audit prerequisite.
