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

## Result log

Pending the single bounded run. Evaluation and adoption decision will use the
recorded endpoint and guards without enlarging the experiment.

## Next lifecycle condition

Preserve measurements and a scoped recommendation after this exploratory pilot.
No formal SUPPORTED recommendation or lifecycle transition follows without the
required human revision acceptance. Additional research requires new direction;
insufficient pilot evidence does not authorize a diagnostic backlog.
