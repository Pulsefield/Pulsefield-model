# Query-conditioned relation matching

The [candidate](../../src/pulsefield_model/research/source_action_modeling/relation_matching.py)
compares two RelationAttention score rules at one fixed backbone schedule. It
tests whether query-dependent relation preference improves held-out source-action
prediction. It uses the four-action schema and the existing
[composition runner](source_action_relation_composition.md).

## Operator and controls

For head $h$ with width $d_h$, the additive operator computes

$$
s_{ijh}=q_{ih}^{T}k_{jh}/\sqrt{d_h}+b_h(e_{ij}).
$$

The query candidate adds one term:

$$
s'_{ijh}=s_{ijh}+q_{ih}^{T}(W_{\mathrm{rel}}e_{ij})_h/\sqrt{d_h}.
$$

Here $e_{ij}$ is the existing 64-channel descriptor: the edge projection plus
the sum of relation-MLP outputs assigned to that edge. The shared module also
accepts an optional extra descriptor, which enters both rules before projection;
the composition backbone does not supply it. The bias projection, relation MLP,
value offset, output projection, normalization and feedforward paths are retained.
The 64×64 matrix mixes descriptor channels before splitting into four heads of
width 16. It has no bias and starts at zero without consuming random numbers.

[Shaw et al.](https://aclanthology.org/N18-2074/) are the closest mechanism
analogue: relative relation vectors participate in query–key scoring and value
aggregation. This experiment adapts that mechanism to the existing beatmap edge
descriptor while retaining its additive bias. It does not propose a new
representation or learning objective.

The selectable arms are `additive` and `query`. Both use the `combined` early
representation, `serial` schedule `U → L1 → L2 → L3 → R → H`, all-bank action
access, the same decoder and FP32. The common 340,913 learned parameters start
identically; `query` adds 4,096 parameters, for 345,009 total. This is a matched
common-parameter comparison, not equal total capacity. Any benefit could include
the effect of the additional trainable matrix and its optimization.

The original `operator_order` configuration remains available. In the matching
runner, `backbone_schedule` is required and selects one schedule for both arms;
the pilot fixes it to `serial`. Warm starts are rejected for this comparison.
Checkpoint policy identities distinguish the matching arms and their schedule.

## Module constraints

[The tests](../../tests/research/source_action_modeling/test_relation_matching.py)
exercise the actual module and prediction path on CPU and available MPS:

- Zero $W_{\mathrm{rel}}$ reproduces additive outputs and common-parameter/input
  gradients, including the relation MLP, bias, value and extra-descriptor paths.
  Its own gradient is nonzero and an optimizer step can activate it.
- With the same neighbor keys, distinct relation descriptors and fixed neighbor
  values, changing the query sign reverses the candidate's relation preference.
  The additive control retains its preference. The test reads the module's actual
  aggregated values before the residual and normalization paths.
- Both full models reproduce the existing four-action serial baseline at
  initialization. A nonzero relation matrix preserves hand exchange and padding;
  saved paired training restores the same next draw and update.

These are computational-capacity and implementation constraints. They establish
neither learned use of this capacity nor benefit on real beatmaps. FP32 rounding
and MPS sparse reductions require small numerical tolerances; tensor initialization
and CPU baseline recovery are checked exactly.

## Bounded exploratory pilot

The [packaged preset](../../src/pulsefield_model/configs/hydra/source_action_relation_matching.yaml)
fixes one seed, 17, and 300 paired updates from fresh initialization. Both arms
receive each of the same eight sampled blocks per update in all three views:
2,400 block draws and 7,200 block/view exposures per arm. The full eligible
training population and the existing uniform song/beatmap/window/scale/start
sampler remain unchanged. Validation fixes 32 identity-selected song groups.
No validation value selects an endpoint.

AdamW uses learning rate `3e-4`, weight decay `1e-4`, global gradient cap `1.0`
and zero dropout. Training is capped at 1,800 seconds and the entire run at
3,600 seconds, with 1,200 seconds reserved for finalization. Fewer than 300
updates makes the run incomplete. Existing guards cap MPS driver allocation at
8 GiB, peak process RSS at 12 GiB and outputs at 4 GiB. The two memory counters
overlap on unified memory. Checks occur between updates or evaluation batches.

The primary measurement is detailed-view mean-row NLL, averaged within song
group and then across represented groups. Positive `additive_minus_query` means
improvement. The screening threshold is at least 0.02 nats per row with a
positive lower endpoint of the paired group 95% bootstrap interval (500 draws).
The interval conditions on one training seed; it does not measure seed uncertainty.

Regression guards are candidate increases of at most 0.05 nats for first-row
NLL and each 4/16/64-row scale, at most 0.05 for frozen human macro NLL and 0.10
for each human concept. Measured candidate update time, summed after the first
10 updates, must be at most 15% above baseline. These gates are specified before
execution. Resource failures or missing stages give an incomplete result.

The runner retains its existing auxiliary evaluations at bounded exposure:
100 steps per frozen human reader, 12 prefix-route pairs, and the fixed synthetic
structural readouts. All use matched trained/untrained controls where already
defined. They are secondary observations. Short readout fitting and synthetic
transfer cannot establish semantic recognition, free-running generation quality,
or player-facing demand.

Run once into a fresh destination:

```sh
uv run --offline --extra mps python -m \
  pulsefield_model.research.source_action_modeling.experiment_hydra \
  --config-name source_action_relation_matching
```

Use `--cfg job` to inspect Hydra values. The runner also validates semantics and
saves resolved config, source identity/copies, common-initialization checks,
draw manifests, initial/final NLL, checkpoints and auxiliary reports. This pilot
includes no automatic seed, capacity, schedule, sampler or architecture search.
An insufficient result ends with its limitations and an adoption decision.
