# Query-conditioned relation matching

The [candidate](../../src/ensomi_model/research/source_action_modeling/relation_matching.py)
compares two RelationAttention score rules at one fixed backbone schedule. It
tests whether query-dependent relation preference improves held-out source-action
prediction. It uses the four-action schema and the existing
[composition runner](source_action_relation_composition.md).

## Pilot result and decision

**Retain additive as the reference; defer adopting query matching.** The candidate
remains selectable. Run `pilot-20260915-01` completed every declared stage at
clean source `643f50519f6000e986cea6f40d36e93172de9159`, with one seed, the serial
schedule and 300 paired updates. The common initializer and zero recovery were
checked against baseline `a7ab19f26cd8dd05480c8376060aaced1af4f2a7`.

All NLL values below are lower-is-better, in nats. Action values average mean-row
NLL within song group and then across 32 groups (192 paired blocks). Human values
use the existing concept/group macro aggregation on 61 cells from 18 groups.

| Measurement | Additive | Query |
| --- | ---: | ---: |
| Initial detailed action NLL | 4.335291 | 4.335291 |
| Final detailed action NLL | 2.633393 | 2.633629 |
| First-position action NLL | 2.914092 | 2.914599 |
| 4-row block NLL | 2.729764 | 2.730039 |
| 16-row block NLL | 2.543504 | 2.543722 |
| 64-row block NLL | 2.626912 | 2.627126 |
| Frozen human macro NLL | 0.954595 | 0.955000 |
| Matched untrained human macro NLL | 0.943776 | 0.943776 |

The paired action gain is **−0.00023594**, with group-bootstrap 95% interval
**[−0.00051673, 0.00000713]**. It misses the predeclared 0.02 improvement threshold
and the positive interval-lower-bound requirement. The data do not establish a
reliable difference in either direction. First-position and scale regressions
remain below 0.000507, versus the 0.05 guard. Human macro regression is 0.000405;
the largest concept regression is 0.000991, both within their guards. The short
human readers do not show a gain over matched untrained controls.

Both arms received 2,400 block draws, 7,200 block/view exposures and 190,836
target-row exposures. These counts include repeated views and possibly overlapping
windows. Training sampled 1,690 of 3,169 groups and 2,061 of 11,564 beatmaps.
Common tensors matched exactly before training; the maximum initial per-row NLL
difference was 9.54e-7 on MPS. The candidate matrix ended with Frobenius norm
0.960672, confirming that it received updates without establishing useful
query-dependent retrieval.

The auxiliary results give no reason to override the primary measurement.
For S4, immediately after serial relation attention, the query arm's combination
holdout MSE is slightly higher for all four fixed structural targets. Equal-pair
prefix-route TV is 0.128975/0.129360 for additive/query across the 12 selected
pairs. These are synthetic readout and routing measurements, with the limitations
of the [existing protocols](source_action_relation_composition.md).

Training took 964.62 seconds; the full run took 1,309.47 seconds. Sampled MPS
driver peak was 2.289 GiB and process peak RSS 2.786 GiB. All time, memory and
storage limits passed. Recorded model-update time over updates 11–300 was
178.19 seconds for additive and 156.06 for query. The runner always executes
additive first; cache and shape warmup can favor the second arm, so these timings
do not isolate intrinsic operator cost or establish a speedup.

The result is limited to one seed, this short training exposure, these validation
groups and teacher-forced action prediction. It cannot exclude gains at another
training horizon or establish generated beatmap quality. This pilot does not
justify replacing the additive reference.

Local evidence is under
`artifacts/source-action-relation-matching/pilot-20260915-01/`: `summary.json`,
the reproducible saved-output auditor `analyze.py`, `report.json`, source/config
snapshots and `seed-17/` measurements/checkpoints. Source aggregate SHA256 is
`a2fdb902dc406a11e82db4fc7a10807a549756866af9dd5372f5e408f6091871`;
validation-manifest SHA256 is
`717e10ecbea9f7228f2bf7b444474f2513d8405a2e6fdafb6256fc5c11fe0254`.
Raw local artifacts may be absent in a fresh clone; the measurements, controls
and decision are preserved here.

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

The [packaged preset](../../src/ensomi_model/configs/hydra/source_action_relation_matching.yaml)
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
  ensomi_model.research.source_action_modeling.experiment_hydra \
  --config-name source_action_relation_matching
```

Use `--cfg job` to inspect Hydra values. The runner also validates semantics and
saves resolved config, source identity/copies, common-initialization checks,
draw manifests, initial/final NLL, checkpoints and auxiliary reports. This pilot
includes no automatic seed, capacity, schedule, sampler or architecture search.
An insufficient result ends with its limitations and an adoption decision.
