# Source-action four-action checkpoints and weight reuse

Source-action uses the [V3 lane language](../formulation/notation.md#row-language-and-long-notes):
`EMPTY=0`, `TAP=1`, `LN_START=2`, `LN_CLOSE=3`. These are categorical codes,
not bit flags. Same-lane simultaneous close/tap and close/head actions are
rejected by the source-action parser and the prepared-row observation adapter.
They are never retimed, merged, or silently discarded from a target block.

Each hand has 16 action pairs. The joint table contains 256 indices in
hand-role order, with index zero reserved for padding. Every real event masks
the all-empty row, leaving at most 255 classes before occupation constraints.
A held lane permits only `EMPTY` or `LN_CLOSE`; an unheld lane permits `EMPTY`,
`TAP` or `LN_START`. Unknown occupation permits their union. Release features
and LN identity remain separate from attack features and recurrence.

This aligns the action language with V3. Supplied event times, bidirectional
conditioning and prefix-only legality still make this a conditional action
reconstruction experiment, not a complete V3 generation pipeline.

## Exact continuation and weight initialization

[`checkpoint.py`](../../src/pulsefield_model/research/source_action_modeling/checkpoint.py)
saves schema 4 with input contract `source-action-visibility-v3-four-actions`.
`load_snapshot` restores a matching four-action model, optimizer, scheduler,
sampler, RNG and optional semantic readout. It rejects six-action snapshots
before changing training state.

`warm_start_snapshot(path, model)` instead loads **weights only**. It accepts:

- schema-3 source-action snapshots with input contract `source-action-visibility-v2`
  and the context-bilinear decoder policy `joint-row/context-bilinear-hand-transpose-v1`;
- matching schema-4 four-action snapshots.

The architecture, representation switches, reader access and `ModelConfig`
must match the destination. Static-Gram, complete-chart style and legacy mapper
checkpoints are not accepted by this migration.

| Component | Six-action to four-action treatment |
| --- | --- |
| Encoder, ActionReader | Copy every learned tensor unchanged |
| Decoder context, unary, pair projection, bilinear matrix and GRU | Copy every learned tensor unchanged |
| Decoder hand-action embedding | Select the 16 semantically matching rows from the old 36-row table |
| Action and hand-token lookup buffers | Rebuild from the four-action schema |
| Optimizer, scheduler, sampler, RNG and update position | Start a new run; do not restore the old training state |
| Fitted ConceptReader | Refit and reevaluate on valid original human inputs |

No new learned tensor needs random initialization when these contracts match.
The discarded embedding rows encode the removed combined actions.

The old lane ordinals were `(EMPTY, TAP, START, CLOSE, CLOSE+TAP, CLOSE+START)`.
Although old `CLOSE` had bit value 4, its **ordinal was 3**. For new hand pair
`(outer, inner)`, the copied row is:

$$
E_{\mathrm{new}}[4\,\mathrm{outer}+\mathrm{inner}]
=E_{\mathrm{old}}[6\,\mathrm{outer}+\mathrm{inner}],
\qquad \mathrm{outer},\mathrm{inner}\in\{0,1,2,3\}.
$$

The old row indices are
`[0,1,2,3,6,7,8,9,12,13,14,15,18,19,20,21]`.
Taking the first 16 rows would assign incorrect meanings to several actions.
The helper checks the historical lookup tables, all tensor keys, dimensions
and dtypes before loading any weights. It records the source path, SHA-256,
source update and conversion policy, and leaves the source file unchanged.

For identical permitted features and a matching decoded prefix, copied weights
retain the logits of surviving legal rows. The softmax renormalizes over the
smaller legal support, so old and new NLL values are not directly comparable.
Corpus membership can also change when incompatible maps are rejected.

## Use the packaged composition runner

Set `warm_start_dir` to a trusted prior run root containing:

```text
previous-run/
  seed-17/serial-latest.pt
  seed-17/interleaved-latest.pt
  seed-29/serial-latest.pt
  seed-29/interleaved-latest.pt
```

Only the declared seeds are required. For seed 17, run from the repository root:

```sh
uv run --offline --extra mps python -m \
  pulsefield_model.research.source_action_modeling.experiment_hydra \
  'seeds=[17]' \
  warm_start_dir=/absolute/path/to/previous-run \
  output_dir=artifacts/source-action-composition/four-action-warm-start
```

Replace the checkpoint root with the actual path and use a fresh output
directory. The runner uses the exact `*-latest.pt` names; it does not silently
choose another checkpoint or fall back to random initialization. It imports
weights before initial evaluation, creates fresh AdamW and sampler state, and
counts new updates from zero. Each seed report records initialization provenance.
The saved resolved YAML and runner configuration include `warm_start_dir`.

The default model dimensions match the packaged six-action composition preset.
If the original run changed a supported dimension, supply the same override;
mismatches fail. To check the complete execution path with bounded exposure,
append the following overrides to the command and choose another fresh output:

```text
validation_groups=4 max_updates=6 min_updates=1 readout_steps=3
bootstrap_samples=20 path_pairs=3 training_seconds_per_seed=300
finalization_seconds_per_seed=900 max_seconds=1800
```

These are whitespace-separated Hydra arguments. A short verification run does
not establish convergence. Longer training should establish a new four-action
validation baseline before interpreting loss or frozen-readout changes.

## Use a single snapshot in Python

For an explicit endpoint file, another model family, or inference-only weight
loading, initialize the matching architecture and call the helper directly:

```python
from pathlib import Path
from pulsefield_model.research.source_action_modeling.checkpoint import warm_start_snapshot
from pulsefield_model.research.source_action_modeling.composition import initialize_composition
from pulsefield_model.research.source_action_modeling.experiment_hydra import compose_config

config = compose_config()
model = initialize_composition(config.model, seed=17)["interleaved"].to("mps")
provenance = warm_start_snapshot(Path("/absolute/path/interleaved-endpoint.pt"), model)
model.eval()
```

The model can now evaluate valid four-action batches or use `score`/`advance`
for prefix decoding. For training, create a fresh optimizer and a sampler over
the revalidated population, then save new schema-4 snapshots. Do not pass old
six-action token IDs or optimizer moments into the new model.

Original source files and prepared lane facts are not rewritten. The full-corpus
catalog is rebuilt with recorded rejection reasons; prepared observations are
validated before masking, and the runner validates human assessment inputs
before action training. An incompatible prepared input fails explicitly and
requires rebuilding the eligible population under the new schema.

Warm-started serial/interleaved arms inherit their individual training histories.
They no longer constitute the experiment's equal-initial-tensor, from-scratch
comparison. Report inherited exposure separately from new updates and refit the
matched readouts. Exact continuation is available only after saving a matching
four-action snapshot with the new optimizer and sampler state.
