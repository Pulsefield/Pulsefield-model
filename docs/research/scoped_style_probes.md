# Scoped style seed 17 probes

The probe entrypoint implements A, B, and C from
[the seed 17 report](scoped_style_seed17_results_and_next_questions.md).
The [experiment postmortem](scoped_style_probe_postmortem.md) records the
completed seed 17 A/B/C results, detection failures, and interpretation limits.
The entrypoint belongs to the scoped-style research classifier. The
[original paired trainer](scoped_style_training.md) retains its machine-only
training contract and checkpoints.

## Run the stages

Run from the repository root. The commands below use MPS on Apple Silicon;
on NVIDIA Linux, replace `--extra mps` with `--extra cuda` and add `device=cuda`.
Every output directory must be fresh. There is no overwrite or resume mode.
A failed or completed directory must be retained under its existing name or
replaced by an explicitly different output path on the next invocation.

```sh
uv run --extra mps python -m ensomi_model.research.scoped_style_modeling.probe_hydra stage=audit

uv run --extra mps python -m ensomi_model.research.scoped_style_modeling.probe_hydra stage=prepare

uv run --extra mps python -m ensomi_model.research.scoped_style_modeling.probe_hydra stage=cache

uv run --extra mps python -m ensomi_model.research.scoped_style_modeling.probe_hydra --config-name=scoped_style_probe_b

uv run --extra mps python -m ensomi_model.research.scoped_style_modeling.probe_hydra --config-name=scoped_style_pilot_c
```

The canonical packaged presets are `scoped_style_probe`, `scoped_style_probe_b`,
and `scoped_style_pilot_c`. Inspect one without loading Torch or running a probe:

```sh
uv run --extra mps python -m ensomi_model.research.scoped_style_modeling.probe_hydra --config-name=scoped_style_pilot_c --cfg job
```

| Stage | Default output | Inputs |
| --- | --- | --- |
| A, `stage=audit` | `artifacts/scoped-style-modeling/probe-a-v1` | `prepare-v1`, both seed 17 `best.pt` files and validation JSON files |
| Preparation, `stage=prepare` | `artifacts/scoped-style-modeling/prepare-probes-v2` | Verified `prepare-v1`, its pinned publication and original `.osu` source cache |
| Cache, `stage=cache` | `artifacts/scoped-style-modeling/seed17-backbone-cache-v2` | Versioned preparation and style-only seed 17 `best.pt` |
| B, `stage=readout` | `artifacts/scoped-style-modeling/probe-b-v1` | Versioned preparation and completed cache |
| C, `stage=pilot` | `artifacts/scoped-style-modeling/pilot-c-v1` | Versioned preparation; models initialized from scratch |

`prepared_dir` names the historical preparation. `probe_prepared_dir` names the
new preparation for B/C and the cache. `checkpoint`, `seed_run_dir`, `cache_dir`,
and `output_dir` can be overridden independently. Changing `output_dir` does not
change the preparation or cache destination. The preparation/cache stages use
`probe_prepared_dir`/`cache_dir`, respectively.

The local dataset and checkpoints are required assets, not repository sources.
The command verifies the publication, cohort, split and loaded chart identities.
It does not download missing files or inspect test charts. The versioned
preparation retains the original cohort and split bytes, copies only verified
training/validation graphs, and adds hashed source timing and target-policy
sidecars. Source actions and original graph features are unchanged. Source timing
is parsed through the latest declared context endpoint for each source; later
points cannot influence these inputs.

## Targets and shared inputs

The policy `human-priority-high-tech-v1` chooses the effective human judgment
for Jack, Stream, Trill and LN coordination when present, with eligible machine
fallback otherwise. Unresolved/conflicting human cells block fallback. Tech
requires the chosen human observation's explicit `high` confidence. Confidence
from previous revisions, evidence rationale or predictions is never inferred.
Agreeing duplicate cells retain the original adapter's chosen observation and
its context. The adapter checks the pinned publication's human exclusions and
rejects unhandled exclusion categories.

`target-policy.json` records eligibility decisions, per-concept/class/layer
support and the selected indices. On the pinned preparation, training contains
811 Jack, 778 Stream, 804 Trill, 26 Tech and 781 LN cells. The primary validation
view has 61 human cells, including six High-confidence Tech cells. Historical
human and machine validation views remain separate diagnostics.

Training draws a concept, source group and eligible cell uniformly at each
level. Distinct source/scope/context/rate inputs are encoded once per batch.
Only sampled concept queries are read; duplicate queries share logits, and the
loss gather preserves every sampled occurrence. Missing concept judgments
contribute no target and no extra denominator. Contexts are never widened to
combine readouts. The frozen publication adapter admits only 1x; rate sweeps
and label transfer are outside this runner.

## Model comparisons

B caches the original encoder in evaluation mode as CPU `[event, hand, 64]`
states. Each entry binds the checkpoint hash, exact input/rate, graph hash and
tensorization version, and stores valid length, section mask and source times.
The manifest records cache time, storage and sampled device/process memory.
B streams cached states from disk and freezes only the backbone. R0 trains a
fresh original assessor. R1 trains the multiscale composition and readout.
Neither R0 nor R1 receives enhanced temporal features.

C uses C0 (original), CT (enhanced time), CM (multiscale), and CTM (both), all at
zero evidence weight. Common original parameters are copied explicitly from a
single initialization. Extension families have deterministic independent
initialization, with common composition/readout parameters equal in CM/CTM.
Each arm's `initial.pt` preserves the starting values for inspection.

Enhanced time adds signed `log1p` performed seconds, signed `log1p` ratios to
median positive attack gaps in radius-2/radius-8 neighborhoods, local redline
beat-length ratios, and adjacent positive-gap log ratios. Availability is
explicit. Lane recurrence, LN ages/endpoints, signed release-to-nearest-attack
offsets, edge intervals and LN relation descriptors receive comparable
coordinates. They enter lane/hand encoding and attention bias/value descriptors.
A query before the first redline has unavailable BPM. Nonpositive/nonfinite
required redlines fail preparation. Inherited scroll-velocity points do not
supply BPM; the distinction follows the
[osu! timing-point format](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29).
A crossing interval uses the redline active at its query, not an integrated
beat count. Local gap statistics consume only complete attacks inside context.

Multiscale composition retains incoming hand states and three residual
kernel-3, stride-1 convolution outputs with dilations 1/2/4. Padding is masked
after every block. Scale metadata includes physical span, attack count,
section-relative anchor position, context availability and section crossing.
CM/R1 receive span metadata; CTM additionally receives local-pace descriptors.
The local readout uses only in-section source-event anchors, with one maximum
response and one attended vector per scale. A joint head reads these summaries
and the ordered section BiGRU summary, adding its logits to the original
ordered head. Empty sections have zero local summaries. Responses are internal
scores, not local probabilities or witness annotations. A concept-independent
learned scale mixture supplies fixed-width memory, also usable by a later
selector comparison; this runner does not train an evidence selector.

## Budgets and checkpoint selection

Both comparisons default to seed 17, AdamW at `0.0003`, weight decay `0.0001`,
batch size 16, dropout `0.1` and gradient norm cap 1. B permits at most 900 charged
seconds per arm; C permits 1,800. Each has an upper bound of 1,000 optimizer
updates, including disposable throughput updates. `max_updates` and
`arm_budget_seconds` can reduce these bounds but cannot exceed them.

The runner measures initial primary validation and up to five disposable
throughput updates (`throughput_updates`). It restores parameters and optimizer
state afterward, then chooses a common update count within the remaining
budget. The final diagnostic reserve scales measured primary-evaluation cost
by the number of final/best validation and training-fit cells, plus 20 seconds
for exports. This is an estimate, not a runtime guarantee. The throughput file
records it alongside measured update/evaluation time and the chosen count.

Preparation, disposable updates, parameter restoration, updates, validation
and final diagnostics are charged to each arm. Shared preparation is charged
in full to every arm. Cache construction is reported separately. Training
stops at a common completed-update barrier when any arm reaches its budget
minus the diagnostic reserve. Validation/export and the last common update
can overrun their estimates; `budget_overshoot_seconds` reports actual excess.
Final diagnostics are completed for every arm so faster arms never gain extra
training. Sampled memory peaks can miss transient allocations between samples,
especially on MPS; process and allocator counters overlap and must not be added.

Checkpoint selection uses primary human group-macro three-class NLL at update
zero, every 100 common updates (`evaluation_every`), and the planned final update.
Strict improvement replaces the selected checkpoint; ties keep the earlier one.
An unexpected budget stop between evaluation points receives a final evaluation
without becoming an extra selection opportunity. `best.pt` and `final.pt` remain
separate. These are bounded development pilots, not convergence or confirmation
runs. Confirmation seeds 29/43 require a separately fixed design and budget.

## Read the outputs

A writes `section-errors.csv`, `scores.json`, `scores.png` and
`joint-readouts.png` for each arm's human, machine and primary-human views.
The CSV preserves input, source-group, checkpoint, label, confidence and
provenance identities. Presence AUROC/AP give each source group equal total
cell weight, with tied scores handled together. Conditional-strength metrics
retain every true positive, including missed positives. A missing binary class
makes AUROC/AP and balanced accuracy unavailable.

B/C write `sampled-cells.json`, `throughput.json`, `updates.jsonl`,
`validation-history.json`, `curves.png`, and `summary.json`. Per-arm `best/` and
`final/` directories contain the same validation reports plus separate human
and machine training-fit reports. The summary includes matched-input changes
and C's final per-concept interaction contrast. With one seed, seed uncertainty
is unavailable. The small High-confidence Tech slice remains visible through
support counts and class-specific metrics; it has no supporting validation cell.

`localized-trill.json` inventories High-confidence human positive sections,
including a descriptive under-two-second stratum. On the pinned train/validation
cohort there are 33 High-confidence Trill cells, 16 positives, and two short
positive sections (382 ms and 1.884 s). These are section labels, not annotated
internal intervals inside a larger labeled section. Their source scopes have
no larger containing human Trill section in the pinned cohort.

Multiscale arms write `position-scales/` JSON/PNG files for those short gold
sections, validation Trill positives and any explicitly supplied localized
cases. Plots mark section boundaries and the strongest anchor's composition
span at every scale. They do not describe attention as causal attribution.

An optional `localized_trill_path=/absolute/path/intervals.json` supplies a JSON
list with `cell_id`, `source_sha256`, `scope`, `context`, `playback_rate`,
`start_ms`, `end_ms`, and `human_reference` for each independently human-localized
positive Trill interval. Its exact input must match an existing human
training/validation judgment and its interval must lie in that section. Evidence
selection endpoints are never converted into localization targets. Without
such a manifest, internal localization remains unavailable while section
classification and position/scale inspection still run.

The runners save executable source snapshots and resolved configurations for
A/B/C. Output hashes, target-policy identity and the frozen split identify the
comparison. No probe evaluates the test partition or claims an architecture
improvement from threshold tuning on validation.
