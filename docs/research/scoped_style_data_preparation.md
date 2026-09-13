# Scoped style data preparation

The isolated package `pulsefield_model.research.scoped_style_modeling` implements
the dataset adapter, source recovery/replay, relation preparation, and fixed
source-group split for [Evidence-supervised scoped style modeling](scoped_style_witness_generation.md).
It does not implement the encoder, task heads, trainer, or pilot. The frozen
study document has SHA-256
`197ae4c5de62d4f7207200c6892562650a47e3dbc0ea80640ac86776e7ea9bdd`.

## Run and consume

Run from the repository root with a fresh output directory. Data preparation
uses CPU dependencies and does not import Torch or the legacy model stack.

```bash
uv run python -m pulsefield_model.research.scoped_style_modeling.prepare_hydra \
  download=true \
  output_dir=artifacts/scoped-style-modeling/prepare-new
```

The packaged Hydra preset is
`src/pulsefield_model/configs/hydra/scoped_style_prepare.yaml`; the accepted
configuration is `PrepareConfig` in the research package. `--help` lists paths,
network access, recovery concurrency, and request timeout. `download=false`
checks the existing cache without network requests. Unknown configuration keys
fail. Output directories must not already exist, and Hydra does not change the
working directory or create its own output tree.

The adapter pins dataset commit
`b22a7a443783e05fee4db4b1d22b8e573ad448ae`, publication schema v4,
Foundation `f-15fa68913bdb2bf3`, and the method named in the study. It verifies the
manifest and all three Parquet tables before decoding. Source files live at
`source_cache/<source_sha256>.osu`; every cached or downloaded file must match the
published hash and byte length. A mutable locator returning a newer chart fails.
An existing exact original may be placed at that cache path; it receives the
same verification. Source-line references use physical one-based `.osu` lines.

The output directory contains:

| File | Contract |
| --- | --- |
| `summary.json` | Readiness, cohort and split hashes, every concept/class/split's cell and group support, evidence availability, rare slices, exposure and failures |
| `source-recovery.json` | One retrieval/hash status and path per published source |
| `split-manifest.json` | Copy of the frozen source-to-group-to-split assignment |
| `assessment-cohort.jsonl` | One resolved exact cell per human or machine layer, original record IDs, origin/confirmation metadata, assessment, chart identity and separate evidence targets |
| `contexts/<chart_key>.json.gz` | Deduplicated source/scope/context inputs and relations, with source objects and decision identities in separate fields |
| `adapter-issues.json` | Agreeing duplicate records, unresolved/unreviewed cells, and conflicting cells |
| `excluded-assessments.json` | Cells whose source could not be recovered or replayed; identical exclusion for both arms |
| `evidence-errors.json` | Invalid reference identity/membership; these cells remain in the assessment cohort and block readiness |
| `row-incompatibilities.json` | Exact same-lane close/head coincidences that cannot become a V3 single-action row |
| `config.json` | Resolved preparation parameters |

Main training must select `layer == "machine"` and `split == "train"` from the
cohort. Human training-group cells are development material; human validation/test
cells are separate assessment evaluation cohorts. Never combine their metrics
with machine metrics or treat retained human selections as independent human gold.
Agreeing exact-cell duplicates within a layer use the lowest record ID's published
context and evidence, retain all record IDs and provenance, and receive one cell
weight. Selections are not unioned. Conflicts have no class target.

Load a chart with its cohort hash:

```python
from pathlib import Path
from pulsefield_model.research.scoped_style_modeling.prepare import load_chart

chart, edges = load_chart(
    Path(output_dir) / "contexts" / f"{cell['chart_key']}.json.gz",
    expected_sha256=cell["chart_sha256"],
)
inputs = chart.inputs
```

Only `inputs` and `edges` supply chart features. `visible_objects` and `decisions`
are source-identity sidecars; source-line values must not be embedded. Concept
queries and supervision come from the separate cohort. Missing evidence has
`evidence_status="missing"` and null masks; explicit empty evidence has
`evidence_status="empty"` and all-zero masks. Invalid evidence is a contract
failure, never ordinary missing supervision. Forced-zero decisions remain in the
timeline. Masks use bits for zero-based source columns; `hand_mask` converts to
canonical outer/inner hand coordinates.

Replay retains complete visible attacks, closes, boundary occupation and original
LN endpoints. Section indices include $a^-$, all source event rows in $[a,b)$,
and $b^-$. A source row at $a$ follows its marker; a head at $b$ is excluded.
Outside-context endpoints have no invented nodes. Strict attack-neighbor features
use only visible attacks, and boundary markers do not replace original event or
attack-group adjacency. Relations combine all applicable types and role pairs
once per query/neighbor pair. The delivery complement `context_note_refs`, true
assessments, and evidence membership never affect these inputs.

The command exits unsuccessfully when recovery, replay, evidence identity,
conflicting assessment, or row compatibility fails. Inspect the error artifacts;
do not train from `data_contract_ready=false`. Zero class support is reported
separately because it limits evaluation rather than source validity.

## Fixed grouping

The default split manifest is `artifacts/scoped-style-modeling/split-v1.json`.
Groups combine exact sources, positive known beatmap/version and beatmap-set IDs,
and identical recovered action arrangements. Optional `known_groups_path` accepts
a JSON mapping from a known song/audio/duplicate identity to source hashes:

```json
{"audio:known-shared-identity": ["<source-sha256-1>", "<source-sha256-2>"]}
```

Each group ID is its lexicographically smallest source hash. Hash UTF-8
`0:<group_id>` with SHA-256, interpret it as an unsigned 256-bit integer divided
by $2^{256}$, then use $[0,0.8)$, $[0.8,0.9)$, and $[0.9,1)$ for
train, validation, and test. Reordered input cannot change this assignment.
An existing manifest must match exactly; changed grouping requires an explicitly
revised manifest path. Calibration/exemplar references are reported as exposure
without joining sources into a transitive exclusion graph.

## Verified snapshot support

Full preparation of the pinned snapshot recovered and replayed **545/545**
sources, prepared **1,176** distinct source/scope/context inputs, and retained
**4,403 machine cells** and **590 human cells**. Two agreeing human duplicates
were collapsed. There were no excluded assessments, invalid references, replay
failures, or incompatible source rows. The published URL for beatmap `2578496`
returned HTTP 403; an existing local original matched the published hash
`31381fc85d58994c15ca949917fab1cf83090ff96fe07d887d78861e0e33b041`.
The different current official revision was rejected.

No published beatmap/version or set identities, or exact recovered arrangements,
joined different sources in this snapshot. The fixed grouping therefore has
**433/63/49** train/validation/test groups, with split SHA-256
`15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a`.
This is source-grouped exploratory evaluation, without comprehensive cross-set
song/audio deduplication.

| Layer | Train cells | Validation cells | Test cells | Evidence |
| --- | ---: | ---: | ---: | --- |
| Machine | 3,526 | 487 | 390 | All 4,403 have nonempty valid selections |
| Human | 466 | 79 | 45 | 587 nonempty and three explicit empty selections; no missing references |

Every concept/assessment combination has validation and test support. The table
below shows **cells / independent groups** in absent, supporting, prominent order.
Human cells are deduplicated, and group counts within a row are not additive.

| Layer / concept | Train (A, S, P) | Validation (A, S, P) | Test (A, S, P) |
| --- | --- | --- | --- |
| Machine / Jack | 291/220, 304/234, 137/113 | 40/28, 35/29, 26/21 | 38/29, 31/26, 14/13 |
| Machine / Stream | 138/121, 179/150, 382/275 | 25/20, 28/21, 47/37 | 14/12, 19/15, 42/30 |
| Machine / Trill | 568/340, 94/83, 42/39 | 78/47, 14/13, 3/3 | 63/39, 11/11, 3/2 |
| Machine / Tech | 629/350, 45/39, 34/25 | 87/49, 6/5, 3/3 | 74/41, 2/2, 5/3 |
| Machine / LN coordination | 573/327, 61/52, 49/40 | 76/45, 13/11, 6/5 | 63/39, 6/5, 5/4 |
| Human / Jack | 25/21, 38/36, 16/16 | 3/3, 6/6, 4/4 | 4/4, 1/1, 2/2 |
| Human / Stream | 10/10, 36/31, 33/30 | 4/4, 2/2, 6/6 | 1/1, 3/3, 3/3 |
| Human / Trill | 64/54, 18/18, 18/18 | 10/10, 1/1, 3/3 | 5/5, 3/3, 1/1 |
| Human / Tech | 61/61, 29/28, 20/16 | 14/14, 4/4, 6/6 | 9/9, 4/4, 1/1 |
| Human / LN coordination | 65/60, 22/21, 11/11 | 10/10, 4/4, 2/2 | 6/6, 1/1, 1/1 |

Machine test Tech supporting has only two groups; machine test Trill prominent
has two. Several human test positive classes have one group. Human test has no
entering-LN cases. The complete rare-slice counts are in `summary.json`; sparse
support limits claims even when all three assessment classes are represented.
Across machine positives, replay recovers the study's 311 entering-LN records,
47 selections containing entering LNs, 41 partial attack-group selections, and
1,229 distributed witnesses. These are descriptive slices, not quality filters.

Published exemplar references occur on 4,758 retained cells and name 112 source
identities: 88 training, 11 validation, eight test, and five unpublished sources.
The report also preserves the five Foundation calibration source identities.
These are exposure limitations, not independent human-evidence certification.
No model fitting or held-out performance evaluation was performed in this stage.

Focused verification:

```bash
uv run --group dev python -m pytest -q \
  tests/research/scoped_style_modeling tests/test_package_layout.py
```

The tests cover source byte/line identity, boundary and outside-context LNs,
explicit empty versus missing evidence, impossible masks, unchanged complete
source facts under changed targets, merged relation roles, mirror equivariance,
empty sections, frozen grouping, origin separation, artifact hash verification,
Hydra projection, and end-to-end failure reporting. Model padding, gradient
isolation, optimization, and paired throughput belong to the next implementation
stage.
