# Source-action Stage 1 verification

The source-action foundation passed its selected software checks and one bounded
real-input run on 2026-09-14. The verified implementation is clean product commit
`da0a0b7404af9d2025bb217e7cbe8af0f825f670`. The
[input/model guide](source_action_stage1.md) owns the contract and execution API.
These results establish trainable wiring under that contract, not a structural
or semantic advantage over another architecture.

## Contract checks

The following command passed **58 tests** in 7.58 seconds: 23 source-action tests
and 35 affected existing replay/model tests.

```sh
uv run --offline --extra mps --group dev pytest -q \
  tests/research/source_action_modeling \
  tests/research/scoped_style_modeling/test_replay.py \
  tests/research/scoped_style_modeling/test_model.py
git diff --check
```

The constructed fixtures cover hidden-target substitutions with identical
encoder tensors and graph topology, including changed LN endpoints attached to
visible rows outside the block. They distinguish unknown actions/occupation
from absence, unavailable time from zero, and real release rows from synthetic
boundaries. Entering/exiting holds, context-start releases, same-lane close/head
coincidences, visible LN identity edges and 64-row blocks are included.

Model checks cover joint probability normalization, prefix-only legality,
hand-exchange equivariance, padding, block/row loss denominators and gradients
to both components. Teacher-forcing changes only subsequent decoder responses.
Snapshot checks retain exact parameters, buffers, optimizer moments, scheduler,
RNG and sampling position; CPU next-update continuation is bit-exact. MPS
probability/gradient comparisons and an independent CPU optimizer check account
for observed device roundoff without omitting any parameter. CUDA was not tested.

## Real-input provenance and bounds

The run used Python 3.10.20, PyTorch 2.11.0, MPS, one CPU thread and seed 17.
The model had 58,516 encoder and 23,465 decoder parameters. AdamW used learning
rate `0.0003`, weight decay `0.0001`, gradient norm cap `1` and batch size eight.
There were **20 updates in 17.3494 seconds**, including data checks, fixed
evaluations, three displacement diagnostics and snapshot creation. The run
stopped at the update bound, below the 120-second time bound.

Local assets were available. Annotation revision
`b22a7a443783e05fee4db4b1d22b8e573ad448ae` and split SHA-256
`15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a`
were checked, along with cohort and original source bytes. Selection was
label-independent, from eight distinct training groups. No validation/test
chart, download, new corpus, resumed run or additional warmup update was used.

Each source below is also its selected group identity. Intervals are half-open
source milliseconds. The used context is also the standalone prediction scope;
the original annotation scope is retained for provenance only.

| Source SHA-256 | Original scope | Original review context | Used context | Real rows |
| --- | --- | --- | --- | ---: |
| `000662977cf314075da22600d2fbabbf140edc15791a5fc5dc473f2f64a58923` | [132497, 134497) | [131997, 134997) | [131997, 134997) | 61 |
| `0123b75a850ebf24136fffe0f975679aa4da8dc46db42e66799a1f494bafaf6e` | [115000, 124572) | [113000, 124572) | [113000, 124572) | 85 |
| `01e47b5f654875fce19ec204fb0975cbe8f4703b3d234fb2e8adeb0e2fb8c0f2` | [265000, 275000) | [263000, 277000) | [263000, 275897) | 128 |
| `03aff5a29072115e809b14970a28c76499a051baf6fa8387dc8c0ab6ba4259f7` | [210998, 212708) | [210377, 213200) | [210377, 213200) | 45 |
| `04c92e42f99b42efbca4f9f1183e94e17199fde141ebbc6eafc3446b238e37f9` | [359532, 360380) | [358900, 360600) | [358900, 360600) | 10 |
| `050c81957d10a01e22f0a91ba64d010cbd284ee1676a5f9c94e546bf7b7541ad` | [130261, 132918) | [128261, 134918) | [128261, 134918) | 68 |
| `0607e9981126b7955de112ce64ae9dd347c0c6edaaf20712649a1f0e65ab0e07` | [240000, 246395) | [238000, 246395) | [238000, 246395) | 41 |
| `0784a87c5d56c6dc07cdc587f89872854d60409734c1c9a0d6723ce132631626` | [50000, 60000) | [48000, 62000) | [48000, 62000) | 102 |

The sampler produced 160 blocks and 2,764 target-row exposures across all eight
groups. These are repeated exposures, not independent samples. The eight fixed
evaluation blocks had lengths 16, 4, 16, 4, 4, 4, 16 and 64, drawn with a separate
seed-17 sampler under the same group/context/feasible-scale/start policy.

## Observations and interpretation

All training losses were finite, ranging from 4.19576 to 4.49884. Encoder and
decoder parameter displacement norms from initialization were respectively
0.834950 and 0.619358. Fixed-block NLL changed as follows, in nats:

| Measure | Initial | After update 20 |
| --- | ---: | ---: |
| Mean within-block per-row NLL, then mean over blocks | 4.466832 | 4.319334 |
| First-row NLL, mean over blocks | 4.535209 | 4.398057 |
| Later-row NLL, pooled over valid later rows | 4.382065 | 4.199906 |

Six fixed blocks improved and two worsened. The later-row number has a different
denominator from the main loss and must not replace it. This short training
check measures neither fit to convergence nor generalization. It does not
separate contextual encoder learning from target-prefix or broad-prior use.

The fixed four-row diagnostic from the first context compared actual normalized
log-likelihood changes with gradients dotted into the actual parameter steps:

| Update | Encoder contribution | Decoder contribution | Actual change | Linearization residual |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.000114028 | 0.001349946 | 0.001340866 | -0.000123108 |
| 2 | -0.000092918 | 0.000953271 | 0.000778198 | -0.000082155 |
| 3 | -0.000483996 | 0.000482501 | -0.000045776 | -0.000044281 |

The third update's opposing module terms nearly cancel, and its residual
accounts for most of the actual change. The first-order decomposition therefore
does not accurately explain that small net response. These observations verify
the diagnostic calculation and its limits; they do not assign causal credit to
examples, slots or semantic mechanisms.

## Reproduction artifacts

The exact invocation was the Python command in the
[bounded local check](source_action_stage1.md#bounded-local-check), from the
repository root with `--offline --extra mps --group dev` and `device='mps'`.
It created run ID `20260914T064307.006731Z` under
`artifacts/source-action-modeling/stage1-smoke/`. The run reported
`product_dirty=false`; no further optimizer updates were performed.

| Artifact | SHA-256 |
| --- | --- |
| `report.json` | `dc890e48d536b040a1207b8fc45a88ed822502ef4036cebe7898fc5c9e94a06c` |
| `contexts.json` | `ad86125a483d3bb6afd2e5d3a85346212e69518b82867fc50e645931ef6866fa` |
| `final.pt` | `da372d3dd896a6da5216da59c70a250cb492b682edc0372b3584163dee4013e0` |

The artifacts are local generated evidence and may be absent in a fresh clone.
The source contract, deterministic selection, commands and observations above
remain understandable without them. The next architecture comparison still
requires the controls and decision criteria listed in the
[foundation guide](source_action_stage1.md#following-comparison).
