# Agent Note: Portable inference and difficulty/LN stability coverage

Note ID: 2026-09-19-native-inference-and-difficulty-coverage
Status: proposed
Kind: investigation
Created: 2026-09-19
Updated: 2026-09-19
Product revision: 50dda55040f51a7afc9a13994b762f953fe3064d on codex/bounded-typed-continuation
Scope: Packaged native generation, real-checkpoint equivalence and retrospective difficulty coverage of the frozen 24-group R1 confirmation
Related: 2026-09-18-bounded-typed-time-three-arm-comparison; 2026-09-19-bounded-typed-r1-unused-group-confirmation; 2026-09-19-bounded-typed-r1-burden-mechanism

## Completion requirements and authority

The human owner clarified the overall goal on 2026-09-19: given a timing skeleton
and prefix context, generate highly playable charts with actual human chart
regularities, consistently through long continuations and across difficulty
levels. Once those conditions are established, the overall task can be completed.
The owner specifically requires independent 4K 2-star to 6-star quality and
stability coverage alongside difficult maps. The LN scope explicitly includes
long holds, complex independent LN organization, short/fragmented holds and
organic integration with taps. A total LN ratio or isolated impressive episode
does not satisfy these requirements.

The standing authorization covers autonomous implementation and bounded
experiments, including revising the supplied skeleton. These clarifications do
not accept a particular Note/Card revision or select a model. The overall goal
remains active. Mechanical consistency and quality consistency are separate.

Required evaluation must cover multiple source groups and generation seeds in
each difficulty band, early/middle/late development and section transitions,
with multi-scale Foundation/human-gold judgments. Source and generated star
ratings need separate measurement with pinned calculator, key count, mods and
clock rate. A source's star band is not the output's difficulty, and exact source
star matching is not an established model objective. Short holds are not
automatically defects; length needs beat and action context. Simultaneous long
holding alone is not evidence of complex independent control. Good aggregate
scores cannot hide long-form degeneration or missing LN/tap modes.

## Packaged inference is implemented and numerically verified

Product commit `50dda55040f51a7afc9a13994b762f953fe3064d` adds two packaged Hydra
entrypoints under `research.bounded_typed_continuation`:

- `condition_hydra` prepares a standalone R/H/seed condition from a pinned
  strictly admitted 4K source. The default seed includes the complete row crossing
  30 note heads. Only typed seed objects expose crossing endpoints.
- `generate_hydra` reads that condition and pinned corpus-training or
  learning-check weights. Generation requires no corpus identity, allocation or
  source suffix labels. Optional source presentation metadata affects export
  only. Native temperature-one sampling and support remain unchanged.

The portable schema, exact input boundary, commands, resource defaults and
recovery semantics are owned by `docs/research/bounded_typed_continuation.md`.
Runtime accepts typed settings without Hydra imports. A run owns a fresh output
directory; resume verifies the parent's durable journal prefixes against its
snapshot and finite raw context, then copies them without modifying the parent.
Failed partial writes do not advance the durable checkpoint. Full outputs pass
an independent task verifier, osu! export and exact reparse. Pauses have a
checkpoint but no partial osu! export. Same source/input/sampling identities are
required for resume; exact numerical recovery also requires the same execution
environment. Audio is neither model input nor bundled output.

Selected local evidence:

```sh
uv run --offline --python 3.10 --extra mps --group dev pytest \
  tests/research/bounded_typed_continuation/test_generate_run.py \
  tests/research/bounded_typed_continuation/test_generation.py \
  tests/research/oracle_time_continuation/test_data.py \
  tests/test_package_layout.py -q
```

Result: 75 tests and 22 import subtests pass. Coverage includes CPU and actual
MPS, all three tasks, O1 committed-availability features, exact recovery beyond
the small test context with an old held seed object, injected journal interruption,
nondurable tails, mismatched inputs, corrupted snapshots/logs and one continuous
resource baseline. Both real CLI `--help`/`--cfg job` paths pass, unknown fields
are rejected, runtime imports succeed with Hydra/OmegaConf blocked, and the
documented JSON condition validates. Packaged YAML resource rules and the final
diff were inspected. No CUDA coverage or full-repository test claim is made.

### Frozen real-checkpoint equivalence

The engineering probe has no accepted research Card and makes no quality claim.
It uses source `50dda55040f51a7afc9a13994b762f953fe3064d`, Python 3.10.20,
Torch 2.11.0, CPU with one thread on the same Mac, native seed 17, candidate
budget 8192 and unscored endpoint likelihoods. Each subprocess has a 180-second
bound and each generation a 120-second pause bound, using the packaged default
resource limits. All destinations are fresh under
`artifacts/bounded-typed-continuation/native-interface-20260919-v1/`.

| Task/model | Source SHA-256 prefix | Weight SHA-256 | Complete physical rows |
| --- | --- | --- | --- |
| R1, initialization 172 at 2M | `231ed043719f` | `9be0adc00cea85058b59f0a2446884d43fc40cd412188ee123e866df6369d83a` | 3363 |
| R0, initialization 171 at 2M | `4991479df07b` | `fbe7a80936d089779a5cae48e48c3c76f36ef2062b70027c963f2a21cf9ae2d1` | 622 |
| O1, initialization 171 at 2M | `4991479df07b` | `759a4e6212e1c47b0039a457d9617bdfa957a7989e31f3715b6f70ff0b18d3b1` | 621 |

For every task, source-file condition preparation equals the original admitted
cache's timing, seed and permitted crossing endpoints. Row journals, decision
journals, exported osu! bytes and final RNG equal the corresponding frozen native
screen output. R1 additionally pauses at candidate cursor 512 and resumes into a
new directory: all three output files and RNG equal the uninterrupted run and
every parent file remains unchanged.

The complete probe takes 20.705 seconds; sampled maximum RSS is 297025536 bytes,
physical footprint 238633992 bytes and swap growth zero. All jobs terminate.
The existing R1 burden findings are reproduced, not repaired by this interface.

Probe identities:

- `verify_cli.py`: `a7aa48be0b9fab9a5edb4f357dca4af32c78941c5b479c1eb9084a0d6914ed1d`.
- `report.json`: `c255d8c2cea249d26d5d4e1199b2db799eee645643ca6c54be7d4a6506a896c4`.
- Original three-arm readout:
  `ed164330cc0a87b8254b8d8c593f8203c808d12cf097017629fb1a8ff0f38b11`.
- Original additional-group readout:
  `b46ff31e677ec370dfc5487da3b56896ad7fd27baaff23a2e8c2af04cb733765`.

## Difficulty measurement calibration

The pinned original catalog contains 13216 entries and no star-rating field.
The repository already owns `osu_core.difficulty.compute_mania_star_rating_20241007`.
At upstream osu! revision `ebaf7e9910ef3755308dec2c7950d916aabc545b`, the
[mania calculator](https://github.com/ppy/osu/blob/ebaf7e9910ef3755308dec2c7950d916aabc545b/osu.Game.Rulesets.Mania/Difficulty/ManiaDifficultyCalculator.cs)
declares version 20241007. This is a pinned comparison, not a promise about a
future server's rating version.

The local calculator exactly matches two values in the upstream native-4K
[difficulty test](https://github.com/ppy/osu/blob/ebaf7e9910ef3755308dec2c7950d916aabc545b/osu.Game.Rulesets.Mania.Tests/ManiaDifficultyCalculatorTest.cs):
2.3493769750220914 at clock rate 1 and 2.797245912537965 at rate 1.5, absolute
error zero on both. The fixture has 137 objects. Initial parsing rejected its
missing AudioFilename; a separate derivative adds only unused audio metadata,
with unchanged HitObjects bytes and no audio access. The original fixture and
upstream test remain unchanged. These are two numerical points on one fixture,
not exhaustive port conformance.

Calibration artifacts are under
`artifacts/bounded-typed-continuation/difficulty-coverage-20260919-v1/`:

- Calculator source SHA: `3faaed183a3ef7f62b6ea58f0b0b156cc707c94b760717f78794224a0fb722e9`.
- Upstream test SHA: `15392e7b4982736daf5a03c19dc7cbb4d85f31e7056a385cece9ac34d33733b2`.
- Original fixture SHA: `3348b119b076d68f1de2fb9e5c932365e7d9ea0730839aea2a622ca45d84bf7c`.
- Adapted fixture SHA: `3b30450f96d51924cd9ce87f0f7cdb592589455bb2694b1a2b3f109275ae921b`.
- `calibrate.py`: `e01c3b852281aa17bb7d091def527bf2036a77c761309ffaac9d7d276fe4431c`.
- `calibration.json`: `898586b3e5ec0999e2a294ac45288e2335c5d4f0828dff71901c309638d729e5`.

## Retrospective coverage of the fixed confirmation collection

The audit rates all 24 existing sources and all 144 existing generations. It
does not select a new cohort, train, generate, filter outputs or revise frozen
judgments. Conditions are native 4K, no mods, clock rate 1 and the pinned local
calculator above; ratings cover the whole chart including the unchanged seed.
Source bytes and all generated osu! digests are checked against the original
condition manifest/readout. Existing suffix LN summaries are joined from that
readout rather than redefined. Online Ranked status is not established by these
ratings. TEST payloads stay unread.

| Source stars | Distinct groups | Outputs per initialization | Largest three-seed star span, 171 | Largest three-seed star span, 172 |
| --- | --- | --- | --- | --- |
| below 2 | 3 | 9 | 0.494 | 0.743 |
| [2,3) | 3 | 9 | 0.440 | 0.678 |
| [3,4) | 4 | 12 | 0.535 | 1.883 |
| [4,5) | 7 | 21 | 0.474 | 0.466 |
| [5,6] | 4 | 12 | 0.719 | 1.246 |
| above 6 | 3 | 9 | 0.797 | 1.406 |

Eighteen groups fall in 2–6 stars, giving 54 outputs per initialization. Their
generated stars range 1.571–5.953 for 171 and 1.539–6.718 for 172. Three of 54
171 outputs and fourteen of 54 172 outputs fall outside 2–6. Boundary crossing
is descriptive, not a retrospective failure threshold; source ratings need
not be exactly reproduced, and a chart just across a band boundary can be good.
This table establishes numerical coverage, not quality in every band. Previous
semantic scopes were selected by ordinary/stress descriptors, not these bands.

The largest 172 seed span in the 2–6-star sources occurs on index 21, the
long-gap source
`6450d6ff083550e03a3aec467e219c3bc6d56c21190c5a73cc31b4e1ad6413ef`:

| Generation seed | Whole-chart stars | Suffix LN-head fraction | Suffix LN median / maximum duration |
| --- | --- | --- | --- |
| 17 | 6.020917 | 90.604% | 91 / 11636 ms |
| 19 | 4.138147 | 0.626% | 227 / 4727 ms |
| 23 | 5.798457 | 75.425% | 91 / 1758 ms |

The source is 3.719836 stars. Identical model weights, timing, seed context and
sampling settings produce a 1.882770-star span when only generation RNG changes.
Both long and short LN durations occur, but that fact does not establish organic
organization or playability. The large LN-use variation makes whole-chart
TAP/LN behavior a useful next diagnostic. It could reflect legitimate multimodal
continuations, learned self-reinforcement, insufficient training/coverage or an
underidentified style condition. These alternatives are not resolved by star
ratings or duration quantiles. Do not label this source a failure without the
appropriate multi-scale inspection.

Audit identities, all under the same difficulty-coverage artifact owner:

- Frozen conditions: `4200b2c27b8d21ba617889d75add008d3a90efff4cb194b805b5c0ca0d5ac5dd`.
- Frozen input readout: `b46ff31e677ec370dfc5487da3b56896ad7fd27baaff23a2e8c2af04cb733765`.
- `audit_confirmation.py`: `5772fa4f5802fd3340d737bcd3dc6909c9139f2a993946146b30795c96db7018`.
- `report.json`: `557707316060149029b7ff93cce8be1ef3deb5a8a8e3e0aa0ea903a4e87dbf6e`.

The read-only audit completes in 3.050 seconds under its two-minute bound. It is
exploratory, with no accepted Card or preregistered quality threshold. Evaluation
remains REFINE; none of the previous masked judgments or model choices changes.

## Next decision

The reusable native entrypoint closes the practical generation-interface gap.
The next research decision concerns learned quality, not another wrapper. Carry
the new completion requirements into the next bounded Card before choosing an
intervention. Evaluate independent difficulty bands and the named LN/tap forms
through full-length behavior, with deterministic coverage and risk-selected
diagnostics distinguished from representative comparisons. Preserve both R1
initializations and all frozen artifacts. The new index-21 witness can test
whether large seed variation is structured alternative mapping or progression
toward incompatible TAP/LN modes; it does not alone select decoding constraints,
additional oracle information, more training or a new architecture.

Product and Notes commits are local only. All processes are terminal. No model
adoption, Note acceptance or overall-goal completion is inferred.
