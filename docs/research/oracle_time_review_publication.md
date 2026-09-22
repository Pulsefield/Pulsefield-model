# Formulation review: source and validation record

This record describes the source bundle accompanying the
[research question](oracle_time_expert_question.md), dated 2026-09-18.
It establishes which implementation the question refers to and which local
checks were run. It does not claim generated-quality acceptance or whole-project
readiness.

## Source scope

The publication branch is `codex/oracle-time-formulation-review`. Its stack
parent and verified merge base is `origin/witness-style-probe` at
`4629686b1d38c6c52d65818a64c2d099046a066f`. The inherited local commit
`f679269b92e96efb5bd7989e748bd42cf069379e` implements the causal backbone and
sequence training. The added scope contains:

- The M3 disk-backed corpus/cache, batched causal execution, training recovery,
  sampled generation, export and resource checks; timing/clock features,
  temporal widening, grouped window sampling and their tests.
- Canonical packaged training/generation presets and the `psutil` dependency.
- Research documentation describing the implementation and measured limitations.
- Seventeen byte-identical endpoint/audit/typed-onset experimental Python
  sources, a source hash manifest, curated numerical evidence, an arithmetic
  verifier, two existing plot pages and the self-contained research question.

The source was prepared in a separate worktree. The original experimental
worktree and its uncommitted state were preserved. No datasets, weights, full
runs or Agent Notes are part of the product branch publication.

All **252 `src/` files** listed in the last experiment's frozen runtime manifest
match this branch byte-for-byte. That runtime manifest has SHA-256
`09469489b1d5b0d3f9e92890fa0ae6f58c3a9ecc2bf78cad977a865a3d16790e`
and contains 258 files overall. This equality covers model/runtime source,
not fresh execution of the past experiments. The 17 copied prototype sources
have their own [manifest](../../experiments/oracle_time_formulation/source-manifest.json).

## Selected local evidence

Environment: Apple M5 / 24 GiB, Python 3.10.20, PyTorch 2.11.0 on macOS 27.0.
The following command completed with **227 passed and 21 subtests passed**,
no skips, in 123.09 seconds:

```sh
uv run --python 3.10 --extra mps --group dev python -m pytest tests/research/oracle_time_continuation tests/research/scoped_style_modeling/test_replay.py tests/test_package_layout.py -q
```

Coverage includes exact replay, causality and mirror behavior, step/chunk and
gradient parity, actual CPU/MPS model paths, training/generation recovery,
interrupted publication, source storage, export, Hydra projection and runner
consumption, and package imports. The package-layout test checks imports; it
does not substitute for a built-wheel deployment test.

The unmodified artifact-independent prototype tests completed with **10 + 4
tests passed**:

```sh
uv run --offline --python 3.10 --extra mps --group dev python -m unittest discover -s experiments/oracle_time_formulation/linked-endpoint-head-v1/scripts -p 'test_*.py'
uv run --offline --python 3.10 --extra mps --group dev python -m unittest discover -s experiments/oracle_time_formulation/onset-endpoint-generation-v1 -p 'test_*.py'
uv run --offline --python 3.10 --extra mps --group dev python experiments/oracle_time_formulation/verify_evidence.py
```

The verifier checks all 17 source hashes and recomputes the 400/1,600-update
paired endpoint bootstrap intervals from the 32 included group summaries.
Original charts and weights are not needed for that arithmetic; the verifier
does not independently recover the original predictions from those assets.

The two operator inspection paths also completed:

```sh
uv run --offline --python 3.10 --extra mps python -m ensomi_model.research.oracle_time_continuation.train_hydra --config-name oracle_time_train_mac --cfg job
uv run --offline --python 3.10 --extra mps python -m ensomi_model.research.oracle_time_continuation.generate_hydra --cfg job
```

These commands inspect composition. The tests above separately exercise typed
projection and actual runner consumption. A plain `--cfg job` is not a corpus
training dry-run. The packaged defaults keep timing and clock paths optional;
the reported experiments explicitly enabled their recorded settings.

Two initial offline environment setup attempts could not obtain uncached
dependencies: first a Python 3.11 Torch wheel, then Python 3.10 `debugpy`.
No tests ran in those attempts. Selecting the experimental Python version and
allowing dependency installation resolved setup; the reported suite is the
subsequent successful run. No failing test was deselected or weakened.

## Limits

Selected local checks pass. CUDA, the full legacy suite, built-wheel installation,
new corpus training and a new generation benchmark were not run for this source
publication. The reported historical measurements and the new exact target
coverage count are summarized in the question and evidence JSON; their original
local assets remain unavailable in a fresh clone. Generated-quality inspection
is partial where the question explicitly says so.

Git history and the final tree are checked to exclude `artifacts/agent-notes/`.
Publication uses a new branch without rewriting history. The remote head is
verified separately against the committed local head; no PR readiness or merge
claim follows from this source publication.
