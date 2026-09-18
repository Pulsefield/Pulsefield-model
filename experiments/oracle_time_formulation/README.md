# Oracle-time formulation evidence and experiment sources

Read the [self-contained research question](../../docs/research/oracle_time_expert_question.md)
for the goal, conditional tasks, measurements, confounders and requested decision.
This directory makes the implemented endpoint probe and typed-onset pilot
inspectable without access to local runs. It is an experimental source record,
not an installed package or a replacement for the packaged training CLI.

## Contents

- `linked-endpoint-head-v1/scripts/`: object conversion, frozen feature
  extraction, original and resumed fitting, endpoint feasibility, resource
  supervisors and ten small tests.
- `onset-endpoint-generation-v1/`: required-onset scheduler, generated endpoint
  obligations, rollout driver, supervisor and four scheduler tests.
- `timing-representation-audit-v1/audit.py`: TRAIN population audit.
- `source-manifest.json`: SHA-256 for all 17 Python files, copied byte-for-byte
  from the versions used by the completed experiments.
- `evidence.json`: curated measurements, all 32 validation-group endpoint
  summaries at both fitting budgets, generation counts, clock recurrence
  comparisons and hashes identifying the original readouts. Dataset paths,
  checkpoints and full chart contents are omitted.
- `verify_evidence.py`: checks the source copy identities, independently
  recomputes both paired endpoint bootstrap summaries from the included group
  values, and checks the principal count identities. It does not reproduce
  inference or validate the original source data.

The two plot images in the research question are unmodified pages from the
same generated/source time window. They illustrate a selected example, not
an independently sampled quality benchmark.

## Checks that work without experimental artifacts

From the repository root on Apple Silicon:

```sh
uv run --python 3.10 --extra mps --group dev python -m unittest discover -s experiments/oracle_time_formulation/linked-endpoint-head-v1/scripts -p 'test_*.py'
uv run --python 3.10 --extra mps --group dev python -m unittest discover -s experiments/oracle_time_formulation/onset-endpoint-generation-v1 -p 'test_*.py'
uv run --python 3.10 --extra mps --group dev python experiments/oracle_time_formulation/verify_evidence.py
```

The first two commands cover synthetic crossed-LN round trips, feature input
boundaries, mirrored features, untruncated candidate support, the prior's
independence from history, exact feasible joint probabilities, rare feasible
mass, skipped candidates, future-onset availability and same-lane legality.
They do not require model weights or corpus data. The tests run on CPU while
the explicit `mps` extra installs the matching Torch dependency on this platform.
Use the `cuda` extra on Linux with NVIDIA.

## What cannot be rerun from this directory alone

The historical drivers deliberately retain their original constants and
artifact-relative imports. They require the original catalog, grouped split,
source charts, frozen runtime, clock100 weights and/or extracted features.
They also verify the relevant SHA-256 identities and require fresh output
directories. Merely copying them into a fresh clone does not reconstruct these
inputs, and the source manifest is not a model checkpoint.

The sequence used for the retained endpoint results was `extract.py`, then
`resume_extract.py` with its verified retained-input manifest, `fit_resumed.py`
(400 updates), and `fit1600.py` (a fresh 1,600-update run reproducing the first
400 updates). An initial extraction attempt failed when passing a structured
NumPy field with a 12-byte stride into Torch. The corrected code owns contiguous
float64 times. A subsequent producer was stopped after 82 charts under its
projected time budget; the resumed producer verified and reused those feature
files and completed the other 78 charts. This directory contains that corrected
code.
`fit.py` retains the original feature-directory default; `fit_resumed.py`
switches it to the completed resumed feature set.

The rollout used `generate.py prior` and `generate.py context` with the pinned
clock100 backbone and the respective 1,600-step endpoint head. Running it
requires recreating the original artifact layout, including the verified
frozen runtime on `PYTHONPATH`; do not remove those assertions to pretend to
reproduce the reported run with different inputs.

All reported results are exploratory. The 400-step failure and adaptive budget
extension are retained alongside the later improvement. A lower conditional
endpoint NLL is not a generated-quality result.
