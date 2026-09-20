# Rebuild vacation inputs

`prepare_inputs.py` consumes the pinned September 15 catalog, original annotation
split and admitted row cache under the repository's `artifacts/`. It checks the
11,563 eligible TRAIN sources, 3,169 groups and 10,735,674 available suffix
onsets before creating a fresh plan with the existing group/chart sampler.
It does not recover historical checkpoint bytes or claim equivalent training.

```sh
uv run --offline --python 3.10 --extra mps python \
  experiments/vacation_rebuild/prepare_inputs.py \
  /absolute/fresh/asset-root
```

The asset root must be fresh and outside the product worktree. Inputs retain
absolute references to the surviving source/cache owner, so that owner must
remain available. The driver copies the small catalog and split manifests;
it does not duplicate the full dataset or audio. `preparation.json` records
source, driver and input identities. The source index only narrows candidates;
selected source stars are recomputed at rate 1 with the pinned local calculator.

There are two independent configurations:

- `config/r1-restore.yaml`: plain R1 followed by observed seed, landmark memory,
  head routing, release routing and response learning. The separate restoration
  worker rebuilds native TRAIN pools before their respective correction stages.
  It uses CPU1, a four-hour cumulative training budget and a 12-hour queue budget.
- `config/teacher35m.yaml`: the separate clean 35M teacher, CPU1, audio caching,
  1M/5M/10M/20M/30M readouts and the documented 16h audio/44h training budgets.

Each configuration has its own output directory. The 35M queue disables independent
fixed-model stress because the previously selected small checkpoint is absent.
The teacher performs fixed native VAL generation at its milestones; restoration
performs those readouts after each of its six stages.
Do not run both configurations concurrently on the same host.

The rebuilt fixed monitor uses eight TRAIN and eight VAL song groups, covering
source bands 2–3/3–4/4–5/5–6 stars with suffix LN-head fractions below/at least
0.1. Suffixes last 180–600 seconds. Likelihood uses early/middle/late 256-onset
crops; eight VAL sources use generation seeds 17 and 23. LN amount is a coverage
stratum, not a semantic judgment. Historical use of these VAL groups is not
excluded, so this is not independent confirmation. No TEST payload is opened.

```sh
./scripts/r1-restore.sh --config-dir /absolute/fresh/asset-root/config \
  --config-name r1-restore mode=preflight
./scripts/r1-restore.sh --config-dir /absolute/fresh/asset-root/config \
  --config-name r1-restore mode=run
```

Use `scripts/vacation-training.sh --config-name teacher35m` with the same config
directory for the separate teacher. A successful preflight
freezes configuration and source but does not create a training ledger or start
the queue's wall clock. After training begins, pause with the selected output
directory's `PAUSE` file, remove it to continue, and use `mode=resume`.

The historical best response candidate followed source training, persistent-seed
conditioning, long-memory continuation, then frozen-base head routing, release
routing and response learning on their respective native TRAIN pools. Ordinary
source supervision of all modules from initialization is a different recipe.
Fresh checkpoints require those corrections and long-form/LN/TAP evaluation
before being described as restored performance.

[historical_recipe.yaml](historical_recipe.yaml) preserves the known six-stage
path to the 6.75M-exposure candidate, including frozen-parameter scopes, native
pool sizes and the final KL anchor. It is a reconstruction reference, not an
executable queue configuration. Use the separate
[r1_restore worker](../../docs/research/r1_staged_restoration.md) to rebuild
native pools and follow that module-addition curriculum. The ordinary vacation
queue remains responsible for the independent 35M teacher.
