# Reconstruct the staged R1 candidate

The `r1_restore` worker rebuilds the computational path to the response6.75M
candidate. It starts with the original small R1 and adds conditioning and
correction modules in their training order. The final architecture has 3,084,432
parameters. The 35M vacation teacher is a separate configuration and is not a
parent of this lineage.

| Stage | Source-onset exposures at completion | Parameters | Parameters trained |
| --- | ---: | ---: | --- |
| Plain R1 | 4,500,000 | 2,281,104 | Entire model |
| Persistent observed seed | 5,000,000 | 2,330,384 | Entire model |
| Landmark memory | 6,000,000 | 2,777,232 | Entire model |
| Head routing | 6,250,000 | 2,917,008 | Added head-routing residual |
| Release routing | 6,500,000 | 3,056,784 | Added release-routing residual |
| Two-onset row response | 6,750,000 | 3,084,432 | Added `frontier2` residual |

The width-128, eight-level backbone, width-256 memory, width-512 routing modules,
model seed 172, sampler seed 471, CPU1, batch 4/microbatch 2, AdamW 0.0003/0.01
and gradient clip 1 follow the recorded candidate. Each extension preserves all
source pins and the exact existing draw prefix. Residuals start at zero output;
the existing runner transfers inherited parameters, named Adam moments and RNG.
The three correction stages use source CE plus native preference weight 0.25,
two queries per update and recovery seed 954. Response also uses a unit-weight
KL anchor to its frozen release parent. Completed correction stages audit that
all inherited parameters and Adam states remain unchanged.

This reconstructs a method, not deleted checkpoint bytes. Source/cache inputs
survived, while historical weights, full sampling files and native pools did not.
New source manifests, execution identities and sampled trajectories are recorded
under a fresh asset owner. The fixed monitoring cohort is newly prepared; it is
not the original 48+16-output comparison. Completed computation does not establish
that historical quality has been restored.

## Native data and stopping rules

Collectors use only TRAIN sources and their original seed/R/H conditions. All
candidate decisions, including empty release candidates, are retained. Ordinary
durable generation supplies checkpoints, mechanically verified exports and exact
decision journals. Preferences are recomputed by replay before training; these
are machine preferences, not human annotations or hard inference restrictions.

- Head routing uses 32 distinct groups selected by `SHA256('953:' + source_sha)`
  among charts with 1,500–6,000 suffix onsets and at least 180 seconds duration.
  Seed 43 generates their complete trajectories. A lane occurring in at least
  28 of the previous 32 H rows is a repetition core; the matching source's maximum
  lane count must be at most 24, and a 1,000 ms rest clears history. Queries are
  separated by at least eight H rows. Alternatives omit a core lane or release
  a persistent blocking hold.
- Release routing excludes the 32 ordinary groups selected with prefix 957,
  then ranks 2–6-star, 180–600-second sources with 256–4,000 suffix onsets by
  sparse-to-dense transition ratio, qualifying count and SHA. The preceding 12
  H intervals span 6–30 seconds, no preceding gap exceeds 2.5 seconds, and the
  following 12 span at most 3 seconds with a ratio of at least 4. Seed 17 generates
  the fixed ordered cohort until a complete-chart prefix supplies enough queries.
  A query requires three unchanged holds through 12 H, all attacks on the remaining
  lane, while the matching source uses at least three masks and no lane over 9/12.
  Alternatives release a blocking lane; queries are spaced by at least four H.
- Response uses source-only seed 20260922 to select 32 groups: eight in each
  4–5/5–6-star band crossed with suffix LN-head fraction below/at least 0.1,
  duration 180–600 seconds and 1,000–6,000 suffix onsets. Generation uses seed 17.
  Queries cover native below-30ms heads and their preceding two-H contexts,
  require a strictly better optimistic response cost, and require no matching
  source short head in the current-through-next-two-H interval. Preferences
  minimize head-count change, LN-count change, response cost and action distance
  in that order, retaining ties and composition-conditional normalization.

Each chart contributes at most 32 temporally distributed queries. Routing and
release need at least 16 queries across four groups; response needs at least 64
across eight groups. The historical pools contained 80/8, 72/4 and 92/9
queries/groups respectively; these are observations, not counts to fabricate.
An insufficient rebuilt pool returns `needs_review` before training that stage.
It does not lower thresholds, reuse VAL examples, or start another search.

## Inputs and execution

Supply a pinned corpus plan, catalog, source root, admitted row cache and fixed
TRAIN/VAL evaluation file in the same formats as the vacation queue. Preflight
rebuilds the source-only census, fixes all three TRAIN selections and prepares
six cumulative plans. It checks source/cache identities and evaluation conditions,
then freezes configuration, implementation and runtime. No training starts during
preflight. Put the output directory outside disposable worktrees.

```yaml
defaults:
  - r1_restore
  - _self_
output_dir: /path/to/stable-assets/r1-restoration/run
base_plan_file: /path/to/base-plan.json
base_plan_sha256: REPLACE_WITH_SHA256
catalog_file: /path/to/catalog.json
catalog_root: /path/to/source-owning-checkout
evaluation_file: /path/to/evaluation.json
evaluation_sha256: REPLACE_WITH_SHA256
training:
  source_cache_dir: /path/to/admitted-row-cache
```

```sh
./scripts/r1-restore.sh --config-dir /path/to/config --config-name r1-restore mode=preflight
./scripts/r1-restore.sh --config-dir /path/to/config --config-name r1-restore mode=run
```

The macOS launcher attaches `caffeinate -is`. The default queue wall budget is
12 hours; cumulative training is bounded at four hours, each head/response harvest
at 30 minutes, release harvest at one hour, and each stage's fixed readouts at
30 minutes. RSS/footprint is limited to 6 GiB, swap growth to 128 MiB, new products
to 20 GiB and remaining disk reserve to 100 GiB. These are budgets, not throughput
or completion guarantees.

Each stage evaluates fixed likelihood windows and complete native VAL cases.
TRAIN collection, training and VAL evaluation run serially. Completed stages and
trajectories are not regenerated. `ledger.json` identifies the active stage,
segments, pools and readouts; `summary.json` retains the final checkpoint identity
and an explicit pending-quality-review status.

An output-directory `PAUSE` file, SIGINT/SIGTERM, power loss or thermal pressure
requests a safe pause. Remove the `PAUSE` file and use `mode=resume` to continue;
`mode=status` is read-only. Paused time consumes wall budgets. Resource failures,
inadequate pools and missing finalized ledgers require inspection; neither failed
stages nor incomplete hard-killed updates are silently retried. A changed source,
configuration or runtime cannot resume the old execution freeze.

## Interpretation and provenance

The earlier all-module `vacation_training_r1_response` preset is ordinary source
pretraining from a different initialization path. It is not the entrypoint for
this staged reconstruction, and its final checkpoint cannot enter the original
module-addition forks without a separately designed adaptation.

The historical experiment record is preserved in
[bounded typed continuation](bounded_typed_continuation.md#native-response-recovery-result-675m)
and the committed `agent-notes` branch, including the long-form, release and
response handoffs at note revision `c4edd10bb486c0cbfb88564f6d4920e14edbbb75`.
Recovered collector/preflight scripts corroborate the selection rules and
optimizer checks. The worker replaces their machine-local paths and one-shot
process control with typed inputs and existing durable runners. It does not
recreate failed control-training arms or copy deleted trajectories.

After computation, inspect long-chart early/middle/late phases, LN/TAP retention,
worst repetition/occupancy cases and all material local-response regressions
against the frozen Lens Foundation and human examples. Mechanical legality,
source loss and a small short-gap count cannot establish restored playability.
