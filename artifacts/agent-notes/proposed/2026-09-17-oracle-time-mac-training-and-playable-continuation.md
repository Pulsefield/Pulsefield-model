# Agent Note: Mac training and playable oracle-time continuation

Note ID: 2026-09-17-oracle-time-mac-training-and-playable-continuation
Status: proposed
Kind: research
Created: 2026-09-17
Updated: 2026-09-18
Product revision: f679269b92e96efb5bd7989e748bd42cf069379e, with uncommitted implementation; frozen source manifests accompany the exploratory runs
Scope: research/oracle_time_continuation M3 runtime, Mac resource envelope, optimizer calibration, corpus exposure and generated structure
Related: 2026-09-15-oracle-time-continuation-resource-and-module-review; 2026-09-18-oracle-time-m3-session-handoff

## Question and authority

Can the causal model learn playable arrangement structure on the available Mac,
corpus and annotations, with supplied event times and no audio? The requested
work includes implementation, actual execution, learning-rate and weight-decay
tuning, resource and checkpoint safeguards, and revision of the stage plan.
The quality objective requires comparison with Beatmap Lens Foundation and
current human gold, not only likelihood or legal output.

The related resource review owns the earlier census and module proposals. This
note owns new runtime measurements, tuning and generated-quality interpretation.
No note lifecycle transition or external publication is requested.

## Baseline, analogues and alternatives

The product baseline implements exact causal replay and the M1/M2 backbone.
Closest architecture analogues are Transformer-XL recurrence and XLNet's
separate query/content duties, as bounded in the product plan. This is an
engineering adaptation, with no novelty claim. osuT5/Mapperatorinator and
Mug-Diffusion remain external quality targets; their audio/timing conditions
differ, and no paired comparative win has been measured.

Live explanations for poor generation are inadequate corpus exposure, unstable
time extrapolation, optimizer step size, and decode truncation removing LN
closure probability. A larger backbone is not the first intervention: the
current default has about 1.28 million parameters, while the first calibration
consumes only 3,943 target rows.

## Exploratory experiment record

Accepted Note revision: none
Accepted Card: none
Card revision: none
Evaluation: REFINE

These runs were explicitly authorized engineering and tuning work, but have no
accepted Experiment Card and use a dirty product tree. They cannot establish
SUPPORTED research evidence. Reproducibility is bounded to frozen source
copies and their manifests in `artifacts/oracle-time-continuation/m3-20260917/`.

### Optimizer and time representation

The comparison uses eight distinct train sources from the existing annotation
allocation, model seed 17 and draw seed 17, 40 updates with effective batch 2,
FP32, chunk 128 and structural objective weight 0.3. The frozen nine validation
windows span three held-out charts and 352 target rows, including one terminal
row with deterministic support. This is a small tuning slice, not corpus-wide
quality evaluation. Each arm consumes the same 3,943 target and 52,402 prefix
rows. The pinned split is
`15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a`.

| Time encoding | LR | Weight decay | Validation nats/row, update 40 |
| --- | --- | --- | --- |
| Linear seconds | 0.0003 | 0.01 | 4.250602 |
| Linear seconds | 0.0001 | 0.01 | 3.996281 |
| Bounded seconds plus asinh | 0.0003 | 0.01 | 4.369458 |
| Bounded seconds plus asinh | 0.0001 | 0.01 | 3.731153 |
| Bounded seconds plus asinh | 0.0001 | 0.001 | 3.731235 |

The lower LR is the provisional default. The weight-decay difference is too
small to support a preference; keep 0.01 pending broader evidence. A controlled
age-only probe of the original encoder gave hidden norms 15.22 at 100 seconds,
60.46 at 400, 151.66 at 1,000 and 607.93 at 4,000. Replacing the linear channel
with seconds/(1+abs(seconds)) preserves exact replay and the asinh channel while
reducing this amplification. This changes weight semantics and has a new schema.

Raw provenance: `calibrate.py`, per-arm `calibration.json` and validation JSONL;
`runtime-snapshot/sha256.json` and `bounded-runtime-snapshot/sha256.json`.

### Long rollout and failure evidence

Three full rollouts of the longest admitted chart (26,976 rows, 4,348.526 seconds)
produced 80,868 generated rows after the seed, with independently verified legal
state transitions and complete exports. Total rollout runtime exceeded 15
minutes. Generation checkpoints remained about 4.66 MB and retained learned
state about 3.59 MB. Training checkpoints were about 15.55 MB and weights about
5.18 MB. Checkpoints own compact tensor storage and use bounded temporary writes,
fsync and atomic replacement; only one durable checkpoint is retained per run.

Those early eight-update, linear-time rollouts fail quality. Raw sampling had a
12,224-row identical run and a 1,838-second LN; top-p 0.95 had a 22,397-row
identical run and a 3,628-second LN. The source maximum LN is 13.2 seconds and
its longest identical run is 10 rows. Truncation removed positive LN-close
probability on 45,204 lane-row occasions in the top-p case. Legal completion
therefore does not establish playable structure.

Raw provenance: `long-rollouts.json`, `quality-baseline.json`, and the three
`long-p*-s*` output directories. Sampling comparison is descriptive, with
different RNG seeds and early weights; it is not an isolated top-p effect estimate.

### Mac execution and resource findings

Hardware is Apple M5, 24 GiB, FP32 Torch 2.11.0/Python 3.10.20. The resource
policy has a 4 GiB MPS driver guard, 8 GiB allocator ceiling, 6 GiB process RSS,
512 MiB available-memory floor and 1 GiB maximum system swap growth. An early
concurrent MPS run stopped on swap growth after two durable updates. It did not
pass the soak criterion.

Time-parallel local/relation frontiers match online reference outputs, states,
causal visibility and parameter gradients. MPS required contiguous paired SDPA
layouts and flattened linear projections. On one real prefix512/target128
window, batch processing reduced MPS forward/backward plus prefix from roughly
26 seconds to 2.6 seconds; CPU took 1.7 seconds. These measurements include
concurrent source admission and are provisional throughput comparisons.

The first full-capacity batched stress case exceeded the 4 GiB driver guard at
the target forward pass: idle prefix allocations remained cached. Boundary
cache release before targets is being checked; increasing the guard is not the
chosen remedy. Variable-shape long-soak, full-corpus exposure and updated
generated-quality evaluation remain required.

## Next evaluation and falsification

Reuse the pinned admitted catalog, never reconstruct train/validation groups.
Its file SHA is
`e31b7e8f4daa044503ef2b8411bc41727ba371eec804c9462a4608be8112ad28`;
it contains 11,564 train charts in 3,169 groups and 1,652 validation charts in
435 groups. Parse to disk with a count-and-byte-bounded LRU and verify original
source, arrangement and row-count identities.

Expand training exposure before increasing model size. Compare fixed held-out
likelihood windows and complete sampled charts. Read current effective human
gold from the canonical Beatmap Lens workspace and compare complete attacks,
rhythm, entering holds, LN roles, motif continuity and section transitions.
Foundation tags identify organization; they are not a playability score.
Degenerate repetition, unmotivated prolonged occupation or destructive action
transitions falsify a quality pass even if likelihood and legality improve.

Completion requires stable full-capacity long execution, bounded disk behavior,
real durable recovery, and defensible qualitative generated structure. Stronger
claims against external models require matched conditions and actual outputs.
Any adoption or lifecycle change remains a separate human decision.

## Additional exploratory evidence: full corpus, allocation ownership and capacity

Accepted Note/Card revision: none. Evaluation remains REFINE. The owner
explicitly authorized increasing capacity in necessary modules after the
baseline measurements. The earlier size restriction is superseded for the
implementation; larger configurations must still earn resource and quality
acceptance independently.

The bounded-time 40-update model completed three more longest-chart rollouts,
80,868 generated rows total, in 439.56/588.53/523.70 seconds. Longest identical
row runs were 4/6/5 rather than the earlier thousands. The raw seed17 sample
still held an LN for 89.564 seconds across a 77.643-second gap; quality is not
accepted. Reset RSS stayed approximately 158–165 MiB, and generation checkpoint
size stayed 4.67–4.68 MB. Both training exposure and time representation changed;
these observations do not isolate either cause.

Boundary cache release enabled a 637.13-second full 1.28M MPS run: 29 updates,
1,600-row prefixes, B2 and target lengths1/17/128 in fixed and varying order.
Peak sampled active/driver/RSS were 1,943,450,368 / 2,838,953,984 / 715,096,064
bytes. Every update boundary had 16,566,272 active bytes, with no additional
swap; unload returned active to zero. This supersedes the unresolved boundary
release measurement above, not the earlier failed runs.

Full corpus admission retained all 11,564 train identities; one chart lacked
the minimum seed, leaving 11,563 eligible sources. Compact disk rows occupied
199 MiB. The 1.28M model completed 500 updates, 44,580 supervised targets and
386,185 logical prefix rows. Twenty-four fixed validation windows on eight
charts, 1,126 targets, measured 2.820800 nats/row at update200 and 2.680243 at
update500. No test payloads were admitted.

The canonical Beatmap Lens effective-observation reader yielded 171 current
High-confidence observations among 58 selected candidate workflows: 124 train,
32 validation and 15 outside the inherited catalog. Foundation SHA:
`15fa68913bdb2bf395a189df7ab433f6d5b126fc35c46c1dbc8e607ce2182e97`.
Confirmed claims are human authority; embedded machine rationale is not called
a human-written comment. The read was limited to relevant High candidates and
verified an unchanged full inventory hash, rather than waiting for unrelated
workflow validation. The sibling annotation workspace was not modified.

Eight complete gold-validation outputs at update500 had LN fractions5.0–8.5%,
despite reference fractions0–71.8%. Three earlier update148 outputs had22–25%.
Ratios do not by themselves make alternative arrangements invalid. Inspected
action renders still showed incidental short holds and unstable chord/return
choices, with weak persistence of the gold examples' LN roles or group exchanges.
No quality pass or external-model advantage is claimed.

Same-update prefix reuse was checked on identical eight-window updates.
Computed prefix work fell5,800→1,834 rows; wall time24.25→9.37 seconds; NLL differed
0.0000153 over324 targets; gradient norm agreed, maximum parameter difference
was1.1e-6. Shared chart cohorts preserve expected batch-average risk but introduce
correlation; sorted positions do not have the recorded pre-sort path density.
Only two same-version prefix carries are retained, and both are cleared before
the optimizer step.

MPS online generation exposed allocation beyond tensors' reported storage:
at256 rows about133.5 MB active vs1.6 MB state payload. Copying bounded carry
again reduced active to6.78 MB including model parameters. Restoring the final
1,226-row checkpoint directly used8.68 MB, whereas the original live run reached
706.65 MB. State deletion released the excess; exact backend retention mechanics
remain unresolved. Generation now re-owns MPS carry at durable checkpoints and
drops old local state references before checking memory. The targeted regression
and CPU/MPS deterministic resume tests pass; a full real-chart replay measurement
is running under the frozen runtime.

The capacity probe enlarged temporal width/layers to256/4,384/6 and512/6 while
keeping local/relation width128 and time-bias MLP64. All used microbatch1/Q64,
two targets of128 rows after1,600-row prefixes, and two AdamW updates per device.
Total parameters were4,110,628 /11,659,464 /19,976,776. Peak MPS driver bytes were
1,656,930,304 /1,880,473,600 /3,115,679,744; every arm added zero system swap and
unloaded to zero active. The largest model's checkpoint was239,927,307 bytes.
A previous256-wide run using a wider bias MLP and B2/Q128 stopped on4,075,356,160
additional swap bytes at first forward; that configuration is not supported.

The selected capacity profile puts19,006,512 parameters in temporal,515,072 in
local,236,552 in relation,78,848 in facts and139,792 in the head. The rank16
head already covers a16-by-16 hand-pair matrix. Four CPU threads completed a
warm capacity update in7.88 seconds vs9.72 with one thread and11.18 on MPS;
overlapping diagnostic work limits hardware-performance attribution.

The 20M real-corpus run now uses effective batch8, microbatch1/Q64, four windows
per chart, prefix reuse, LR1e-4 with20 warmup updates, WD0.01, and a512 MiB
checkpoint cap. It checkpoints every25 complete updates and at requested stage
ends, reducing500 updates' periodic optimizer writes to about4.8 GB. Recovery
replays completed but unsaved updates and restores the warmup sequence; targeted
failure/recovery tests compare final weights exactly. The scheduled readouts
are100/200/500 updates on the same validation windows. Larger-model LR/decay
preference and generated structure remain open.

Provenance under `artifacts/oracle-time-continuation/m3-20260917/`:

- Baseline corpus runtime manifest:
  `8a7e3a6f9bfc41a004f8bcb4e3d6db9393f3febd4d0e8a80dd0dd750c4100630`.
- Update500 weights:
  `e3527ebcbbf6810d0abbe66d43d2747bcd0813c54c7cab83810e60975f1a64b4`.
- Gold generation runtime manifest:
  `177e0454c7c275c7b5fa02a5fef72eb83bab3b38b636569dc0162d353c574a61`.
- Expanded corpus runtime manifest:
  `2c7cf6086cdcae932c13f3390ecc7b8358beb40693a54895023a6ef2844ab4a3`.
- `capacity-probe-v2/`, `capacity-threads-4/`, `prefix-reuse-bench.json`,
  `mps-live-storage.json`, `mps-state-storage.json`, `quality-u500/`, and
  `corpus-mac20m-lr1e4-wd1e2/` retain detailed measurements and outputs.

## Expanded training readouts and prefix-conditioning probe

Accepted Note/Card revision: none. Evaluation remains REFINE. The active task
authorizes implementation and exploratory parameter/generation runs; no quality
acceptance, Note acceptance or external-model comparison has occurred.

The paired 20M arms share model seed17, draw seed17, B8/cohort4, warmup20,
WD0.01 and marginal weight0.3. At update100 both have36,134 targets and273,010
semantic prefix rows; reuse computes139,793. Fixed24-window/1,126-target
validation gives3.9517212018 nats/row for LR1e-4 and3.2886056040 for3e-5.
The1e-4 arm at200 has74,354 targets,600,441 semantic prefix rows and297,035
computed prefix rows, with validation3.0316332822. These settings differ from
the 1.28M baseline in more than capacity and are not a pure size ablation.

Full held-out generation on5b69/e67f/ecc uses raw sampling/seed17. At1e-4/u100,
LN fractions are.0253/.0200/.0203; at1e-4/u200 they are.8932/.7133/.7626;
at3e-5/u100 they are.0986/.1026/.0814. Max LNs in these later full runs remain
below two seconds. The output mix changes substantially across checkpoints and
does not maintain source-dependent organization. Ratios alone are not a style
or playability verdict. Complete visual inspection of the1e-4/u100 e67f review
context shows outer-lane dominated attacks/rapid returns, incidental short
holds and weak flowing organization. The other latest scopes still need complete
render/action review; no global quality pass is recorded.

Read-only head diagnostics on the exact5b69/e67f validation sources found
query RMS near1. At5b69 row640,1e-4/u100 entropy is1.3313 nats and mean max
probability.5922;3e-5/u100 has2.4089/.3062. Hidden-amplitude explosion is not
supported by this small probe. The first diagnostic accidentally selected a
different e67-prefix source; it was corrected to exact gold source matching and
the final file was regenerated. Only the exact-source final file is used above.

New optional conditioning retains counts from the complete minimum seed:
per-lane attacks, LN heads and a0–4-press histogram, including release-only rows.
These counts stop changing at the complete row reaching30 notes. Fourteen
normalized relative-role features enter a shared zero matrix in facts, adding
1,792 parameters without changing the initial shared weights or outputs.
No future skeleton feature, endpoint, annotation or suffix target is consumed.
The hypothesis is that explicit prefix evidence can reduce free-running style
drift; the alternative is that insufficient training or optimizer noise dominates.
This path cannot itself establish motif quality and remains disabled by default.

The bounded warm-start comparison will use the3e-5/u200 weights for both arms,
fresh AdamW and draw seed29,100 updates, the same fixed validation windows and
three initial complete gold-reference outputs. Backbone LR is3e-5; the new
zero matrix alone uses3e-4 to learn a new path while preserving the trained
backbone's smaller steps. This is a conditioning-plus-adapter-optimization
intervention, not an isolated claim about one scalar feature. Accept only a
useful qualitative improvement with no material likelihood regression; retain
failures and expand to the eight gold contexts and multiple decode policies
before any structure pass. Source initialization and conversion are pinned in
`seed-anchor-init/manifest.json` once the200-update source is available.

`seed-anchor-runtime-v2/sha256.json` has SHA
`e50ea424926bc21e5222513d5c195e06ffa92c3163d8eff8a4daff8ce18db7c4`.
It contains the separate seed LR, complete replay/checkpoint support and512-row
generation publication interval. The earlier `seed-anchor-runtime-snapshot`
is unused for training. The default generation interval is now independent of
128-row resource checks/MPS carry ownership refresh: expanded generation state
is about45 MB, and old128-row publishing wrote404–633 MB for each tested
1,226–1,877-row chart. Different write intervals preserve row/export bytes in
targeted checks. The long-run CPU training remains under its resource guards;
new feature/cadence real-run measurements are still required.

The201 relevant owner tests plus21 subtests passed with one explicit-input
allowlist test requiring the new summary field; the updated28-test data owner
then passed. After adding the separate adapter LR,36 training/Hydra/runtime
tests passed, including warmup and exact interruption recovery. A separate prior
test attempt correctly rejected resume after source files were edited during
the test; subsequent verification used fixed sources. Long runs always use
frozen runtime snapshots and were unaffected.

## Completed conditioning comparison and known-time revision

Accepted revision/Card: none. Evaluation: REFINE. The owner additionally
permits larger necessary modules and use of this Mac, with playable output as
the deciding goal. This authorizes exploratory implementation and runs without
accepting this Note or any Experiment Card.

The seed-count comparison completed 100 additional updates per arm, each with
36,609 targets after the common 20M/3e-5/u200 weights. Control validation is
2.8826876811 nats/row; the seed projection gives 2.8770584777. Full raw/seed17
outputs on 5b69/e67f/ecc have LN fractions .4053/.3141/.3358 for control and
.3663/.3252/.3190 with the feature. There is no useful demonstrated generation
benefit, and the optional feature is removed from the product path. Frozen
`seed-anchor-runtime-v2` and both arms remain available for reproduction.
Each arm has about 1,854–1,856 seconds of measured update execution, but about
15,672 seconds between resource timestamps; those wall spans include inactivity
and must not be called continuous overnight training. Peak RSS is below 2.5 GB
and measured swap growth is zero for both arms.

The 20M/1e-4 arm completed 500 updates and 183,849 target rows. Its fixed
24-window/1,126-row validation is 2.7590298638. Immutable weights are
`quality-mac20m-u500/weights.pt`, SHA-256
`f3c991bc2c19f9ad2eb2bb3b4de6f93231a95a67d6fe8eaaebc896a398a59359`.
Three full outputs remain legal; LN fractions are .0807/.0403/.0622. These
metrics do not establish playable organization or a capacity-only improvement.

The 20M/3e-5/u200 longest-chart run completed 26,956 generated rows in
1,277.968 seconds, with a 45,507,463-byte continuation checkpoint and
344,981,504-byte RSS after unload. Its maximum LN is 77,643 ms. At event 10,349,
time 1,679,374 ms, generated actions (2,3,2,0) begin two LNs immediately before
that empty gap; the next event closes both. `mac20m-long-gap-readout.json`
records the exact source and generated actions. This falsifies a playability
claim despite resource and legality success. The 20M MPS recovery run also
completed: 347.583 seconds, 44,727,943-byte checkpoint, peak active/driver/RSS
518,425,344 / 1,217,904,640 / 827,883,520 bytes, zero swap growth and byte-identical
row/export regeneration after restoring the saved 512-row boundary with a
later output tail present. This is checkpoint-tail recovery, not literal SIGKILL.

The next bounded intervention permits time-only context from the already supplied
skeleton while preserving strict action causality. This deliberately revises the
initial future-timing prohibition: up to 16 ordered offsets and successive gaps
enter a shared 110,848-parameter MLP, with a zero final projection. No lane,
future action, source event type or LN endpoint pairing is provided. The nearest
information-set analogue is known-future covariates in
[Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363); only that input
separation transfers. Its forecasting architecture and reported results do not
validate this chart generator. This is an adaptation, with no novelty claim.

The exploratory paired procedure starts both arms from the pinned u500 weights,
reinitializes AdamW, uses draw seed41, backbone LR3e-5, WD.01, lambda_struct.3,
B8/cohort4 and warmup20. The enabled arm alone uses timing LR1e-3. Readouts at
100 and 300 added updates compare the same fixed 24 windows, three initial full
gold-reference charts and the longest-chart gap behavior. A useful timing
response with no material NLL regression (0.05 nats/row) is the initial continuation
gate; a structure verdict still requires full gold-context inspection. Stop on
nonfinite loss, resource guard failure or output cap; use fresh output folders,
checkpoint at completed updates and preserve both negative and positive runs.
An exploratory capacity probe separately measures two complete updates at
512/768/1024 temporal width, six layers and the same frontiers, on CPU and MPS.
It changes size for a resource decision, not a matched learned-quality claim.

## Temporal widening and large-model continuation

The six isolated capacity-v3 runs completed without additional swap. Including
110,848 timing parameters, widths512/768/1024 have20,087,624 /43,800,136 /
76,949,832 parameters. Their second CPU updates took6.68/7.58/8.42 seconds;
MPS took7.38/7.69/8.16 seconds. CPU RSS peaks were1.80/2.93/3.75 GB; largest
MPS active/driver peaks were2.62/4.13 GB. Full AdamW checkpoints were241,262,123 /
525,799,979 /923,584,043 bytes. These are two-update capacity results, not soak
or learned-quality results. The77M preset therefore caps checkpoints at1 GiB,
checks at6 GiB driver/8 GiB RSS/minimum2 GiB available, and publishes every100
completed updates to keep write volume per update near the20M/25-update policy.
Training now rejects an inadequate weights-plus-AdamW tensor budget before
constructing the optimizer; actual serialization remains independently bounded.

A bounded migration adapts [Net2Net](https://arxiv.org/abs/1511.05641)'s
function-preserving widening to the temporal attention module: duplicate residual
and FF channels, duplicate channels within each Q/K/V head, scale Q/K by2^-1/4,
and split outgoing columns. Seeded zero-sum column noise (.01 times source RMS)
breaks gradient symmetry. Facts, local/relation operators and timing remain
unchanged. The transformation does not reuse old neural caches or AdamW state.
Tiny-model tests cover BOS, coarse memory, cached/chunk distributions, source/RNG
immutability and distinct gradient copies. The trained u500/timing-zero model
was also checked on all24 fixed real windows: pooled NLL changes by only
5.2934856498e-10 nats/row across1,126 targets. Converted77M weights are307,874,913
bytes, SHA `cbfaccdfa2de4ac48e92eb9b613fb9bc3c611b47fe9757df9ae392d7cf742fca`.

The next exploratory large arm uses these weights, the same draw seed41,
LR3e-5/timing1e-3/WD.01/warmup20/B8/cohort4/structural.3 as the20M timing arm,
and readouts at100 and300 additional updates. Width and its preserving weight
parameterization are the intervention; this is not a random-from-scratch size
comparison. It retains the same NLL/regression guards and full generation review.
The exact script and initialization are pinned in
`skeleton-time-runtime-v4/large-experiment.json`; source runtime manifest SHA is
`c23912b30bcc47d3f3e08e689eadd338f02eb8297842a44b12a7279dbc58eb17`.
The20M pair continues on frozenv3. Caffeinate prevents idle sleep during these
runs; external forced sleep would still require distinguishing wall and active
time. Accepted Note/Card revision remains none; recommendation remains REFINE.

## Time exposure, gap curriculum and publication recovery

The 20M pair completed 100 added updates with 35,441 identical supervised rows.
Fixed 24-window validation is 2.5614803192 for control and 2.5590380577 with
time lookahead. Pinned weights are respectively
`3215d5026c9c95f5cdd9667f8ef1ca96aad1089955a63a121e93f2598313c9d8`
and `5587eb3cff9ced0a46a71038a67b2a46816cac32625a355aa2f50da67675389e`.
Three complete generated charts per arm passed independent replay. LN fractions
and visual inspection do not yet establish a consistent structural gain.

`gap-exposure-u100.json` counts actual target positions: 119 following gaps at
least 1 second, 18 at least 2 seconds, one at least 4 seconds, and none at least
8 seconds. The known-time input therefore received almost no long-gap exposure.
`gap-response-u100.json` holds generated history fixed before two gaps and
compresses only future times from 84,735 or 77,643 ms to 125 ms. The final 16
prefix rows are recomputed under each time condition. Control probabilities are
identical, as required. Timing any-hold probabilities change .146628 to .146268
and .708764 to .708848; these differences do not demonstrate useful learned gap
avoidance. No future source actions enter this probe.

The next exploratory intervention mixes 25% time-stratified gap windows with
75% original draws. Feasible bands are [2,8), [8,32), and [32,infinity) seconds;
choose band/group/chart/gap/start uniformly within available populations. Starts
cover up to 32 prior rows and include the boundary in the longest half-open
horizon. Full prefix replay and true terminal semantics remain unchanged. This
is a changed population risk, without importance correction. Exact path-mass,
merged-interval, action-isolation, RNG/cohort, Hydra-consumption and update-resume
checks cover the implementation. The focused CPU selection passed 37 tests with
two MPS cases deselected; the earlier full selection passed 205 CPU tests and
four MPS tests before this sampler addition.

The bounded run starts from timing-u100 above with fresh AdamW, draw seed 53,
LR 3e-5, timing LR 1e-3, WD .01, warmup 20, structural weight .3, B8/cohort4,
CPU four threads and the measured 20M resource caps. First readout is 50 updates,
with at most 100 before reassessment. Stop on existing resource/nonfinite/output
guards. Compare the unchanged 24 validation windows (regression bound .05
nats/row), actual gap exposure, the fixed-history time response and complete
generation. This warm-start curriculum combines optimizer restart and changed
exposure; it is not a clean sampler-only comparison. Output is a fresh
`skeleton-time-gap25` directory; checkpoints preserve whole updates. Frozen v6
manifest SHA is `4bf72030c03591c97dab2709daee6736c82950950e28cd726720e441da411f80`;
`gap_train.py` SHA is `36e421d52733ad77bdac2efcd3be9bb7339efde7c5d239278b83922bb3f0c0da`.
Accepted Note/Card fields remain none and recommendation remains REFINE.

A temperature .85/raw-p1/seed17 probe generated the complete 5b69 and e67f
charts from the same timing-u100 weights. The e67f full-chart LN fraction is
145/1800 versus 660/1747 at temperature 1. Its entire 32,071–39,845 ms review
context was visually inspected with all four Beatmap Lens pages and action
records: moving taps and short directional cells occupy the first five seconds,
then a short-LN passage starts around 37.4 seconds. This is locally identifiable
organization, not a whole-chart playability verdict or proof that matching the
source LN fraction is desirable. The 5b69 full-chart fraction is 1305/2009 versus
1249/2183. Its context still requires review; no default temperature change yet.

Five matched initial 77M updates took CPU 60.08/11.88/12.71/26.09/18.73 seconds
and MPS 85.97/29.05/30.39/46.83/62.69 seconds under different concurrent loads.
Both used identical targets. MPS saved a durable update-5 checkpoint and weights,
with fixed CPU validation 2.6360789300. CPU had 15 completed but unsaved updates;
their journal was preserved before resuming exactly from durable update zero.
The large continuation now uses CPU. These measurements support an operational
device choice, not an isolated speed comparison or MPS stability failure.

Publication now uses one locked staging directory per destination. Three actual
subprocess SIGKILLs inside PyTorch serialization left the old checkpoint intact
and at most one incomplete staging directory; the next save reclaimed it and
published readable weights. Independent concurrent-publication coverage checks
that an active writer's stage is never removed. This bounds repeated-crash
temporary debris; run journals still require a single writer per output folder.
Frozen v5/v6 contain the fix; ongoing v3/v4 training retains earlier publication
code. The v5 source manifest is
`d5e1697bcdf3cbb60560a54c2f44715a9f095baa1e0bdccc533f9d31a964f437`.

## Gap exposure result and generation-state mechanism probes

Accepted revision/Card: none. Evaluation: REFINE. The owner explicitly prioritizes
understanding the model's information flow and generated structure over a quick
completion claim. Exploratory implementation and bounded execution remain
authorized; the Note stays proposed.

The 20M pair completed 300 added updates on the same 104,003 target rows.
Control fixed-window NLL is 2.5335913921; timing NLL is 2.5367861164. Their pinned
weight hashes are `984d71a9f8ce9ca2138b9a40ed373ec67d14ff5e13a3b809b09fe2c14b32f387`
and `008acf4f73c173b0f525e85664674a1188dd4917d263cc6d276ba044b76f550c`.
This does not demonstrate a likelihood gain from timing. The 77M arm's first
100 updates have NLL 2.6037101239 on the same 35,441 training targets as the
20M/u100 pair. Its weights hash is
`b84ee573594ce65cc559753807441bb6f91172c31f0009664d3c2be22ee6aa25`.
Full raw/seed17 generation produces 3,368/2,341/2,683 notes and 440/230/177 LNs
on 5b69/e67f/ecc. All replay legally; these counts do not establish quality.
The already bounded large run continues to 300 for its second readout.

The gap25 arm stopped at its first 50-update readout, with 16,844 targets and
fixed-window NLL 2.5636788519 versus the starting timing-u100 value 2.5590380577.
Pinned weights hash is
`4edaf2363e6b4afafffce793495296111d20277f978dad2f38eeb411083e8c44`.
Training has 7,263/212/9 eligible gaps across 1,459/110/7 groups in the three
bands. A separately constructed validation manifest fixes 17 boundaries:
eight distinct groups per first two bands and the sole third-band group.
Its SHA is `7be8253271318dc6fbd2bdde5bfce19e098094d351970cfaca7460a44ad9ee9a`.
The selection uses skeleton times, draw seed67 and earliest feasible context
within 32 rows; it is fixed across checkpoints. Pooled gap-window NLL improves
2.5338106804 to 2.5234816213, but mean boundary NLL worsens 3.5715677054 to
3.5812723251 and per-lane post-row occupancy Brier worsens .1342577122 to
.1482326535. Several source cases deliberately sustain LNs across 8–10 seconds;
unconditionally closing before every gap would be an invalid target.

The fixed generated-history counterfactual also fails: after gap25 training,
any-hold probability at the 84.735-second gap is .1933552623 versus .1897228956
with future time compressed to125 ms; at77.643 seconds it is .8153457642 versus
.8140622973. These are small changes in the wrong direction for these generated
histories. The earlier baseline probabilities were .1466278280/.1462675630 and
.7087639570/.7088475820. This result rejects treating exposure alone as a
demonstrated solution. No additional gap25 updates are scheduled before tracing
the decision mechanism.

The next bounded diagnostic holds timing-u100 weights fixed and uses one
current-gold pure-tap validation chart, exact SHA
`e67f9856f62cf3cf648a8c1107ebcb61b58738df62a395a40712fa0265cccef5`.
At zero-based positions256 and512, replay the complete true prefix and assert
all lanes are closed. Compare three legal one-row interventions: the source
tap row, replacing its first tap with LN_START, and rotating its tap columns
to a different tap-only row. Generate the following128 rows independently
under each branch with common RNG seeds17/19/23/29 and temperatures1/.85.
There are 48 branches and6,144 sampled rows. The true chart terminal is retained;
the probe horizon never forces a close. Record original forced-row probability,
LN starts, occupancy area, first forced-LN duration/censoring, marginal close
hazards, and prefix-length16/32/64/128 summaries. Same RNG is a paired variance
control, not a guarantee of identical actions after the intervention.

The immediate post-intervention read also separates upstream neural effects
from legality renormalization: compare the full LN-branch distribution to the
tap-branch hidden vector evaluated with the LN-branch occupancy mask. This is
a diagnostic hybrid, never a production decode policy. Trace fact, timing,
local, relation, temporal projection and six temporal-layer activations; verify
manual tracing against the normal engine before interpreting deltas. A related
fixed-history time probe compares actual versus compressed future times with
the final16 prefix rows recomputed, and current-query timing removal. Query-only
gradients and activation replacement measure whether information reaches the
head without retaining a full-prefix graph. Attention weights and activation
norms alone are descriptive, not causal explanations.

Run on CPU/one Torch thread with existing memory/swap/disk guards, a 30-minute
active-time bound, a fresh `mechanism-probe-v1` output owner and at most128 MiB
of diagnostic artifacts. Frozen runtime v7 has manifest SHA
`efd670e285bef9a7987f8882e5baf09277549dbf334dc5883c9bfdab4b06a3a8`.
Stop on source/trace mismatch, nonfinite values, illegal action or a resource
guard. Scripts and output identities are appended after execution. This can
distinguish immediate legality effects, learned short-term feedback and weak
timing transmission; it cannot prove whole-chart playability or generalize
from one chart. A relation-edge row-age feature is still linear in event age;
measure its actual source/generated range before selecting any saturation fix.

## Additional runtime and context evidence

The timing-u100 longest-chart run completes26,956 generated rows over4,348.526
seconds of supplied time in1,483.618 active seconds. Peak RSS is492,994,560 bytes,
swap growth is zero and the final continuation checkpoint is45,524,359 bytes.
It emits32,692 notes/17,225 LNs, including an89,356 ms LN. This remains a quality
counterexample despite bounded resource use and legal output. Indexed re-export
reproduces the1,147,955-byte osu file exactly in.6483 seconds, SHA
`27349bfed6576dec335c817b26f3533759235c14c26a8a64e562e574b9b1f476`.

SQLite export now caps database pages before insertion and stores notes in a
WITHOUT ROWID primary key ordered by(time,lane). The export query needs no
auxiliary sort, verified by its query plan and a late-LN-close ordering test.
Writes check the final output cap before each header/note. Existing output is
preserved on staging failure. The current runtime CPU selection passes25 tests
with two MPS cases deselected. Before this exporter edit, the v6 selected suite
passes210 CPU tests plus21 subtests and four separate MPS cases.

An actual Hydra CLI generation on e67f with timing-u100/temp.85/seed19 is killed
at observed row640 and resumed through the CLI. Final row journal and osu bytes
match the uninterrupted reference. The three child runs take149.764 seconds;
final checkpoint is44,706,567 bytes. The complete output has1,874 notes/35 LNs,
maximum607 ms. Its entire four-page32,071–39,845 ms review context shows moving
tap flow with short inner-lane returns and chord accents. The full five-page
5b69/temp.85/seed17 context,112,106–122,107 ms, has staggered short LN movements
and occasional overlap; its source has stronger paired repetition and more
sustained multi-lane holds. These are descriptive Lens/Foundation judgments,
not a whole-chart playability pass. All page/action hashes and the Foundation
identity are retained in `temperature085-context-review.json`. Default decode
temperature remains1 pending broader evidence.

## One-row intervention result and session handoff

Accepted revision/Card: none. Evaluation: REFINE. The owner requests a durable
session handoff while reserving more resources for the next session. The latest
goal permits parameter scaling as needed, keeps this stage audio-free with a
provided time skeleton, and requires Beatmap Lens/current human gold judgments
before any final playable-quality decision. The standalone process Note
`2026-09-18-oracle-time-m3-session-handoff` records authority, implementation,
negative evidence, active processes, frozen-runtime recovery and next priorities.

The bounded branch probe completes all48 branches/6,144 sampled rows in164.749
seconds. On e67f, all16 forced LNs close within1–7 rows/61–425ms. At128 rows,
the single-LN intervention changes subsequent LN heads by paired means+4.125
at temperature1 and−1 at.85; the rotated-tap control gives+8.5 and0. These cases
do not support a strong persistent LN attractor triggered by one LN. Two
positions in one chart cannot establish absence of feedback in general.

Immediate any-new-head probability changes .014132 to.107598 at position256,
and .050777 to.210903 at512. The tap-hidden/LN-mask diagnostic gives .087974
and .167755; full LN-hidden versus this hybrid JS is.004633/.004265. Much of
the immediate effect can be reproduced by support renormalization, but the
hybrid evaluates logits outside their original historical support. It is not
evidence of a legality bug or authority to remove legality constraints.

Results are `mechanism-probe-v1/branch-readout.json`, SHA
`de0e4e6b54300dc45a5107431af68df2fa3f8010629041861a90b740479afcd5`;
manifest SHA `010f5b5920d8587e76c601191e231ba2e0811b3331178f0a285e69151dc53355`.
Common/branch/timing script SHAs are respectively
`d50ca79bc2fe52b493d83bbe9d3c5f49c20943c4417776bb16a9b26dc2c5583f`,
`84ed4be5c3bc12247cbfbbc890a19b9b462e360882de27da85f39a32d1ce6832`,
`80620454efb9c20d22bab75f62a8eb27098349efc28d64587274b52296f87b60`.
The timing trace is still running at the handoff observation, with9/12 completed
cases. Interpret only a complete `timing-readout.json` as the full run.

At02:39 CST, large-model training has computed236 updates/81,773 targets;
the durable checkpoint is update200 with69,897 targets,923,599,593 bytes.
The separate exported weights remain update100 until the stage300 driver ends.
Do not restart the existing100 stage or attach another writer to this output.
The product implementation remains uncommitted at base
`f679269b92e96efb5bd7989e748bd42cf069379e`. Its50 changed/untracked files are
preserved by `session-handoff-20260918/manifest.json`, SHA
`06741fefea5ef0665660f1caec6e838672eda03633bcdcc3932d2aa27f1717fd`;
the small source-only archive SHA is
`2f9221fa33bda4520652932df2c4fed7d9176b69d241d63dd139381651a3c6a4`.
No product commit/push, Note acceptance, research adoption or goal completion
is implied by this handoff.

## Completed handoff runs and broader evaluation

Accepted revision/Card: none. Evaluation: REFINE. The active playable-model
task authorizes continued implementation and bounded execution. Product base
remains `f679269b92e96efb5bd7989e748bd42cf069379e` with the preserved uncommitted
implementation. No lifecycle transition or remote publication is requested.

The 77M continuation completed update300 and104,003 targets. Pinned weights in
`quality-skeleton-large-u300/weights.pt` have SHA
`bb848e65b27a664d852ea3df2eb5a8ccee27912bc8018621c18169ed7d8675d6`.
Its fixed24-window/1,126-row NLL is2.4773810562, versus2.5367861164 for the
20M timing arm and2.5335913921 for the20M control at the same target exposures.
The capacity ordering reverses relative to update100; the early result did
not establish that widening cannot help. These windows cover only seven
validation song groups and cannot establish corpus-wide or generated quality.
The large run accumulated4,823.360 active update seconds,452,045 actually
computed prefix rows and885,324 logical prefix rows. Logged peak RSS is
3,658,989,568 bytes, minimum available9,061,548,032 bytes, and maximum swap
growth zero. Concurrency changed during the run, so elapsed times are not a
controlled device or model-size throughput comparison. The300 readout SHA is
`05a674669eee0ab9a8b9dc8545233636107b4d734ac4ddbfa7e1d8df873c8db3`.

The timing probe completed all12 cases in566.065 seconds. Its complete readout
SHA is `9a19f815b490aa6006da4d49e361b20caafbfbc40785143cb7bbb655f9a1fd28`.
Current-query timing changes reach the head: the local fusion reduces their
RMS to roughly one tenth, but does not eliminate them. At timing-u300, timing
RMS is about5.4–5.6 percent of history-facts RMS at the four inspected queries.
Replacing only current-query timing changes any-hold probability by at most
0.000672 in those four cases. Recomputing the preceding16 rows produces a
larger change in one generated-history case:0.161893 actual versus0.148354
compressed, a direction that still does not solve the long-gap failure.
These are two positions of one TRAIN chart, not a general causal explanation.
No dead gradient path or legality defect is established.

The next evaluation separates insufficient exposure from premature capacity
conclusions. Freeze128 distinct validation song groups selected uniformly
without replacement, one uniformly selected chart per group, then use the
existing uniform feasible-context-stratum/start/horizon choices with seed
20260918. The128-window manifest is fixed before loading any model. Compare
20M/u500,20M timing/added-u300 and77M timing/added-u300 using identical full
true prefixes and targets. Report pooled NLL, equal-group mean NLL and paired
group-bootstrap95% intervals; inspect all groups, not just previously selected
gold examples. This is a broader diagnostic, not an untouched final test set.
Use immutable runtime v7, CPU/two Torch threads, a30-minute active bound,
existing resource guards with2GiB available-memory reserve and a128MiB total
diagnostic-output budget. Output owner is `coverage-evaluation-v1`; files use
exclusive creation. Stop on source mismatch, nonfinite result, resource limit
or incomplete evaluation. No training or optimizer changes occur in this run.

In parallel, generate complete5b69/e67f/ecc4 outputs from the pinned77M/u300
weights at temperature1/top-p1/seed17, matching the existing77M/u100 cases.
The v7 `quality_scaled.py` runner uses CPU/one thread and the large generation
profile. A wrapper adds a30-minute process-tree bound and4GiB aggregate output
limit; resource/checkpoint guards remain enabled. The existing
`quality-skeleton-large-u300` directory currently contains only pinned weights;
each per-source output must be absent before starting. Do not overwrite or
attach a second writer. Full Lens review contexts and action records, including
entry/exit holds, are required for qualitative interpretation. A positive
three-chart result would trigger broader eight-gold/multiple-seed assessment,
not a playable-quality pass.

## Broad validation result and supervised-exposure pilot

Accepted revision/Card: none. Evaluation: REFINE. `coverage-evaluation-v1`
completed the fixed128-group validation with5,709 targets and48,807 prefix rows.
Pooled NLL is2.5427614065 for20M/u500,2.3179136000 for20M timing/added-u300,
and2.3349896527 for77M timing/added-u300. Equal-group means are respectively
2.6076878098,2.3425084491 and2.3470542066. The two continuations improve103
and105 of128 groups over the starting weights. Their paired pooled delta95%
intervals, from5,000 group-bootstrap replicates/seed20260918, are
[-0.291704,-0.163115] and[-0.268784,-0.148582]. Fresh optimizer, lower LR,
new draws and the timing path accompany these continuations; this is not a
pure exposure-only ablation against the500-step model.

For77M minus20M timing, the pooled delta is+0.017076 and its95% interval is
[-0.005530,+0.040169]; the equal-group delta is+0.004546 with interval
[-0.022249,+0.030873]. Exactly64 groups favor each capacity. Thus the apparent
77M advantage on the original seven groups does not establish a broader gain.
Retain both checkpoints; use20M for the next sampling-cost probe without
declaring a size ceiling or rejecting a later capacity experiment.

Merging target intervals from the500-step ancestor and300-step continuation
gives287,852 supervised exposures but249,676 distinct rows across1,460 charts
and1,276 groups. Those rows are2.1669% of the11,522,113 train event rows.
This is a coverage fraction, not an epoch estimate under the nonuniform
group/chart/stratum risk. No-grad prefix replay is not extra supervision.
The timing AdamW state at update300 has nonzero moments and reconstructed
per-weight update RMS around7.37e-5–1.67e-4 for its first projection and
8.28e-5–1.56e-4 for its output projection. Epsilon domination is absent in
20M and affects about0.0021% of the77M first-layer weights; optimizer epsilon
is not a demonstrated general obstruction. These are moment-based update
estimates before decay, not newly observed backward gradients.

The three77M/u300 full raw/seed17 samples complete in162.326 supervisor
seconds. 5b69/e67f/ecc4 contain2,150/1,604/1,756 notes and1,282/884/818 LNs,
with maximum LN durations1,875/1,457/1,214ms. All26 source/generated context
pages were inspected. Staggered holds and independent inner attacks/releases
are identifiable; the three outputs also share a short-LN organization despite
different reference episodes. Source-style equality is not the objective, and
LN counts do not decide quality. This supports continued investigation of
local structure, not whole-chart acceptance or a fix for the72-minute stress
chart. The canonical revalidation and exact qualitative witnesses are recorded
with the completed cost probe below.

### Bounded sampling-cost comparison

Question: can longer supervised horizons materially improve actual target
exposure per active second while keeping exact full-prefix replay and Q64
TBPTT? The closest mechanism analogue is
[Transformer-XL](https://aclanthology.org/P19-1285/): bounded segment gradients
with forward recurrence. This probe changes the sampled target horizons, not
that mechanism. It is an adaptation of ordinary likelihood training, with no
novelty claim. Exposure-bias literature also gives competing outcomes; the
[self-recovery study](https://arxiv.org/abs/1905.10617) is a reason to test
free-running behavior rather than assume error accumulation is the sole cause.
Neither text-generation result proves a chart-generation response.

Use pinned20M timing-u300 weights and immutable runtime v7. Baseline horizons
are1/4/16s; the sole intervention uses4/16/64s. Both use B8/cohort4/microbatch1,
Q64, fresh AdamW, LR3e-5/timing1e-3, weight decay.01, warmup20, clip1,
structural weight.3 and the unchanged fixed loss denominator1024. Thus this
explicitly changes training risk, target exposure and gradient accumulation;
it is not a behavior-neutral optimization or an equal-token quality test.
Seed20260918 selects the same groups/charts/starts/horizon indices in both arms.
Twelve updates per arm alternate execution order by update to limit order
effects. Keep separate models/optimizers and never share learned prefix states.

Record supervised targets, computed prefix rows, active time, unclipped norm,
per-module clipped-gradient RMS and actual parameter-delta RMS. Primary
throughput criterion is at least1.5x targets/second over updates3–12, reported
also including warmup. Failure rejects a larger run for efficiency; a positive
result only permits a further bounded quality comparison, not adoption.
Stop on source mismatch, nonfinite/illegal results, resource limit or30-minute
active bound. CPU/four threads, per-process RSS8GiB, minimum available2GiB,
swap growth at most1GiB, diagnostic-output cap128MiB. No optimizer checkpoint
or final weights are retained from this short cost probe. Output owner is
`horizon-throughput-v1`; exclusive creation prevents overwriting evidence.

## Horizon throughput result and continued training comparison

Accepted revision/Card: none. Evaluation: REFINE. The twelve matched updates
per arm completed in304.220 total process seconds. Baseline targets total4,708
in99.417 update seconds; long targets total16,152 in197.057 update seconds.
Each arm computes exactly18,802 prefix rows. Excluding the first two updates,
throughput is48.194 versus83.002 targets/second, a1.7222x ratio that exceeds
the prespecified1.5x criterion. All twelve steps remain in the20-step LR
warmup; omitting the first two concerns cache/setup cost, not completion of
optimizer warmup. Canonical annotation reading overlapped part of the probe;
alternating arm order reduces but does not eliminate desktop-load confounding.
Peak process RSS is2,711,371,776 bytes, minimum available9,804,595,200 bytes,
maximum swap growth zero. Both arms clip all twelve updates; median pre-clip
norms are2.6769 and8.5773. Facts/local/relation/temporal/head/timing all receive
gradients and measured parameter updates. Median relative timing-update RMS
is.004646/.004917; comparable update magnitude does not imply equal direction
or equal training risk. Readout SHA:
`4d9564a2ed4ddda187b2e9d3598af4fc4e4543752ce15ec1494c97306321d8f1`.

The initial whole-workspace human reader was deliberately stopped while still
live because it exceeded this comparison's required source scope. A fresh
eight-source snapshot pins the original canonical document bytes before and
after reading; the same canonical reader expands content-addressed objects
and verifies their hashes through its owning adapter. This takes7.051 seconds,
returns32 current High judgments, and confirms all eight gold sources' prior
observation identities unchanged. Snapshot SHA:
`e3afbae97c50f03a4abf967e1c2d88962b54f374b074bb83fded84547b21bc95`.
No human source was edited. The26-page machine context review, with entering
holds, exact action witnesses and all render/action hashes, is
`quality-skeleton-large-u300/context-review.json`, SHA
`01cc110b3586885c280b9edcadd0061c64c4c5bd8983044f4a62711ebb1b25e7`.

The next run tests whether the throughput gain survives a meaningful learning
interval without worse held-out prediction or generated organization. Both
arms start from20M timing-u300 SHA
`008acf4f73c173b0f525e85664674a1188dd4917d263cc6d276ba044b76f550c`,
with fresh AdamW, sampling seed91, and identical model, optimizer, objective,
B8/cohort4/Q64 settings from the cost probe. Baseline horizons1/4/16s and long
horizons4/16/64s remain the only configured arm difference. This compares a
practical training allocation at equal update counts, not equal tokens or
equal compute; report all three explicitly. No gradient-norm or LR change is
introduced to conceal the longer arm's larger summed target loss.

Run base100, long100, base300, long300 sequentially in immutable runtime v7,
preserving each arm's AdamW and sampler RNG between its100 and300 readouts.
Each stage pins weights and scores the original24 windows; the300 stages also
score the fixed128-group manifest. Use full true prefixes in all validation.
Primary guard: long-arm pooled and equal-group broad NLL may not exceed base
by more than.02 nats/row; report paired group uncertainty rather than declaring
equivalence from a nonsignificant result. Maintain at least1.5x observed target
throughput as an efficiency criterion, with execution-order confounding stated.
Passing those numerical guards is only a candidate-selection condition.
Compare full5b69/e67f/ecc4 generations at seed17/temp1 first, then all eight
current-gold contexts and multiple seeds for any candidate worth retaining.
Check recurrence, moving attacks, coordinated holds/releases and transitions;
do not substitute LN fraction or source-tag equality for quality. Retain the
long-gap stress counterexample until the relevant candidate is tested.

Output owner is `exposure-continuation-v1`, with separate `base`/`long` run
directories and immutable stage-weight directories. One process lock protects
the experiment from duplicate drivers. CPU/four threads, RSS6GiB, minimum
available2GiB, swap-growth1GiB,512MiB checkpoint cap,25-update publications,
6GiB aggregate output budget and120-minute total process bound. Abort at a
resource/identity/nonfinite/legality failure; preserve the last atomic
checkpoint and report incomplete work. Do not overwrite pinned stages or
restart an already completed100 stage. This is an authorized exploratory
comparison, with no adoption, lifecycle transition or readiness claim.

### Live continuation and recovery observation

At2026-09-18 03:25 CST, the paired driver is live as Python PID57801
(tool session23227). The base arm has computed53 updates/21,516 targets;
its last durable checkpoint is update50,241,277,801 bytes. Logged peak RSS is
2,453,061,632 bytes, maximum swap growth zero, and the latest available memory
is11,330,617,344 bytes. These are observation-time values, not a later status
guarantee. The planned order remains base100, long100, base300, long300.
Poll this exact process or session and the stage readouts before any recovery;
do not launch a second driver while it is live.

The frozen driver is `exposure-continuation-v1/run.py`, SHA
`61c2d7e9dd51e21df7b4243bfd9fea61f316b4462ab22609cd8f26967d5e584a`.
Its persisted dataclass identity contains JSON lists where Python originally
used tuples. The separately checked `resume.py` normalizes that metadata before
comparison, leaving the original driver and model runtime untouched. Wrapper
SHA is `a5743fb1c460c5f241452950083a8b2ffd7a8eab0727cfab9dd3f3f8d1afa695`.
The normalized base configuration matches the saved manifest exactly. If the
process is terminal and work remains, first inspect incomplete stage directories
and checkpoints, then use:

```sh
PYTHONPATH=artifacts/oracle-time-continuation/m3-20260917/skeleton-time-runtime-v7/src caffeinate -i uv run --offline --extra mps python artifacts/oracle-time-continuation/m3-20260917/exposure-continuation-v1/resume.py
```

The driver skips completed hash-verified stage readouts and resumes the owning
run's AdamW/RNG through `run_training`. A directory interrupted between weight
copy and readout publication needs inspection before retry; do not overwrite it
or confuse its presence with a completed stage. The process lock remains the
second-writer guard. Each invocation's120-minute timer is separate; further
execution still requires reassessing the active comparison and resource state.

The20M timing-u300 starting weights now also have complete matched raw/seed17
5b69/e67f/ecc4 generations. They contain2,880/1,715/2,417 notes and913/861/288
LNs, maximum938/1,295/1,286ms, and complete in102.236 supervisor seconds while
base training overlaps. Readout SHA:
`3b0501e5c7cb06077cdfe3a0576b2e1ec71f11b1ea0542a34b65ea6d29fdab8d`.
All context pages/actions have been rendered in `quality-skeleton-timing-u300`;
their machine qualitative review remains pending. Counts alone do not establish
a quality advantage over77M. Product `oracle_time_m3_validation.md` now contains
the completed broad validation, coverage, timing/branch findings,77M context
review and horizon-cost measurements. Product code remains at the preserved
uncommitted implementation; no new product commit or remote push was made.

## Conditional timing signal diagnostic

Accepted revision/Card: none. The previous goal turn made progress: it completed
new controlled measurements, persisted their implications, and started the
bounded horizon comparison. Its existing driver remains live; no restart is
authorized by a mere observation timeout.

The20M starting model's thirteen generated context pages have now been inspected
against the previously checked, byte-identical source context. The ecc4 episode
contains clearer repeated-column chord organization than the77M sample;
5b69 mixes taps/chords with short holds and some independent LN interactions.
e67f still changes the reference's sustained tap flow into short-LN movement.
These are scoped machine observations, not a whole-chart pass or proof that
one model size is generally better. Exact actions and hashes will accompany
the context-review record.

A read-only population diagnostic asks whether known following-gap duration
adds measurable source action information after conditioning on pre-row
occupancy and the preceding gap. It does not change a model or decoding policy.
The analogue is an empirical conditional-frequency predictor, used as an
input-signal control rather than an alternative generator. Exclude seed rows
and the true terminal row. Verify canonical source/row-array identities through
the existing corpus admission API; process one chart at a time with at most1MiB
of row-array payload, never a whole-corpus tensor.

Use fixed gap boundaries0/31.25/62.5/125/250/500/1000/2000/8000/32000/infinity
milliseconds. Count ground-truth per-lane post-row occupancy, separated by
pre-row occupancy. Fit train-only smoothed frequencies for occupancy alone,
occupancy+preceding-gap, and occupancy+preceding+following-gap, with a.5/.5
Bernoulli pseudocount. Evaluate all admitted validation charts without fitting
on them. Report pooled and equal-group Brier, paired435-group bootstrap
uncertainty, per-gap-band errors, and raw start/continue/close counts. An
improvement with a paired group interval below zero establishes predictive
information in this control; it does not establish a neural transmission
defect, causal mapper preference, or a safe occupancy penalty. Sparse cells,
source-style mixing and the much larger exposure of this diagnostic than the
neural model remain confounders. Never infer that every long gap must be empty.

Output owner is `conditional-timing-signal-v1`, using immutable runtime v7,
CPU/one Torch thread, a30-minute active bound, RSS2GiB, minimum available2GiB,
swap-growth1GiB and128MiB output cap. Stop on identity, replay, numeric or resource
failure. Store aggregate counts and per-group scores; do not copy raw charts.

### Timing-information result

The diagnostic completed in 8.566 seconds, scoring 11,219,325 training and
1,827,293 validation nonterminal post-seed rows. Pooled validation Brier is
.06446172 for occupancy alone, .06382779 with the preceding gap, and .06344166
with both preceding/following gaps. The last improvement is -.00038614 with
paired group-bootstrap 95% interval [-.00047127, -.00030112]; 320 of 435 groups
improve. Equal-group improvement is -.00040498, interval
[-.00048923, -.00032417]. This establishes a small predictive signal in this
frequency control, not a causal attribution for the neural model.

The relation is not monotone: per previously closed lane, train LN-start rates
are .0674 before 125–250 ms gaps and .2524 before 2–8 second gaps. Held-out
2–8 second boundaries include post-row occupancy in 573 of 890 rows. At
8–32 seconds, 10 of 26 validation boundaries retain occupancy. The nine train
boundaries beyond 32 seconds and the sole validation boundary have no post-row
occupancy, but the sparse frequency model actually worsens that one validation
case because some conditioning cells are unseen. These observations reject a
general close-before-long-gap rule and retain the extreme tail as an evidence
gap. No model weights or decode rules changed. Readout SHA:
`b9a0934ce462976dc8064fd8c73ef187f0abf787449d5244e5eba0c5a1ebcab9`.

The completed 20M context review is
`quality-skeleton-timing-u300/context-review.json`, SHA
`eb94ee889a02090622e64ad2f727627ba27fd04d031ced8e65347c790a0f9b69`.
All eight canonical source-document hashes remained unchanged at review time.
Its scoped Jack judgment is a machine hypothesis; no human record was changed.

## Bounded shifted-stream feasibility probe

The ongoing horizon comparison remains unchanged. A separate prototype tests
whether less work per event can make broader training practical. The closest
analogue is the shifted autoregressive decoder in
[Attention Is All You Need, section 3.1](https://arxiv.org/html/1706.03762v7),
combined with the existing bounded recurrent temporal bank. An input at event i
contains only actions before i and permitted skeleton times, so its self-visible
token can be used to predict row i without reading that target. This changes
the neural cache meaning from post-content to pre-row history. It is not a
function-preserving rewrite and cannot reuse an old cache or claim an exact
resume from the existing architecture.

Compare the current two-stream model with two exploratory arms: a shifted
temporal stream retaining facts/local/relation inputs, and a shifted stream
using facts plus known timing directly. The latter removes learned local and
relation representations from the execution path; exact replay/occupancy and
joint legality remain. Retain the existing parameter tensors for this cost
probe, record which receive gradients, and do not publish its weights as a
compatible production checkpoint. Transfer likelihood is descriptive only;
the representation changes require a later controlled learning comparison.

Before timing, verify target/future-action isolation, mirror equivariance,
step/chunk forward and parameter-gradient parity, inference-cache parity across
archive creation, and parameter-version invalidation on a small deterministic
legal fixture. A prediction may prepare private cache state, but commit alone
advances the authoritative exact/neural row count. Failed legality must not
publish that state. Prototype outputs use a distinct cache signature.

Use the pinned 20M timing-u300 weights and frozen v7 operators. Select the two
lexicographically first TRAIN source SHAs with at least 1,800 events plus the
known longest TRAIN source; inspect full prefixes of 512 and 1,600 rows with
128 target rows each. Use Q64, B1, AdamW LR3e-5, weight decay.01 and structural
weight.3; compare the same cases in alternating arm order, on CPU/four threads
and MPS. Report prefix, target/backward, optimizer and total time separately.
An arm needs at least 1.3x total target throughput to justify a larger learning
trial; failing correctness rejects it regardless of speed. Concurrent corpus
training limits absolute device comparisons. No superiority or novelty claim
follows from this probe.

Output owner `shifted-stream-probe-v1` has a 30-minute active bound, 128MiB
diagnostic cap, 4GiB MPS driver/6GiB RSS limits, minimum available 2GiB and
swap-growth 1GiB. Stop on any invariant, nonfinite, identity or resource failure.
Record prototype source hashes before execution and leave the product runtime,
ongoing experiment scripts and current architecture plan unchanged until the
evidence warrants an explicit revision.

### Shifted-stream result and quality coverage

Accepted revision/Card: none. Evaluation: REFINE. The prototype's small legal
fixture passes the declared causal/cache/gradient/commit comparisons: the
target/future-action perturbation changes preceding logits by exactly zero;
maximum step/chunk gradient differences are 4.77e-7 and 3.58e-7. CPU/MPS
forward agreement is checked on the same fixture. Prototype SHA is
`a243c655bd8b5140cbdaf41206d6331c0f4b614f61dac6c3ebec78f4549bbffd`;
validation readout SHA is
`f5e8665eafc3ec6385bede98b685e8e820a49cf92a9c6e0bbe6ea22ee47080c7`.

The six matched B1 cost updates per arm/device complete in 212.010 process
seconds. CPU total update times are 35.955/32.395/24.817 seconds for two-stream,
shifted local/relation and shifted facts/time; MPS totals are
44.347/41.767/29.323. The full-input shifted arm gives only 1.11x/1.06x
CPU/MPS throughput and fails the prespecified 1.3x threshold on these cases.
The facts/time arm gives 1.45x/1.51x and qualifies for a separate learning
comparison. Prefix replay dominates this B1 workload; these ratios cannot be
multiplied by the horizon gain or substituted for actual B8/cohort4 costs.
Bypassing local/relation changes representation and transfer likelihood,
not just implementation. No production weights or model adoption result exists.
Readout SHA:
`ea7a42db12e4550218c50addbc60da24cbe28aa465d105a1adc7169fef89b9af`.

All three models/optimizers were retained in one process. CPU peak RSS is
2,503,786,496 bytes; MPS peak active/driver is 2,674,753,536/4,277,829,632 bytes,
with no added swap and at least 8,267,071,488 bytes available. The driver peak
is close to the 4GiB stop and is not a single-model or long-run envelope.
The raw path gives gradients to 19,332,928 parameters while retaining the
20,087,624-parameter allocation. The original product runtime is unchanged.

The horizon comparison's base100/long100 stages consume 41,133/143,020 targets
and the same 163,530 computed prefix rows. Update timers total 773.103/1,538.657
seconds. Ordinary 24-window NLL is 2.512699/2.545021: more target exposure has
not established a prediction gain at this checkpoint. The broad 300-step
comparison remains the decision point; do not silently adopt the longer risk.
Matched raw/seed17 complete generations for both100 stages are finished at
`quality-exposure-{base,long}-u100`; readout SHAs are respectively
`9bad189ea2fc318a429c9387cf1e4b6009564d86b22bc2a69283b97a1a3159ca` and
`25ada41b9ab2e53783bc8b31f4e81ef182706085abe70e7a17267db72013ad5c`.
All contexts are rendered. Full e67f context pages for both arms show short-LN
movement; the longer arm adds more tap/chord attacks around those holds. This
does not establish a whole-chart quality preference. The other two sources'
new generated contexts remain unreviewed.

At 04:27 CST the base300 stage is complete with 115,215 targets. Its ordinary
NLL is 2.5140093082; broad pooled/equal-group NLL is 2.3086116030/2.3373207177.
The starting model's broad values were 2.3179136000/2.3425084491, a small further
improvement whose uncertainty has not yet been estimated. Pinned base300 weights
SHA is `7665fc0bcb284af871cd065ff3824facbc88aa157d99ae3bae7182e4b7ab3604`.
The original PID57801 remains live and has moved to long300; inspect current
state before acting. Its120-minute invocation bound may require a later exact
resume. A timeout must first be confirmed terminal, then the existing checkpoint
and stage identities inspected; do not duplicate the live writer.

### Eight-gold temperature evaluation

Use the completed base300 weights above for the next free-running evaluation.
Temperature .85 has prior positive local evidence at timing-u100 but no broad
quality pass. Hold top-p1/beta0 and use seeds17/19 on all eight fixed current-gold
validation sources, generating complete charts from their minimum seeds.
The eight skeletons contain 19,148 rows per seed, or 38,296 total including seed
rows. This tests a practical candidate and decode choice, not a pure temperature
ablation against all prior checkpoints. Do not equate matching source tags or
LN fractions with quality.

Output owner is `quality-exposure-base-u300-t085`; it must be absent at launch.
Use frozen v7 generation, CPU/one thread, the existing resource/checkpoint guards,
an additional2GiB available-memory reserve,3GiB aggregate output budget and
30-minute process-tree bound. The supervisor records source/weight identities
and stops on nonzero child exit or a resource/deadline guard. Partial chart
readouts cannot be treated as the complete16-generation result. Inspect all
eight full gold review contexts for both seeds and broader chart transitions
before any final quality decision; retain unreviewed scopes explicitly.
The ongoing horizon training remains independent, and no remote publication,
note lifecycle transition or goal completion is authorized by this evaluation.

### Updated execution and uncertainty

The base300 versus its 20M timing-u300 starting weights comparison has now been
bootstrapped on the same 128 groups. Pooled change is -.0093020 nats/row, with
95% interval [-.0350849, +.0123033]; equal-group change is -.0051877, interval
[-.0372087, +.0267725]. Only 66 of128 groups improve. Thus the point estimate
does not establish a further general prediction gain from this continuation.
The full evidence is `exposure-continuation-v1/base300-vs-initial-broad.json`.
Low supervised coverage remains a limitation, not a proven sole explanation
of the generated-organization failures.

The paired e67f update100 review covers all eight generated pages plus four
byte-verified source pages. Base uses more release-only exits; long interleaves
more attacks/chords with its short holds. No whole-chart preference is assigned.
Review SHA is `b70f65fa7e281862d43148ea39f50d0e18ef1ccc84e7beee7d0beb04341912bc`;
file `exposure-continuation-v1/e67f-u100-context-review.json` retains scope,
action witnesses and image hashes. The two other sources' new100-step contexts
remain explicitly unreviewed.

At 2026-09-18 04:40 CST, original training PID57801/session23227 is still live.
The long arm has computed134 updates/196,967 targets; its durable checkpoint
is update125,241,277,929 bytes. The eight-gold temperature evaluation is also
live as Python child PID243, supervised by tool session55417. Four of sixteen
source/seed generations have complete readouts; the rest are not yet complete.
PID numbers can be reused, so verify their command lines or poll the original
session handles before acting on these snapshots.

The temperature supervisor is
`exposure-continuation-v1/generate_gold_085.py`, SHA
`cc71c61892cca9464df42ae5f7f335782a3b14b5d86a129259879ef8ef1534ba`.
Its `gold085-manifest.json` pins all eight sources, both seeds, weights and
the unchanged canonical-human document hashes. `gold085-status.json` appears
only after all sixteen outputs are verified. If its30-minute bound stops it,
retain completed outputs and inspect the incomplete chart's checkpoint; do
not rerun the fresh-output script against an existing target directory.

All32 gold observations were checked to occupy exactly one scope/context pair
per source, so the existing renderer's first-context selection covers the eight
intended contexts. On completion, render the complete `quality-exposure-base-u300-t085`
readout, inspect both seeds at every gold context and broader transitions, then
test any surviving quality candidate against the long-gap failure. The shifted
facts/time prototype remains a separate candidate for a properly identified
learning/inference trial; its cost gain is not permission to call the playable
model goal complete. Product analysis now includes the conditional-time control
and shifted-stream cost evidence, with no product runtime change or remote push.

### Paired horizon result and broader quality failure

Accepted revision/Card: none. Evaluation: REFINE. Both arms complete 300 updates
with all 2,400 stochastic window choices paired and 476,759 computed prefix rows
each. Base supervises 115,215 targets; long supervises 408,820. Broad 128-group
pooled NLL improves 2.3086116030 to 2.2597199507, difference −.0488917 with paired
95% bootstrap interval [−.0719151, −.0273644]. Equal-group NLL improves 2.3373207177
to 2.3015570823, difference −.0357636, interval [−.0589207, −.0131351]; 81/128 groups
improve. These results pass the declared prediction regression guard. Ordinary
24-window NLL also improves 2.5140093082 to 2.4295380055. This supports the horizon
policy on these prediction metrics, not playable generation or a pure
supervised-exposure mechanism: the longer target span also changes training risk
and gradient aggregation.

Retained update timers total 2,399.348/4,576.653 seconds, giving 1.860x target-row
throughput. This excludes discarded work: the original invocation hit its
7,200-second deadline during update 298 and exited. After verifying process exit,
the released driver lock and complete 275-step checkpoint, the unchanged runtime
resumed through the JSON-normalization wrapper. Updates 276–297 were recomputed;
the observed loss, gradient norm, row counts and cumulative targets at 297 match
the pre-timeout values exactly. Recovery plus remaining validation takes 470.544
seconds. Concurrent probes/generation and discarded work prevent treating the
retained-step ratio as total wall throughput. Peak logged RSS is 2.749 GB, minimum
available 8.267 GB, and logged swap growth 0. Final long weights SHA is
`7160e330d64d9224362c5c923144ac37fb1e7966f527c933b64c09ffc79eb9f4`.
`exposure-continuation-v1/paired300-readout.json` SHA:
`ef29cd6c91a4183172ed2bc2971213ca54b8ffc13489dcebc6b7e465ca9e27de`.

Base300 temperature .85 evaluation completes all 16 full charts in 1,042.159
supervisor seconds. All 58 generated gold-context pages and 16 additional source
pages were inspected; 13 previously inspected source pages were byte-verified.
The eight canonical human documents remain unchanged and all 32 references are
current. Local moving tap flow and independent LN control are present, with
large seed differences; source-style mismatch is not itself a failure. The
review retains unresolved dimensions rather than manufacturing negative labels.

A full-chart recurrence locator then identifies concrete additional failures.
On 85058a/seed17, column 2 taps at 74634/74658/74682/74706 ms, four attacks 72 ms apart
from first to last. On 871955/seed19, column 0 taps at 137342/137360/137386/137407 ms,
four attacks in 65 ms amid dense changing chords. The corresponding source events
include release-only rows and separate LN heads. On ecc496/seed17, source moving
single attacks at 137500/137535/137571/137607/137643/137678 ms become repeated chords;
column 1 attacks on all six times over 178 ms. Complete surrounding 4-second
contexts were inspected for all three, 12 source/generated pages, retaining
entering holds and raw endpoints. These are post hoc failure locators, not
universal Foundation label thresholds. The candidate fails whole-chart playable
quality despite valid export and some coherent gold-context passages.

`quality-exposure-base-u300-t085/context-review.json` SHA:
`0e0663a151fbb4aaf54c708c88f42c3307369369630423a89a155ea52b01973b`.
Its full-generation readout SHA is
`4c405571f5db76dbad12374f52185e243b36ab65f09589c6b9b3cffc044642fa`.
Other whole-chart passages remain unreviewed, no player trial occurred, and the
prior 89-second-LN stress case has not been retested on this checkpoint.

### Long300 generation at the new counterexamples

Before broad adoption, generate the complete 85058a, 871955 and ecc496 validation
charts from the pinned long300 weights, seeds 17/19, temperature .85/top-p1/beta0,
with the identical minimum seed and frozen v7 generation implementation. The
three sources are selected post hoc from the base300 failures, so this is a
targeted failure check, not an unbiased new quality estimate. First compare the
complete gold contexts and the previously located rapid-recurrence neighborhoods,
then inspect each new output's own fastest recurrence. A surviving candidate
still requires the remaining five gold sources and the long-gap stress case.

Use fresh owner `quality-exposure-long-u300-risk-t085`, CPU/one thread,
30-minute process-tree bound, 3 GiB aggregate output cap, minimum 2 GiB available
memory and existing generation/checkpoint guards. Keep completed per-chart
outputs on failure; do not rerun the fresh-output supervisor over them. Do not
modify sampling, insert timing restrictions or copy source actions in response
to the discovered failures. Record the supervisor identity before execution;
no note lifecycle transition, product adoption or remote publication follows.

The bounded supervisor is `exposure-continuation-v1/generate_long_risk_085.py`;
its SHA-256 is
`4e165415a65121192334b4816c68d8d145db424bdabf714175f999d0837f0610`.
The run manifest also records the six source/seed pairs, pinned weights and
current-human snapshot identity.

### Rapid-recurrence probability diagnostic

The six long300 generations complete, but recurrence extrema still reach four
same-column attacks in 65–96 ms on 85058a/871955. Before changing the model or
decoder, test whether these are low-probability sample tails or high-probability
decisions under generated history. At the three base300 failure locations,
replay each full source prefix and each fixed base300 generated prefix under
both base300 and long300 weights, then inspect four consecutive original rows.
Record raw and temperature.85 lane-attack probabilities, expected attack count,
multi-attack probability and LN-start probability. Source/generated legality
differs; do not interpret their difference as an isolated history effect.

For each state, compare the actual query with two valid time-only counterfactuals:
insert a 250-ms previous-event gap while translating future times to preserve
their offsets, and multiply future offsets by four while preserving the current
time and all history. These query-only interventions are never committed and
do not retime generated outputs. The first changes all elapsed query clocks,
not just one encoder; the second changes only available future-time information.
Report probability changes and JS distance, without attributing attention or
activation size to causality. This is a post hoc three-case diagnostic, not a
global calibration estimate or evidence for a blanket timing restriction.

Use frozen v7, CPU/one thread, a 10-minute active bound, 2GiB RSS limit,
minimum 2GiB available memory and 128MiB output cap in fresh owner
`rapid-recurrence-probe-v1`. Verify source/generated time alignment, model hashes,
unchanged legality across each time intervention and direct-model/engine
agreement for the actual query. Save no weights and change no product runtime.

### Long300 quality and recurrence-probe results

Accepted revision/Card: none. Evaluation: REFINE. All six long300 risk outputs
complete in 247.041 supervisor seconds; generation readout SHA is
`76ebf3b7bc546b0995ea734765977133e69e76d940ce31866d920363fa765a2d`.
The review covers all 20 generated gold pages, 12 generated pages at the fixed
failure contexts, and eight source/generated pages around ecc496's two own
recurrence extrema. Ten gold-source and six fixed-failure source pages were
byte-verified against already inspected pages. The first four outputs' own
extrema fall within the fixed inspected contexts; their separately rendered
wider contexts remain unreviewed. All canonical human documents remain unchanged.

Long300/871955 now has coherent moving taps in both gold contexts. ecc496/seed17
has strong changing-chord recurrence, while seed19 has mainly moving taps. These
are machine hypotheses under the frozen Foundation, not new human labels.
Nevertheless, 85058a's fastest four same-column attacks span 72/96 ms across
seeds17/19, and both871955 outputs span 65 ms. ecc496/seed17 retains the rapid
chord burst near137500 and has another near92357; its fastest six attacks span
178 ms. ecc496/seed19 distributes the prior burst across columns and has a
214-ms fastest-four span elsewhere. The targeted generation comparison therefore
fails the whole-chart quality gate despite the likelihood improvement. This is
not a blanket rejection of longer horizons as an efficiency policy. Review SHA:
`23b5b64fa8d3f69c826d9fd089ef19647c51db3d49f0bde3dfb7e40a4b4a00f9`.

The read-only recurrence probe completes 48 original-row cases and 144 query
distributions in 43.241 seconds. On the ten selected generated-history rows
whose inspected lane was attacked at most40 ms earlier, temperature.85
repeat-attack probabilities range .141–.650 for base300 and .305–.798 for
long300; means are .435/.554. Both models are evaluated on the identical
base300 histories here, not their separately sampled trajectories. These
decisions cannot all be explained as extremely unlikely sample tails.

Inserting the250-ms previous gap increases those probabilities by .021–.100
and .026–.068 respectively: elapsed clocks do affect the output in this test.
Multiplying current-query future offsets by four changes the inspected lane's
probability by at most .001923/.005291. This small direct response does not
prove that all future-time information is unused: earlier cached history still
contains its original future-time inputs. Source-history predictions also
overpredict attack multiplicity at ecc496's moving burst, so generated-history
drift alone does not explain the failure. Source/generated legal support differs
on LN examples and is not a controlled occupancy intervention.

Probe SHA is `a692d5621a4249b50886ea324376fbfc714f3fe5a7721c437c65356d7bc9b876`;
`rapid-recurrence-probe-v1/readout.json` SHA is
`954e947a37ec259b490f9df6cdb4eb532e85b374bfd10d725bab9a08f300263f`.
Peak RSS is 1,175,994,368 bytes, minimum available10,978,951,168 bytes and swap
growth zero. Source hashes/time alignment, query legality and direct/engine
agreement pass the recorded invariants. No weights, sampling rule, product
runtime or human annotation changed. All training/generation/probe processes
described in this result are complete; there is no pending writer to resume.

The next design question is whether a direct learned physical-clock path into
the joint output can make dense-time decisions trainable without being diluted
by the history stack. A zero-initialized residual readout could preserve the
current function at initialization and permit a matched comparison against
continued long-horizon training. This remains a hypothesis, not an implemented
model or an accepted experiment. Preserve alternatives: insufficient exposure,
the event-union requirement under all-closed generated history, and history
dependence can also contribute. Do not replace the learned task with an
unvalidated minimum-key-interval rule, and do not infer final playability from
the improved validation loss. The goal remains active and unmet.

## Proposed experiment: direct physical-clock readout

Card ID: oracle-time-clock-readout-v1. Revision: 1. Accepted revision: none.
The user-authorized implementation/run remains exploratory against product
revision `f679269b92e96efb5bd7989e748bd42cf069379e` plus the recorded dirty
worktree. Preserve a preimplementation file manifest and freeze a new runtime
before execution; existing v7 and its checkpoints remain immutable. Evaluation
cannot be SUPPORTED or an acceptance claim from this dirty baseline.

Question: can a short learned path from physical query clocks to joint-row
scores improve fast-time decisions without giving up the history model's
organization? Current long300 broad NLL is 2.259719951 over 128 groups/5,709
targets. Its .85 outputs fail the selected recurrence cases, and the fixed
base300 histories give long300 .305–.798 repeat probability on ten rows with
lane-attack age at most40 ms. Those selected diagnostics motivate the experiment
but are not an unbiased population estimate.

Intervention: add a shared-hand MLP producing 16 additive hand-pair unary scores
directly from pre-row lane LN/attack/release ages, hand attack/release ages,
previous-event gap, elapsed chart time, prior complete-row actions, occupancy,
terminal flag and the existing time-only lookahead offsets/gaps. Use the same
bounded/asinh clock basis and relative hand coordinates. A hidden width of128
with16 lookahead rows adds152,080 parameters. Zero-initialize the final layer;
copy every existing parameter from long300 and verify identical initial logits.
The joint 256-row family, coupling, legality, event skeleton and cache state
remain unchanged. No future action, source LN pairing or annotation is input.

This is an additive conditional-output adaptation of the existing joint head,
not a new output family or novelty claim. Hypotheses remain distinct: a shorter
gradient path may learn timing-sensitive scores; continued exposure alone may
give the same benefit; or the all-events task/history dependence may require a
different intervention. Failure to improve the paired quality cases leaves the
direct-readout hypothesis unproven even if its parameters receive gradients.

Control and intervention start from long300 weights
`7160e330d64d9224362c5c923144ac37fb1e7966f527c933b64c09ffc79eb9f4`, with fresh
AdamW, seed20260919, B8/cohort4/microbatch1, Q64 and horizons4/16/64 seconds.
Keep LR3e-5, timing LR1e-3, weight decay.01, warmup20, clip1, structural weight.3
and denominator1,024. Only the new clock-readout parameters receive a separate
LR1e-3. Existing groups otherwise retain their settings; global clipping means
the new branch can also change the effective backbone step. Pair all group,
chart, stratum, start and horizon draws. Preserve each arm's optimizer/RNG
between pinned stages; do not call fresh-optimizer continuation exact resume.

First run100 updates per arm and score the fixed ordinary and128-group windows.
Report pooled/equal-group NLL and paired group-bootstrap uncertainty. Stop and
refine if the new arm is worse by more than .05 nats/row in either broad
aggregation at100; otherwise the possible300-stage continuation requires review
of the100-step evidence. The prediction guard for a retained candidate is no
worse than control by .02. Full-chart .85/top-p1/beta0 generations on the same
three failure skeletons and seeds17/19 must be inspected for both arms. Rapid
repeated-chord failures must improve without simply replacing the chart with
another broken organization. A surviving candidate still needs all eight gold
contexts, multiple seeds, the long-gap counterexample and wider transitions.

Before corpus training, verify zero-initialization equivalence, action causality,
mirror equivariance, physical-time translation, step/chunk/gradient agreement,
typed Hydra projection and real runner consumption. Exercise the new optimizer
group in exact training resume and CPU/MPS generation/export recovery. Check
that both clock layers learn after the zero-output first update. This code
change must not weaken existing legality or cache/version guards.

Use CPU/four threads for training and one for rollout. Each invocation has a
120-minute bound, checkpoint every25 updates, 512MiB checkpoint cap, RSS6GiB,
minimum2GiB available memory and swap-growth1GiB. Keep the training owner
`clock-readout-v1` below6GiB; separate generation owners receive3GiB each and
30-minute process bounds. Stop on identity, nonfinite, causality, resource or
writer-conflict failures. Resume only after confirming the old process terminal
and inspecting partial stage publication. Record source/runtime/init identities
before launching; no remote publication or lifecycle transition is implied.

### Clock-readout implementation and initial parity

The optional path is implemented in the product worktree with default width0;
seven runtime/config files, four test owners and two interface/plan documents
change relative to the preimplementation snapshot. The snapshot contains the
50 inherited modified/untracked files, with manifest SHA
`e623642066483791b87baf4f5e295f209d2dab639645e7d794d3f1dbc246c928`.
`clock-readout-v1/implementation-manifest.json` records the scoped final file
hashes and diff, SHA
`84974b3131ef004a58b4157be1eff5a2d6b9f53435b2bb20eaf530e713e35966`.

Selected evidence comprises55 model/batch/Hydra tests, three periodic-resume
and CPU/MPS generation/export recovery cases, and12 training/package tests
with21 subtests. The actual-runner test was then extended and rerun to verify
zero first-layer gradient at update1 and positive gradients in both clock
layers at update2. `selected-checks.json` records exact commands and results;
SHA `d5100d23ffc097a72939f11afdac5f80706debc6f701ddd56ed2de3f4b152e72`.
This is selected evidence for the scoped change, not full inherited M3 readiness.

Frozen runtime v8 manifest SHA is
`09469489b1d5b0d3f9e92890fa0ae6f58c3a9ecc2bf78cad977a865a3d16790e`.
The initial128-group validation reproduces2.259719950719731 NLL for both arms,
with exactly zero per-case NLL difference over5,709 targets. It takes271.186
seconds and changes no existing parameter. The new model has20,239,704
parameters, including152,080 added clock parameters. Initialized clock weights
SHA is `99a5ae9f72ba86ee3d477d47d238c2c8a950541af5f197fde2125c4ffe192750`;
initial readout SHA is
`1aa1d21093d85d51a6881c4c1247b4a558053aad4e121c733fc85ea3dabbf4e6`.

The training driver `clock-readout-v1/run.py` SHA is
`6b5cdce61b79b7ff3799725068daab15380684c9572a0895cacb262c4f5863e6`;
shared helper SHA is
`c26a477bdd4dbb9bf01112cc45e60d90eb90ba9e63b2c84a3d9c9d748eee08b9`.
The first invocation requests100 updates per arm. Sampling and added-branch
initialization use seed20260919; model construction retains preset seed17 and
then loads every pinned parameter. There is no dropout or other training draw
from that construction RNG. The driver records post-clip clock gradients and
actual parameter-update RMS without changing the optimizer step. Its identity
comparison normalizes dataclass tuples before comparing persisted JSON, and
its kernel lock protects against duplicate drivers. Resume uses this same
entrypoint with100 only after terminal confirmation;300 is a separate reviewed
continuation. No default sampling or trained-model adoption changed.

### Live clock-readout comparison state

At 2026-09-18 09:30 CST, training PID35757/tool session24456 is confirmed live
under `clock-readout-v1/run.py 100`. Control100 is complete with129,843 targets,
ordinary NLL2.4413434481, broad pooled/equal-group NLL2.2923368436/2.3293640517.
Its weights SHA is
`2d033d4c39e9ebcb14ce5723e819709c3a42e5cb23cf715246cf0dc5e77de88d`.
The clock arm has computed14 updates/24,632 targets and is still training.
Both arms' first update has exactly the same1,705 targets and2.3339998066 NLL.

On real corpus update2, the new input/output weight gradient RMS values after
clipping are4.10e-7/9.94e-4; actual update RMS values are6.03e-5/8.05e-5.
At update1 the input gradient is zero and its tiny8e-9 movement is decay, while
the zero output layer learns. These observations distinguish learned updates
from mere parameter presence or decay, but do not establish quality benefit.

The control generation supervisor is also live: tool session50833, Python child
PID52843. It is producing the six complete outputs in
`quality-clockreadout-control-u100-t085`. The supervisor is
`clock-readout-v1/generate_quality.py`, SHA
`667016338e507cd29e616e03294af8c40ef6762d6be13c81d21591dccf492e4b`.
It supports `control|clock 100|300`, checks the pinned stage readout and canonical
human document hashes, and writes a completion status only after all six charts
are verified. Fresh-output semantics prohibit blindly rerunning a partial target.

The read-only post-training diagnostic `clock-readout-v1/probe.py` is prepared
but not run; SHA
`ae181e3be3c053069d571e8d71e3f6836f77ccc29e8c8e27380b4991feaf7cdb`.
After both100 readouts exist, invoke it with100 using frozen v8. It compares
control/clock on the same fixed base300 generated histories and source prefixes,
preserving the previous query-only intervention limits. Do not substitute these
fixed-history probabilities for free-running quality.

Training resource timestamps contain long gaps while retained update timers
are much shorter: observed neighboring samples differ by931/381/174 seconds.
The caffeinate process is verified to hold a PreventUserIdleSystemSleep
assertion. The cause of these gaps is not established; report wall spans and
update timers separately, and do not label them model computation or compare
uncontrolled wall throughput. The process is advancing with no logged swap
growth. The last explicitly loaded control checkpoint was update50,241,277,929
bytes; control subsequently finished100 and published its stage normally.

Continue by checking the original live handles, collecting both100-stage
readouts and the paired bootstrap, then generating the clock arm through the
same supervisor and reviewing the six matched outputs per arm. Existing
`render_quality.py` and `inspect_quality_recurrence.py` can operate on the new
quality tags under v8. Preserve source/inspection hashes and unresolved scopes.
If training terminates early, inspect its latest atomic checkpoint and any
partial stage, then resume `run.py 100` with the same frozen v8 and identities;
never duplicate a live writer. Product code and the validation report remain
uncommitted, note status remains proposed, and the playable-model goal is active.

### Control100 quality and event-union support audit

Accepted revision: none. Card: oracle-time-clock-readout-v1 revision 1 for the
paired training/generation; the descriptive support audit is exploratory.
Evaluation: REFINE. All six control100 generations finish in 513.293 supervisor
seconds, readout SHA
`e3974ab54e1212bf723bc43958ab4ee82c81e16b7cd5bdbeb539170a96f9406e`.
Inspection covers 20 generated gold pages, 24 own-extremum source/generated
pages and 12 generated fixed-failure pages: 56 files/54 distinct images. Another 32
source-page files match the previously inspected long300 sources byte for byte.
All eight canonical human documents retain their recorded hashes. The review
SHA is `7e40ae8770c7d3e38c513d9eb942662d428deaf0edb7bf7bdb10e649a5587f7b`.

All six gold contexts have coherent moving taps with small chord accents;
unresolved Jack/Trill/Tech dimensions remain unresolved. The fixed ecc496 burst
near 137500–137678 ms is improved across both seeds. Nevertheless, 85058a/seed17
still repeats one column four times in 72 ms, and both 871955 outputs repeat a
column four times in 66 ms, amid larger repeated-chord bursts. Control100 fails
the whole-chart quality gate. Other full-chart passages, five remaining gold
sources and the 89-second-LN stress case remain unreviewed for this checkpoint.

`clock-readout-v1/recurrence_metrics.py` adds full-chart descriptive counts and
rolling-span quantiles; script SHA
`6f380d82dc0f614721c49850eaf7ca92eb00c707e1edd8af5085ec9cd113dd3d`.
The control metrics SHA is
`e6529f9b0e5569ee510c2dab2596d4ecc6f2866a5275aeaac8f487297c23e1b3`.
For 871955 seeds 17/19, same-column adjacent attack gaps below 40 ms number 301/241,
versus 472/497 under the long300 initialization. These counts overlap in episodes
and are locators, not Foundation thresholds or a standalone quality score.

The research scope is reopened to challenge the entire model, experiment
design and learning process, including the skeleton definition and explicit LN
conditions. A read-only audit therefore compares source event roles with the
generated pre-row occupancy on all 12 completed long300/control100 outputs.
It replays legal state transitions, checks time alignment and terminal closure,
and uses source roles only as analysis labels. No generation input changes.
The audit has a 120-second bound and 32 MiB output cap; it completes in .391 seconds.
Its script SHA is
`dd92ddc9cceae7f8f91f1c93874c927bee1d00ede7ab665c7e1f9ffddcdd5f94`;
`clock-readout-v1/support-audit.json` SHA is
`4b7770d2727f05c8a94df77cd5c2c9708c4dd6760413dfeeed70dff59f0f91eb`.

In 85058a's continuation, 247 source rows contain releases only. Both control
seeds reach 242 of these times with all four lanes closed. In 871955 the analogous
counts are 451/400 out of 584. Given closed lanes, exact nonempty event-union
coverage forces at least one attack. It does not force any particular column,
multiple attacks, or the earlier decision not to open an LN. For 871955, 117/92
of 301/241 fast reattacks end at source release-only times; 94/63 occur there with
all generated lanes closed. These are descriptive intersections, not an
identified causal fraction. The older long300 ecc496/seed17 output has 73 fast
reattacks and none end at release-only times, defeating a single-cause account.
The product validation report now preserves this conditional support mechanism.

### Reopened timing contract and learning problem

The goal remains playable generation on the available Mac and corpus. Existing
module structure, exact source-event coverage, 16-row timing lookahead and the
training-window policy are revisable research choices. Preserve source identity,
split isolation, valid gameplay actions, declared inputs and honest whole-chart
quality evaluation. The V3 formulation already allows a planning representation
to encode row absence; it forbids materializing an all-zero chart row. A
candidate scheduler can therefore distinguish NO_EVENT from per-lane EMPTY
without changing that chart language.

Live alternatives answer different questions:

- A declared oracle LN skeleton can expose event roles, counts or complete
  head/end pairs while withholding lanes. It isolates choreography from LN
  planning. Exact pair timing is substantially more information than an LN
  proportion or mode, and success would be conditional on that oracle. Release
  tags alone do not guarantee consistency with a generated closed state.
- A candidate-time skeleton permits NO_EVENT and learns event selection. It
  needs real absence supervision and the same candidate-generation process in
  training and inference. Adding easily recognized random decoys may teach only
  decoy rejection; it does not automatically teach recovery from mismatched LN
  histories. Prevent an all-empty or sparse-tap collapse from counting as success.
- A note-object or hierarchical event plan predicts LN duration/end choices at
  the opening decision and learns lane choreography conditioned on that plan.
  Releases then follow generated obligations. This changes the model's
  factorization and requires evaluating predicted plans, not only teacher plans.
- A larger known-time encoder can read global or phrase-scale skeleton features
  without seeing future source actions. The current 16-row limit is not an
  information constraint in the user's task. Its cost must be compared with
  usable supervision throughput and full-chart behavior.

The closest local analogue for candidate absence is the formulation's optional
row representation; paired LN objects already exist in source parsing but are
currently excluded from inference inputs. These are adaptations/factorizations,
not novelty claims. Broader approaches use different supplied information:
Mug-Diffusion's primary README describes audio and LN-ratio/style/difficulty
conditioning, while Mapperatorinator uses spectrogram inputs. Their public
descriptions do not establish a matched quality comparison with this oracle-time
task. Sources: [Mug-Diffusion](https://github.com/Keytoyze/Mug-Diffusion) and
[Mapperatorinator](https://github.com/OliBomby/Mapperatorinator), inspected 2026-09-18.

No replacement contract or experiment is accepted or implemented by this
exploration. Complete the already-running 100-step readout and its planned
quality comparison; do not automatically expand to 300. Select the next bounded
test by whether it separates supplied-information, legal-support and learning
failures. Compare supervision exposure and fixed compute as well as update
counts, and do not compare per-candidate NLL containing many null targets directly
against the current event-only NLL. Keep oracle-condition success separate from
end-to-end generation quality.

### Clock100 prediction result and launched quality follow-up

The original training session 24456 exits successfully after both 100-stage
readouts and the paired comparison complete. Each arm has 129,843 targets and
800 exactly matched window draws. Clock ordinary NLL is 2.4212722215; broad
pooled/equal-group NLL is 2.2752402496/2.3118684842. Against control100, pooled
delta is −.0170966 with paired 95% bootstrap interval [−.0280513,−.0065015], and
equal-group delta is −.0174956 with interval [−.0276027,−.0072840]. 72/128 groups
improve. This passes the planned prediction guard; both broad clock values
remain above the long300 initialization's 2.2597199507/2.3015570823. A paired
improvement is not evidence of a new overall best model or a quality pass.

Clock weights SHA is
`29a2a29e8dd901b72447f24884336adf78349264bd7d61846976d7e4f6856641`;
`clock-readout-v1/paired100-readout.json` SHA is
`3d1ae52ecb6dee4af15108db6bda28eaa40a3ea4190e66aa46ec4eecca7787fc`.
The driver records a perf_counter duration of 3654.111 seconds; wall-clock
gaps still prevent a clean elapsed-throughput comparison.

At 10:05 CST, the existing bounded supervisor launches
`generate_quality.py clock 100`, tool session 12313, generation child PID 65373,
for the six full charts in
`quality-clockreadout-clock-u100-t085`. The prepared fixed-history `probe.py 100`
also launches, tool session 16760. Both use frozen v8 and their recorded guards.
Inspect their exact handles and completion files before any further action;
never restart over a partial output. No 300-stage training is launched.

The probe session 16760 subsequently completes all 48 original-row cases and
144 query distributions, in 65.232 seconds. Its readout SHA is
`64146253ef1f190201aeb81a2892660e4a4941c1a2a21d6a03a032442f556e1a`.
On the ten fixed base300 generated-history rows with inspected-lane attack age
at most 40 ms, temperature .85 repeat probability has control mean .439021,
range .294352–.600548; clock mean .234077, range .026611–.486257. A 250-ms prior
gap increases it by .0387–.0837 and .0049–.1333 respectively. Current-query
future-offset expansion changes it by at most .014567/.011048. All previous
query-only and fixed-history limitations remain. The clock arm changes shared
weights as well as the new branch, so this is not an isolated readout ablation.
It does establish room for a learned response on these states, alongside the
conditional support constraint established by the event-role audit. Whole-chart
clock quality is still pending; generation child PID 65373/session 12313 was
confirmed live at 10:06 CST. Both findings are preserved in the product report.

### Clock100 generated recurrence and broader follow-up

The user explicitly supports free exploration of these directions and permits
redefining the supplied time skeleton for experiments. Exact event-union coverage
is not a fixed requirement of the user goal. This grants experiment scope, not
an automatic Note acceptance, model adoption or remote publication.

All six clock100 generations complete in 390.889 supervisor seconds. Readout
SHA is `266c531e16e98558bc954c24a162d09abca0c8e3406c91a3b1348181796c3ca7`.
Review covers all 20 generated gold pages, 24 source/generated own-extremum
pages and 12 generated fixed-failure pages. Another 32 source-page files match
the inspected control references, and the eight human documents are unchanged.
Review SHA: `38edc3861d404e67b326b862934a0e655fad593410cec11ee51f29de5b7f0ff9`.
Metrics SHA: `2084d4910d324f887f87c45f67b93073758865d6cfd90dc8f0fc303100da33d8`.

For 85058a/871955/ecc496, seeds 17/19 respectively, the fastest four attacks in
one column span 209/169, 131/174 and 250/214 ms, versus control's 72/145, 66/66
and 178/143 ms. Full-chart adjacent same-column attack gaps below 40 ms decrease
from 583 to 45 across the six outputs; 871955 alone decreases from 301/241 to
16/17. These are descriptive locators with overlapping windows, not quality
thresholds. The fixed dense repeated-chord failures are substantially reduced,
and the complete contexts retain moving single/chord organization. In the
85058a/seed17 context, columns 0/3 hold over 102254–103215 and 102446–103312 ms
with intervening inner-column taps; seed19 instead remains mainly tap flow.
In ecc496, seed19 has stronger moving chord accents than seed17. Machine labels
and unresolved strengths are retained separately from the source human gold.

Residual concerns include the four column0 attacks over 131 ms near 257320 in
871955/seed17 and isolated 24–35 ms pairs elsewhere. This is a meaningful
targeted improvement, not a whole-chart playability pass. Keep clock100 as a
candidate for broader inspection; do not infer that the existing task definition
is either sufficient or impossible from this slice. No 300-step extension starts.

Complete the existing Card's broader candidate check with the other five gold
validation sources (5b69,713ef9,98357f,e67f,ece738), seeds 17/19, plus one seed17
full generation of the known TRAIN long-gap stress source 1022f1. Use the pinned
clock100 weights, temperature .85/top-p1/beta0, minimum 30-note source seed and
unchanged frozen v8. The remaining gold results are candidate checks without
new control100 generations on those five charts; the older 89-second-LN result
also differs in checkpoint and may differ in decoding. Neither is a new causal
ablation, and the TRAIN stress result cannot establish held-out generalization.

The fresh owners are `quality-clockreadout-clock-u100-gold-rest-t085` and
`quality-clockreadout-clock-u100-long-t085`. The supervisor
`clock-readout-v1/generate_coverage.py` takes `gold-rest` or `long`, runs CPU/one
thread, bounds each child process tree to 30 minutes, requires 2 GiB available,
caps each owner at 3 GiB and retains the runner's checkpoint/replay guards.
It validates pinned weights/runtime/catalog/split and current human documents
before launching. Per-chart completion is journaled; a final readout/status is
published only after every requested chart finishes. Preserve partial outputs
and inspect the child before any resume; never rerun the fresh-output supervisor
over a partial owner. No weights, decoder or human labels change in this check.
The supervisor SHA is
`2f37dbf6e070b2ab6a5a85dbdadcb7121d88e397812576b7f75b75f26cf485de`.

### Exploratory representation audit: onset skeleton and learned LN endpoints

The next concrete design branch supplies onset times and learns note types,
columns and LN endpoints as linked decisions. Generated endpoints create their
own release obligations; unused release candidates need not materialize rows.
This supplies onset-role information beyond the old untyped event union, while
leaving LN pairing and column assignment as outputs. It is a changed conditional
task, permitted by the explicit skeleton-redefinition scope. No product contract
or implementation is changed by selecting this audit.

The closest analogue is REMI's replacement of separated note-off events with
note-duration prediction, described in section3.2 of
[Pop Music Transformer](https://arxiv.org/pdf/2002.00212). The transferable idea
is representing onset and duration together; the paper's piano/metrical-time
representation and quality results do not establish 4K choreography quality.
Pulsefield would retain raw milliseconds and explicit lane occupancy. Supplying
future source endpoints at inference is a separate oracle diagnostic, not this
learned-endpoint proposal. This is an adaptation, not a novelty claim.

Before implementing a duration or endpoint head, audit all11,564 admitted TRAIN
charts under the pinned catalog/split and frozen v8. Count event/onset/release-only
rows, LN instances, releases coincident with onsets, simultaneous LN heads with
different ends, and distributions of physical duration and endpoint rank in the
original event union. Verify each lane's ordered start/end pairing and final
closure. Retain per-chart summaries and a bounded set of extreme examples. No
validation/test payload or model weights are read. The known population is
11,522,113 TRAIN event rows; this audit supplies the currently missing proposed
representation counts, not a prediction or quality comparison.

A local endpoint candidate set is plausible only if its observed coverage and
cost justify it. Even high coverage cannot justify dropping rare long targets:
a shortlisted design needs an explicit long-tail path or full candidate access.
An onset-conditioned model must also keep future required attacks feasible when
four lanes could otherwise remain held; source validity does not automatically
make arbitrary generated duration choices valid. These are design obligations,
not reasons to insert true endpoints as an undisclosed fallback.

Do not build negative candidate examples by blindly taking unions of charts in
one song group. The catalog groups are transitive published-group/set/beatmap/
normalized-song components; audio deduplication is not established, and that
grouping is not a guarantee of temporal alignment. Candidate absence remains a
live alternative requiring a declared, matched training/inference construction.

Audit ID: timing-representation-audit-v1, revision1, accepted revision none.
Baseline product is f679269b92e96efb5bd7989e748bd42cf069379e plus the recorded
dirty worktree; the read-only audit imports immutable v8. Its fresh output owner
is `timing-representation-audit-v1/results`, script
`timing-representation-audit-v1/audit.py`, SHA
`cf6662ba4f2d37dc9c10bbac98e6fb6aded84c9a6e00f77dd68410a94dff1598`.
Run once with the v8 PYTHONPATH and `uv run --offline --extra mps python`.
Bounds are10 minutes,2 GiB RSS,2 GiB minimum available memory and128 MiB output.
Stop on identity, source-pairing, resource or count inconsistency; keep partial
output and do not overwrite it. Broader gold and long-gap generation run
concurrently, so timing is operational only. The result can refine the endpoint
representation and its feasibility test; it cannot establish learned benefit or
select a final architecture. Evaluation remains REFINE.

### Broader clock100 review and completed representation audit

The remaining ten gold-source generations and the one TRAIN stress generation
completed, in814.953 and923.023 supervisor seconds respectively. Their readout
SHAs are `7c6b936ab25f3bd1fc8e92c157bb61ca78881550c4030ab041ceaca5df779c2a`
and `2344a168b1af638439e6be4f407b875168c7dd126b466a9957fc3c0160e9d4c4`.
All38 additional generated gold-context pages were visually inspected; all38
source pages match the previously inspected references byte-for-byte, and the
eight canonical human documents remain unchanged. The broader review SHA is
`d9210e6e61556b5c4b5782021e19a5322478857a014c7253fd6d93ad7b6d95f9` in
`quality-clockreadout-clock-u100-gold-rest-t085/context-review.json`.

In5b69/seed17, overlapping holds, staggered independent releases and intervening
attacks persist through the context: a medium-confidence machine hypothesis of
prominent LN coordination with supporting moving-head organization. Seed19
instead produces coherent tap flow. The other four sources mainly produce
moving singles and occasional chords; their source gold tags are not transferred
to generated outputs. ece738/seed19's single short hold does not establish LN
coordination. In713ef9/seed17, the4924ms final LN has exactly the source closing
phrase's start/end times315465–320389, on a different column; duration alone is
not evidence of a runaway hold. All58 generated gold pages across eight sources
and two seeds have now been inspected. This remains local structural evidence,
not whole-chart playability or controlled difficulty/style evidence.

Full-output descriptive locators for these11 outputs are retained in
`clock-readout-v1/coverage-recurrence.json`, SHA
`5ac0c3eb337e70d95237062cce2179646aa60a72386e3b5a496af1bbe6b761dc`.
For5b69/713ef9/98357f/e67f/ece738, fastest four same-column attacks span
313/365,307/231,353/383,303/203,435/543ms for seeds17/19. Their adjacent attack
pairs below40ms total7. These extrema have not all been visually inspected and
are locators, not pass thresholds. The TRAIN long chart has one such pair,
four-attack minimum222ms, and549LNs. Neither84.735s nor77.643s event gap is
spanned by a hold. Its longest LN is13200ms at4335854–4349054, exactly the source
closing LN's times on a different column. It also adds a terminal tap. This is
one TRAIN seed, with a different checkpoint/policy from the earlier89s failure;
no causal or held-out quality conclusion follows. Its visual quality is unreviewed.

The all-TRAIN representation audit completed in9.741s with readout SHA
`7ef9903f8856074642d92cf73f142a293e72654872bf436e2de655ef29fd016a`.
Across11564charts,11522113event rows contain11014010onset rows and508103
release-only rows (4.41%). There are2548286LNs;1852846releases coincide with an
onset, and175872 of349291 multi-LN-head rows have different endpoints. Endpoint
rank in the original event union is at most16 for99.9483% of LNs,32 for99.9958%,
64 for99.9995%, and128 for all TRAIN LNs. Maximum physical duration is35375ms.
Dropping release-only rows alone offers little sequence-length reduction. The
plausible benefit is linked start/end credit assignment and freedom to leave
unused candidate times unmaterialized. No rare target will be silently dropped
or replaced by a true endpoint. The next probe uses all future candidates.

### Exploratory Card: linked-endpoint-head-v1, revision1

Owner is this proposed Note; accepted revision is none. The user's explicit
authorization covers exploratory implementation, runs and skeleton redefinition.
The immutable v8 runtime and pinned clock100 weights provide the reproducible
baseline over the inherited dirty product tree; there is no clean intervention
OID or formal Card acceptance. Evaluation can therefore end only in REFINE.

Question: after a chosen LN head, can causal history and already chosen object
endpoints improve prediction of its endpoint beyond a generic timing prior?
This tests a prerequisite for onset-conditioned, linked-duration generation,
not its superiority in free rollout. The analogue is REMI's linked duration
representation above. Full source LN pairing as an inference input is rejected;
candidate NO_EVENT learning remains a separate branch. Full-candidate scoring
avoids an arbitrary128-row cutoff despite the TRAIN coverage result.

Baseline identity: product f679269b92e96efb5bd7989e748bd42cf069379e plus the
recorded dirty state; immutable runtime09469489 and clock100 weights29a2a29e,
with full SHAs recorded above. Cataloge31b7e8f and split15175f45 are unchanged.
Select128 TRAIN and32 validation song groups deterministically by SHA-ranked
seed20260920, one eligible chart per group,16–6000event rows and at least16LN
heads after the seed. Sample at most64 heads per chart without replacement.
Selection uses counts and identities, never gold labels or model errors. Replay
the full source prefix with frozen clock100 weights; retain pre-row hidden
features only for selected heads. Do not read test payloads. Record the exact
selection, targets, frozen features, code and runtime digests before fitting.

Both arms score every future event-union time with the same candidate features
(raw-time transforms, relative rank, candidate onset role and local timing).
Baseline uses a learned constant query; intervention adds the frozen pre-row
history, chosen current row actions, current lane and previously chosen LN
endpoints. Prior LN endpoints are teacher-forced object decisions in this new
factorization; a generated decoder must use its own predictions. The current
head's endpoint and any later source actions or pairings are never input.
Simultaneous heads are conditionally independent for this feasibility probe.
Onset roles are declared additional skeleton information, beyond old untyped Γ.

Fit each arm for400 AdamW updates, batch32, learning rate.001, weight decay.01,
clip1, shared example draws and construction seed20260920; frozen backbone is
never updated. Primary metric is equal-group mean endpoint negative log
likelihood on the fixed32 held-out groups. Report per-head NLL, exact endpoint
accuracy, timing error and5000 paired group-bootstrap replicates separately.
No baseline endpoint metric exists yet; measure the trained generic-prior arm
in this paired run rather than substituting row NLL. A decrease of at least.05
nats with the95% paired CI below zero and no more than5percentage-point exact
accuracy loss merits a generated-decoder pilot. No improvement rejects this
particular context head; a noisy or conflicting result refines it. No quality
or whole-model claim follows from teacher-forced endpoint accuracy.

Procedure: implement artifact-local dataset/feature extraction and endpoint
scoring, verify linked encoding reconstructs canonical rows, then run one fresh
extraction and paired fitting. Use `uv run --offline --extra mps python` with
v8 PYTHONPATH, CPU/four threads on this24GiB Mac. Bound extraction to30minutes,
fitting to20minutes, each to6GiB RSS and2GiB minimum available memory, total
owner to2GiB. Fresh owner is `linked-endpoint-head-v1`; implementation scripts
and outputs have separate subdirectories. Stop on hash/round-trip mismatch,
non-finite loss, impossible candidate targets or any resource bound. Preserve
partial outputs; never overwrite or relaunch an existing output owner blindly.
Record exact commands and source digests before launching. Sampling a moderate
LN-rich slice, frozen features, known onset roles, teacher-forced prior endpoints
and independent simultaneous duration heads limit all conclusions. Qualitative
quality requires subsequent complete generated contexts and long-gap checks.

Implementation is artifact-local under `linked-endpoint-head-v1/scripts`.
The exact extraction command is:

```sh
PYTHONPATH=artifacts/oracle-time-continuation/m3-20260917/skeleton-time-runtime-v8/src caffeinate -i uv run --offline --extra mps python artifacts/oracle-time-continuation/m3-20260917/linked-endpoint-head-v1/scripts/supervise.py extract
```

Fitting uses the same command with final argument `fit`, after a complete
verified feature manifest exists. Initial extraction stopped while serializing
the first chart: the structured disk row's float64 time view has a12-byte stride,
which PyTorch cannot consume directly. `attempt-r1` preserves the original code,
selection, partial output and failure. An owned contiguous float64 copy fixes
storage without altering times, selection or the scientific comparison. The
retry's exact selected sources/targets equal the failed attempt's. Five focused
checks pass: crossing-hold reconstruction and prior-plan boundary, mirrored
context parity, untruncated future support/padding gradients, generic-prior
history isolation, and exact independent storage of structured timestamps.
`selected-checks.json` SHA is
`ca8b36a1b035d3ab78497a5a78943a5f33fc14a3056b3fafa08c3338ea08a548`.

Current source SHAs are core.py
`3d6ef6e4bb6137b2fd009336ef6140a42d935ab6b12b0d79dbb0b130e70b4276`,
extract.py `00688da5d6d0e2bbc4761e6ca69cc000d3cbabbcf4dd27f07ff9a19ae0965ce0`,
fit.py `e54bcfbdc7dd6a1459e1151e3a7254f1af924603fbe1e51f8b43ab029e54ad9b`,
supervise.py `ae460bde70899c2cdfdb3e7ed72d86ae65b101f557dc63bc56f17b121c1a67a7`.
The retry selection SHA is
`ee581d536d08d32b0a1269e8159672b1c179c893fc6d295e19b6383925130289`:
128 TRAIN charts/7010 selected heads and32 validation charts/1794heads, with
155036/30969 total event rows respectively. Context width is1096. Extraction
remains active at this record; no fitting result or generated-decoder claim exists.

During extraction, the three additional outputs with a fastest-four locator
below250ms received complete ±2s source/generated context inspection:12pages,
review `clock-readout-v1/coverage-extrema/review.json`, SHA
`d718c5a99f630d97bea790b1b581fa56f4562cbe1f18f62ce00d447e2bc13d35`.
713ef9/seed19 repeats column0 at196773/196850/196927/197004ms (77ms gaps).
The TRAIN long output repeats column2 at606593/606667/606741/606815ms (74ms).
e67f/seed19 repeats column2 at98629/98710/98791/98832ms (81/81/41ms).
Moving structure persists around these short concentrations, unlike the earlier
20ms repeated-chord bursts, but uniform difficulty is not established. Preserve
these as additional regressions to inspect, without converting the locator
threshold into a Foundation tag or universal quality threshold. Human gold is
unchanged; other uninspected chart passages remain open.

### Further branch to refine: learned context versus supervised exposure

Do not treat exact full-BOS learned-memory reconstruction as synonymous with
causality. A finite learned context can still read only past actions and retain
exact occupancy, open-LN starts, elapsed clocks, earlier planned endpoints and
known skeleton times. It changes the model's available history and may lose
phrase-scale relationships; it is not a hidden future-action shortcut. A cropped
window must never impersonate true BOS. This is a research alternative to the
current complete learned-history contract, not an implementation exception to it.

The existing coverage measurement is a concrete reason to test this tradeoff:
the500-update20M ancestor plus its300-update timing continuation have287852
supervised exposures, covering249676 distinct rows (2.17% of TRAIN) across1460
charts/1276groups. Later horizon and clock updates improve exposure, but prefix
replay without gradients is still not supervised coverage. The long300 arm
alone has408820 targets and476759 computed prefix rows. These observations do
not establish that insufficient training causes all generation problems; they
make data exposure per Mac compute budget a live competing explanation.

The closest architecture analogue is the relative local-attention variant in
[Music Transformer](https://arxiv.org/pdf/1809.04281), section3.5 and Table3.
It attends within neighboring blocks rather than all positions at once. The
transferable idea is bounded learned context with relative timing; its piano
data and conclusions do not establish 4K action quality. A second representation
analogue, [Compound Word Transformer](https://arxiv.org/pdf/2101.02402), groups
related note attributes and uses typed output heads. Its large sequence-length
savings cannot be transferred to this already row-compacted representation,
where removing release-only rows saves only4.41%. Both would be adaptations,
not a novelty claim.

A discriminating future comparison should hold the conditional task, corpus
allocation, loss targets and generation policy fixed, then compare an efficient
finite-context baseline against the current architecture at equal real compute
budget, reporting supervised exposures/unique coverage and complete generated
quality alongside likelihood. If greater exposure does not improve structure,
or phrase/hold relations regress despite exact replay state, the alternative
loses support. Context length, exact timing/occupancy features and training
coverage must be recorded; parameter count alone is not the comparison axis.
Outcome is REFINE. There is no new Card, implementation, training run or adoption
for this branch; finish the linked-endpoint probe before selecting another run.
