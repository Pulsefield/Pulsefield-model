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
