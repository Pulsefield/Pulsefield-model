# Agent Note: Mac training and playable oracle-time continuation

Note ID: 2026-09-17-oracle-time-mac-training-and-playable-continuation
Status: proposed
Kind: research
Created: 2026-09-17
Updated: 2026-09-17
Product revision: f679269b92e96efb5bd7989e748bd42cf069379e, with uncommitted implementation; frozen source manifests accompany the exploratory runs
Scope: research/oracle_time_continuation M3 runtime, Mac resource envelope, optimizer calibration, corpus exposure and generated structure
Related: 2026-09-15-oracle-time-continuation-resource-and-module-review

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
