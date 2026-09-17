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
