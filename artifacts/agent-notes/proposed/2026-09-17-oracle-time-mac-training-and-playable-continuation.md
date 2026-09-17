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
