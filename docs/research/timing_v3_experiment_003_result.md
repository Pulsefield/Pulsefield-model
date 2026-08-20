# Timing v3 Experiment 003 Result: Joint Anchor/BPM Phase Projection

Date: 2026-08-11

Decision: reject both Family C candidates under this card. C1 repaired the
exposed 80-audio set, but protocol-v2 holdout100 failed the frozen fallback
guard: C0 and C1 each fell back on `9/100` rows, above the `5%` cap. Do not
relax the 5% BPM-adjustment guard, do not materialize broad500, and do not
switch `TimingV3Fitter` to Family C. Proceed to Experiment 004.

`.osu` red timing remains a weak evaluation comparator, not audio truth. The
result below uses it only for paired oracle metrics and keeps projection,
fallback, source-only, and comparator denominators separate.

## Protocol Repair

The original holdout-v1 protocol was contaminated before the formal run. During
evaluator plumbing, a worker ran `--limit 3` and exposed aggregate output for
the first three v1 rows. No algorithm, objective, metric, threshold, or
candidate changed after seeing those values, and the remaining 97 rows were not
formally run. The three exposed rows were all from the small post-pilot
`ramp_audit` pool, so the v1 holdout was invalidated as a held-out gate.

The v1 identity artifacts remain audit evidence only:

| Artifact | SHA-256 |
| --- | --- |
| `timing_v3_exp003_holdout100_v1_manifest.json` | `25d925fbcbd33682eb81338f6b3cce3b33b60649a28adcd8a39ea6fa8154afbe` |
| v1 manifest fingerprint | `5248279c3076a224cae15cd6c5ae7f2439de76c25bbaa034b44bc532e65189c6` |
| `timing_v3_exp003_holdout100_v1_labels.jsonl` | `5427d79ba1b02e3090793d757fe86c3f8056acbc2447ddcddc49a376035edee9` |
| `timing_v3_exp003_holdout100_v1_v2_baseline.jsonl` | `613b4a7eddafe5d2189bb100adb8e921234687d5d290d3d83db7f7dd03f89b99` |
| `timing_v3_exp003_protocol_exclusion_v1.json` | `120ff805ccbb925ca338045cd3bc40df7f973ac4eab9899e3959ae26d77ee5c7` |

The protocol-v2 repair froze a new seed,
`timing-v3-exp003-holdout100-v2`, before formal C metrics were inspected. It
excludes the 80 repair audio keys and the three exposed v1 keys, then selects
exclusive priority quotas in order `ramp_audit/anomaly/long/dense/jump/stable`
as `4/10/11/10/25/40`. The lost fifth ramp slot was reassigned to `long`
before selection because only four unexposed ramp candidates remained.

| Protocol-v2 identity | Value |
| --- | --- |
| Manifest SHA-256 | `87fc944f22abaf39ae5762dca57ec4153840b33a86839b17b2104fcd4211b5c4` |
| Manifest fingerprint | `7ae093565b18876e55c057fffddf710306c8f7dc0473d686cfaa3c2c0983d400` |
| Selected full label rows SHA-256 | `d109a064ee2c72aa07d3a6091f5b20bf7b74c8703a7980bad9ba2b503071c0b7` |
| Frozen-v2 baseline subset SHA-256 | `3b0151c6ff745335131318a777a13a7e06629314f3f6ffa5257ea88e27bf60f5` |
| Pilot exclusion-set SHA-256 | `3a2504bbe9a0d632c4cbffc8fe1de17123e3a34f3c972c149711fb21348304a0` |
| Protocol exclusion-set SHA-256 | `cec724e02837371cba4934c28a11b3fb52c3c28a5bbcff43bd5fd44bea559b60` |
| Label source SHA-256 | `cce8287a8270e49913d37ade2f8b1e2ec75d4724573d67dca1be0d8e1be6ddd8` |
| Pilot source rows SHA-256 | `cdb5e2af87d99c8af3bbff71a0985bb490fdeed169d10cb0a64b17a8ca0296d7` |

## Frozen Code and Config

The formal holdout was opened only after repair80 and these hashes were frozen:

| Component | SHA-256 / fingerprint |
| --- | --- |
| Joint projection source, `pulsefield_model.timing.v3.joint_projection` | `9fde5804a193d9858b1e1a97d4be561f8f2f0158fc0b9e4c8e04a741fc6a7ccf` |
| Source-only comparator, `pulsefield_model.timing.evaluation.source_projection` | `db7434462be79d2bc52049be0179f51bd8ce2df0348b94f8ac49c6334fcb7c2e` |
| Projection evaluator, `pulsefield_model.timing.evaluation.v3_projection` | `d7bca7629049186f5a08991f772a551ba24af4db6337b12865556f8bd0ae2326` |
| Split verifier, `pulsefield_model.timing.evaluation.splits` | `b09655524c514b6bfd912ec9ca62b8fd5f00e52e99ae5a525833ef2b322907f9` |
| Projection config fingerprint | `80906a2b3f081dcc223ba2258e2d8c5db67ac3a96a41488713cffc85551b1c88` |
| Evaluator behavior fingerprint | `f08300b3b1a407ecf1c04d0e9fd7935817bcaf56464f160f0506d5014b7a94d9` |

Frozen Family C guards were 20-1000 BPM source/projected range, `0.05`
maximum relative projected BPM adjustment, `0.500000000001` adjacent-local-beat
anchor displacement, and `1e-10` solver normalized residual.

## Repair80 Result

Repair80 is the exposed Experiment 002 pilot rerun with Family C. It is a repair
and regression set, not a new holdout. All 80 stored fits were
projection-evaluable; 77 audio groups had at least one usable stored `.osu`
comparator, and three remained comparator-unavailable.

| Candidate | Projection success | Fallback | Paired audio | Mean ratio vs v2 | p90 ratio vs v2 | Max projected seam | Max BPM adjust | Guard result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C0 fixed counts | 76/80 | 4/80 = 5.00% | 75 | 1.030863 | 0.998847 | `2.99e-11 ms` | 3.7397% | fail |
| C1 nearby counts | 76/80 | 4/80 = 5.00% | 75 | 1.026240 | 0.998847 | `5.68e-11 ms` | 3.7397% | pass |

C0 narrowly fails exactly one mandatory repair guard: the
`alias_p90_abs_30s_relative_drift_ms` mean ratio is `1.250209`, just above the
`1.25` cap. Its p90 for that metric passes at `1.018298`, and all other C0
phase, fallback, seam, section-count, BPM-adjustment, solver, replay, and
serialization checks pass.

C1 passes every available repair80 mandatory check. Its worst drift ratio is
also the 30-second alias drift mean at `1.248925`, below the `1.25` cap. Solver
residuals pass on all 76 successful C1 projections; the maximum normalized
residual is `7.317850437019743e-17`. The 4 fallbacks are split as two
`relative_bpm_adjustment_exceeded` and two
`adjacent_beat_identity_displacement_exceeded`.

## Holdout100 Result

Protocol-v2 holdout denominators are fixed and exact:

| Denominator | Count |
| --- | ---: |
| Projection-evaluable rows | 100 |
| Comparison-eligible audio groups | 92 |
| Comparator-unavailable audio groups | 8 |
| Selected C-or-v2 paired difficulty comparisons | 261 |

C0 and C1 candidate-only paired comparisons are `245` because failed C
projections are not counted as successful C comparisons. The selected
`C1-or-fallback-v2` path still has 261 paired comparisons, matching all usable
stored comparator rows.

| Candidate | Projection success | Fallback | Paired audio | Mean ratio vs v2 | p90 ratio vs v2 | Max projected seam | Serialization seam | Max BPM adjust | Solver |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C0 fixed counts | 91/100 | 9/100 = 9.00% | 84 | 1.007480 | 1.011642 | `5.68e-11 ms` | `0.0 ms` | 4.4978% | pass |
| C1 nearby counts | 91/100 | 9/100 = 9.00% | 84 | 1.007513 | 1.011642 | `5.68e-11 ms` | `0.0 ms` | 4.8465% | pass |

Both C0 and C1 pass the aggregate phase, seam, section-count, serialization,
BPM-adjustment, source-comparison, search-convergence, fingerprint, and solver
checks. Both fail only the mandatory fallback-rate guard:
`9.00% > 5.00%`. C0's duration-weighted absolute relative BPM adjustment is
`0.002029`; C1's is `0.002242`. For both, all 91 successful projections pass
the solver residual guard, with maximum normalized residual
`7.761157332137371e-17`.

Global drift guards pass for both candidates on the matched holdout
population. C1's largest global alias drift ratio is
`alias_abs_endpoint_relative_drift_ms` p90 at `1.160525`, below the `1.25` cap;
C0's largest is `alias_max_abs_prefix_relative_drift_ms` p90 at `1.220189`.

## Holdout Strata

The manifest quotas are exclusive, but the summary strata are not all
exclusive. In particular, `label_stratum=stable` has 44 rows because several
long-quota rows still carry stable labels. The long guard uses
`source_long_track=true`.

| Stratum | Audio | Comparable | C1 fallback | C1 mean ratio | C1 p90 ratio | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `label_stratum=stable` | 44 | 44 | 1/44 = 2.27% | 1.001829 | 1.022118 | Phase passes; one adjacent-displacement fallback. |
| `label_stratum=jump_candidate` | 25 | 25 | 0/25 = 0.00% | 1.004342 | 1.038590 | Phase and fallback pass locally. |
| `source_long_track=true` | 11 | 11 | 4/11 = 36.36% | 1.000804 | 1.002614 | Phase passes; fallback concentration fails the safety story. |
| `label_stratum=dense` | 17 | 17 | 6/17 = 35.29% | 1.032856 | 1.006716 | Main fallback concentration; four of six are also long. |
| `label_stratum=ramp_candidate` | 4 | 4 | 1/4 = 25.00% | 1.002818 | 1.016341 | Mandatory audit only; not ramp truth. |
| `label_stratum=ambiguous` | 10 | 2 | 1/10 = 10.00% | 1.009005 | 1.013030 | Eight rows are comparator-unavailable. |

Dense, ramp-audit, and ambiguous rows do not establish ramp precision or recall.
They are robustness reports against weak comparator queues.

## Fallback Attribution

All nine holdout fallbacks are emitted by the Family C joint-projection module,
not by top-level evaluator failure or `.osu` comparison. The formal module hash
is `pulsefield_model.timing.v3.joint_projection`
`9fde5804a193d9858b1e1a97d4be561f8f2f0158fc0b9e4c8e04a741fc6a7ccf`.
The selected result for each row is the tagged v2 fallback
`selected_family_c1_or_fallback_v2`.

| Row | Stratum | Long | Comparable | Reason | Failed boundary | Max rel BPM | Max anchor beats | Audio |
| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | `ramp_candidate` | false | yes | `relative_bpm_adjustment_exceeded` | 8 | 0.051856 | 0.451555 | `dataset/0/2046911/14 BIG SHOT (Deltarune Chapter 2).mp3` |
| 12 | `ambiguous` | false | no | `relative_bpm_adjustment_exceeded` | 5 | 0.098643 | 0.363595 | `dataset/0/512591/audio.mp3` |
| 14 | `dense` | true | yes | `relative_bpm_adjustment_exceeded` | 1 | 0.056110 | 0.426864 | `dataset/0/486590/audio.mp3` |
| 15 | `dense` | true | yes | `adjacent_beat_identity_displacement_exceeded` | 11 | 0.069484 | 0.511579 | `dataset/0/682489/audio.mp3` |
| 22 | `dense` | true | yes | `relative_bpm_adjustment_exceeded` | 15 | 0.070190 | 0.482037 | `dataset/0/2064066/audio.mp3` |
| 23 | `dense` | true | yes | `relative_bpm_adjustment_exceeded` | 16 | 0.084019 | 0.407531 | `dataset/0/1462798/audio.ogg` |
| 29 | `dense` | false | yes | `relative_bpm_adjustment_exceeded` | 5 | 0.063880 | 0.351918 | `dataset/0/1602283/audio.ogg` |
| 32 | `dense` | false | yes | `relative_bpm_adjustment_exceeded` | 6 | 0.058582 | 0.384496 | `dataset/0/1624523/audio.mp3` |
| 75 | `stable` | false | yes | `adjacent_beat_identity_displacement_exceeded` | 0 | 0.029100 | 0.508991 | `dataset/0/1316855/B!TTF 1.300x.mp3` |

Reason counts are seven `relative_bpm_adjustment_exceeded` and two
`adjacent_beat_identity_displacement_exceeded`. By label stratum, the fallbacks
are six dense, one ramp-audit, one ambiguous, and one stable. By long-track
flag, four are long and five are not.

## Replay and Hash Notes

The durable per-audio outputs are:

| Artifact | SHA-256 |
| --- | --- |
| `timing_v3_projection_pilot80_experiment003_v1.jsonl` | `253cb672725a25b141664cd4ece10f052ea5dfdc3fb32d0e5641ae5106c6beb3` |
| `timing_v3_projection_pilot80_experiment003_v1_summary.json` | `8fc813ff0f350d1af1676ba342211b4e091be2b554f86080c18f07214eeccb5c` |
| `timing_v3_projection_exp003_holdout100_v2.jsonl` | `0f4b066fbbb101de0619d1c41d2c955f707a9a7d6353f4766d430718c5dd8457` |
| `timing_v3_projection_exp003_holdout100_v2_summary.json` | `c277d956e9a96a5c6642dcc61adebd841e835b987fdc9aed7b528f8c3b259e33` |

The summaries were regenerated as zero-compute replays over existing JSONL
rows: repair80 has `processed_count=0` and `skipped_success_count=80`;
holdout100 has `processed_count=0` and `skipped_success_count=100`. Across two
zero-compute replays, the projection JSONL byte hashes stayed stable. Summary
file hashes can still change when runtime fields such as `started_at_unix`,
`finished_at_unix`, and `total_seconds` change; that is runtime metadata churn,
not a mathematical replay failure. Deterministic mathematical replay remains
present in the row payloads: all successful C0/C1 projections have
mathematical-grid, integer-search, and replay fingerprints recorded, and the
summary guard `all_successful_fingerprints_present` passes.

## Stage Decision

The stage stops at holdout100. No broad500 or Experiment 003 full5050 projection
artifacts are present under `artifacts/reports/timing/`, and broad500 must
remain forbidden for this card because the replacement holdout gate failed.

The production fitter was not switched to Family C. The result rejects the
cache-only Family C joint anchor/BPM adapter as a promotion candidate. Keep the
frozen guards intact, kill C0/C1 for this loop, and proceed to the
BeatThis-supported global constant/jump path in Experiment 004.
