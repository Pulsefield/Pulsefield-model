# Mapper v2.1 44k Decoder Inference Postmortem

Date: 2026-05-20

Status: frozen snapshot of the first v2.1 sparse full-song rollout experiments.

## Context

The goal was to investigate lower-level decoder inference issues for the v2.1 sparse mapper checkpoint at step 44,000:

- Whether token speed degrades with autoregressive prefix length.
- Whether latency is dominated by many small framework/kernel launches.
- Whether constraint decoding or sampling is materially expensive.
- Whether no-TS-penalty greedy full-song rollout creates long empty spans.
- Whether EOS is actually learned by model logits or only enforced by grammar.
- Whether rendered generated `.osu` output shows repetition, emptiness, or other qualitative failures.

The instrumentation intentionally avoided per-token JSONL traces. Aggregate JSON summaries and PyTorch profiler `record_function` scopes were used instead. Generated maps were exported to `.osu` and rendered with Reamber.

## Checkpoints

Mapper checkpoint:

```text
artifacts/runs/stage2_mapper_v2_1/stage2_mapper_v2_1_phase_b_sparse_global_d384_l4_b2/checkpoints/checkpoint_step_044000.pt
```

Control checkpoint:

```text
artifacts/runs/stage2_control_demo/stage2_control_demo_global_d384_l3_stride16_b6/checkpoints/checkpoint_step_002000.pt
```

Runtime detected the mapper checkpoint as v2.1 and loaded `MapperV21Model` with `MapperV21Vocab`.

## Code State

The experiments depended on a new sparse v2.1 rollout path:

- `src/pulsefield_model/inference/mapper_v2_1_rollout.py`
- `src/pulsefield_model/evals/mapper_v21_decoder_profiler.py`
- `src/pulsefield_model/evals/mapper_render_reamber.py`
- `src/pulsefield_model/inference/model_runtime.py`

The current rollout path is true v2.1 sparse-token generation/export, but it is not incremental decode. Each autoregressive token step still rebuilds and forwards a full prefix.

## Artifact Index

Bounded synthetic/zero-control profiler run:

```text
artifacts/evals/mapper_v21_decoder_44000_bounded/
```

Key files:

- `mapper_v21_prefix_length_sweep.json`
- `mapper_v21_kernel_overhead_probe.json`
- `mapper_v21_constraint_sampling_split.json`
- `mapper_v21_no_ts_full_rollout_metrics.json`
- `mapper_v21_eos_probe.json`
- `mapper_v21_no_ts_full_rollout.osu`
- `reamber/*.png`

Real eval-split real-audio/control run:

```text
artifacts/evals/mapper_v21_decoder_44000_real_eval_riria/
```

Key files:

- `mapper_v21_44000_real_eval_riria_summary.json`
- `mapper_v21_44000_real_eval_riria_no_ts_greedy.osu`
- `reamber/*.png`

## Experiment 1: Bounded Synthetic/Zero-Control Profiler

Command shape:

```text
uv run pytest tests/evals/test_mapper_v21_decoder_profiler.py -q \
  --run-mapper-v21-decoder-evals \
  --mapper-v21-decoder-checkpoint artifacts/runs/stage2_mapper_v2_1/stage2_mapper_v2_1_phase_b_sparse_global_d384_l4_b2/checkpoints/checkpoint_step_044000.pt \
  --mapper-v21-decoder-eval-device mps \
  --mapper-v21-decoder-eval-prefix-lengths 16,32,64,128,256,512,1024 \
  --mapper-v21-decoder-eval-rollout-ms 16000 \
  --mapper-v21-decoder-eval-rollout-max-tokens-per-window 64 \
  --mapper-v21-decoder-eval-output-dir artifacts/evals/mapper_v21_decoder_44000_bounded
```

Result:

```text
5 passed in 45.97s
```

This run used synthetic probe windows and the zero-control batch provider. It is a decoder/grammar canary, not a real-song quality eval.

### Prefix Length Sweep

Measured on MPS with PyTorch profiler enabled.

| Prefix len | Wall ms | grammar_mask ms | decode_with_global_context ms | ln_close_adapter ms | state_prior_adapter ms |
|---:|---:|---:|---:|---:|---:|
| 16 | 236.96 | 54.13 | 21.82 | 16.53 | 15.59 |
| 32 | 166.00 | 98.78 | 13.00 | 10.75 | 8.14 |
| 64 | 252.85 | 179.18 | 14.56 | 14.01 | 8.05 |
| 128 | 417.23 | 343.66 | 15.04 | 13.87 | 8.53 |
| 256 | 734.51 | 663.74 | 14.83 | 15.46 | 8.42 |
| 512 | 1404.64 | 1318.16 | 14.53 | 22.97 | 8.85 |
| 1024 | 2799.04 | 2648.04 | 14.77 | 87.83 | 9.25 |

Interpretation:

The dominant measured growth is in `mapper_v21.grammar_mask`, not in `mapper_v21.decode_with_global_context`. This points to grammar/valid-mask construction and replay/state work as the first speed target. The 16 to 32 ms non-monotonic wall result is likely warmup/profiler noise; the 64 to 1024 trend is clear.

Important caveat:

This was MPS. PyTorch profiler on MPS gives limited visibility into true device kernel time. The wall-time trend is still useful, but CUDA should be used before making final GPU-utilization claims.

### Constraint/Sampling Split

| Section | Wall ms | Notes |
|---|---:|---|
| forward_logits | 68.60 | model forward for one synthetic prefix |
| constraints | 53.27 | valid token count at last step: 21 |
| sampling | 3.02 | selected token id: 12 |

Interpretation:

Sampling itself is not the bottleneck in this run. Constraint construction is large enough to matter almost as much as forward logits for the tested prefix.

### Kernel Overhead Probe

| Probe | Wall ms |
|---|---:|
| empty_record_function_scope | 0.038 |
| tiny_tensor_kernel | 4.466 |

Interpretation:

Framework/device overhead exists, but this probe does not by itself explain the much larger rollout latency. The larger signal is still grammar/constraint cost.

### No-TS Full Rollout Canary

Setup:

- Synthetic zero-control provider.
- `chart_end_ms=16000`.
- `max_tokens_per_window=64`.
- Greedy decode.
- `time_shift_length_penalty_alpha=0`.

Metrics:

| Metric | Value |
|---|---:|
| completed | false |
| dead_end | false |
| max_tokens_exceeded | true |
| window_count | 1 |
| completed_window_count | 0 |
| token_count | 64 |
| lane_action_count | 28 |
| eos_count | 0 |
| timepoint_count | 18 |
| longest_event_gap_ms | 12990 |
| rollout_wall_ms | 10729.45 |

Generated behavior:

The `.osu` export contains a regular early pattern from about 120 ms to 3010 ms, then silence for the rest of the 16 s chart. The rollout hit the 64-token cap before finishing the first 8 s write window.

Interpretation:

Under zero-control, no-TS greedy can enter a token-heavy repetitive pattern and fail to terminate the first window. This is a useful stress signal, but it should not be treated as final real-song behavior.

### Synthetic EOS Probe

| Position | current_ms | EOS allowed | EOS logit | EOS rank |
|---:|---:|---|---:|---:|
| 61 | 2400 | true | -3.045 | 7 |

Interpretation:

In this synthetic terminal state, EOS was grammar-allowed but not strongly preferred by the model.

## Experiment 2: Real Eval-Split Full-Song Rollout

Purpose:

Check whether the zero-control failure was an artifact of synthetic control by running a real held-out song through real audio, BeatThis timing, control encoder features, v2.1 mapper rollout, `.osu` export, and Reamber rendering.

Selected map:

```text
dataset/0/1942086/Riria. - Shitsuren Song Takusan Kiite Naite Bakari no Watashi wa Mou. (TV Size) (Kibitz) [Stay With Me, Don't Let Go].osu
```

Selected audio:

```text
dataset/0/1942086/audio.mp3
```

Selection hygiene:

The training split is map-based in the repo. For this run, the split was reconstructed using the same `split_train_eval_dataset(..., eval_fraction=0.01, seed=1337)` logic, then the selected candidate was additionally checked for audio disjointness.

| Check | Value |
|---|---:|
| eligible_windows | 174515 |
| train_windows | 172649 |
| eval_windows | 1866 |
| train_unique_maps | 9142 |
| eval_unique_maps | 96 |
| train_unique_audio | 3625 |
| eval_unique_audio | 95 |
| target_eval_window_count | 11 |
| target_total_window_count | 11 |
| target_in_train_maps | false |
| target_audio_in_train_audio | false |

Reference chart:

| Field | Value |
|---|---:|
| beatmap_id | 4017175 |
| beatmap_set_id | 1942086 |
| difficulty | 2.36 |
| normalized_difficulty | -0.82 |
| audio_length_ms | 90020 |
| chart_end_ms | 87200 |
| reference_hitobject_count | 489 |
| reference_timepoint_count | 398 |

Decode setup:

- Real audio.
- BeatThis timing fit.
- Real control encoder features.
- Full-song v2.1 sparse rollout.
- Greedy decode.
- `time_shift_length_penalty_alpha=0`.
- `max_tokens_per_window=256`.
- Device: MPS.

Runtime timing:

| Stage | Wall ms |
|---|---:|
| runtime_load | 290.57 |
| prepare_audio | 7710.07 |
| prepare_full_control | 921.77 |
| rollout | 3649.86 |
| eos_probe | 1199.16 |

Rollout metrics:

| Metric | Value |
|---|---:|
| completed | true |
| dead_end | false |
| max_tokens_exceeded | false |
| window_count | 11 |
| completed_window_count | 11 |
| empty_window_count | 10 |
| token_count | 34 |
| time_shift_token_count | 27 |
| lane_action_count | 6 |
| eos_count | 1 |
| timepoint_count | 2 |
| longest_event_gap_ms | 86750 |

Window distribution:

| Window start ms | Token count | Lane action count | Terminal current ms |
|---:|---:|---:|---:|
| 0 | 2 | 0 | 8000 |
| 8000 | 2 | 0 | 16000 |
| 16000 | 2 | 0 | 24000 |
| 24000 | 2 | 0 | 32000 |
| 32000 | 2 | 0 | 40000 |
| 40000 | 2 | 0 | 48000 |
| 48000 | 2 | 0 | 56000 |
| 56000 | 2 | 0 | 64000 |
| 64000 | 2 | 0 | 72000 |
| 72000 | 2 | 0 | 80000 |
| 80000 | 14 | 6 | 87200 |

Generated hitobjects:

```text
86750ms: lanes 1,2,4
87100ms: lanes 1,3,4
```

Interpretation:

Real control changed the failure mode. The model did not get stuck on token cap. Instead, greedy no-TS mostly fast-forwarded through every write window with time-shift tokens, emitted almost nothing, and placed a tiny chord burst at the end.

This is a stronger real-inference failure than the zero-control canary: the decoder can complete structurally while producing an unusably empty chart.

### Real EOS Probe

The EOS probe inspected teacher-forced real-chart states and compared pre-grammar logits against final grammar-masked logits.

| Probe | write_start_ms | current_ms | EOS allowed | pre-grammar EOS rank | pre-grammar EOS prob | final EOS rank | final EOS prob |
|---|---:|---:|---|---:|---:|---:|---:|
| middle_window_last_state | 40000 | 47800 | false | 10 | 0.00547 | n/a | 0 |
| near_end_previous_window_last_state | 72000 | 79800 | false | 10 | 0.00877 | n/a | 0 |
| exact_chart_end_eos_state | 80000 | 87200 | true | 23 | 0.00103 | 1 | 1.0 |

Interpretation:

EOS is not strongly preferred by pre-grammar model logits even at the exact chart end. At chart end, final EOS wins because the grammar mask forces it, not because the decoder assigned it high probability before grammar.

## Findings

1. Grammar/constraint cost is the first speed target.

The prefix sweep shows `grammar_mask` growing from 54 ms at prefix 16 to 2648 ms at prefix 1024. The decoder scope stays near 15 ms in the same profiler view. Before optimizing sampling or transformer FLOPs, inspect and optimize v2.1 grammar mask construction, replay-state reconstruction, and any per-prefix Python loops.

2. Sampling is currently not the problem.

The constraint/sampling split measured sampling at about 3 ms, much lower than constraints and forward logits.

3. The no-TS greedy failure is real, but mode-dependent.

Zero-control canary: repetitive early pattern, token cap hit, no EOS.

Real eval-split audio/control: structurally completes all windows, but skips almost all musical content and emits only two late timepoints.

4. EOS appears grammar-forced rather than learned.

The real exact-chart-end probe had pre-grammar EOS rank 23 and probability 0.00103. Final EOS rank became 1 only after grammar masking.

5. Full-song v2.1 rollout is now runnable but still algorithmically inefficient.

The real 87.2 s song rollout took 3.65 s on MPS with only 34 generated tokens. That timing is acceptable only because the decode collapsed to very few tokens. It does not prove the path is ready for dense outputs.

## Caveats

- MPS profiling is not enough for final GPU kernel conclusions. Repeat core speed probes on CUDA.
- The real eval result is one held-out song. It is enough to prove a failure mode, not enough to estimate prevalence.
- The real eval used no TS penalty by design. This intentionally exposes greedy time-shift bias and should be compared against penalty-enabled decode.
- The zero-control run is not a quality eval; it is a decoder/grammar stress canary.
- The current full-song rollout uses no incremental decoder KV cache and reprocesses full prefixes per token.
- Current cross-window v2.1 rollout uses empty LN carry at window boundaries in this path, matching the initial streamer behavior but not necessarily the best full-song behavior.

## Immediate Next Experiments

1. Run the exact same real eval map with `time_shift_length_penalty_alpha` sweep.

Suggested values:

```text
0.00, 0.02, 0.05, 0.10, 0.20
```

Primary metrics:

- empty windows
- lane actions/window
- longest event gap
- token count/window
- max token hits
- Reamber visual QA

2. Run the same map with small stochastic decode.

Suggested start:

```text
temperature=0.8
top_p=0.9
time_shift_length_penalty_alpha in {0.05, 0.10}
```

This tests whether the empty rollout is purely greedy argmax bias.

3. Teacher-forced density/logit probe on real eval windows.

For selected real windows, record aggregate ranks/probs for:

- next real lane-action tokens
- best time-shift tokens
- EOS
- grammar valid count
- density_teacher_8s summary

Do this as aggregate JSON, not per-token JSONL traces.

4. Optimize grammar/replay before model FLOPs.

Concrete targets:

- avoid rebuilding full replay state from token prefix every step
- vectorize valid-mask computation where possible
- cache invariant per-window tensors
- separate grammar-only wall time from model forward wall time during real rollout

5. Expand held-out real-song eval to a small fixed panel.

Use 5 to 10 audio-disjoint eval maps covering difficulty and length buckets. Report aggregate medians plus worst-case Reamber panels.

## Current Conclusion

The 44k v2.1 decoder has two confirmed issues:

1. Inference speed is dominated by v2.1 grammar/constraint work as prefix length grows.
2. No-TS greedy real-control rollout can structurally complete while producing an almost empty chart.

The next decision point should be a penalty/sampling sweep on the same held-out real map. If modest TS penalty restores density, the immediate fix is decode policy plus grammar optimization. If it stays empty, the model/control conditioning or training objective is the more likely root cause.
