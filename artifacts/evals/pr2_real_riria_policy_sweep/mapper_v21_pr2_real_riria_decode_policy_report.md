# Mapper v2.1 PR2 Real Riria Decode Policy Report

This report freezes the real Riria PR2 decode policy runs into a compact, reviewable artifact. It intentionally records aggregate metrics only; no per-token JSONL trace is used.

## Dataset And Reference

- Map: `dataset/0/1942086/Riria. - Shitsuren Song Takusan Kiite Naite Bakari no Watashi wa Mou. (TV Size) (Kibitz) [Stay With Me, Don't Let Go].osu`
- Audio: `dataset/0/1942086/audio.mp3`
- Chart end: `87200 ms`; audio length: `90020 ms`; baseline normalized difficulty: `-0.82`
- Reference lane actions: `713`; timepoints: `398`; longest gap: `800 ms`; empty windows: `0`
- Reference events per 8s window: `[58, 61, 69, 65, 60, 61, 84, 64, 64, 66, 61]`

## Executive Result

- Flat TS penalty alone is not usable: it jumps from empty maps to capped/overdense maps.
- Length-scaled TS penalty fixes the original empty-window failure: `flat=0.0`, `delta=1.25..2.0` completes with no empty windows, no dead ends, and no token caps.
- The best reranked candidate is low-temperature sampling: `flat=0.0`, `delta=1.25`, `temperature=0.4`, `top_p=0.95`, `seed=0`.
- Remaining risk: timepoint count and dominant pattern repetition are still higher than reference, so this is a viable decode policy candidate, not a full musical-quality solution.

## Best Candidate

| score | flat | delta | temp | top_p | seed | completed | dead | cap | lane | lane_ratio | timepoints | gap_ms | dominant_pattern |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.666 | 0.000 | 1.250 | 0.400 | 0.950 | 0 | True | False | False | 722 | 1.013 | 555 | 310 | 0.294 |

Exported osu: `/Users/l/projects/Pulsefield-model/artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_best_low_temp_seed0.osu`

## Flat Stress Sweep

| flat | completed | dead | cap | empty | lane | lane_ratio | gap_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.800 | True | False | False | 10 | 6 | 0.008 | 86750 |
| 1.500 | True | False | False | 10 | 6 | 0.008 | 86750 |
| 3.000 | True | False | False | 10 | 6 | 0.008 | 86700 |
| 5.000 | False | False | True | 0 | 986 | 1.383 | 31900 |
| 7.000 | True | False | False | 0 | 1235 | 1.732 | 300 |
| 9.000 | True | False | False | 0 | 1235 | 1.732 | 300 |

## Length-Scaled Greedy Sweep

| delta | completed | dead | cap | empty | lane | lane_ratio | timepoints | gap_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.500 | True | False | False | 0 | 651 | 0.913 | 534 | 1340 |
| 0.750 | True | False | False | 0 | 652 | 0.914 | 534 | 1340 |
| 1.000 | True | False | False | 0 | 682 | 0.957 | 534 | 1340 |
| 1.250 | True | False | False | 0 | 741 | 1.039 | 543 | 270 |
| 1.500 | True | False | False | 0 | 741 | 1.039 | 543 | 270 |
| 1.750 | True | False | False | 0 | 742 | 1.041 | 543 | 270 |
| 2.000 | True | False | False | 0 | 742 | 1.041 | 543 | 270 |

## Low-Temperature Sampling

| temp | top_p | seed | completed | dead | lane | lane_ratio | gap_ms | dominant_pattern |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.400 | 0.950 | 0 | True | False | 722 | 1.013 | 310 | 0.294 |
| 0.400 | 0.950 | 1 | True | False | 779 | 1.093 | 250 | 0.223 |
| 0.600 | 0.950 | 0 | False | True | 103 | 0.144 | 79210 | 0.154 |
| 0.600 | 0.950 | 1 | False | True | 580 | 0.813 | 15210 | 0.241 |
| 0.600 | 0.900 | 0 | False | True | 103 | 0.144 | 79210 | 0.231 |

## Difficulty Sensitivity

| difficulty | completed | lane | lane_ratio | timepoints | gap_ms | dominant_pattern |
| --- | --- | --- | --- | --- | --- | --- |
| -1.200 | True | 579 | 0.812 | 542 | 340 | 0.264 |
| -0.820 | True | 741 | 1.039 | 543 | 270 | 0.276 |
| -0.400 | True | 959 | 1.345 | 565 | 240 | 0.319 |

## Artifacts

- `phase_a`: `artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_phase_a_policy_sweep.json`
- `flat`: `artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_flat_stress_policy_sweep.json`
- `length`: `artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_length_scaled_flat0_incremental_policy_sweep.json`
- `sampling`: `artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_sampling_incremental_policy_sweep.json`
- `low_temp`: `artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_low_temp_sampling_policy_sweep.json`
- `difficulty`: `artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_difficulty_delta125_policy_sweep.json`
- `rerank`: `artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_candidate_rerank_summary.json`
- `export_metrics`: `artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_best_low_temp_seed0_export_metrics.json`
- frozen data: `artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_decode_policy_report_data.json`

## Reamber Render

- first 30s: `artifacts/evals/pr2_real_riria_policy_sweep/reamber_best_low_temp_seed0/mapper_v21_pr2_real_riria_best_low_temp_seed0__first_30s.png`
- middle 30s: `artifacts/evals/pr2_real_riria_policy_sweep/reamber_best_low_temp_seed0/mapper_v21_pr2_real_riria_best_low_temp_seed0__middle_30s.png`
- last 30s: `artifacts/evals/pr2_real_riria_policy_sweep/reamber_best_low_temp_seed0/mapper_v21_pr2_real_riria_best_low_temp_seed0__last_30s.png`
- longest empty span: `artifacts/evals/pr2_real_riria_policy_sweep/reamber_best_low_temp_seed0/mapper_v21_pr2_real_riria_best_low_temp_seed0__longest_empty_span.png`
- most repetitive span: `artifacts/evals/pr2_real_riria_policy_sweep/reamber_best_low_temp_seed0/mapper_v21_pr2_real_riria_best_low_temp_seed0__most_repetitive_span.png`
