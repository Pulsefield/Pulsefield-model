# Mapper v2.1 PR2 Real Riria Timing Drift Audit

This report audits timing alignment for the exported best PR2 candidate. It compares the generated `.osu` against the original Riria reference chart using parsed osu!mania hitobjects and red timing points.

## Files

- Generated export: `artifacts/evals/pr2_real_riria_policy_sweep/mapper_v21_pr2_real_riria_best_low_temp_seed0.osu`
- Reference chart: `dataset/0/1942086/Riria. - Shitsuren Song Takusan Kiite Naite Bakari no Watashi wa Mou. (TV Size) (Kibitz) [Stay With Me, Don't Let Go].osu`
- Generated policy: `flat=0.0`, `delta=1.25`, `temperature=0.4`, `top_p=0.95`, `seed=0`

## Method

- Parsed red timing points from each `.osu`.
- Parsed 4K osu!mania hitobjects from each `.osu`.
- Measured each hitobject start time against the active red timing grid.
- Residual means the distance in milliseconds to the nearest grid line at `beat_length / subdivision`.
- Also measured all note times, meaning starts plus hold ends.
- Checked same-lane overlaps separately.

## Timing Points

| file | red offset ms | beat length ms | meter |
| --- | ---: | ---: | ---: |
| reference | 1000 | 800 | 4 |
| generated | 1020 | 800 | 4 |

The generated export has the same beat length as the reference but places the red timing offset `20 ms` later.

Changing the generated audit grid from offset `1020 ms` to the reference offset `1000 ms` does not explain the poor alignment. Against `1000 ms`, generated starts still have only `11.2%` within `5 ms` of `1/16`, and only `42.1%` within `5 ms` of `1/48`.

## Hitobject Summary

| file | objects | taps | holds | first start ms | last note time ms | starts before red offset | same-lane overlaps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 489 | 265 | 224 | 800 | 87200 | 2 | 0 |
| generated | 696 | 670 | 26 | 170 | 87190 | 3 | 0 |

The generated file is mechanically valid in this audit: it has no same-lane overlaps. It also starts much earlier than the reference, with three objects before `800 ms`.

## Start-Time Grid Alignment

Residuals below are measured against each file's own red timing point.

| subdivision | reference mean ms | reference median ms | reference p90 ms | reference <=5ms | generated mean ms | generated median ms | generated p90 ms | generated <=5ms | generated <=10ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/4 | 3.401 | 0.000 | 0.000 | 95.1% | 49.986 | 50.000 | 90.000 | 2.0% | 4.3% |
| 1/8 | 1.507 | 0.000 | 0.000 | 96.1% | 25.187 | 30.000 | 40.000 | 2.6% | 7.0% |
| 1/12 | 0.560 | 0.000 | 0.000 | 97.8% | 16.739 | 16.667 | 30.000 | 17.5% | 28.4% |
| 1/16 | 0.436 | 0.000 | 0.000 | 97.3% | 14.670 | 20.000 | 20.000 | 11.4% | 20.4% |
| 1/24 | 0.219 | 0.000 | 0.000 | 98.8% | 8.769 | 10.000 | 13.333 | 34.2% | 56.5% |
| 1/32 | 0.229 | 0.000 | 0.000 | 97.3% | 5.963 | 5.000 | 10.000 | 52.2% | 78.4% |
| 1/48 | 0.014 | 0.000 | 0.000 | 100.0% | 3.975 | 3.333 | 6.667 | 69.4% | 100.0% |
| 1/80 | 0.096 | 0.000 | 0.000 | 100.0% | 0.000 | 0.000 | 0.000 | 100.0% | 100.0% |

The generated starts align perfectly to `1/80` because `800 ms / 80 = 10 ms`, matching the model/export time grid. They do not align well to ordinary musical subdivisions such as `1/8`, `1/12`, or `1/16`.

## Start And Hold-End Alignment

The same pattern holds when note starts and hold ends are both included. The generated file has only `26` holds, so all-time alignment is close to start-time alignment.

| subdivision | reference all-times mean ms | reference all-times <=5ms | generated all-times mean ms | generated all-times <=5ms |
| --- | ---: | ---: | ---: | ---: |
| 1/4 | 5.418 | 93.3% | 49.986 | 2.1% |
| 1/8 | 1.314 | 96.5% | 25.222 | 2.8% |
| 1/12 | 1.229 | 95.9% | 16.764 | 17.5% |
| 1/16 | 0.439 | 97.3% | 14.584 | 11.6% |
| 1/24 | 0.154 | 99.2% | 8.758 | 34.1% |
| 1/32 | 0.227 | 97.3% | 5.963 | 51.8% |
| 1/48 | 0.014 | 100.0% | 3.975 | 69.1% |
| 1/80 | 0.094 | 100.0% | 0.000 | 100.0% |

## Onset Gap Distribution

Gaps are measured between unique sorted start times, so chords at the same timestamp are collapsed.

| reference gap ms | count |
| ---: | ---: |
| 200 | 267 |
| 400 | 63 |
| 50 | 7 |
| 134 | 6 |
| 100 | 5 |
| 133 | 5 |
| 350 | 3 |
| 800 | 3 |
| 300 | 2 |
| 266 | 2 |
| 66 | 2 |
| 67 | 1 |

| generated gap ms | count |
| ---: | ---: |
| 160 | 245 |
| 150 | 207 |
| 170 | 46 |
| 310 | 26 |
| 210 | 2 |
| 230 | 1 |
| 240 | 1 |

The reference is dominated by `200 ms` and `400 ms` gaps. The generated file is dominated by `150-170 ms` gaps, especially `160 ms`.

## Audit Conclusion

The generated `.osu` is structurally valid and has no same-lane overlap in this audit. It does not fit the song timing well.

The object times are exactly aligned to a `10 ms` grid, but that is not the same as aligning to the chart's beat grid. The reference chart is near-perfect on normal beat subdivisions; the generated chart is not. The generated export therefore looks complete and dense, but its hitobject timing is rhythmically drifted/off-grid relative to the Riria reference timing.
