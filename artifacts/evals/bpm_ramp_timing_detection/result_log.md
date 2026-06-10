# BPM Ramp Timing Detection Result

## Status

- Attempted rows: 562
- Successful rows: 562
- Failed rows: 0
- Positives: 62
- Negatives: 500
- Recall: 100.0% (62/62)
- False positives: 0/500
- False-positive rate: 0.0%
- Mean total seconds/map: 0.001146
- P95 total seconds/map: 0.002917
- Max total seconds/map: 0.091667
- Positive signal observed: `True`

## Interpretation

- The eval measures cheap structural ramp recognition over timing grids parsed from `.osu` red timing.
- It does not run BeatThis audio inference and does not prove audio-ground-truth novelty.
- Borderline ramp-audit rows are intentionally excluded from the primary recall metric.

## False Negatives

[]

## False Positives

[]
