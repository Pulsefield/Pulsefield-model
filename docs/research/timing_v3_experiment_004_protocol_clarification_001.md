# Timing v3 Experiment 004 Protocol Clarification 001

- Date: 2026-08-11
- Status: pre-holdout repair clarification
- Original Experiment Card:
  `docs/research/timing_v3_experiment_004_global_constant_jump.md`
- Original Experiment Card SHA-256:
  `0421de8abb1a016a215002ec08903282d7ed2500a9fb85a49ab0dc9fb4c1230e`

## Scope

This addendum resolves two implementation ambiguities found while auditing the
first repair80 execution. It does not change candidate extraction, candidate
caps, cost weights, tie-breaks, stage gates, the numeric attempt cap, the
per-audio timeout, or the fallback policy. No Experiment 004 holdout, broad, or
full-corpus result was created or inspected before this clarification.

The original card remains immutable. Every Experiment 004 execution after this
clarification must bind both the original card bytes and this addendum bytes in
its behavior provenance. The repaired implementation must restart repair80 to
new immutable output paths before any holdout is selected or executed.

## Attempt-cap scope

The hard cap of `120000` scored section or terminal-tail attempts applies
independently to each `(cache_audio_key, variant)` fit. A row still evaluates
all frozen variants in order: `CJ0`, `CJ1`, `CJ2`, and `CJ3`. Each variant must
independently return either an accepted grid or a deterministic tagged core
fallback on a cache-valid row after candidate extraction succeeds, unless a
row-level hard guard terminates execution first.

At most `120000` distinct section-score equivalence classes may be evaluated on
cache misses for one audio-variant fit. An attempt to evaluate the `120001`st
class is rejected before scoring and returns `edge_attempt_cap_exceeded`. The
fitter may not return a best-so-far or anytime accepted path after the cap is
exhausted.

The `180` second timeout remains row-scoped. It includes cache loading, the
current-v2 comparator, candidate extraction, and all four variant fits. A
timeout does not grant a later variant extra attempt budget and does not change
the selected `CJ3`-or-v2 fallback policy. A timeout immediately terminates the
row: already completed variant results are retained, while the active variant
and every later variant receive deterministic `not_run` payloads tagged
`timeout`. A tagged core fallback from one variant does not stop later variants;
timeout is the only normal early termination during the frozen variant loop.
Existing cache, configuration, source-integrity, candidate-extraction,
serialization, and unexpected-internal-failure guards retain their fail-closed
`not_run` semantics. Those are protocol violations or unavailable inputs, not
core fallbacks.

This interpretation is required to keep variant attempt budgets independent. A
cap shared across variants would make later variants depend on earlier
variants' search cost and would turn their fallback status into an
evaluation-order artifact.

## Section-score cache equivalence

For an interior section, the card's frozen key
`(left_anchor_id, right_anchor_id, N, variant)` denotes one geometric section
score for `CJ1` and `CJ2`. Absolute beat number and downbeat phase are not score
inputs for those variants and must not split equivalent cache entries.

For every downbeat-aware score (`CJ0` terminal tails and `CJ3` interior sections
or terminal tails), the only phase identity permitted in the cache key is the
effective global-downbeat residue:

```text
phase_class = (left_beat_at_anchor - global_downbeat_phase) mod 4
```

The complete phase-class domain is `{0, 1, 2, 3, none}`. Absolute beat number
and raw global-downbeat phase must not split entries within one phase class.
The `none` downbeat case remains distinct from residues `0` through `3`.
Implementations must use a canonical residue-based numerical form so
mathematically equivalent phase classes produce exactly identical score inputs
rather than differing through large-integer floating-point addition.

The complete `CJ3` interior key is
`(left_anchor_id, right_anchor_id, N, variant, phase_class)`. The non-downbeat
`CJ1` and `CJ2` interior key remains
`(left_anchor_id, right_anchor_id, N, variant)`.

For `CJ1` and `CJ2`, the complete terminal-tail key is
`(left_anchor_id, terminal_sentinel, N, variant, round(bpm, 6))`. Absolute beat
number and downbeat phase must not split that key. For `CJ0` and `CJ3`, the key
adds the effective phase class from `{0, 1, 2, 3, none}`. The exclusive endpoint
`E` is fixed for one audio fit and is represented by the terminal sentinel.

Terminal-tempo identity as `round(bpm, 6)` is mandatory. The terminal BPM is
selected independently from the frozen terminal tempo set, and one tail beat
count can correspond to multiple BPM values with different lattices or prior
costs.

The attempt counter advances only when the corresponding complete section-score
equivalence class is evaluated on a cache miss. Cache hits do not consume the
attempt budget. Variant-independent beat-count candidate sets may be shared
across the four variants; section scores and attempt counters remain
variant-local.

## Unchanged constraints

- One declared, shift-zero BeatThis cache remains the only inference input.
- `.osu`, objects, redlines, metadata, network data, and raw audio remain
  forbidden in prediction and candidate assembly.
- The numeric attempt cap remains `120000` for every audio-variant fit.
- The row timeout remains `180` seconds.
- Exceeding the cap still yields a tagged fallback, never a truncated success.
- Candidate generation, weights, priors, section limits, beam width, expansion
  tie-breaks, evaluation metrics, and stage gates are unchanged.
