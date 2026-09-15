# Oracle-time continuation: causal data and exact replay

The `research/oracle_time_continuation` package implements M0 of the
[continuation plan](Pulsefield_oracle_time_causal_continuation_plan.md#9-里程碑与依赖):
verified source rows, a time skeleton, the 30-note seed, exact pre/post-row state,
and complete-chart terminal legality. This is the data foundation for a causal
continuation model. The learned backbone, training objective, decode policy and
corpus generation runs remain M1–M4 work.

The [V3 formulation](../formulation/notation.md) owns lane actions, simultaneous
rows, occupancy and committed-prefix semantics. The source-action package's
strict raw `.osu` parser supplies byte verification and four-action admission.

## Data ownership

| Owner | Contents |
| --- | --- |
| `ContinuationSource` | Verified source/arrangement SHA-256, existing song-group ID and split, complete supervision rows and their skeleton |
| `TimeSkeleton` | Scheduler-owned, strictly increasing union of original attack and release times |
| `ContinuationState` | Skeleton and exact replay; next position is the committed row count |
| `PredictionInput` | Current time, true skeleton terminal flag, and immutable committed replay facts |
| `ExactReplayState` | First/last committed row, row/note counts, four open LN start times, lane attack/release clocks and completion status |

[`admit_source()`](../../src/pulsefield_model/research/oracle_time_continuation/data.py)
requires the expected source digest and the caller's existing song-group
assignment. It preserves those identities and does not assign new splits.
Every simultaneous action is merged into one complete row. Release-only rows
are retained. Negative/nonfinite times, unsupported sources, overlapping objects
and same-lane close/attack coincidences are rejected without retiming.

Source targets and the complete skeleton stay outside `PredictionInput`.
In particular, it contains no original LN close time, remaining duration,
following gap, future event type, section/window coordinates or source identity.
An open LN contributes only its committed head time and occupancy. The scheduler
provides `is_terminal` only at the true last skeleton row.

## Seed and prefix construction

`minimum_seed()` counts each TAP or LN_START as one original hit object. A chord
counts every attacking lane; LN_CLOSE contributes zero. The seed ends at the
first complete row reaching at least 30 notes and includes all earlier release
rows. `SeedSelection` records `seed_note_count`, `seed_row_count` and
`seed_elapsed_ms`, measured from the first source row through the seed's last row.

Charts with fewer than 30 notes receive `fewer-than-30-notes`. Reaching the
threshold on the last row receives `no-target-suffix`. Both are explicitly
ineligible; `prefix_state()` rejects them. A release-only suffix is eligible.

`prefix_state(target_start)` replays every row from the chart's beginning through
the row preceding that target position. Omitting `target_start` chooses the
minimum seed. Later windows keep the original first-row time, clocks, counts and
open LNs. They cannot begin before the minimum seed or leave an empty target.

## Query, commit and time

[`ContinuationState.query()`](../../src/pulsefield_model/research/oracle_time_continuation/engine.py)
returns the pre-row prediction input without consuming a target or changing
state. Repeated calls return equal inputs. A teacher-forcing caller must score
this input before reading and committing the corresponding true row; a decode
caller commits its chosen row through the same `commit()` method.

```python
def teacher_replay(source, predict, score):
    state = source.prefix_state()
    while not state.finished:
        distribution = predict(state.query())
        target = source.targets[state.next_index]
        score(distribution, target)
        state = state.commit(target)
    return state
```

`commit()` returns a new state. It validates the whole row against the same
pre-state, then updates every lane simultaneously. An illegal row, duplicate or
out-of-order time, or a time differing from the next skeleton slot raises
`ContractError`; the previous state remains unchanged.

`query.clocks` computes elapsed milliseconds since the previous row, first row,
each active LN head, and each lane/hand's latest attack and release. Canonical
hands use lanes `(0, 1)` and `(3, 2)`. Missing predecessor times use `None`; a
known elapsed time of zero remains zero. Querying a long silent interval advances
ages without releasing LNs or modifying history.

Exact replay always has known history from true BOS. Unknown rows and padding
cannot be materialized as `CompleteRow`; all-empty rows are rejected.
`prefill()` requires consecutive positions from skeleton index zero, so cropped
prefixes cannot impersonate BOS. Learned-memory truncation and batch padding
representations belong to M1; they do not erase exact replay facts.

## Legal support and terminal closure

`query.legal_actions` enumerates legal nonempty joint rows. On ordinary rows,
EMPTY preserves occupancy, TAP/LN_START require a closed lane, and LN_CLOSE
requires an open lane. A chunk or sample horizon has no closure effect.

At the true final skeleton row, every open lane must LN_CLOSE, and each closed
lane may TAP or EMPTY. New LN_START actions are excluded. This leaves at least
one nonempty legal candidate for every occupancy state. Commit marks completion;
querying an exhausted skeleton is an error. Seed LN endpoints are determined by
the subsequently committed rows, even when they differ from source endpoints.

Replay retains a fixed number of exact facts. Full source rows remain in their
CPU supervision owner. Parsed-source cache budgets, neural memory, RNG snapshots
and optimizer-version checks will be defined by the later runtime stages.

## Verification

Run the focused contract suite with:

```sh
uv run --offline --group dev pytest -q tests/research/oracle_time_continuation
```

The suite exhaustively checks four-lane action legality across all 16 occupancy
states, including terminal support. It also covers threshold chords and short
seeds, release-only suffixes, simultaneous pre/post clocks, missing versus zero
time, invalid sources, immutable queries, changed future LN endpoints/actions,
changed future skeleton times, mirrored replay, prefix/chunk parity, and a
continuation that replaces the seed's original LN endpoints. An import check
keeps the data path independent of model, legacy training/inference, and masked
feature-building modules.

[`verify_source()`](../../src/pulsefield_model/research/oracle_time_continuation/verification.py)
checks every pre/post-state against an independent source-side oracle. The oracle
uses bisection over raw attack/release times and LN intervals, including exact
endpoint equality. It also compares selected full-prefix constructions with
continuous replay. A mismatch raises `ContractError`. The returned report keeps
source identity, counts, seed eligibility and checked prefix positions.

For a locally available source from the existing pinned split:

```sh
uv run --offline --group dev python - <<'PY'
import json
from pathlib import Path
from pulsefield_model.research.oracle_time_continuation.verification import verify_source
from pulsefield_model.research.scoped_style_modeling.dataset import canonical_json, digest

root = Path('artifacts/scoped-style-modeling')
split = json.loads((root / 'prepare-v1/split-manifest.json').read_text())
expected = '15175f45e91cf7299a9a30166731bf38ee7361399b346fb692cf68e76de5992a'
actual = digest(canonical_json({k: v for k, v in split.items() if k != 'sha256'}).encode())
assert actual == split['sha256'] == expected
sha = '000662977cf314075da22600d2fbabbf140edc15791a5fc5dc473f2f64a58923'
assignment = split['sources'][sha]
report = verify_source((root / 'sources' / (sha + '.osu')).read_bytes(), sha,
                       group_id=assignment['group_id'], split=assignment['split'])
print(json.dumps(report, indent=2))
PY
```

Local source files and the split manifest are required; this API performs no
downloads. The check uses CPU replay and does not require an accelerator extra.

On 2026-09-15, complete source replay was checked for the eight source identities
in the [pinned real-input table](source_action_stage1_verification.md#real-input-provenance-and-bounds),
under that split digest. All **16,800 rows** matched in both pre- and post-state,
covering **23,901 hit objects**, **2,018 LNs** and **417 release-only rows**.
The eight minimum seeds contained 30–31 notes over 15–30 rows. Every source had
an eligible suffix and closed occupancy after its final row. These are data and
replay correctness checks; no model was trained or evaluated.
