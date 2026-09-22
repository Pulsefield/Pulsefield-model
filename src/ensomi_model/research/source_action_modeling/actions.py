"""The V3 single-lane action schema at source-action data boundaries.

Raw source inspection may retain incompatible coincidences for other research;
source-action training and assessment reject them without retiming or merging.
"""
from ..scoped_style_modeling.dataset import ContractError
from ..scoped_style_modeling.replay import Row, SourceChart, parse_source as _parse_source

EMPTY, TAP, LN_START, LN_CLOSE = range(4)
LANE_ACTIONS = (EMPTY, TAP, LN_START, LN_CLOSE)
ATTACK_ACTIONS = (TAP, LN_START)
HAND_CLASSES = len(LANE_ACTIONS) ** 2
ROW_CLASSES = HAND_CLASSES ** 2
ACTION_SCHEMA = "v3-lane/empty-tap-ln-start-ln-close-v1"


class SourceActionSchemaError(ContractError):
    """A source row cannot be represented by the V3 single-action lane schema."""


def parse_source(data: bytes, expected_sha256: str) -> SourceChart:
    """Verify original bytes and reject same-lane close/tap or close/head pairs."""
    source = _parse_source(data, expected_sha256)
    if source.row_incompatibilities:
        first, second = source.row_incompatibilities[0]
        raise SourceActionSchemaError(
            f"V3 lane schema rejects simultaneous close+tap/close+head at source lines {first}/{second}")
    return source


def source_actions(row: Row) -> tuple[int, ...]:
    """Translate complete replay facts before masking; never collapse coincidences."""
    actions = []
    for lane, facts in enumerate(row.lanes):
        if sum((facts.tap, facts.ln_start, facts.ln_close)) > 1:
            raise SourceActionSchemaError(
                f"V3 lane schema requires one action per lane: lane {lane + 1} at {row.time_ms} ms")
        actions.append(TAP if facts.tap else LN_START if facts.ln_start else LN_CLOSE if facts.ln_close else EMPTY)
    return tuple(actions)
