from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

from pulsefield_model.models.mapper.shared.vocab import KEY_COUNT, LaneAction, MapperTupleVocab, coerce_lane_action


MapperProfileName: TypeAlias = Literal["v2_tuple", "v2_1_sparse"]
MapperProfileConfig: TypeAlias = Literal["auto", "v2_tuple", "v2_1_sparse"]

MAPPER_PROFILE_AUTO = "auto"
DEFAULT_MAPPER_PROFILE_NAME: MapperProfileName = "v2_1_sparse"
MAPPER_TOKEN_CONTRACT_VERSION = 2
MAPPER_PROTOCOL_CAPABILITY_NAME = "mapper.tuple_tokens"
HITOBJECT_TOKEN_MANIFEST_V2_PATH = Path(__file__).with_name("hitobject_token_manifest_v2.json")


@dataclass(frozen=True)
class HitObjectToken:
    token_id: int
    token_name: str
    ms_in_ref_audio: int
    actions: tuple[str, ...]

    def message_token(self) -> list[int]:
        return [self.token_id, self.ms_in_ref_audio]


@dataclass(frozen=True)
class MapperProtocolContract:
    capability_name: str = MAPPER_PROTOCOL_CAPABILITY_NAME
    token_contract_version: int = MAPPER_TOKEN_CONTRACT_VERSION
    manifest_path: Path = HITOBJECT_TOKEN_MANIFEST_V2_PATH


@dataclass(frozen=True)
class MapperProfileSpec:
    name: MapperProfileName
    checkpoint_version: str
    model_family: str
    vocab_contract: str
    grammar_contract: str
    bundle_model_id: str
    default_checkpoint_path: Path
    aliases: frozenset[str]
    checkpoint_aliases: frozenset[str]
    protocol_contract: MapperProtocolContract = MapperProtocolContract()


DEFAULT_MAPPER_PROTOCOL_CONTRACT = MapperProtocolContract()
MAPPER_PROFILE_SPECS: Mapping[MapperProfileName, MapperProfileSpec] = {
    "v2_tuple": MapperProfileSpec(
        name="v2_tuple",
        checkpoint_version="v2",
        model_family="mapper_v2",
        vocab_contract="tuple_event_tokens",
        grammar_contract="tuple_event_grammar",
        bundle_model_id="mapper/v2_tuple",
        default_checkpoint_path=Path(
            "artifacts/runs/stage2_mapper_v2/"
            "stage2_mapper_v2_phase_b_global_d768_l8_b1/checkpoint.pt",
        ),
        aliases=frozenset(("v2", "2", "mapper_v2", "tuple", "tuple_event", "tuple_event_tokens")),
        checkpoint_aliases=frozenset(("v2", "mapper_v2", "2")),
    ),
    "v2_1_sparse": MapperProfileSpec(
        name="v2_1_sparse",
        checkpoint_version="v2_1",
        model_family="mapper_v2_1",
        vocab_contract="sparse_lane_actions",
        grammar_contract="sparse_lane_action_grammar",
        bundle_model_id="mapper/v2_1_sparse",
        default_checkpoint_path=Path(
            "artifacts/runs/stage2_mapper_v2_1/"
            "stage2_mapper_v2_1_phase_b_sparse_global_d384_l4_b2/checkpoint.pt",
        ),
        aliases=frozenset(
            (
                "v21",
                "v2_1",
                "2_1",
                "mapper_v2_1",
                "sparse",
                "sparse_lane_action",
                "sparse_lane_actions",
            ),
        ),
        checkpoint_aliases=frozenset(("v2_1", "mapper_v2_1", "2_1")),
    ),
}
DEFAULT_MAPPER_PROFILE_SPEC = MAPPER_PROFILE_SPECS[DEFAULT_MAPPER_PROFILE_NAME]
_MAPPER_PROFILE_BY_ALIAS: Mapping[str, MapperProfileName] = {
    alias: spec.name
    for spec in MAPPER_PROFILE_SPECS.values()
    for alias in (spec.name, *spec.aliases)
}
_MAPPER_PROFILE_BY_CHECKPOINT_ALIAS: Mapping[str, MapperProfileName] = {
    alias: spec.name
    for spec in MAPPER_PROFILE_SPECS.values()
    for alias in (spec.checkpoint_version, *spec.checkpoint_aliases)
}


class MapperProtocolTranslator(Protocol):
    profile_name: MapperProfileName
    protocol_contract: MapperProtocolContract

    def consume_window(self, generated: Any) -> tuple[HitObjectToken, ...]:
        ...

    def flush(self) -> tuple[HitObjectToken, ...]:
        ...


class TupleEventProtocolTranslator:
    """Translates v2 tuple EVENT tokens directly into the protobuf tuple contract."""

    profile_name: MapperProfileName = "v2_tuple"

    def __init__(
        self,
        *,
        source_vocab: Any,
        protocol_vocab: MapperTupleVocab | None = None,
        protocol_contract: MapperProtocolContract = DEFAULT_MAPPER_PROTOCOL_CONTRACT,
    ) -> None:
        self.source_vocab = source_vocab
        self.protocol_vocab = MapperTupleVocab() if protocol_vocab is None else protocol_vocab
        self.protocol_contract = protocol_contract

    def consume_window(self, generated: Any) -> tuple[HitObjectToken, ...]:
        tokens: list[HitObjectToken] = []
        for token_id, state_before in _generated_token_states(generated):
            if not self.source_vocab.is_event_token(token_id):
                continue
            actions = tuple(coerce_lane_action(action) for action in self.source_vocab.decode_event(token_id))
            tokens.append(
                HitObjectToken(
                    token_id=int(token_id),
                    token_name=str(self.source_vocab.token_name(token_id)),
                    ms_in_ref_audio=_state_current_ms(state_before),
                    actions=tuple(action.value for action in actions),
                ),
            )
        return tuple(tokens)

    def flush(self) -> tuple[HitObjectToken, ...]:
        return ()


class SparseLaneActionTupleProtocolTranslator:
    """Buffers v2.1 sparse lane-action tokens until a 4-lane tuple is ready."""

    profile_name: MapperProfileName = "v2_1_sparse"

    def __init__(
        self,
        *,
        source_vocab: Any,
        protocol_vocab: MapperTupleVocab | None = None,
        protocol_contract: MapperProtocolContract = DEFAULT_MAPPER_PROTOCOL_CONTRACT,
    ) -> None:
        self.source_vocab = source_vocab
        self.protocol_vocab = MapperTupleVocab() if protocol_vocab is None else protocol_vocab
        self.protocol_contract = protocol_contract
        self._pending_ms: int | None = None
        self._pending_actions: list[LaneAction] = [LaneAction.NONE] * KEY_COUNT

    def consume_window(self, generated: Any) -> tuple[HitObjectToken, ...]:
        emitted: list[HitObjectToken] = []
        for token_id, state_before in _generated_token_states(generated):
            if self.source_vocab.is_lane_action_token(token_id):
                emitted.extend(self._accept_lane_action(token_id, _state_current_ms(state_before)))
                continue
            if self._pending_ms is not None:
                emitted.extend(self.flush())
        if self._pending_ms is not None:
            emitted.extend(self.flush())
        return tuple(emitted)

    def flush(self) -> tuple[HitObjectToken, ...]:
        if self._pending_ms is None:
            return ()
        time_ms = int(self._pending_ms)
        actions = tuple(self._pending_actions)
        self._pending_ms = None
        self._pending_actions = [LaneAction.NONE] * KEY_COUNT
        if all(action == LaneAction.NONE for action in actions):
            return ()
        token_id = int(self.protocol_vocab.encode_event(actions))
        return (
            HitObjectToken(
                token_id=token_id,
                token_name=self.protocol_vocab.token_name(token_id),
                ms_in_ref_audio=time_ms,
                actions=tuple(action.value for action in actions),
            ),
        )

    def _accept_lane_action(self, token_id: int, time_ms: int) -> tuple[HitObjectToken, ...]:
        emitted: tuple[HitObjectToken, ...] = ()
        if self._pending_ms is not None and int(self._pending_ms) != int(time_ms):
            emitted = self.flush()
        if self._pending_ms is None:
            self._pending_ms = int(time_ms)

        lane, action = self.source_vocab.decode_lane_action(token_id)
        lane = int(lane)
        lane_action = coerce_lane_action(action)
        if self._pending_actions[lane] != LaneAction.NONE:
            raise ValueError(f"duplicate sparse lane-action at {time_ms}ms lane {lane}")
        self._pending_actions[lane] = lane_action
        return emitted


def normalize_mapper_profile_name(value: object) -> MapperProfileConfig:
    if value is None:
        return MAPPER_PROFILE_AUTO
    if isinstance(value, bool):
        raise ValueError("mapper_profile must be a string")
    raw = str(value).strip().lower().replace("-", "_").replace(".", "_")
    if raw in {"", "auto"}:
        return MAPPER_PROFILE_AUTO
    try:
        return _MAPPER_PROFILE_BY_ALIAS[raw]
    except KeyError as exc:
        raise ValueError(f"unsupported mapper_profile: {value!r}") from exc


def mapper_profile_for_checkpoint_version(version: object) -> MapperProfileName:
    normalized = str(version).strip().lower().replace("-", "_").replace(".", "_")
    try:
        return _MAPPER_PROFILE_BY_CHECKPOINT_ALIAS[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported mapper checkpoint version: {version!r}") from exc


def resolve_mapper_profile(
    requested: object = MAPPER_PROFILE_AUTO,
    *,
    checkpoint_version: object | None = None,
) -> MapperProfileSpec:
    requested_name = normalize_mapper_profile_name(requested)
    if requested_name == MAPPER_PROFILE_AUTO:
        if checkpoint_version is None:
            return DEFAULT_MAPPER_PROFILE_SPEC
        return MAPPER_PROFILE_SPECS[mapper_profile_for_checkpoint_version(checkpoint_version)]

    if checkpoint_version is not None:
        checkpoint_profile = mapper_profile_for_checkpoint_version(checkpoint_version)
        if requested_name != checkpoint_profile:
            raise ValueError(
                "mapper_profile does not match checkpoint version: "
                f"{requested_name!r} requested for {checkpoint_version!r} checkpoint",
            )
    return MAPPER_PROFILE_SPECS[requested_name]


def infer_mapper_profile_name_from_vocab(vocab: Any) -> MapperProfileName:
    if callable(getattr(vocab, "is_lane_action_token", None)) and callable(getattr(vocab, "decode_lane_action", None)):
        return "v2_1_sparse"
    if callable(getattr(vocab, "is_event_token", None)) and callable(getattr(vocab, "decode_event", None)):
        return "v2_tuple"
    raise ValueError(f"cannot infer mapper profile from vocab: {type(vocab).__name__}")


def build_mapper_protocol_translator(
    profile: MapperProfileSpec | MapperProfileName | object,
    *,
    source_vocab: Any,
    protocol_vocab: MapperTupleVocab | None = None,
) -> MapperProtocolTranslator:
    protocol_contract = (
        profile.protocol_contract if isinstance(profile, MapperProfileSpec) else DEFAULT_MAPPER_PROTOCOL_CONTRACT
    )
    profile_name = _profile_name(profile)
    if profile_name == "v2_tuple":
        return TupleEventProtocolTranslator(
            source_vocab=source_vocab,
            protocol_vocab=protocol_vocab,
            protocol_contract=protocol_contract,
        )
    if profile_name == "v2_1_sparse":
        return SparseLaneActionTupleProtocolTranslator(
            source_vocab=source_vocab,
            protocol_vocab=protocol_vocab,
            protocol_contract=protocol_contract,
        )
    raise ValueError(f"unsupported mapper profile for protocol translation: {profile!r}")


def _profile_name(profile: MapperProfileSpec | MapperProfileName | object) -> MapperProfileName:
    if isinstance(profile, MapperProfileSpec):
        return profile.name
    normalized = normalize_mapper_profile_name(profile)
    if normalized == MAPPER_PROFILE_AUTO:
        return DEFAULT_MAPPER_PROFILE_NAME
    return normalized


def _generated_token_states(generated: Any) -> Iterator[tuple[int, Any]]:
    tokens = getattr(generated, "tokens")
    states_before = getattr(generated, "states_before")
    for token_id, state_before in zip(tokens, states_before, strict=True):
        yield int(token_id), state_before


def _state_current_ms(state: Any) -> int:
    try:
        return int(getattr(state, "current_ms"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"generated token state must expose integer current_ms: {state!r}") from exc


__all__ = [
    "DEFAULT_MAPPER_PROFILE_NAME",
    "DEFAULT_MAPPER_PROFILE_SPEC",
    "DEFAULT_MAPPER_PROTOCOL_CONTRACT",
    "HITOBJECT_TOKEN_MANIFEST_V2_PATH",
    "HitObjectToken",
    "MAPPER_PROFILE_SPECS",
    "MAPPER_PROFILE_AUTO",
    "MAPPER_PROTOCOL_CAPABILITY_NAME",
    "MAPPER_TOKEN_CONTRACT_VERSION",
    "MapperProfileSpec",
    "MapperProfileConfig",
    "MapperProfileName",
    "MapperProtocolContract",
    "MapperProtocolTranslator",
    "SparseLaneActionTupleProtocolTranslator",
    "TupleEventProtocolTranslator",
    "build_mapper_protocol_translator",
    "infer_mapper_profile_name_from_vocab",
    "mapper_profile_for_checkpoint_version",
    "normalize_mapper_profile_name",
    "resolve_mapper_profile",
]
