from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from pulsefield_model.inference.stream_with_cache import DecoderWindow, StreamWithCache, StreamWithCacheConfig
from pulsefield_model.inference.mapper_protocol import (
    DEFAULT_MAPPER_PROFILE_SPEC,
    MAPPER_PROFILE_SPECS,
    MapperProtocolContract,
    SparseLaneActionTupleProtocolTranslator,
    TupleEventProtocolTranslator,
    infer_mapper_profile_name_from_vocab,
    normalize_mapper_profile_name,
    resolve_mapper_profile,
)
from pulsefield_model.inference.protocol_adapter import PulsefieldProtocolAdapter
from pulsefield_model.inference.service_models import HitObjectTokenEvent
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab
from pulsefield_model.models.mapper.v2_1 import MapperV21Vocab


def test_mapper_profile_specs_are_the_metadata_source_of_truth() -> None:
    v2 = MAPPER_PROFILE_SPECS["v2_tuple"]
    v21 = MAPPER_PROFILE_SPECS["v2_1_sparse"]

    assert DEFAULT_MAPPER_PROFILE_SPEC is v21
    assert v2.bundle_model_id == "mapper/v2_tuple"
    assert v2.default_checkpoint_path.name == "checkpoint.pt"
    assert v21.bundle_model_id == "mapper/v2_1_sparse"
    assert normalize_mapper_profile_name("tuple_event_tokens") == v2.name
    assert normalize_mapper_profile_name("sparse-lane-actions") == v21.name


def test_mapper_profile_resolution_matches_checkpoint_version() -> None:
    assert resolve_mapper_profile("auto", checkpoint_version="v2").name == "v2_tuple"
    assert resolve_mapper_profile("auto", checkpoint_version="v2_1").name == "v2_1_sparse"
    assert resolve_mapper_profile("v2.1", checkpoint_version="mapper_v2_1").name == "v2_1_sparse"

    with pytest.raises(ValueError, match="mapper_profile does not match checkpoint version"):
        resolve_mapper_profile("v2_tuple", checkpoint_version="v2_1")


def test_mapper_profile_can_be_inferred_from_vocab_shape() -> None:
    assert infer_mapper_profile_name_from_vocab(MapperTupleVocab()) == "v2_tuple"
    assert infer_mapper_profile_name_from_vocab(MapperV21Vocab()) == "v2_1_sparse"


def test_tuple_event_protocol_translator_passes_tuple_event_tokens_through() -> None:
    vocab = MapperTupleVocab()
    event_token = vocab.encode_event(("TAP", "NONE", "HOLD_START", "NONE"))
    generated = SimpleNamespace(
        tokens=[vocab.time_shift_token_id(100), event_token],
        states_before=[_state(0), _state(100)],
    )

    tokens = TupleEventProtocolTranslator(source_vocab=vocab).consume_window(generated)

    assert len(tokens) == 1
    assert tokens[0].token_id == event_token
    assert tokens[0].token_name == vocab.token_name(event_token)
    assert tokens[0].ms_in_ref_audio == 100
    assert tokens[0].actions == ("TAP", "NONE", "HOLD_START", "NONE")


def test_sparse_lane_action_translator_buffers_until_tuple_boundary() -> None:
    source_vocab = MapperV21Vocab()
    protocol_vocab = MapperTupleVocab()
    generated = SimpleNamespace(
        tokens=[
            source_vocab.time_shift_token_id(100),
            source_vocab.lane_action_token_id(0, "TAP"),
            source_vocab.lane_action_token_id(2, "HOLD_START"),
            source_vocab.time_shift_token_id(10),
            source_vocab.lane_action_token_id(1, "TAP"),
        ],
        states_before=[_state(0), _state(100), _state(100), _state(100), _state(110)],
    )

    tokens = SparseLaneActionTupleProtocolTranslator(
        source_vocab=source_vocab,
        protocol_vocab=protocol_vocab,
    ).consume_window(generated)

    assert [token.ms_in_ref_audio for token in tokens] == [100, 110]
    assert [token.actions for token in tokens] == [
        ("TAP", "NONE", "HOLD_START", "NONE"),
        ("NONE", "TAP", "NONE", "NONE"),
    ]
    assert [token.token_id for token in tokens] == [
        protocol_vocab.encode_event(("TAP", "NONE", "HOLD_START", "NONE")),
        protocol_vocab.encode_event(("NONE", "TAP", "NONE", "NONE")),
    ]


def test_stream_with_cache_uses_sparse_profile_translator_for_generated_windows() -> None:
    source_vocab = MapperV21Vocab()
    protocol_vocab = MapperTupleVocab()
    backend = _SparseGeneratedStream(source_vocab)
    backend._session_runtimes["s1"] = object()

    async def collect_tokens():
        tokens = []
        async for token in backend.iter_hitobject_tokens(
            session_id="s1",
            audio_path=Path("/tmp/song.wav"),
            audio_length_ms=2_000,
            window=DecoderWindow(start_ms=0, end_ms=8_000),
        ):
            tokens.append(token)
        return tokens

    tokens = asyncio.run(collect_tokens())

    assert [token.ms_in_ref_audio for token in tokens] == [100]
    assert tokens[0].actions == ("TAP", "NONE", "TAP", "NONE")
    assert tokens[0].token_id == protocol_vocab.encode_event(("TAP", "NONE", "TAP", "NONE"))


def test_protocol_adapter_uses_configured_mapper_contract_for_stream_begin() -> None:
    adapter = PulsefieldProtocolAdapter(
        mapper_contract=MapperProtocolContract(token_contract_version=7),
    )

    envelopes = list(
        adapter.outbound_envelopes_for_event(
            HitObjectTokenEvent(session_id="s1", token_id=10, ms_in_ref_audio=100),
        ),
    )

    assert envelopes[0].WhichOneof("payload") == "mapper_stream_begin"
    assert envelopes[0].mapper_stream_begin.token_contract_version == 7
    assert envelopes[1].WhichOneof("payload") == "hit_object_token"


def _state(current_ms: int) -> SimpleNamespace:
    return SimpleNamespace(current_ms=int(current_ms))


class _SparseGeneratedStream(StreamWithCache):
    def __init__(self, vocab: MapperV21Vocab) -> None:
        super().__init__(StreamWithCacheConfig(token_send_interval_s=0.0, mapper_profile="v2_1_sparse"))
        self.vocab = vocab
        self.model_runtime = SimpleNamespace(
            vocab=vocab,
            mapper_profile=resolve_mapper_profile("v2_1_sparse"),
        )
        self.models_ready = True

    def _vocab(self) -> MapperV21Vocab:
        return self.vocab

    def _generate_window(self, session_id, session_runtime, window, audio_length_ms):
        del session_id, session_runtime, window, audio_length_ms
        return SimpleNamespace(
            tokens=[
                self.vocab.time_shift_token_id(100),
                self.vocab.lane_action_token_id(0, "TAP"),
                self.vocab.lane_action_token_id(2, "TAP"),
            ],
            states_before=[_state(0), _state(100), _state(100)],
        )
