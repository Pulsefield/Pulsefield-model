import asyncio
import errno
import json
import tempfile
import unittest
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import torch
from pulsefield.protocol.v1 import envelope_pb2, inference_pb2

from pulsefield_model.inference.stream_with_cache import (
    DecoderWindow,
    HitObjectToken,
    StreamWithCache,
    _apply_time_shift_length_penalty,
    _mapper_v2_logits_fn,
    _time_shift_length_penalty_tensors,
    clamp_decoder_window_to_audio,
    decoder_windows_until_audio_end,
)
from pulsefield_model.inference.routed_backend import (
    RoutedInferenceBackend,
    TimingMockStreamBackend,
)
from pulsefield_model.inference.ws_endpoint import (
    InferenceEndpoint,
    InferenceError,
    ProtocolError,
    ReferenceClock,
    SESSION_TRANSITION_RULES,
    SessionState,
    WsEndpointConfig,
    audio_end_reset_host_time_ms,
    audio_path_from_message,
    clamp_decoder_window_for_policy,
    choose_decoder_window,
    current_host_time_ms,
    decoder_window_policy_for_route,
    difficulty_from_message,
    infer_message_type,
    inference_route_from_message,
    host_time_ms_reached,
    reference_clock_from_message,
    session_transition_target,
    ws_status_log_payload,
)
from pulsefield_model.inference.protocol_adapter import PulsefieldProtocolAdapter
from pulsefield_model.inference.service_models import (
    AudioCommand,
    EndOfStreamEvent,
    HitObjectTokenEvent,
    MapperStreamBeginEvent,
    ReadyCommand,
    ReferenceTimeCommand,
    StopCommand,
    event_to_endpoint_payload,
)
from pulsefield_model.inference.protobuf_transport import (
    MAPPER_TOKEN_CONTRACT_VERSION,
    envelope_to_command,
    outbound_payload_to_envelope,
)
from pulsefield_model.inference.ws_server import _handle_websocket_client
from pulsefield_model.data.control_windows import normalize_difficulty
from pulsefield_model.models.mapper.shared.generation import MapperGeneratedWindow, MapperGenerationStep
from pulsefield_model.models.mapper.shared.replay import empty_ln_carry_state
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab
from pulsefield_model.timing.canonicalization import TIMING_CANONICALIZATION_BPM_80_160


MANIFEST_PATH = Path("src/pulsefield_model/inference/hitobject_token_manifest_v2.json")


def _timing_report() -> dict[str, object]:
    return {
        "source_path": "song.mp3",
        "provider": "unit-test",
        "checkpoint_path": "fake",
        "device": "cpu",
        "canonicalization": "bpm_80_160",
        "frame_count": 110,
        "frame_rate_hz": 50.0,
        "fit_seconds": 0.001,
        "score": 1.0,
        "diagnostics": {},
        "segments": [
            {
                "offset_ms": 120.0,
                "beat_length_ms": 500.0,
                "bpm": 120.0,
                "meter": 4,
            },
        ],
    }


class FakePeer:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send_event(self, event) -> None:
        self.messages.append(dict(event_to_endpoint_payload(event)))


class DisconnectingPeer:
    async def send_event(self, event: dict) -> None:
        del event
        raise OSError(errno.ENOTCONN, "Socket is not connected")


class FakeIncrementalMapperModel:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = int(vocab_size)
        self.calls: list[tuple[int, int]] = []
        self.control_attention_kv_cache_args: list[object] = []

    def create_empty_decode_state(self, *, batch_size: int, device: torch.device):
        del batch_size, device
        return SimpleNamespace(steps=0)

    def incremental_decode_next_token(self, *, decode_state, decoder_input_token, position: int, **kwargs):
        token_id = int(decoder_input_token.reshape(-1)[0].item())
        self.calls.append((int(position), token_id))
        self.control_attention_kv_cache_args.append(kwargs.get("control_attention_kv_cache"))
        logits = torch.zeros((1, self.vocab_size), dtype=torch.float32)
        logits[0, token_id] = 1.0
        return SimpleNamespace(
            decode_state=SimpleNamespace(steps=int(getattr(decode_state, "steps", 0)) + 1),
            logits_final=logits,
        )


class WsEndpointProtocolTests(unittest.TestCase):
    def test_infer_message_type_accepts_protocol_message_types(self) -> None:
        self.assertEqual(infer_message_type({"type": "ready"}), "ready")
        self.assertEqual(infer_message_type({"type": "node_hello"}), "node_hello")
        self.assertEqual(infer_message_type({"type": "audio"}), "audio")
        self.assertEqual(infer_message_type({"type": "reference_time"}), "reference_time")
        self.assertEqual(infer_message_type({"type": "stop"}), "stop")
        with self.assertRaisesRegex(ProtocolError, "message must include type"):
            infer_message_type({"session_id": "s1", "audio_path": "/tmp/song.wav"})
        with self.assertRaisesRegex(ProtocolError, "message must include type"):
            infer_message_type({"control": "ready"})
        with self.assertRaisesRegex(ProtocolError, "unsupported message type"):
            infer_message_type({"type": "audio_path", "session_id": "s1", "audio_path": "/tmp/song.wav"})
        with self.assertRaisesRegex(ProtocolError, "unsupported message type"):
            infer_message_type({"type": "unknown"})

    def test_audio_path_from_message_reads_normalized_local_path(self) -> None:
        self.assertEqual(audio_path_from_message({"audio_path": "/tmp/a.wav"}), "/tmp/a.wav")
        self.assertIsNone(audio_path_from_message({"audio": {"path": "/tmp/a.wav"}}))

    def test_inference_route_from_message_defaults_mapper(self) -> None:
        self.assertEqual(inference_route_from_message({"audio_path": "/tmp/a.wav"}), "mapper")

    def test_inference_route_from_message_accepts_protocol_route_names(self) -> None:
        self.assertEqual(inference_route_from_message({"route": "mapper"}), "mapper")
        self.assertEqual(inference_route_from_message({"route": "INFERENCE_ROUTE_MAPPER"}), "mapper")
        self.assertEqual(inference_route_from_message({"route": "timing_mock"}), "timing_mock")
        self.assertEqual(
            inference_route_from_message({"route": "INFERENCE_ROUTE_TIMING_MOCK"}),
            "timing_mock",
        )
        with self.assertRaisesRegex(ProtocolError, "unsupported inference route"):
            inference_route_from_message({"route": "unknown"})

    def test_protocol_envelope_audio_request_decodes_to_service_command(self) -> None:
        envelope = envelope_pb2.Envelope(session_id="s1")
        envelope.audio.audio.local_path = "/tmp/song.wav"
        envelope.audio.audio.audio_length_ms = 2_000
        envelope.audio.difficulty = 5.0
        envelope.audio.route = inference_pb2.INFERENCE_ROUTE_MAPPER

        command = envelope_to_command(envelope)

        self.assertEqual(
            command,
            AudioCommand(
                session_id="s1",
                audio_path="/tmp/song.wav",
                audio_length_ms=2_000,
                difficulty=5.0,
                route="mapper",
            ),
        )

    def test_session_transition_rules_define_endpoint_lifecycle(self) -> None:
        self.assertEqual(
            {transition: rule.to_status for transition, rule in SESSION_TRANSITION_RULES.items()},
            {
                "start_audio": "audio_preparing",
                "audio_prepared": "audio_ready",
                "reference_time": "streaming",
                "replace_audio": "no_session",
                "stop": "no_session",
                "prepare_failed": "no_session",
                "peer_disconnect": "no_session",
            },
        )
        self.assertEqual(
            SESSION_TRANSITION_RULES["reference_time"].allowed_from,
            frozenset({"audio_ready", "streaming"}),
        )
        self.assertEqual(
            SESSION_TRANSITION_RULES["replace_audio"].allowed_from,
            frozenset({"audio_ready", "streaming"}),
        )

    def test_session_transition_target_rejects_invalid_state_moves(self) -> None:
        self.assertEqual(session_transition_target("start_audio", "no_session"), "audio_preparing")
        self.assertEqual(session_transition_target("audio_prepared", "audio_preparing"), "audio_ready")
        self.assertEqual(session_transition_target("reference_time", "audio_ready"), "streaming")
        self.assertEqual(session_transition_target("reference_time", "streaming"), "streaming")
        self.assertEqual(session_transition_target("replace_audio", "audio_ready"), "no_session")
        self.assertEqual(session_transition_target("stop", "no_session"), "no_session")

        with self.assertRaisesRegex(ProtocolError, "reference_time.*no_session.*audio_ready, streaming"):
            session_transition_target("reference_time", "no_session")
        with self.assertRaisesRegex(ProtocolError, "replace_audio.*audio_preparing.*audio_ready, streaming"):
            session_transition_target("replace_audio", "audio_preparing")

    def test_protocol_outbound_hitobject_token_uses_pypi_envelope(self) -> None:
        envelope = outbound_payload_to_envelope(
            {
                "type": "hit_object_token",
                "session_id": "s1",
                "token_id": 10,
                "ms_in_ref_audio": 1_240,
                "token_index": 3,
            },
            sequence=7,
        )

        roundtrip = envelope_pb2.Envelope()
        roundtrip.ParseFromString(envelope.SerializeToString())

        self.assertEqual(roundtrip.session_id, "s1")
        self.assertEqual(roundtrip.sequence, 7)
        self.assertEqual(roundtrip.WhichOneof("payload"), "hit_object_token")
        self.assertEqual(roundtrip.hit_object_token.token_id, 10)
        self.assertEqual(roundtrip.hit_object_token.ms_in_ref_audio, 1_240)
        self.assertEqual(roundtrip.hit_object_token.token_index, 3)

    def test_protocol_adapter_preserves_stream_begin_token_index_and_sequence_behavior(self) -> None:
        adapter = PulsefieldProtocolAdapter()

        envelopes = list(
            adapter.outbound_envelopes_for_event(
                HitObjectTokenEvent(session_id="s1", token_id=10, ms_in_ref_audio=1_240),
            ),
        )
        envelopes.extend(
            adapter.outbound_envelopes_for_event(
                EndOfStreamEvent(session_id="s1", audio_length_ms=2_000, complete_through_ms=2_000),
            ),
        )

        self.assertEqual([envelope.WhichOneof("payload") for envelope in envelopes], [
            "mapper_stream_begin",
            "hit_object_token",
            "end_of_stream",
        ])
        self.assertEqual(envelopes[0].mapper_stream_begin.token_contract_version, MAPPER_TOKEN_CONTRACT_VERSION)
        self.assertEqual(envelopes[1].hit_object_token.token_index, 0)
        self.assertEqual([envelope.sequence for envelope in envelopes], [1, 2, 3])

    def test_protocol_adapter_can_reset_stream_state_before_session_reuse(self) -> None:
        adapter = PulsefieldProtocolAdapter()

        first = list(
            adapter.outbound_envelopes_for_event(
                HitObjectTokenEvent(session_id="s1", token_id=10, ms_in_ref_audio=1_240),
            ),
        )
        adapter.reset_session_stream("s1")
        second = list(
            adapter.outbound_envelopes_for_event(
                HitObjectTokenEvent(session_id="s1", token_id=11, ms_in_ref_audio=1_500),
            ),
        )

        self.assertEqual([envelope.WhichOneof("payload") for envelope in first], [
            "mapper_stream_begin",
            "hit_object_token",
        ])
        self.assertEqual([envelope.WhichOneof("payload") for envelope in second], [
            "mapper_stream_begin",
            "hit_object_token",
        ])
        self.assertEqual(first[1].hit_object_token.token_index, 0)
        self.assertEqual(second[1].hit_object_token.token_index, 0)
        self.assertEqual([envelope.sequence for envelope in first + second], [1, 2, 3, 4])

    def test_protocol_adapter_surfaces_outbound_validation_as_protocol_error(self) -> None:
        adapter = PulsefieldProtocolAdapter()

        with self.assertRaisesRegex(ProtocolError, "token_contract_version must be positive"):
            list(
                adapter.outbound_envelopes_for_event(
                    MapperStreamBeginEvent(session_id="s1", token_contract_version=-1),
                ),
            )

    def test_difficulty_from_message_validates_supported_mapper_range(self) -> None:
        self.assertEqual(difficulty_from_message({}, default=4.0), 4.0)
        self.assertEqual(difficulty_from_message({"difficulty": 5.0}, default=4.0), 5.0)
        with self.assertRaisesRegex(ProtocolError, "difficulty"):
            difficulty_from_message({"difficulty": 7.0}, default=4.0)

    def test_current_host_time_ms_uses_monotonic_milliseconds(self) -> None:
        first = current_host_time_ms()
        second = current_host_time_ms()

        self.assertGreaterEqual(first, 0.0)
        self.assertGreaterEqual(second, first)

    def test_reference_clock_accepts_pulsefield_host_time(self) -> None:
        clock = reference_clock_from_message(
            {
                "session_id": "s1",
                "ref_time_ms": 1_000,
                "local_host_time_send_ms": 50_000.25,
            },
        )

        self.assertEqual(clock.ref_time_ms, 1_000)
        self.assertEqual(clock.local_host_time_send_ms, 50_000.25)

    def test_audio_end_reset_deadline_uses_sent_host_time_and_ref_time_ms(self) -> None:
        deadline = audio_end_reset_host_time_ms(
            reference_clock=ReferenceClock(
                ref_time_ms=1_000,
                local_host_time_send_ms=50_000.25,
                received_local_host_time_ms=50_250.25,
            ),
            audio_length_ms=10_000,
            reset_after_audio_end_ms=2_000,
        )

        self.assertEqual(deadline, 61_000.25)
        self.assertFalse(host_time_ms_reached(deadline, now_ms=61_000.0))
        self.assertTrue(host_time_ms_reached(deadline, now_ms=61_000.25))

    def test_ws_status_log_payload_includes_status_transition(self) -> None:
        payload = ws_status_log_payload(
            session_id="s1",
            from_status="audio_ready",
            to_status="streaming",
            reason="reference_time",
            ref_time_ms=1_234,
            reset_local_host_time_ms=90_000.25,
        )

        self.assertEqual(payload["event"], "ws_status")
        self.assertIn("local_host_time_ms", payload)
        self.assertEqual(payload["session_id"], "s1")
        self.assertEqual(payload["from"], "audio_ready")
        self.assertEqual(payload["to"], "streaming")
        self.assertEqual(payload["reason"], "reference_time")
        self.assertEqual(payload["ref_time_ms"], 1_234)
        self.assertEqual(payload["reset_local_host_time_ms"], 90_000.25)

    def test_choose_decoder_window_rounds_up_to_later_control_window(self) -> None:
        clock = ReferenceClock(
            ref_time_ms=1_234,
            local_host_time_send_ms=10_000.0,
            received_local_host_time_ms=10_500.0,
        )

        window = choose_decoder_window(
            clock,
            WsEndpointConfig(decoder_window_ms=8_000, decoder_lead_ms=2_000),
        )

        self.assertEqual(window, DecoderWindow(start_ms=8_000, end_ms=16_000))

    def test_choose_decoder_window_keeps_exact_later_boundary(self) -> None:
        clock = ReferenceClock(
            ref_time_ms=5_500,
            local_host_time_send_ms=10_000.0,
            received_local_host_time_ms=10_500.0,
        )

        window = choose_decoder_window(
            clock,
            WsEndpointConfig(decoder_window_ms=8_000, decoder_lead_ms=2_000),
        )

        self.assertEqual(window, DecoderWindow(start_ms=8_000, end_ms=16_000))

    def test_choose_decoder_window_can_disable_lead_and_boundary_roundup(self) -> None:
        clock = ReferenceClock(
            ref_time_ms=1_234,
            local_host_time_send_ms=10_000.0,
            received_local_host_time_ms=10_500.0,
        )

        window = choose_decoder_window(
            clock,
            WsEndpointConfig(decoder_window_ms=8_000, decoder_lead_ms=2_000),
            decoder_lead_ms=0,
            align_to_decoder_window=False,
        )

        self.assertEqual(window, DecoderWindow(start_ms=1_734, end_ms=9_734))

    def test_decoder_window_policy_disables_live_lead_for_timing_mock_by_default(self) -> None:
        config = WsEndpointConfig(decoder_lead_ms=2_000)

        self.assertEqual(decoder_window_policy_for_route("mapper", config).decoder_lead_ms, 2_000)
        self.assertTrue(decoder_window_policy_for_route("mapper", config).align_to_decoder_window)
        self.assertEqual(decoder_window_policy_for_route("timing_mock", config).decoder_lead_ms, 0)
        self.assertFalse(decoder_window_policy_for_route("timing_mock", config).align_to_decoder_window)

    def test_unaligned_decoder_window_clamp_does_not_round_near_audio_end_back_to_boundary(self) -> None:
        config = WsEndpointConfig(decoder_window_ms=8_000)
        policy = decoder_window_policy_for_route("timing_mock", config)

        window = clamp_decoder_window_for_policy(
            DecoderWindow(start_ms=9_400, end_ms=17_400),
            policy=policy,
            audio_length_ms=9_500,
            config=config,
        )

        self.assertEqual(window, DecoderWindow(start_ms=9_400, end_ms=17_400))

    def test_clamp_decoder_window_keeps_reference_window_inside_audio(self) -> None:
        config = WsEndpointConfig(decoder_window_ms=8_000)

        window = clamp_decoder_window_to_audio(
            DecoderWindow(start_ms=24_000, end_ms=32_000),
            audio_length_ms=18_500,
            config=config,
        )

        self.assertEqual(window, DecoderWindow(start_ms=16_000, end_ms=24_000))

    def test_decoder_windows_continue_from_selected_window_until_audio_end(self) -> None:
        config = WsEndpointConfig(decoder_window_ms=8_000)

        windows = decoder_windows_until_audio_end(
            DecoderWindow(start_ms=8_000, end_ms=16_000),
            audio_length_ms=18_500,
            config=config,
        )

        self.assertEqual(
            windows,
            (
                DecoderWindow(start_ms=8_000, end_ms=16_000),
                DecoderWindow(start_ms=16_000, end_ms=24_000),
            ),
        )

    def test_time_shift_length_penalty_applies_flat_scalar_to_all_ts_tokens(self) -> None:
        vocab = MapperTupleVocab()
        logits = torch.zeros(vocab.size, dtype=torch.float32)
        ts_50 = vocab.time_shift_token_id(50)
        ts_1000 = vocab.time_shift_token_id(1000)
        ts_200 = vocab.time_shift_token_id(200)
        event_id = vocab.event_token_ids[0]
        logits[event_id] = 4.0

        penalty = _time_shift_length_penalty_tensors(
            vocab,
            alpha=0.5,
            device=torch.device("cpu"),
        )
        adjusted = _apply_time_shift_length_penalty(logits, time_shift_penalty=penalty)

        self.assertAlmostEqual(float(adjusted[event_id].item()), 4.0)
        self.assertAlmostEqual(float(adjusted[ts_50].item()), -0.5)
        self.assertAlmostEqual(float(adjusted[ts_200].item()), -0.5)
        self.assertAlmostEqual(float(adjusted[ts_1000].item()), -0.5)

    def test_mapper_v2_logits_fn_incremental_decode_appends_only_new_prefix_token(self) -> None:
        vocab = MapperTupleVocab()
        model = FakeIncrementalMapperModel(vocab.size)
        ln_carry_in = empty_ln_carry_state(0)
        ln_carry_out = empty_ln_carry_state(8000)
        control_batch = {
            "density_teacher_8s": torch.zeros((1, 400, 1), dtype=torch.float32),
            "projected_control_memory_8s": torch.zeros((1, 400, 16), dtype=torch.float32),
            "control_attention_kv_cache": ((torch.zeros((1, 1, 400, 16)), torch.zeros((1, 1, 400, 16))),),
        }
        logits_fn = _mapper_v2_logits_fn(
            model=model,
            vocab=vocab,
            device=torch.device("cpu"),
            normalized_difficulty=0.0,
            audio_batch={},
            control_batch=control_batch,
            ln_carry_in=ln_carry_in,
            ln_carry_out=ln_carry_out,
            is_full_chart_start=True,
            is_full_chart_end=False,
            use_incremental_decode=True,
            time_shift_length_penalty_alpha=0.0,
        )

        logits_fn(
            MapperGenerationStep(
                decoder_input_tokens=torch.tensor([vocab.bos_id], dtype=torch.long),
                generated_tokens=(),
                state=ln_carry_in,
                valid_token_mask=torch.ones(vocab.size, dtype=torch.bool),
                token_index=0,
                write_start_ms=0,
                write_end_ms=8000,
                ln_carry_in=ln_carry_in,
                ln_carry_out=ln_carry_out,
            ),
        )
        logits = logits_fn(
            MapperGenerationStep(
                decoder_input_tokens=torch.tensor(
                    [vocab.bos_id, vocab.time_shift_token_id(10)],
                    dtype=torch.long,
                ),
                generated_tokens=(vocab.time_shift_token_id(10),),
                state=empty_ln_carry_state(10),
                valid_token_mask=torch.ones(vocab.size, dtype=torch.bool),
                token_index=1,
                write_start_ms=0,
                write_end_ms=8000,
                ln_carry_in=ln_carry_in,
                ln_carry_out=ln_carry_out,
            ),
        )

        self.assertEqual(model.calls, [(0, vocab.bos_id), (1, vocab.time_shift_token_id(10))])
        self.assertTrue(
            all(cache is control_batch["control_attention_kv_cache"] for cache in model.control_attention_kv_cache_args)
        )
        self.assertEqual(int(torch.argmax(logits).item()), vocab.time_shift_token_id(10))


class WsEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_path_requires_ready(self) -> None:
        endpoint = InferenceEndpoint(
            config=WsEndpointConfig(token_send_interval_s=0.0),
            backend=FakeInferenceBackend(),
        )

        with self.assertRaisesRegex(ProtocolError, "send ready"):
            await endpoint.handle_message(
                {"type": "audio", "session_id": "s1", "audio_path": "/tmp/song.wav"},
                FakePeer(),
            )

    async def test_audio_path_backend_failure_is_inference_error_and_cleans_session(self) -> None:
        backend = FailingPrepareBackend(
            RuntimeError("Expected one of cpu device type at start of device string: auto"),
        )
        endpoint = InferenceEndpoint(
            config=WsEndpointConfig(token_send_interval_s=0.0),
            backend=backend,
        )
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready"}, peer)
        with self.assertRaises(InferenceError) as caught:
            await endpoint.handle_message(
                {
                    "type": "audio",
                    "session_id": "s1",
                    "audio_path": "/tmp/song.wav",
                    "audio_length_ms": 2_000,
                },
                peer,
            )

        error = caught.exception
        self.assertEqual(error.session_id, "s1")
        self.assertEqual(error.phase, "prepare_audio")
        self.assertEqual(error.route, "mapper")
        self.assertEqual(error.code, "invalid_device")
        self.assertEqual(error.to_event()["error_kind"], "inference")
        self.assertNotIn("s1", endpoint.sessions)
        self.assertEqual(backend.reset_sessions, ["s1"])

    async def test_default_backend_router_ready_does_not_load_route_backends(self) -> None:
        endpoint = InferenceEndpoint(config=WsEndpointConfig(token_send_interval_s=0.0))
        backend = endpoint.backend
        assert isinstance(backend, RoutedInferenceBackend)

        await endpoint.handle_message({"type": "ready"}, FakePeer())

        self.assertTrue(backend.models_ready)
        self.assertFalse(backend.mapper_backend.models_ready)
        self.assertFalse(backend.timing_mock_backend.models_ready)

    async def test_routed_backend_lazy_startup_serializes_concurrent_first_prepare_same_route(self) -> None:
        mapper_backend = FakeInferenceBackend(startup_delay_s=0.01)
        timing_mock_backend = FakeInferenceBackend()
        backend = RoutedInferenceBackend(
            WsEndpointConfig(token_send_interval_s=0.0),
            mapper_backend=mapper_backend,
            timing_mock_backend=timing_mock_backend,
        )
        await backend.startup()

        await asyncio.gather(
            *(
                backend.prepare_audio(
                    session_id=f"s{index}",
                    audio_path=Path(f"/tmp/song-{index}.wav"),
                    audio_length_ms=2_000,
                    difficulty=4.0,
                    route="mapper",
                )
                for index in range(8)
            ),
        )

        self.assertEqual(mapper_backend.startup_calls, 1)
        self.assertEqual(len(mapper_backend.prepared_audio), 8)
        self.assertEqual(timing_mock_backend.startup_calls, 0)

    async def test_routed_backend_reset_waits_for_in_flight_prepare_same_session(self) -> None:
        route_backend = BlockingPrepareBackend()
        backend = RoutedInferenceBackend(
            WsEndpointConfig(token_send_interval_s=0.0),
            mapper_backend=route_backend,
            timing_mock_backend=FakeInferenceBackend(),
        )
        await backend.startup()

        prepare_task = asyncio.create_task(
            backend.prepare_audio(
                session_id="s1",
                audio_path=Path("/tmp/song.wav"),
                audio_length_ms=2_000,
                difficulty=4.0,
                route="mapper",
            ),
        )
        await route_backend.prepare_started.wait()
        reset_task = asyncio.create_task(backend.reset_session("s1"))
        await asyncio.sleep(0)
        self.assertFalse(reset_task.done())

        route_backend.release_prepare.set()
        await prepare_task
        await reset_task

        self.assertFalse(backend.registry.has_session("s1"))
        self.assertEqual(route_backend.reset_sessions, ["s1"])

    async def test_routed_backend_reset_waits_for_lazy_startup_before_prepare_same_session(self) -> None:
        route_backend = BlockingStartupBackend()
        backend = RoutedInferenceBackend(
            WsEndpointConfig(token_send_interval_s=0.0),
            mapper_backend=route_backend,
            timing_mock_backend=FakeInferenceBackend(),
        )
        await backend.startup()

        prepare_task = asyncio.create_task(
            backend.prepare_audio(
                session_id="s1",
                audio_path=Path("/tmp/song.wav"),
                audio_length_ms=2_000,
                difficulty=4.0,
                route="mapper",
            ),
        )
        await route_backend.startup_started.wait()
        reset_task = asyncio.create_task(backend.reset_session("s1"))
        await asyncio.sleep(0)
        self.assertFalse(reset_task.done())

        route_backend.release_startup.set()
        await prepare_task
        await reset_task

        self.assertFalse(backend.registry.has_session("s1"))
        self.assertFalse(backend.registry.has_session_lock("s1"))
        self.assertEqual(route_backend.reset_sessions, ["s1"])

    async def test_reference_time_starts_hitobject_token_stream(self) -> None:
        config = WsEndpointConfig(token_send_interval_s=0.0)
        backend = FakeInferenceBackend(
            tokens=(
                HitObjectToken(10, "EVENT_A", 1_240, ("tap", "none", "none", "none")),
                HitObjectToken(11, "EVENT_B", 1_500, ("none", "tap", "none", "none")),
            ),
        )
        endpoint = InferenceEndpoint(config=config, backend=backend)
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready"}, peer)
        await endpoint.handle_message(
            {
                "type": "audio",
                "session_id": "s1",
                "audio_path": "/Users/ken/audio/song1.wav",
                "audio_length_ms": 180_000,
                "difficulty": 5.0,
            },
            peer,
        )
        await endpoint.handle_message(
            {
                "type": "reference_time",
                "session_id": "s1",
                "ref_time_ms": 1_234,
                "local_host_time_send_ms": current_host_time_ms(),
            },
            peer,
        )
        task = endpoint.sessions["s1"].stream_task
        assert task is not None
        await task

        token_messages = [message for message in peer.messages if message["type"] == "hit_object_token"]
        self.assertEqual(
            [message["type"] for message in peer.messages],
            ["hit_object_token", "hit_object_token", "end_of_stream"],
        )
        self.assertTrue(all(message["session_id"] == "s1" for message in peer.messages))
        self.assertTrue(
            all(set(message) == {"type", "session_id", "token_id", "ms_in_ref_audio"} for message in token_messages),
        )
        self.assertEqual(
            [(message["token_id"], message["ms_in_ref_audio"]) for message in token_messages],
            [(10, 1_240), (11, 1_500)],
        )
        self.assertEqual(
            peer.messages[-1],
            {
                "type": "end_of_stream",
                "session_id": "s1",
                "audio_length_ms": 180_000,
                "complete_through_ms": 180_000,
            },
        )
        self.assertEqual(
            backend.prepared_audio,
            [
                {
                    "session_id": "s1",
                    "audio_path": Path("/Users/ken/audio/song1.wav"),
                    "audio_length_ms": 180_000,
                    "difficulty": 5.0,
                },
            ],
        )
        self.assertEqual(len(backend.iter_calls), 1)
        self.assertEqual(backend.iter_calls[0]["audio_path"], Path("/Users/ken/audio/song1.wav"))
        self.assertEqual(backend.iter_calls[0]["audio_length_ms"], 180_000)
        self.assertIsInstance(backend.iter_calls[0]["window"], DecoderWindow)
        await endpoint.stop_session("s1")

    async def test_handle_command_starts_hitobject_token_stream(self) -> None:
        backend = FakeInferenceBackend(
            tokens=(
                HitObjectToken(10, "EVENT_A", 1_240, ("tap", "none", "none", "none")),
            ),
        )
        endpoint = InferenceEndpoint(
            config=WsEndpointConfig(token_send_interval_s=0.0, decoder_lead_ms=0),
            backend=backend,
        )
        peer = FakePeer()

        await endpoint.handle_command(ReadyCommand(), peer)
        await endpoint.handle_command(
            AudioCommand(
                session_id="s1",
                audio_path="/tmp/song.wav",
                audio_length_ms=2_000,
                difficulty=5.0,
            ),
            peer,
        )
        await endpoint.handle_command(
            ReferenceTimeCommand(
                session_id="s1",
                ref_time_ms=0,
                local_host_time_send_ms=current_host_time_ms(),
            ),
            peer,
        )
        task = endpoint.sessions["s1"].stream_task
        assert task is not None
        await task

        self.assertEqual([message["type"] for message in peer.messages], ["hit_object_token", "end_of_stream"])
        self.assertEqual(peer.messages[0]["token_id"], 10)
        self.assertEqual(peer.messages[-1]["complete_through_ms"], 2_000)
        self.assertEqual(backend.prepared_audio[0]["difficulty"], 5.0)
        await endpoint.stop_session("s1")

    async def test_reference_time_rejects_negative_ref_time_without_starting_tasks(self) -> None:
        endpoint = InferenceEndpoint(
            config=WsEndpointConfig(token_send_interval_s=0.0, decoder_lead_ms=0),
            backend=FakeInferenceBackend(),
        )
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready"}, peer)
        await endpoint.handle_message(
            {
                "type": "audio",
                "session_id": "s1",
                "audio_path": "/tmp/song.wav",
                "audio_length_ms": 2_000,
            },
            peer,
        )

        with self.assertRaisesRegex(ProtocolError, "ref_time_ms must be non-negative"):
            await endpoint.handle_message(
                {
                    "type": "reference_time",
                    "session_id": "s1",
                    "ref_time_ms": -1,
                    "local_host_time_send_ms": current_host_time_ms(),
                },
                peer,
            )
        with self.assertRaisesRegex(ProtocolError, "ref_time_ms must be non-negative"):
            await endpoint.handle_command(
                ReferenceTimeCommand(
                    session_id="s1",
                    ref_time_ms=-1,
                    local_host_time_send_ms=current_host_time_ms(),
                ),
                peer,
            )

        session = endpoint.sessions["s1"]
        self.assertIsNone(session.reference_clock)
        self.assertIsNone(session.decoder_window)
        self.assertIsNone(session.stream_task)
        self.assertIsNone(session.wall_clock_reset_task)
        await endpoint.stop_session("s1")

    async def test_handle_command_validates_stop_session_id_and_reason(self) -> None:
        endpoint = InferenceEndpoint(
            config=WsEndpointConfig(token_send_interval_s=0.0),
            backend=FakeInferenceBackend(),
        )
        peer = FakePeer()

        with self.assertRaisesRegex(ProtocolError, "session_id"):
            await endpoint.handle_command(StopCommand(session_id=""), peer)
        with self.assertRaisesRegex(ProtocolError, "stop reason"):
            await endpoint.handle_command(StopCommand(session_id="s1", reason=""), peer)

    async def test_reference_time_transition_requires_audio_ready_or_streaming_state(self) -> None:
        endpoint = InferenceEndpoint(
            config=WsEndpointConfig(token_send_interval_s=0.0),
            backend=FakeInferenceBackend(),
        )
        peer = FakePeer()

        with self.assertRaisesRegex(ProtocolError, "reference_time.*no_session.*audio_ready, streaming"):
            await endpoint.handle_command(
                ReferenceTimeCommand(
                    session_id="s1",
                    ref_time_ms=0,
                    local_host_time_send_ms=current_host_time_ms(),
                ),
                peer,
            )

        self.assertNotIn("s1", endpoint.sessions)

    async def test_replace_audio_transition_rejects_audio_preparing_state(self) -> None:
        endpoint = InferenceEndpoint(
            config=WsEndpointConfig(token_send_interval_s=0.0),
            backend=FakeInferenceBackend(),
        )
        peer = FakePeer()

        await endpoint.handle_command(ReadyCommand(), peer)
        endpoint.sessions["s1"] = SessionState(
            session_id="s1",
            audio_path=Path("/tmp/in-flight.wav"),
            audio_length_ms=2_000,
        )

        with self.assertRaisesRegex(ProtocolError, "replace_audio.*audio_preparing.*audio_ready, streaming"):
            await endpoint.handle_command(
                AudioCommand(
                    session_id="s1",
                    audio_path="/tmp/replacement.wav",
                    audio_length_ms=2_000,
                ),
                peer,
            )

        self.assertEqual(endpoint.sessions["s1"].audio_path, Path("/tmp/in-flight.wav"))
        self.assertEqual(endpoint.backend.reset_sessions, [])

    async def test_endpoint_streams_timing_mock_route_through_reference_time(self) -> None:
        def timing_fit_fn(audio_path, **kwargs):
            del audio_path, kwargs
            return _timing_report()

        mapper_backend = FakeInferenceBackend()
        timing_mock_backend = TimingMockStreamBackend(
            WsEndpointConfig(decoder_window_ms=1_000, token_send_interval_s=0.0),
            timing_fit_fn=timing_fit_fn,
        )
        backend = RoutedInferenceBackend(
            WsEndpointConfig(decoder_window_ms=1_000, token_send_interval_s=0.0, decoder_lead_ms=0),
            mapper_backend=mapper_backend,
            timing_mock_backend=timing_mock_backend,
        )
        endpoint = InferenceEndpoint(
            config=WsEndpointConfig(decoder_window_ms=1_000, token_send_interval_s=0.0, decoder_lead_ms=0),
            backend=backend,
        )
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready"}, peer)
        await endpoint.handle_message(
            {
                "type": "audio",
                "session_id": "s1",
                "audio_path": "/tmp/song.mp3",
                "audio_length_ms": 1_700,
                "route": "timing_mock",
            },
            peer,
        )
        await endpoint.handle_message(
            {
                "type": "reference_time",
                "session_id": "s1",
                "ref_time_ms": 1,
                "local_host_time_send_ms": current_host_time_ms(),
            },
            peer,
        )
        task = endpoint.sessions["s1"].stream_task
        assert task is not None
        await task

        self.assertFalse(mapper_backend.models_ready)
        token_messages = [message for message in peer.messages if message["type"] == "hit_object_token"]
        self.assertEqual([message["type"] for message in peer.messages], ["hit_object_token"] * 4 + ["end_of_stream"])
        self.assertTrue(
            all(set(message) == {"type", "session_id", "token_id", "ms_in_ref_audio"} for message in token_messages),
        )
        self.assertEqual([message["ms_in_ref_audio"] for message in token_messages], [120, 620, 1_120, 1_620])
        self.assertEqual(peer.messages[-1]["audio_length_ms"], 1_700)
        await endpoint.stop_session("s1")

    async def test_endpoint_clamps_forwarded_tokens_at_audio_end_before_eos(self) -> None:
        backend = FakeInferenceBackend(
            tokens=(
                HitObjectToken(10, "EVENT_A", 900, ("tap", "none", "none", "none")),
                HitObjectToken(11, "EVENT_B", 1_000, ("none", "tap", "none", "none")),
                HitObjectToken(12, "EVENT_C", 1_500, ("none", "none", "tap", "none")),
            ),
        )
        endpoint = InferenceEndpoint(config=WsEndpointConfig(token_send_interval_s=0.0), backend=backend)
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready"}, peer)
        await endpoint.handle_message(
            {
                "type": "audio",
                "session_id": "s1",
                "audio_path": "/tmp/song.wav",
                "audio_length_ms": 1_000,
            },
            peer,
        )
        await endpoint.handle_message(
            {
                "type": "reference_time",
                "session_id": "s1",
                "ref_time_ms": 0,
                "local_host_time_send_ms": current_host_time_ms(),
            },
            peer,
        )
        task = endpoint.sessions["s1"].stream_task
        assert task is not None
        await task

        self.assertEqual([message["type"] for message in peer.messages], ["hit_object_token", "end_of_stream"])
        self.assertEqual(peer.messages[0]["token_id"], 10)
        self.assertEqual(peer.messages[0]["ms_in_ref_audio"], 900)
        self.assertEqual(peer.messages[1]["complete_through_ms"], 1_000)
        await endpoint.stop_session("s1")

    async def test_stream_token_socket_disconnect_finishes_task_quietly(self) -> None:
        config = WsEndpointConfig(token_send_interval_s=0.0)
        backend = FakeInferenceBackend()
        endpoint = InferenceEndpoint(config=config, backend=backend)

        await endpoint.handle_message({"type": "ready"}, FakePeer())
        await endpoint.handle_message(
            {
                "type": "audio",
                "session_id": "s1",
                "audio_path": "/Users/ken/audio/song1.wav",
                "audio_length_ms": 180_000,
            },
            FakePeer(),
        )
        await endpoint.handle_message(
            {
                "type": "reference_time",
                "session_id": "s1",
                "ref_time_ms": 1_234,
                "local_host_time_send_ms": current_host_time_ms(),
            },
            DisconnectingPeer(),
        )
        for _ in range(20):
            if "s1" not in endpoint.sessions:
                break
            await asyncio.sleep(0.01)

        self.assertNotIn("s1", endpoint.sessions)
        self.assertEqual(backend.reset_sessions, ["s1"])

    async def test_raw_tcp_disconnect_before_websocket_handshake_is_ignored(self) -> None:
        endpoint = InferenceEndpoint(
            config=WsEndpointConfig(token_send_interval_s=0.0),
            backend=FakeInferenceBackend(),
        )
        errors: list[BaseException] = []
        handler_tasks: list[asyncio.Task] = []

        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = asyncio.current_task()
            assert task is not None
            handler_tasks.append(task)
            try:
                await _handle_websocket_client(endpoint, reader, writer)
            except BaseException as exc:
                errors.append(exc)

        server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
        try:
            assert server.sockets is not None
            host, port = server.sockets[0].getsockname()[:2]
            _reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            for _ in range(20):
                if handler_tasks:
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(handler_tasks)
            await asyncio.wait_for(asyncio.gather(*handler_tasks), timeout=1.0)
        finally:
            server.close()
            await server.wait_closed()

        self.assertEqual(errors, [])

    async def test_websocket_disconnect_after_audio_path_resets_owned_session(self) -> None:
        backend = FakeInferenceBackend()
        endpoint = InferenceEndpoint(
            config=WsEndpointConfig(token_send_interval_s=0.0),
            backend=backend,
        )
        errors: list[BaseException] = []
        handler_tasks: list[asyncio.Task] = []

        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = asyncio.current_task()
            assert task is not None
            handler_tasks.append(task)
            try:
                await _handle_websocket_client(endpoint, reader, writer)
            except BaseException as exc:
                errors.append(exc)

        server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
        try:
            assert server.sockets is not None
            host, port = server.sockets[0].getsockname()[:2]
            reader, writer = await asyncio.open_connection(host, port)
            writer.write(
                b"GET / HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Upgrade: websocket\r\n"
                b"Connection: Upgrade\r\n"
                b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                b"Sec-WebSocket-Version: 13\r\n"
                b"\r\n"
            )
            await writer.drain()
            self.assertIn(b"101 Switching Protocols", await reader.readuntil(b"\r\n\r\n"))

            ready = envelope_pb2.Envelope()
            ready.ready.SetInParent()
            audio = envelope_pb2.Envelope(session_id="s1")
            audio.audio.audio.local_path = "/tmp/song.wav"
            audio.audio.audio.audio_length_ms = 2_000
            audio.audio.route = inference_pb2.INFERENCE_ROUTE_MAPPER
            writer.write(_client_binary_frame(ready))
            writer.write(_client_binary_frame(audio))
            await writer.drain()
            for _ in range(20):
                if backend.prepared_audio:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(len(backend.prepared_audio), 1)

            writer.close()
            await writer.wait_closed()
            for _ in range(20):
                if handler_tasks and all(task.done() for task in handler_tasks):
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(handler_tasks)
            await asyncio.wait_for(asyncio.gather(*handler_tasks), timeout=1.0)
        finally:
            server.close()
            await server.wait_closed()

        self.assertEqual(errors, [])
        self.assertNotIn("s1", endpoint.sessions)
        self.assertEqual(backend.reset_sessions, ["s1"])

    async def test_websocket_inference_error_returns_structured_error_without_handler_failure(self) -> None:
        backend = FailingPrepareBackend(RuntimeError("invalid device: auto"))
        endpoint = InferenceEndpoint(
            config=WsEndpointConfig(token_send_interval_s=0.0),
            backend=backend,
        )
        errors: list[BaseException] = []
        handler_tasks: list[asyncio.Task] = []

        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = asyncio.current_task()
            assert task is not None
            handler_tasks.append(task)
            try:
                await _handle_websocket_client(endpoint, reader, writer)
            except BaseException as exc:
                errors.append(exc)

        server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
        try:
            assert server.sockets is not None
            host, port = server.sockets[0].getsockname()[:2]
            reader, writer = await asyncio.open_connection(host, port)
            writer.write(
                b"GET / HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Upgrade: websocket\r\n"
                b"Connection: Upgrade\r\n"
                b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                b"Sec-WebSocket-Version: 13\r\n"
                b"\r\n"
            )
            await writer.drain()
            self.assertIn(b"101 Switching Protocols", await reader.readuntil(b"\r\n\r\n"))

            ready = envelope_pb2.Envelope()
            ready.ready.SetInParent()
            audio = envelope_pb2.Envelope(session_id="s1")
            audio.audio.audio.local_path = "/tmp/song.wav"
            audio.audio.audio.audio_length_ms = 2_000
            audio.audio.route = inference_pb2.INFERENCE_ROUTE_MAPPER
            writer.write(_client_binary_frame(ready))
            writer.write(_client_binary_frame(audio))
            await writer.drain()

            envelope = await _read_server_envelope_frame(reader)
            self.assertEqual(envelope.WhichOneof("payload"), "error")
            self.assertEqual(envelope.session_id, "s1")
            self.assertEqual(envelope.error.error_kind, "inference")
            self.assertEqual(envelope.error.phase, "prepare_audio")
            self.assertEqual(envelope.error.route, inference_pb2.INFERENCE_ROUTE_MAPPER)
            self.assertEqual(envelope.error.code, "invalid_device")

            writer.close()
            await writer.wait_closed()
            await asyncio.wait_for(asyncio.gather(*handler_tasks), timeout=1.0)
        finally:
            server.close()
            await server.wait_closed()

        self.assertEqual(errors, [])
        self.assertNotIn("s1", endpoint.sessions)
        self.assertEqual(backend.reset_sessions, ["s1"])

    async def test_websocket_text_json_frame_returns_protobuf_protocol_error(self) -> None:
        endpoint = InferenceEndpoint(
            config=WsEndpointConfig(token_send_interval_s=0.0),
            backend=FakeInferenceBackend(),
        )
        errors: list[BaseException] = []
        handler_tasks: list[asyncio.Task] = []

        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = asyncio.current_task()
            assert task is not None
            handler_tasks.append(task)
            try:
                await _handle_websocket_client(endpoint, reader, writer)
            except BaseException as exc:
                errors.append(exc)

        server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
        try:
            assert server.sockets is not None
            host, port = server.sockets[0].getsockname()[:2]
            reader, writer = await asyncio.open_connection(host, port)
            writer.write(
                b"GET / HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Upgrade: websocket\r\n"
                b"Connection: Upgrade\r\n"
                b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                b"Sec-WebSocket-Version: 13\r\n"
                b"\r\n"
            )
            await writer.drain()
            self.assertIn(b"101 Switching Protocols", await reader.readuntil(b"\r\n\r\n"))

            writer.write(_client_text_frame(b'{"control":"ready"}'))
            await writer.drain()

            envelope = await _read_server_envelope_frame(reader)
            self.assertEqual(envelope.WhichOneof("payload"), "error")
            self.assertEqual(envelope.error.code, "protocol_error")
            self.assertIn("binary protobuf", envelope.error.message)

            writer.close()
            await writer.wait_closed()
            await asyncio.wait_for(asyncio.gather(*handler_tasks), timeout=1.0)
        finally:
            server.close()
            await server.wait_closed()

        self.assertEqual(errors, [])

    async def test_websocket_binary_protobuf_round_trip_streams_envelopes(self) -> None:
        backend = FakeInferenceBackend(
            tokens=(
                HitObjectToken(10, "EVENT_A", 1_240, ("tap", "none", "none", "none")),
            ),
        )
        endpoint = InferenceEndpoint(
            config=WsEndpointConfig(token_send_interval_s=0.0, decoder_lead_ms=0),
            backend=backend,
        )
        errors: list[BaseException] = []
        handler_tasks: list[asyncio.Task] = []

        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = asyncio.current_task()
            assert task is not None
            handler_tasks.append(task)
            try:
                await _handle_websocket_client(endpoint, reader, writer)
            except BaseException as exc:
                errors.append(exc)

        server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
        try:
            assert server.sockets is not None
            host, port = server.sockets[0].getsockname()[:2]
            reader, writer = await asyncio.open_connection(host, port)
            writer.write(
                b"GET / HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Upgrade: websocket\r\n"
                b"Connection: Upgrade\r\n"
                b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                b"Sec-WebSocket-Version: 13\r\n"
                b"\r\n"
            )
            await writer.drain()
            self.assertIn(b"101 Switching Protocols", await reader.readuntil(b"\r\n\r\n"))

            ready = envelope_pb2.Envelope()
            ready.ready.SetInParent()
            audio = envelope_pb2.Envelope(session_id="s1")
            audio.audio.audio.local_path = "/tmp/song.wav"
            audio.audio.audio.audio_length_ms = 2_000
            audio.audio.difficulty = 5.0
            audio.audio.route = inference_pb2.INFERENCE_ROUTE_MAPPER
            reference = envelope_pb2.Envelope(session_id="s1")
            reference.reference_time.ref_time_ms = 0
            reference.reference_time.local_host_time_send_ms = int(current_host_time_ms())

            writer.write(_client_binary_frame(ready))
            writer.write(_client_binary_frame(audio))
            writer.write(_client_binary_frame(reference))
            await writer.drain()

            begin = await _read_server_envelope_frame(reader)
            token = await _read_server_envelope_frame(reader)
            eos = await _read_server_envelope_frame(reader)

            self.assertEqual(begin.session_id, "s1")
            self.assertEqual(begin.WhichOneof("payload"), "mapper_stream_begin")
            self.assertEqual(begin.mapper_stream_begin.token_contract_version, MAPPER_TOKEN_CONTRACT_VERSION)
            self.assertEqual(token.WhichOneof("payload"), "hit_object_token")
            self.assertEqual(token.hit_object_token.token_id, 10)
            self.assertEqual(token.hit_object_token.ms_in_ref_audio, 1_240)
            self.assertEqual(token.hit_object_token.token_index, 0)
            self.assertEqual(eos.WhichOneof("payload"), "end_of_stream")
            self.assertEqual(eos.end_of_stream.audio_length_ms, 2_000)
            self.assertEqual(eos.end_of_stream.complete_through_ms, 2_000)
            self.assertEqual([begin.sequence, token.sequence, eos.sequence], [1, 2, 3])

            writer.close()
            await writer.wait_closed()
            await asyncio.wait_for(asyncio.gather(*handler_tasks), timeout=1.0)
        finally:
            server.close()
            await server.wait_closed()

        self.assertEqual(errors, [])
        self.assertEqual(backend.prepared_audio[0]["audio_path"], Path("/tmp/song.wav"))
        self.assertEqual(backend.prepared_audio[0]["difficulty"], 5.0)

    async def test_websocket_stop_allows_same_session_reuse_with_fresh_stream_begin(self) -> None:
        backend = FakeInferenceBackend(
            tokens=(
                HitObjectToken(10, "EVENT_A", 1_240, ("tap", "none", "none", "none")),
            ),
            block_after_first=True,
        )
        endpoint = InferenceEndpoint(
            config=WsEndpointConfig(token_send_interval_s=0.0, decoder_lead_ms=0),
            backend=backend,
        )
        errors: list[BaseException] = []
        handler_tasks: list[asyncio.Task] = []

        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            task = asyncio.current_task()
            assert task is not None
            handler_tasks.append(task)
            try:
                await _handle_websocket_client(endpoint, reader, writer)
            except BaseException as exc:
                errors.append(exc)

        server = await asyncio.start_server(handle_client, host="127.0.0.1", port=0)
        try:
            assert server.sockets is not None
            host, port = server.sockets[0].getsockname()[:2]
            reader, writer = await asyncio.open_connection(host, port)
            writer.write(
                b"GET / HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Upgrade: websocket\r\n"
                b"Connection: Upgrade\r\n"
                b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
                b"Sec-WebSocket-Version: 13\r\n"
                b"\r\n"
            )
            await writer.drain()
            self.assertIn(b"101 Switching Protocols", await reader.readuntil(b"\r\n\r\n"))

            ready = envelope_pb2.Envelope()
            ready.ready.SetInParent()
            audio = envelope_pb2.Envelope(session_id="s1")
            audio.audio.audio.local_path = "/tmp/song-a.wav"
            audio.audio.audio.audio_length_ms = 2_000
            audio.audio.route = inference_pb2.INFERENCE_ROUTE_MAPPER
            reference = envelope_pb2.Envelope(session_id="s1")
            reference.reference_time.ref_time_ms = 0
            reference.reference_time.local_host_time_send_ms = int(current_host_time_ms())

            writer.write(_client_binary_frame(ready))
            writer.write(_client_binary_frame(audio))
            writer.write(_client_binary_frame(reference))
            await writer.drain()

            first_begin = await _read_server_envelope_frame(reader)
            first_token = await _read_server_envelope_frame(reader)
            self.assertEqual(first_begin.WhichOneof("payload"), "mapper_stream_begin")
            self.assertEqual(first_token.WhichOneof("payload"), "hit_object_token")
            self.assertEqual(first_token.hit_object_token.token_index, 0)

            stop = envelope_pb2.Envelope(session_id="s1")
            stop.stop_session.SetInParent()
            writer.write(_client_binary_frame(stop))
            await writer.drain()
            for _ in range(20):
                if backend.reset_sessions:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(backend.reset_sessions, ["s1"])

            replacement_audio = envelope_pb2.Envelope(session_id="s1")
            replacement_audio.audio.audio.local_path = "/tmp/song-b.wav"
            replacement_audio.audio.audio.audio_length_ms = 2_000
            replacement_audio.audio.route = inference_pb2.INFERENCE_ROUTE_MAPPER
            replacement_reference = envelope_pb2.Envelope(session_id="s1")
            replacement_reference.reference_time.ref_time_ms = 0
            replacement_reference.reference_time.local_host_time_send_ms = int(current_host_time_ms())
            writer.write(_client_binary_frame(replacement_audio))
            writer.write(_client_binary_frame(replacement_reference))
            await writer.drain()

            second_begin = await _read_server_envelope_frame(reader)
            second_token = await _read_server_envelope_frame(reader)

            self.assertEqual(second_begin.WhichOneof("payload"), "mapper_stream_begin")
            self.assertEqual(second_token.WhichOneof("payload"), "hit_object_token")
            self.assertEqual(second_token.hit_object_token.token_index, 0)
            self.assertEqual(second_begin.sequence, first_token.sequence + 1)
            self.assertEqual(second_token.sequence, second_begin.sequence + 1)
            self.assertEqual([call["audio_path"] for call in backend.prepared_audio], [
                Path("/tmp/song-a.wav"),
                Path("/tmp/song-b.wav"),
            ])

            writer.close()
            await writer.wait_closed()
            await asyncio.wait_for(asyncio.gather(*handler_tasks), timeout=1.0)
        finally:
            server.close()
            await server.wait_closed()

        self.assertEqual(errors, [])

    async def test_audio_path_requires_length_or_readable_audio_file(self) -> None:
        endpoint = InferenceEndpoint(
            config=WsEndpointConfig(token_send_interval_s=0.0),
            backend=FakeInferenceBackend(),
        )
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready"}, peer)
        with self.assertRaisesRegex(ProtocolError, "audio_length_ms"):
            await endpoint.handle_message(
                {"type": "audio", "session_id": "s1", "audio_path": "/tmp/nonexistent-song.wav"},
                peer,
            )

    async def test_audio_path_file_duration_allows_reference_time_without_message_length(self) -> None:
        backend = FakeInferenceBackend()
        endpoint = InferenceEndpoint(config=WsEndpointConfig(token_send_interval_s=0.0), backend=backend)
        peer = FakePeer()

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "song.wav"
            _write_silent_wav(audio_path, duration_ms=1_000)

            await endpoint.handle_message({"type": "ready"}, peer)
            await endpoint.handle_message(
                {"type": "audio", "session_id": "s1", "audio_path": str(audio_path)},
                peer,
            )
            await endpoint.handle_message(
                {
                    "type": "reference_time",
                    "session_id": "s1",
                    "ref_time_ms": 0,
                    "local_host_time_send_ms": current_host_time_ms(),
                },
                peer,
            )
            task = endpoint.sessions["s1"].stream_task
            assert task is not None
            await task

            self.assertEqual(endpoint.sessions["s1"].audio_length_ms, 1_000)
            self.assertEqual(backend.prepared_audio[0]["audio_length_ms"], 1_000)
            self.assertEqual(backend.prepared_audio[0]["difficulty"], 4.0)
            self.assertEqual(backend.iter_calls[0]["audio_length_ms"], 1_000)
            await endpoint.stop_session("s1")

    async def test_audio_route_timing_mock_omits_mapper_difficulty(self) -> None:
        backend = FakeInferenceBackend()
        endpoint = InferenceEndpoint(config=WsEndpointConfig(token_send_interval_s=0.0), backend=backend)
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready"}, peer)
        await endpoint.handle_message(
            {
                "type": "audio",
                "session_id": "s1",
                "audio_path": "/tmp/song.wav",
                "audio_length_ms": 2_000,
                "route": "timing_mock",
                "difficulty": 7.0,
            },
            peer,
        )

        self.assertEqual(endpoint.sessions["s1"].route, "timing_mock")
        self.assertIsNone(endpoint.sessions["s1"].difficulty)
        self.assertEqual(backend.prepared_routes, ["timing_mock"])
        self.assertIsNone(backend.prepared_audio[0]["difficulty"])

    async def test_timing_mock_route_streams_timing_grid_tokens_without_mapper_backend(self) -> None:
        timing_calls = []

        def timing_fit_fn(audio_path, **kwargs):
            timing_calls.append((Path(audio_path), kwargs))
            return _timing_report()

        config = WsEndpointConfig(
            decoder_window_ms=800,
            decoder_lead_ms=0,
            token_send_interval_s=0.0,
        )
        mapper_backend = FakeInferenceBackend()
        backend = RoutedInferenceBackend(
            config,
            mapper_backend=mapper_backend,
            timing_mock_backend=TimingMockStreamBackend(config, timing_fit_fn=timing_fit_fn),
        )
        endpoint = InferenceEndpoint(config=config, backend=backend)
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready"}, peer)
        self.assertFalse(mapper_backend.models_ready)
        await endpoint.handle_message(
            {
                "type": "audio",
                "session_id": "s1",
                "audio_path": "/tmp/song.wav",
                "audio_length_ms": 1_300,
                "route": "timing_mock",
            },
            peer,
        )
        await endpoint.handle_message(
            {
                "type": "reference_time",
                "session_id": "s1",
                "ref_time_ms": 0,
                "local_host_time_send_ms": current_host_time_ms() + 1_000.0,
            },
            peer,
        )
        task = endpoint.sessions["s1"].stream_task
        assert task is not None
        await task

        vocab = MapperTupleVocab()
        token_messages = [message for message in peer.messages if message["type"] == "hit_object_token"]
        self.assertEqual(
            [(message["token_id"], message["ms_in_ref_audio"]) for message in token_messages],
            [
                (vocab.encode_event(("NONE", "NONE", "TAP", "TAP")), 120),
                (vocab.encode_event(("TAP", "TAP", "NONE", "NONE")), 620),
                (vocab.encode_event(("NONE", "NONE", "TAP", "TAP")), 1120),
            ],
        )
        self.assertEqual(peer.messages[-1]["type"], "end_of_stream")
        self.assertEqual(peer.messages[-1]["audio_length_ms"], 1_300)
        self.assertEqual(timing_calls[0][0], Path("/tmp/song.wav"))
        self.assertEqual(mapper_backend.prepared_routes, [])
        await endpoint.stop_session("s1")

    async def test_reference_time_difficulty_does_not_override_audio_path_difficulty(self) -> None:
        backend = FakeInferenceBackend()
        endpoint = InferenceEndpoint(config=WsEndpointConfig(token_send_interval_s=0.0), backend=backend)
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready"}, peer)
        await endpoint.handle_message(
            {
                "type": "audio",
                "session_id": "s1",
                "audio_path": "/tmp/song.wav",
                "audio_length_ms": 2_000,
                "difficulty": 5.0,
            },
            peer,
        )
        await endpoint.handle_message(
            {
                "type": "reference_time",
                "session_id": "s1",
                "ref_time_ms": 0,
                "local_host_time_send_ms": current_host_time_ms(),
                "difficulty": 3.0,
            },
            peer,
        )
        task = endpoint.sessions["s1"].stream_task
        assert task is not None
        await task

        self.assertEqual(endpoint.sessions["s1"].difficulty, 5.0)
        self.assertEqual(backend.prepared_audio[0]["difficulty"], 5.0)
        await endpoint.stop_session("s1")

    async def test_audio_path_rejects_cross_owner_session_replace(self) -> None:
        backend = FakeInferenceBackend()
        endpoint = InferenceEndpoint(config=WsEndpointConfig(token_send_interval_s=0.0), backend=backend)
        peer = FakePeer()
        owner_a = object()
        owner_b = object()

        await endpoint.handle_message({"type": "ready"}, peer, owner=owner_a)
        await endpoint.handle_message(
            {
                "type": "audio",
                "session_id": "s1",
                "audio_path": "/tmp/song-a.wav",
                "audio_length_ms": 1_000,
            },
            peer,
            owner=owner_a,
        )

        with self.assertRaisesRegex(ProtocolError, "owned"):
            await endpoint.handle_message(
                {
                    "type": "audio",
                    "session_id": "s1",
                    "audio_path": "/tmp/song-b.wav",
                    "audio_length_ms": 2_000,
                },
                peer,
                owner=owner_b,
            )

        self.assertEqual(endpoint.sessions["s1"].owner, owner_a)
        self.assertEqual(endpoint.sessions["s1"].audio_path, Path("/tmp/song-a.wav"))
        self.assertEqual(endpoint.sessions["s1"].audio_length_ms, 1_000)
        self.assertEqual(len(backend.prepared_audio), 1)
        self.assertEqual(backend.reset_sessions, [])

    async def test_reference_time_rejects_audio_length_change_after_prepare(self) -> None:
        backend = FakeInferenceBackend()
        endpoint = InferenceEndpoint(config=WsEndpointConfig(token_send_interval_s=0.0), backend=backend)
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready"}, peer)
        await endpoint.handle_message(
            {
                "type": "audio",
                "session_id": "s1",
                "audio_path": "/tmp/song.wav",
                "audio_length_ms": 1_000,
            },
            peer,
        )

        with self.assertRaisesRegex(ProtocolError, "audio_length_ms"):
            await endpoint.handle_message(
                {
                    "type": "reference_time",
                    "session_id": "s1",
                    "ref_time_ms": 0,
                    "local_host_time_send_ms": current_host_time_ms(),
                    "audio_length_ms": 2_000,
                },
                peer,
            )
        self.assertIsNone(endpoint.sessions["s1"].stream_task)

    async def test_stop_waits_for_in_flight_prepare_and_resets_routed_session(self) -> None:
        route_backend = BlockingPrepareBackend()
        config = WsEndpointConfig(token_send_interval_s=0.0)
        backend = RoutedInferenceBackend(
            config,
            mapper_backend=route_backend,
            timing_mock_backend=FakeInferenceBackend(),
        )
        endpoint = InferenceEndpoint(config=config, backend=backend)
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready"}, peer)
        prepare_task = asyncio.create_task(
            endpoint.handle_message(
                {
                    "type": "audio",
                    "session_id": "s1",
                    "audio_path": "/tmp/song.wav",
                    "audio_length_ms": 1_000,
                },
                peer,
            ),
        )
        await route_backend.prepare_started.wait()
        stop_task = asyncio.create_task(endpoint.stop_session("s1"))
        await asyncio.sleep(0)
        self.assertFalse(stop_task.done())

        route_backend.release_prepare.set()
        await prepare_task
        await stop_task

        self.assertNotIn("s1", endpoint.sessions)
        self.assertFalse(backend.registry.has_session("s1"))
        self.assertNotIn("s1", endpoint._session_locks)
        self.assertFalse(backend.registry.has_session_lock("s1"))
        self.assertEqual(route_backend.reset_sessions, ["s1"])

    async def test_session_locks_are_released_after_stop(self) -> None:
        route_backend = FakeInferenceBackend()
        config = WsEndpointConfig(token_send_interval_s=0.0)
        backend = RoutedInferenceBackend(
            config,
            mapper_backend=route_backend,
            timing_mock_backend=FakeInferenceBackend(),
        )
        endpoint = InferenceEndpoint(config=config, backend=backend)
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready"}, peer)
        await endpoint.handle_message(
            {
                "type": "audio",
                "session_id": "s1",
                "audio_path": "/tmp/song.wav",
                "audio_length_ms": 1_000,
            },
            peer,
        )
        self.assertIn("s1", endpoint._session_locks)
        self.assertTrue(backend.registry.has_session_lock("s1"))

        await endpoint.stop_session("s1")

        self.assertNotIn("s1", endpoint.sessions)
        self.assertFalse(backend.registry.has_session("s1"))
        self.assertNotIn("s1", endpoint._session_locks)
        self.assertFalse(backend.registry.has_session_lock("s1"))

    async def test_mapper_v2_backend_prepares_session_runtime_from_ws_audio(self) -> None:
        loader_configs = []
        created = []
        fake_session = FakeSessionRuntime()

        def runtime_loader(config):
            loader_configs.append(config)
            return SimpleNamespace(device="cpu", vocab=MapperTupleVocab(), mapper_model=None)

        def session_factory(session_id, model_runtime, config):
            created.append((session_id, model_runtime, config))
            return fake_session

        config = WsEndpointConfig(
            mapper_checkpoint_path="mapper.pt",
            control_checkpoint_path="control.pt",
            beatthis_checkpoint="custom-beatthis",
            device="cpu",
            token_send_interval_s=0.0,
            canonicalization=TIMING_CANONICALIZATION_BPM_80_160,
            timing_mode="v3_shadow",
            timing_max_supported_audio_duration_seconds=480.0,
        )
        backend = StreamWithCache(
            config,
            runtime_loader=runtime_loader,
            session_runtime_factory=session_factory,
        )

        await backend.startup()
        await backend.prepare_audio(
            session_id="s1",
            audio_path=Path("/tmp/song.wav"),
            audio_length_ms=1_234,
            difficulty=5.0,
        )

        self.assertTrue(backend.models_ready)
        self.assertEqual(loader_configs[0].mapper_checkpoint_path, Path.cwd() / "mapper.pt")
        self.assertEqual(loader_configs[0].control_checkpoint_path, Path.cwd() / "control.pt")
        self.assertEqual(loader_configs[0].beatthis_checkpoint, "custom-beatthis")
        self.assertEqual(created[0][0], "s1")
        self.assertAlmostEqual(created[0][2].default_normalized_difficulty, normalize_difficulty(5.0))
        self.assertEqual(created[0][2].grid_fitter_config.canonicalization, TIMING_CANONICALIZATION_BPM_80_160)
        self.assertFalse(created[0][2].grid_fitter_config.canonicalize_tempo_aliases)
        self.assertEqual(created[0][2].timing_mode, "v3_shadow")
        self.assertEqual(created[0][2].timing_max_supported_audio_duration_seconds, 480.0)
        self.assertEqual(fake_session.prepare_audio_calls, [(Path("/tmp/song.wav"), 1_234, 0)])

    async def test_mapper_v2_backend_streams_selected_window_through_music_end(self) -> None:
        backend = StreamingBackend(
            WsEndpointConfig(
                decoder_window_ms=8_000,
                token_send_interval_s=0.0,
            ),
        )
        session_runtime = FakeSessionRuntime()
        backend._session_runtimes["s1"] = session_runtime

        tokens = []
        async for token in backend.iter_hitobject_tokens(
            session_id="s1",
            audio_path=Path("/tmp/song.wav"),
            audio_length_ms=18_500,
            window=DecoderWindow(start_ms=8_000, end_ms=16_000),
        ):
            tokens.append(token)

        self.assertEqual(
            backend.generated_windows,
            [
                DecoderWindow(start_ms=8_000, end_ms=16_000),
                DecoderWindow(start_ms=16_000, end_ms=24_000),
            ],
        )
        self.assertEqual([token.ms_in_ref_audio for token in tokens], [8_000, 16_000])

    async def test_timing_mock_backend_streams_mock_tokens_from_selected_reference_window(self) -> None:
        fit_calls = []

        def timing_fit_fn(audio_path, **kwargs):
            fit_calls.append((Path(audio_path), kwargs))
            return _timing_report()

        backend = TimingMockStreamBackend(
            WsEndpointConfig(
                decoder_window_ms=1_000,
                token_send_interval_s=0.0,
                beatthis_checkpoint="custom-beatthis",
                beatthis_device="cpu",
                canonicalization=TIMING_CANONICALIZATION_BPM_80_160,
            ),
            timing_fit_fn=timing_fit_fn,
        )

        await backend.startup()
        await backend.prepare_audio(
            session_id="s1",
            audio_path=Path("/tmp/song.mp3"),
            audio_length_ms=1_700,
            difficulty=None,
            route="timing_mock",
        )

        tokens = []
        async for token in backend.iter_hitobject_tokens(
            session_id="s1",
            audio_path=Path("/tmp/song.mp3"),
            audio_length_ms=1_700,
            window=DecoderWindow(start_ms=1_000, end_ms=2_000),
        ):
            tokens.append(token)

        self.assertEqual([token.ms_in_ref_audio for token in tokens], [1_120, 1_620])
        self.assertEqual(
            [token.actions for token in tokens],
            [
                ("NONE", "NONE", "TAP", "TAP"),
                ("TAP", "TAP", "NONE", "NONE"),
            ],
        )
        self.assertEqual(fit_calls[0][0], Path("/tmp/song.mp3"))
        self.assertEqual(fit_calls[0][1]["checkpoint_path"], "custom-beatthis")
        self.assertEqual(fit_calls[0][1]["device"], "cpu")
        self.assertEqual(
            fit_calls[0][1]["fitter_config"].canonicalization,
            TIMING_CANONICALIZATION_BPM_80_160,
        )

    async def test_timing_mock_backend_does_not_round_near_end_reference_back_to_boundary(self) -> None:
        def timing_fit_fn(audio_path, **kwargs):
            del audio_path, kwargs
            return _timing_report()

        backend = TimingMockStreamBackend(
            WsEndpointConfig(decoder_window_ms=1_000, token_send_interval_s=0.0),
            timing_fit_fn=timing_fit_fn,
        )

        await backend.startup()
        await backend.prepare_audio(
            session_id="s1",
            audio_path=Path("/tmp/song.mp3"),
            audio_length_ms=1_700,
            difficulty=None,
            route="timing_mock",
        )

        tokens = []
        async for token in backend.iter_hitobject_tokens(
            session_id="s1",
            audio_path=Path("/tmp/song.mp3"),
            audio_length_ms=1_700,
            window=DecoderWindow(start_ms=1_201, end_ms=2_201),
        ):
            tokens.append(token)

        self.assertEqual([token.ms_in_ref_audio for token in tokens], [1_620])

    def test_hitobject_token_manifest_matches_full_mapper_event_vocab(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        mapping = manifest["event_token_id_to_lane_action"]
        vocab = MapperTupleVocab()

        expected = {
            str(token_id): [action.value for action in vocab.decode_event(token_id)]
            for token_id in vocab.event_token_ids
        }

        self.assertEqual(mapping, expected)
        self.assertEqual(manifest["event_token_count"], len(vocab.event_token_ids))
        self.assertEqual(manifest["event_token_id_range"], [min(vocab.event_token_ids), max(vocab.event_token_ids)])

    async def test_stop_cancels_stream_and_resets_session(self) -> None:
        config = WsEndpointConfig(token_send_interval_s=1.0)
        backend = FakeInferenceBackend(block_after_first=True)
        endpoint = InferenceEndpoint(config=config, backend=backend)
        peer = FakePeer()

        await endpoint.handle_message({"type": "ready"}, peer)
        await endpoint.handle_message(
            {
                "type": "audio",
                "session_id": "s1",
                "audio_path": str(Path("/tmp/song.wav")),
                "audio_length_ms": 180_000,
            },
            peer,
        )
        await endpoint.handle_message(
            {
                "type": "reference_time",
                "session_id": "s1",
                "ref_time_ms": 0,
                "local_host_time_send_ms": current_host_time_ms(),
            },
            peer,
        )
        await asyncio.sleep(0)
        await endpoint.handle_message({"type": "stop", "session_id": "s1"}, peer)

        self.assertNotIn("s1", endpoint.sessions)
        self.assertEqual(backend.reset_sessions, ["s1"])

    async def test_wall_clock_resets_session_after_audio_end_grace(self) -> None:
        config = WsEndpointConfig(
            token_send_interval_s=0.0,
            reset_after_audio_end_ms=20,
            wall_clock_check_interval_s=0.01,
        )
        endpoint = InferenceEndpoint(config=config, backend=FakeInferenceBackend())
        peer = FakePeer()
        now_ms = current_host_time_ms()

        await endpoint.handle_message({"type": "ready"}, peer)
        await endpoint.handle_message(
            {
                "type": "audio",
                "session_id": "s1",
                "audio_path": "/tmp/song.wav",
                "audio_length_ms": 1,
            },
            peer,
        )
        await endpoint.handle_message(
            {
                "type": "reference_time",
                "session_id": "s1",
                "ref_time_ms": 1,
                "local_host_time_send_ms": now_ms,
            },
            peer,
        )

        await asyncio.sleep(0.2)

        self.assertNotIn("s1", endpoint.sessions)


class FakeInferenceBackend:
    def __init__(
        self,
        *,
        tokens: tuple[HitObjectToken, ...] | None = None,
        block_after_first: bool = False,
        startup_delay_s: float = 0.0,
    ) -> None:
        self.models_ready = False
        self.tokens = (
            HitObjectToken(10, "EVENT_A", 0, ("tap", "none", "none", "none")),
        ) if tokens is None else tokens
        self.block_after_first = bool(block_after_first)
        self.startup_delay_s = float(startup_delay_s)
        self.startup_calls = 0
        self.prepared_audio: list[dict[str, object]] = []
        self.prepared_routes: list[str] = []
        self.iter_calls: list[dict[str, object]] = []
        self.reset_sessions: list[str] = []

    async def startup(self, *, route: str = "mapper") -> None:
        del route
        self.startup_calls += 1
        if self.startup_delay_s:
            await asyncio.sleep(self.startup_delay_s)
        self.models_ready = True

    async def prepare_audio(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        difficulty: float | None,
        route: str = "mapper",
    ) -> None:
        self.prepared_routes.append(route)
        self.prepared_audio.append(
            {
                "session_id": session_id,
                "audio_path": audio_path,
                "audio_length_ms": audio_length_ms,
                "difficulty": difficulty,
            },
        )
        await asyncio.sleep(0)

    async def iter_hitobject_tokens(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        window: DecoderWindow,
    ) -> AsyncIterator[HitObjectToken]:
        self.iter_calls.append(
            {
                "session_id": session_id,
                "audio_path": audio_path,
                "audio_length_ms": audio_length_ms,
                "window": window,
            },
        )
        for index, token in enumerate(self.tokens):
            yield token
            if index == 0 and self.block_after_first:
                await asyncio.sleep(10)

    async def reset_session(self, session_id: str) -> None:
        self.reset_sessions.append(session_id)


class FailingPrepareBackend(FakeInferenceBackend):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self.exc = exc

    async def prepare_audio(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        difficulty: float | None,
        route: str = "mapper",
    ) -> None:
        del session_id, audio_path, audio_length_ms, difficulty
        self.prepared_routes.append(route)
        raise self.exc


class BlockingPrepareBackend(FakeInferenceBackend):
    def __init__(self) -> None:
        super().__init__()
        self.prepare_started = asyncio.Event()
        self.release_prepare = asyncio.Event()

    async def prepare_audio(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        difficulty: float | None,
        route: str = "mapper",
    ) -> None:
        self.prepare_started.set()
        await self.release_prepare.wait()
        await super().prepare_audio(
            session_id=session_id,
            audio_path=audio_path,
            audio_length_ms=audio_length_ms,
            difficulty=difficulty,
            route=route,
        )


class BlockingStartupBackend(FakeInferenceBackend):
    def __init__(self) -> None:
        super().__init__()
        self.startup_started = asyncio.Event()
        self.release_startup = asyncio.Event()

    async def startup(self, *, route: str = "mapper") -> None:
        del route
        self.startup_calls += 1
        self.startup_started.set()
        await self.release_startup.wait()
        self.models_ready = True


class StreamingBackend(StreamWithCache):
    def __init__(self, config: WsEndpointConfig) -> None:
        super().__init__(config)
        self.vocab = MapperTupleVocab()
        self.generated_windows: list[DecoderWindow] = []

    def _vocab(self) -> MapperTupleVocab:
        return self.vocab

    def _generate_window(
        self,
        session_id: str,
        session_runtime: object,
        window: DecoderWindow,
        audio_length_ms: int,
    ) -> MapperGeneratedWindow:
        del session_id, session_runtime, audio_length_ms
        self.generated_windows.append(window)
        event_token = int(self.vocab.event_token_ids[0])
        state_before = empty_ln_carry_state(int(window.start_ms))
        return MapperGeneratedWindow(
            write_start_ms=int(window.start_ms),
            write_end_ms=int(window.end_ms),
            ln_carry_in=state_before,
            ln_carry_out=empty_ln_carry_state(int(window.end_ms)),
            tokens=[event_token],
            states_before=[state_before],
            states_after=[state_before],
            terminal_state=empty_ln_carry_state(int(window.end_ms)),
            completed=True,
            dead_end=False,
            max_tokens_exceeded=False,
        )


class FakeSessionRuntime:
    def __init__(self) -> None:
        self.prepare_audio_calls: list[tuple[Path, int, int]] = []

    def prepare_audio(
        self,
        audio_path: str | Path,
        *,
        audio_length_ms: int,
        start_ms: int = 0,
    ) -> SimpleNamespace:
        self.prepare_audio_calls.append((Path(audio_path), audio_length_ms, start_ms))
        return SimpleNamespace(audio_length_ms=audio_length_ms)

    def reset_audio_cache(self) -> None:
        pass


def _write_silent_wav(path: Path, *, duration_ms: int, sample_rate: int = 8_000) -> None:
    frame_count = int(round(sample_rate * duration_ms / 1000.0))
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * frame_count)


def _client_binary_frame(envelope: envelope_pb2.Envelope) -> bytes:
    data = envelope.SerializeToString()
    return _client_frame(data, opcode=0x2)


def _client_text_frame(payload: bytes) -> bytes:
    return _client_frame(payload, opcode=0x1)


def _client_frame(data: bytes, *, opcode: int) -> bytes:
    length = len(data)
    if length < 126:
        prefix = bytes([0x80 | opcode, length])
    elif length <= 0xFFFF:
        prefix = bytes([0x80 | opcode, 126]) + length.to_bytes(2, "big")
    else:
        prefix = bytes([0x80 | opcode, 127]) + length.to_bytes(8, "big")
    return prefix + data


async def _read_server_envelope_frame(reader: asyncio.StreamReader) -> envelope_pb2.Envelope:
    header = await reader.readexactly(2)
    opcode = header[0] & 0x0F
    length = header[1] & 0x7F
    if length == 126:
        length = int.from_bytes(await reader.readexactly(2), "big")
    elif length == 127:
        length = int.from_bytes(await reader.readexactly(8), "big")
    payload = await reader.readexactly(length)
    envelope = envelope_pb2.Envelope()
    envelope.ParseFromString(payload)
    if opcode != 0x2:
        raise AssertionError(f"expected binary protobuf frame, got opcode {opcode}")
    return envelope


if __name__ == "__main__":
    unittest.main()
