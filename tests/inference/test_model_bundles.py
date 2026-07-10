from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

from pulsefield_model.inference.mapper_protocol import resolve_mapper_profile
from pulsefield_model.inference.model_bundles import (
    DEFAULT_MAPPER_MODEL_ID,
    DEFAULT_TIMING_MOCK_MODEL_ID,
    MapperV21SparseBundle,
    MapperV21SparseStreamWithCache,
    MapperV2TupleBundle,
    MapperV2TupleStreamWithCache,
)
from pulsefield_model.inference.routed_backend import RoutedInferenceBackend
from pulsefield_model.inference.stream_with_cache import DecoderWindow, HitObjectToken
from pulsefield_model.inference.ws_endpoint import InferenceEndpoint, WsEndpointConfig
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab
from pulsefield_model.models.mapper.v2_1 import MapperV21Vocab


class ModelBundleLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_supervisor_ready_does_not_mount_route_bundles(self) -> None:
        mapper_backend = FakeRouteBackend()
        timing_backend = FakeRouteBackend()
        backend = RoutedInferenceBackend(
            WsEndpointConfig(token_send_interval_s=0.0),
            mapper_backend=mapper_backend,
            timing_mock_backend=timing_backend,
        )

        await backend.startup()

        self.assertTrue(backend.models_ready)
        self.assertFalse(mapper_backend.models_ready)
        self.assertFalse(timing_backend.models_ready)
        self.assertEqual(
            [(status.model_id, status.state) for status in backend.bundle_status()],
            [(DEFAULT_MAPPER_MODEL_ID, "cold"), (DEFAULT_TIMING_MOCK_MODEL_ID, "cold")],
        )

    async def test_configured_route_model_ids_become_bundle_ids(self) -> None:
        backend = RoutedInferenceBackend(
            WsEndpointConfig(
                mapper_model_id="mapper/custom",
                timing_mock_model_id="timing_mock/custom",
            ),
            mapper_backend=FakeRouteBackend(),
            timing_mock_backend=FakeRouteBackend(),
        )

        self.assertEqual(
            [status.model_id for status in backend.bundle_status()],
            ["mapper/custom", "timing_mock/custom"],
        )

    async def test_prepare_audio_mounts_only_selected_bundle_and_holds_session_lease(self) -> None:
        mapper_backend = FakeRouteBackend()
        timing_backend = FakeRouteBackend()
        backend = RoutedInferenceBackend(
            WsEndpointConfig(token_send_interval_s=0.0),
            mapper_backend=mapper_backend,
            timing_mock_backend=timing_backend,
        )
        await backend.startup()

        await backend.prepare_audio(
            session_id="s1",
            audio_path=Path("/tmp/song.wav"),
            audio_length_ms=2_000,
            difficulty=4.0,
            route="mapper",
        )

        self.assertEqual(mapper_backend.startup_calls, 1)
        self.assertEqual(timing_backend.startup_calls, 0)
        self.assertEqual(backend.mapper_bundle.state, "ready")
        self.assertEqual(backend.mapper_bundle.lease_count, 1)
        self.assertIn("s1", backend._session_backends)

        await backend.reset_session("s1")

        self.assertEqual(backend.mapper_bundle.lease_count, 0)
        self.assertNotIn("s1", backend._session_backends)

    async def test_default_mapper_route_uses_v2_1_sparse_concrete_bundle(self) -> None:
        backend = RoutedInferenceBackend(WsEndpointConfig(token_send_interval_s=0.0))

        self.assertIsInstance(backend.mapper_bundle, MapperV21SparseBundle)
        self.assertIsInstance(backend.mapper_backend, MapperV21SparseStreamWithCache)
        self.assertEqual(backend.mapper_backend.config.mapper_profile, "v2_1_sparse")

    async def test_explicit_v2_mapper_route_uses_v2_tuple_concrete_bundle(self) -> None:
        backend = RoutedInferenceBackend(
            WsEndpointConfig(
                token_send_interval_s=0.0,
                mapper_profile="v2_tuple",
            ),
        )

        self.assertIsInstance(backend.mapper_bundle, MapperV2TupleBundle)
        self.assertIsInstance(backend.mapper_backend, MapperV2TupleStreamWithCache)
        self.assertEqual(backend.mapper_backend.config.mapper_profile, "v2_tuple")

    async def test_concrete_mapper_streams_pass_fixed_profile_to_runtime_loader(self) -> None:
        v2_configs = []
        v21_configs = []

        def load_v2(config):
            v2_configs.append(config)
            return SimpleNamespace(
                device="cpu",
                mapper_profile=resolve_mapper_profile("v2_tuple"),
                vocab=MapperTupleVocab(),
            )

        def load_v21(config):
            v21_configs.append(config)
            return SimpleNamespace(
                device="cpu",
                mapper_profile=resolve_mapper_profile("v2_1_sparse"),
                vocab=MapperV21Vocab(),
            )

        v2_stream = MapperV2TupleStreamWithCache(
            WsEndpointConfig(token_send_interval_s=0.0, mapper_profile="auto"),
            runtime_loader=load_v2,
        )
        v21_stream = MapperV21SparseStreamWithCache(
            WsEndpointConfig(token_send_interval_s=0.0, mapper_profile="v2_tuple"),
            runtime_loader=load_v21,
        )

        await v2_stream.startup()
        await v21_stream.startup()

        self.assertEqual(v2_configs[0].mapper_profile, "v2_tuple")
        self.assertEqual(v21_configs[0].mapper_profile, "v2_1_sparse")
        self.assertEqual(v2_stream.config.mapper_profile, "v2_tuple")
        self.assertEqual(v21_stream.config.mapper_profile, "v2_1_sparse")

    async def test_concrete_mapper_stream_context_manager_releases_runtime(self) -> None:
        loader_configs = []

        def runtime_loader(config):
            loader_configs.append(config)
            return SimpleNamespace(
                device="cpu",
                mapper_profile=resolve_mapper_profile("v2_tuple"),
                vocab=MapperTupleVocab(),
            )

        stream = MapperV2TupleStreamWithCache(
            WsEndpointConfig(token_send_interval_s=0.0),
            runtime_loader=runtime_loader,
        )

        async with stream as active_stream:
            self.assertIs(active_stream, stream)
            self.assertTrue(stream.models_ready)
            self.assertIsNotNone(stream.model_runtime)

        self.assertFalse(stream.models_ready)
        self.assertIsNone(stream.model_runtime)
        self.assertEqual(loader_configs[0].mapper_profile, "v2_tuple")

    async def test_prepare_failure_keeps_supervisor_ready_and_does_not_block_sibling_bundle(self) -> None:
        mapper_backend = FakeRouteBackend(prepare_exc=RuntimeError("mapper failed"))
        timing_backend = FakeRouteBackend()
        backend = RoutedInferenceBackend(
            WsEndpointConfig(token_send_interval_s=0.0),
            mapper_backend=mapper_backend,
            timing_mock_backend=timing_backend,
        )
        await backend.startup()

        with self.assertRaisesRegex(RuntimeError, "mapper failed"):
            await backend.prepare_audio(
                session_id="mapper-session",
                audio_path=Path("/tmp/song.wav"),
                audio_length_ms=2_000,
                difficulty=4.0,
                route="mapper",
            )

        self.assertTrue(backend.models_ready)
        self.assertEqual(backend.mapper_bundle.lease_count, 0)
        self.assertEqual(mapper_backend.reset_sessions, ["mapper-session"])

        await backend.prepare_audio(
            session_id="timing-session",
            audio_path=Path("/tmp/song.wav"),
            audio_length_ms=2_000,
            difficulty=None,
            route="timing_mock",
        )

        self.assertEqual(timing_backend.startup_calls, 1)
        self.assertEqual(backend.timing_mock_bundle.state, "ready")
        await backend.reset_session("timing-session")

    async def test_unmount_waits_for_session_lease_to_drain(self) -> None:
        mapper_backend = FakeRouteBackend()
        backend = RoutedInferenceBackend(
            WsEndpointConfig(token_send_interval_s=0.0),
            mapper_backend=mapper_backend,
            timing_mock_backend=FakeRouteBackend(),
        )
        await backend.startup()
        await backend.prepare_audio(
            session_id="s1",
            audio_path=Path("/tmp/song.wav"),
            audio_length_ms=2_000,
            difficulty=4.0,
            route="mapper",
        )

        unmount_task = asyncio.create_task(backend.unmount_model(DEFAULT_MAPPER_MODEL_ID))
        await asyncio.sleep(0)
        self.assertFalse(unmount_task.done())
        self.assertEqual(backend.mapper_bundle.state, "draining")

        await backend.reset_session("s1")
        await unmount_task

        self.assertEqual(backend.mapper_bundle.state, "cold")
        self.assertFalse(mapper_backend.models_ready)

    async def test_endpoint_exposes_model_lifecycle_controls(self) -> None:
        mapper_backend = FakeRouteBackend()
        backend = RoutedInferenceBackend(
            WsEndpointConfig(token_send_interval_s=0.0),
            mapper_backend=mapper_backend,
            timing_mock_backend=FakeRouteBackend(),
        )
        endpoint = InferenceEndpoint(config=WsEndpointConfig(token_send_interval_s=0.0), backend=backend)

        await endpoint.mount_model(DEFAULT_MAPPER_MODEL_ID)
        self.assertTrue(mapper_backend.models_ready)
        self.assertEqual(endpoint.bundle_status()[0].state, "ready")

        await endpoint.unmount_model(DEFAULT_MAPPER_MODEL_ID)
        self.assertFalse(mapper_backend.models_ready)
        self.assertEqual(endpoint.bundle_status()[0].state, "cold")

    async def test_endpoint_shutdown_stops_sessions_and_unmounts_bundles(self) -> None:
        mapper_backend = FakeRouteBackend()
        backend = RoutedInferenceBackend(
            WsEndpointConfig(token_send_interval_s=0.0),
            mapper_backend=mapper_backend,
            timing_mock_backend=FakeRouteBackend(),
        )
        endpoint = InferenceEndpoint(config=WsEndpointConfig(token_send_interval_s=0.0), backend=backend)

        await endpoint.handle_message({"type": "ready"}, FakePeer())
        await endpoint.handle_message(
            {
                "type": "audio",
                "session_id": "s1",
                "audio_path": "/tmp/song.wav",
                "audio_length_ms": 2_000,
            },
            FakePeer(),
        )

        await endpoint.shutdown()

        self.assertFalse(backend.models_ready)
        self.assertFalse(mapper_backend.models_ready)
        self.assertEqual(endpoint.sessions, {})
        self.assertEqual(backend.mapper_bundle.state, "cold")
        self.assertEqual(mapper_backend.reset_sessions, ["s1"])


class FakeRouteBackend:
    def __init__(self, *, prepare_exc: Exception | None = None) -> None:
        self.models_ready = False
        self.prepare_exc = prepare_exc
        self.startup_calls = 0
        self.prepared_audio: list[str] = []
        self.reset_sessions: list[str] = []

    async def startup(self) -> None:
        self.startup_calls += 1
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
        del audio_path, audio_length_ms, difficulty, route
        if self.prepare_exc is not None:
            raise self.prepare_exc
        self.prepared_audio.append(session_id)

    async def iter_hitobject_tokens(
        self,
        *,
        session_id: str,
        audio_path: Path,
        audio_length_ms: int,
        window: DecoderWindow,
    ) -> AsyncIterator[HitObjectToken]:
        del session_id, audio_path, audio_length_ms, window
        yield HitObjectToken(10, "EVENT_A", 0, ("tap", "none", "none", "none"))

    async def reset_session(self, session_id: str) -> None:
        self.reset_sessions.append(session_id)


class FakePeer:
    async def send_event(self, event) -> None:
        del event
