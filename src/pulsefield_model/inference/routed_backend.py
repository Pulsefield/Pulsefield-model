from __future__ import annotations

from pulsefield_model.inference.model_bundles import (
    RouteBackend,
    TimingMockModelBundle,
    TimingMockSession,
    TimingMockStreamBackend,
    mapper_bundle_for_config,
)
from pulsefield_model.inference.model_bundles.timing_mock import (
    TimingFitFn,
    _grid_fitter_config_for_canonicalization,
    _hitobject_token_from_timepoint,
    _timepoints_in_window,
)
from pulsefield_model.inference.service_models import InferenceRoute
from pulsefield_model.inference.stream_with_cache import StreamWithCacheConfig
from pulsefield_model.inference.supervisor import InferenceSupervisor


class RoutedInferenceBackend(InferenceSupervisor):
    """Compatibility router backed by the supervisor/model-bundle lifecycle."""

    def __init__(
        self,
        config: StreamWithCacheConfig,
        *,
        mapper_backend: RouteBackend | None = None,
        timing_mock_backend: RouteBackend | None = None,
    ) -> None:
        self.config = config
        self.mapper_bundle = mapper_bundle_for_config(
            config,
            model_id=config.mapper_model_id,
            backend=mapper_backend,
        )
        self.mapper_backend = self.mapper_bundle.backend
        self.timing_mock_backend = (
            TimingMockStreamBackend(config) if timing_mock_backend is None else timing_mock_backend
        )
        self.timing_mock_bundle = TimingMockModelBundle(
            config,
            model_id=config.timing_mock_model_id,
            backend=self.timing_mock_backend,
        )
        super().__init__((self.mapper_bundle, self.timing_mock_bundle))

        # Kept for existing tests and callers that inspect router internals.
        self._session_backends = self.registry._session_backends
        self._session_locks = self.registry._session_locks
        self._session_bindings = self.registry._session_bindings

    def _backend_for_route(self, route: InferenceRoute) -> RouteBackend:
        if route == "mapper":
            return self.mapper_backend
        if route == "timing_mock":
            return self.timing_mock_backend
        raise ValueError(f"unsupported inference route: {route!r}")

    async def _ensure_route_started(self, route: InferenceRoute, backend: RouteBackend) -> None:
        del backend
        await self.registry.bundle_for_route(route).mount()


__all__ = [
    "RouteBackend",
    "RoutedInferenceBackend",
    "TimingFitFn",
    "TimingMockSession",
    "TimingMockStreamBackend",
    "_grid_fitter_config_for_canonicalization",
    "_hitobject_token_from_timepoint",
    "_timepoints_in_window",
]
