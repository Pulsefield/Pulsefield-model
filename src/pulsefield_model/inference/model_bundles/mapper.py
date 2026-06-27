from __future__ import annotations

from pulsefield_model.inference.mapper_protocol import resolve_mapper_profile
from pulsefield_model.inference.model_bundles.base import RouteBackend
from pulsefield_model.inference.model_bundles.mapper_base import StreamWithCacheMapperBundle
from pulsefield_model.inference.model_bundles.mapper_v2_1_sparse import (
    MAPPER_V2_1_SPARSE_MODEL_ID,
    MapperV21SparseBundle,
)
from pulsefield_model.inference.model_bundles.mapper_v2_tuple import (
    MAPPER_V2_TUPLE_MODEL_ID,
    MapperV2TupleBundle,
)
from pulsefield_model.inference.stream_with_cache import StreamWithCacheConfig


DEFAULT_MAPPER_MODEL_ID = "mapper/default"


def mapper_bundle_for_config(
    config: StreamWithCacheConfig,
    *,
    model_id: str = DEFAULT_MAPPER_MODEL_ID,
    backend: RouteBackend | None = None,
) -> StreamWithCacheMapperBundle:
    profile = resolve_mapper_profile(config.mapper_profile)
    if profile.name == "v2_tuple":
        return MapperV2TupleBundle(config, model_id=model_id, backend=backend)
    if profile.name == "v2_1_sparse":
        return MapperV21SparseBundle(config, model_id=model_id, backend=backend)
    raise ValueError(f"unsupported mapper profile: {profile.name!r}")


class MapperModelBundle:
    """Compatibility factory for the concrete mapper stream-with-cache bundles."""

    def __new__(
        cls,
        config: StreamWithCacheConfig,
        *,
        model_id: str = DEFAULT_MAPPER_MODEL_ID,
        backend: RouteBackend | None = None,
    ) -> StreamWithCacheMapperBundle:
        return mapper_bundle_for_config(config, model_id=model_id, backend=backend)


__all__ = [
    "DEFAULT_MAPPER_MODEL_ID",
    "MAPPER_V2_1_SPARSE_MODEL_ID",
    "MAPPER_V2_TUPLE_MODEL_ID",
    "MapperModelBundle",
    "MapperV21SparseBundle",
    "MapperV2TupleBundle",
    "StreamWithCacheMapperBundle",
    "mapper_bundle_for_config",
]
