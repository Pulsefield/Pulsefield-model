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
_MAPPER_BUNDLE_TYPES = (MapperV2TupleBundle, MapperV21SparseBundle)
_MAPPER_BUNDLE_TYPE_BY_PROFILE = {bundle_type.profile.name: bundle_type for bundle_type in _MAPPER_BUNDLE_TYPES}


def mapper_bundle_for_config(
    config: StreamWithCacheConfig,
    *,
    model_id: str = DEFAULT_MAPPER_MODEL_ID,
    backend: RouteBackend | None = None,
) -> StreamWithCacheMapperBundle:
    profile = resolve_mapper_profile(config.mapper_profile)
    try:
        bundle_type = _MAPPER_BUNDLE_TYPE_BY_PROFILE[profile.name]
    except KeyError as exc:
        raise ValueError(f"unsupported mapper profile: {profile.name!r}") from exc
    return bundle_type(config, model_id=model_id, backend=backend)


__all__ = [
    "DEFAULT_MAPPER_MODEL_ID",
    "MAPPER_V2_1_SPARSE_MODEL_ID",
    "MAPPER_V2_TUPLE_MODEL_ID",
    "MapperV21SparseBundle",
    "MapperV2TupleBundle",
    "StreamWithCacheMapperBundle",
    "mapper_bundle_for_config",
]
