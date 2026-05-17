"""Inference runtimes and export paths."""

from importlib import import_module
from typing import Any


_EXPORTS = {
    "DecoderWindow": "stream_with_cache",
    "HitObjectToken": "stream_with_cache",
    "StreamWithCache": "stream_with_cache",
    "StreamWithCacheConfig": "stream_with_cache",
    "audio_length_ms_from_file": "stream_with_cache",
    "decoder_windows_until_audio_end": "stream_with_cache",
    "run_cached_stream_sample": "stream_with_cache",
    "InferenceEndpoint": "ws_endpoint",
    "WsEndpointConfig": "ws_endpoint",
    "serve_forever": "ws_server",
    "ModelRuntime": "model_runtime",
    "ModelRuntimeConfig": "model_runtime",
    "load_model_runtime": "model_runtime",
    "OsuExportMetadata": "osu_export",
    "decode_mapper_tokens_to_timepoints": "osu_export",
    "format_osu_export": "osu_export",
    "SessionRuntime": "session_runtime",
    "SessionRuntimeConfig": "session_runtime",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
