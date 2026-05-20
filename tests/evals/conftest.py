from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture(scope="session")
def mapper_v21_decoder_eval_output_dir(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Path:
    raw_output_dir = request.config.getoption("--mapper-v21-decoder-eval-output-dir")
    if raw_output_dir:
        return Path(raw_output_dir).expanduser()
    return tmp_path_factory.mktemp("mapper_v21_decoder_evals")


@pytest.fixture(scope="session")
def mapper_v21_decoder_eval_options(request: pytest.FixtureRequest) -> dict[str, Any]:
    return {
        "device": request.config.getoption("--mapper-v21-decoder-eval-device"),
        "repeat": int(request.config.getoption("--mapper-v21-decoder-eval-repeat")),
        "warmup": int(request.config.getoption("--mapper-v21-decoder-eval-warmup")),
        "prefix_lengths": _parse_int_csv(request.config.getoption("--mapper-v21-decoder-eval-prefix-lengths")),
        "prefix_sweep_apply_grammar_mask": not bool(
            request.config.getoption("--mapper-v21-decoder-eval-skip-internal-grammar-mask"),
        ),
        "use_profiler": not bool(request.config.getoption("--mapper-v21-decoder-eval-no-profiler")),
        "rollout_ms": int(request.config.getoption("--mapper-v21-decoder-eval-rollout-ms")),
        "rollout_max_tokens_per_window": int(
            request.config.getoption("--mapper-v21-decoder-eval-rollout-max-tokens-per-window"),
        ),
        "policy_alphas": _parse_float_csv(request.config.getoption("--mapper-v21-decoder-policy-alphas")),
        "policy_delta_alphas": _parse_float_csv(
            request.config.getoption("--mapper-v21-decoder-policy-delta-alphas"),
        ),
        "policy_temperatures": _parse_float_csv(
            request.config.getoption("--mapper-v21-decoder-policy-temperatures"),
        ),
        "policy_top_ps": _parse_optional_float_csv(request.config.getoption("--mapper-v21-decoder-policy-top-ps")),
        "policy_seeds": _parse_optional_int_csv(request.config.getoption("--mapper-v21-decoder-policy-seeds")),
        "policy_candidate_indices": _parse_optional_int_csv(
            request.config.getoption("--mapper-v21-decoder-policy-candidate-indices"),
        ),
        "render_reamber": bool(request.config.getoption("--mapper-v21-decoder-eval-render-reamber")),
    }


@pytest.fixture(scope="session")
def mapper_v21_decoder_model_state(request: pytest.FixtureRequest) -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("mapper v2.1 decoder evals require torch")
    import torch

    from pulsefield_model.evals.mapper_v21_decoder_profiler import tiny_mapper_v21_config
    from pulsefield_model.models.mapper.v2_1 import MapperV21Config, MapperV21Model, MapperV21Vocab

    torch.manual_seed(20260520)
    vocab = MapperV21Vocab()
    checkpoint_path = request.config.getoption("--mapper-v21-decoder-checkpoint")
    checkpoint = None
    if checkpoint_path:
        checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=True)
    raw_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    config = MapperV21Config(**raw_config) if isinstance(raw_config, dict) else tiny_mapper_v21_config()
    model = MapperV21Model(config, vocab=vocab)
    if checkpoint is not None:
        model.load_state_dict(_checkpoint_state_dict(checkpoint), strict=True)
    model.eval()
    return {
        "config": config,
        "vocab": vocab,
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
    }


@pytest.fixture()
def mapper_v21_decoder_model(mapper_v21_decoder_model_state: dict[str, Any]) -> Any:
    from pulsefield_model.models.mapper.v2_1 import MapperV21Model

    model = MapperV21Model(
        mapper_v21_decoder_model_state["config"],
        vocab=mapper_v21_decoder_model_state["vocab"],
    )
    state = {key: value.detach().clone() for key, value in mapper_v21_decoder_model_state["state_dict"].items()}
    model.load_state_dict(state)
    model.eval()
    return model


def _parse_int_csv(value: str) -> tuple[int, ...]:
    lengths = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not lengths:
        raise pytest.UsageError("--mapper-v21-decoder-eval-prefix-lengths must include at least one integer")
    if any(length <= 0 for length in lengths):
        raise pytest.UsageError("--mapper-v21-decoder-eval-prefix-lengths values must be positive")
    return lengths


def _parse_float_csv(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not values:
        raise pytest.UsageError("mapper v2.1 decoder float CSV option must include at least one value")
    return values


def _parse_optional_float_csv(value: str) -> tuple[float | None, ...]:
    values: list[float | None] = []
    for item in str(value).split(","):
        cleaned = item.strip().lower()
        if not cleaned:
            continue
        if cleaned in {"none", "null"}:
            values.append(None)
        else:
            values.append(float(cleaned))
    if not values:
        raise pytest.UsageError("--mapper-v21-decoder-policy-top-ps must include at least one value")
    return tuple(values)


def _parse_optional_int_csv(value: str) -> tuple[int | None, ...]:
    values: list[int | None] = []
    for item in str(value).split(","):
        cleaned = item.strip().lower()
        if not cleaned:
            continue
        if cleaned in {"none", "null"}:
            values.append(None)
        else:
            values.append(int(cleaned))
    if not values:
        raise pytest.UsageError("--mapper-v21-decoder-policy-seeds must include at least one value")
    return tuple(values)


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return _filter_control_encoder_state(candidate)
        return _filter_control_encoder_state(checkpoint)
    raise TypeError(f"unsupported checkpoint payload type: {type(checkpoint).__name__}")


def _filter_control_encoder_state(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in state_dict.items()
        if not str(key).startswith("control_encoder.")
    }
