from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from importlib import resources
from pathlib import Path

import pytest

from pulsefield_model.inference.config import (
    InferenceServiceConfig,
    default_inference_service_config,
    inference_service_config_from_mapping,
    project_to_ws_endpoint_config,
)
from pulsefield_model.inference.defaults import (
    DEFAULT_CONTROL_CHECKPOINT_PATH,
    DEFAULT_HOST,
    DEFAULT_MAPPER_CHECKPOINT_PATH,
    DEFAULT_PORT,
    DEFAULT_RUNTIME_DEVICE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_default_inference_service_projection_matches_current_endpoint_defaults() -> None:
    config = default_inference_service_config()

    endpoint_config = project_to_ws_endpoint_config(config)

    assert endpoint_config.host == DEFAULT_HOST
    assert endpoint_config.port == DEFAULT_PORT
    assert endpoint_config.mapper_profile == "v2_1_sparse"
    assert endpoint_config.mapper_model_id == "mapper/default"
    assert endpoint_config.timing_mock_model_id == "timing_mock/default"
    assert endpoint_config.mapper_checkpoint_path == DEFAULT_MAPPER_CHECKPOINT_PATH
    assert endpoint_config.control_checkpoint_path == DEFAULT_CONTROL_CHECKPOINT_PATH
    assert endpoint_config.beatthis_checkpoint == "final0"
    assert endpoint_config.device == DEFAULT_RUNTIME_DEVICE


def test_inference_identity_and_timing_checkpoint_overrides_reach_endpoint_config() -> None:
    config = _compose(
        [
            "mapper.model_id=mapper/custom",
            "timing_mock.model_id=timing_mock/custom",
            "timing_mock.timing_checkpoint_path=custom-checkpoint",
        ],
    )

    endpoint_config = project_to_ws_endpoint_config(config)

    assert endpoint_config.mapper_model_id == "mapper/custom"
    assert endpoint_config.timing_mock_model_id == "timing_mock/custom"
    assert endpoint_config.beatthis_checkpoint == "custom-checkpoint"


def test_mapping_adapter_rejects_unknown_keys_without_hydra_dependency() -> None:
    with pytest.raises(ValueError, match="unknown InferenceMapperConfig key"):
        inference_service_config_from_mapping({"mapper": {"unexpected": True}})


def test_default_hydra_composition_exposes_reviewable_contract() -> None:
    config = _compose()

    assert config.mapper.model_id == "mapper/default"
    assert config.mapper.bundle_model_id == "mapper/v2_1_sparse"
    assert config.mapper.profile == "v2_1_sparse"
    assert config.mapper.checkpoint_path == DEFAULT_MAPPER_CHECKPOINT_PATH.as_posix()
    assert config.mapper.control_checkpoint_path == DEFAULT_CONTROL_CHECKPOINT_PATH.as_posix()
    assert config.mapper.vocab_contract == "sparse_lane_actions"
    assert config.mapper.grammar_contract == "sparse_lane_action_grammar"
    assert config.protocol.mapper_capability_name == "mapper.tuple_tokens"
    assert config.protocol.mapper_token_contract_version == 2
    assert Path(config.protocol.mapper_manifest_path).name == "hitobject_token_manifest_v2.json"


def test_hydra_configs_are_package_resources() -> None:
    package_root = resources.files("pulsefield_model")

    assert package_root.joinpath("configs/inference/service.yaml").is_file()
    assert package_root.joinpath("configs/hydra/mapper_training.yaml").is_file()


def test_hydra_mapper_groups_select_supported_profiles() -> None:
    v2_config = _compose(["mapper=v2_tuple"])
    v21_config = _compose(["mapper=v2_1_sparse"])

    assert v2_config.mapper.profile == "v2_tuple"
    assert v2_config.mapper.bundle_model_id == "mapper/v2_tuple"
    assert v2_config.mapper.checkpoint_version == "v2"
    assert v2_config.mapper.vocab_contract == "tuple_event_tokens"
    assert v2_config.mapper.grammar_contract == "tuple_event_grammar"

    assert v21_config.mapper.profile == "v2_1_sparse"
    assert v21_config.mapper.bundle_model_id == "mapper/v2_1_sparse"
    assert v21_config.mapper.checkpoint_version == "v2_1"
    assert v21_config.mapper.vocab_contract == "sparse_lane_actions"
    assert v21_config.mapper.grammar_contract == "sparse_lane_action_grammar"


@pytest.mark.parametrize("override", ("mapper.unexpected=true", "+mapper.unexpected=true"))
def test_hydra_composition_rejects_unknown_nested_keys(override: str) -> None:
    from pulsefield_model.inference.hydra_entry import compose_inference_service_config

    with pytest.raises(Exception, match="unexpected|Could not override"):
        compose_inference_service_config([override])


def test_hydra_composition_rejects_invalid_canonicalization() -> None:
    from pulsefield_model.inference.hydra_entry import compose_inference_service_config

    with pytest.raises(ValueError, match="canonicalization must be one of"):
        compose_inference_service_config(["runtime.canonicalization=bogus"])


def test_hydra_composition_rejects_disabled_timing_mock_route() -> None:
    from pulsefield_model.inference.hydra_entry import compose_inference_service_config

    with pytest.raises(ValueError, match="timing_mock.enabled must be true"):
        compose_inference_service_config(["timing_mock.enabled=false"])


@pytest.mark.parametrize(
    "override",
    (
        "mapper.model_id=''",
        "timing_mock.model_id=''",
        "timing_mock.timing_checkpoint_path=''",
    ),
)
def test_hydra_composition_rejects_blank_runtime_identity_fields(override: str) -> None:
    from pulsefield_model.inference.hydra_entry import compose_inference_service_config

    with pytest.raises(ValueError, match="must be a non-empty string"):
        compose_inference_service_config([override])


def test_hydra_composition_rejects_auto_mapper_profile() -> None:
    from pulsefield_model.inference.hydra_entry import compose_inference_service_config

    with pytest.raises(ValueError, match="mapper.profile must be explicit"):
        compose_inference_service_config(["mapper.profile=auto"])


@pytest.mark.parametrize(
    ("override", "match"),
    (
        ("server.port=-1", "server.port"),
        ("server.port=65536", "server.port"),
        ("server.token_send_interval_s=0", "server.token_send_interval_s"),
        ("server.decoder_lead_ms=-1", "server.decoder_lead_ms"),
        ("server.timing_mock_decoder_lead_ms=-1", "server.timing_mock_decoder_lead_ms"),
        ("server.reset_after_audio_end_ms=-1", "server.reset_after_audio_end_ms"),
        ("server.wall_clock_check_interval_s=0", "server.wall_clock_check_interval_s"),
        ("mapper.decoder_window_ms=0", "mapper.decoder_window_ms"),
        ("mapper.max_tokens=0", "mapper.max_tokens"),
        ("mapper.temperature=nan", "mapper.temperature"),
        ("mapper.temperature=-0.1", "mapper.temperature"),
        ("mapper.top_p=0", "mapper.top_p"),
        ("mapper.top_p=2.0", "mapper.top_p"),
        ("mapper.time_shift_length_penalty_alpha=nan", "mapper.time_shift_length_penalty_alpha"),
        ("mapper.time_shift_length_penalty_alpha=-0.1", "mapper.time_shift_length_penalty_alpha"),
        ("runtime.default_difficulty=nan", "runtime.default_difficulty"),
        ("runtime.default_difficulty=7.0", "runtime.default_difficulty"),
        ("runtime.max_control_batch_size=0", "runtime.max_control_batch_size"),
    ),
)
def test_hydra_composition_rejects_invalid_numeric_boundaries(override: str, match: str) -> None:
    from pulsefield_model.inference.hydra_entry import compose_inference_service_config

    with pytest.raises(ValueError, match=match):
        compose_inference_service_config([override])


def test_hydra_help_stays_import_light_without_torch(tmp_path: Path) -> None:
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        """
import importlib.abc
import sys

BLOCKED_PREFIXES = (
    "torch",
    "pulsefield_model.inference.model_bundles",
    "pulsefield_model.inference.model_runtime",
    "pulsefield_model.inference.session_runtime",
    "pulsefield_model.inference.stream_with_cache",
    "pulsefield_model.inference.ws_endpoint",
    "pulsefield_model.inference.ws_server",
)


class ImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in BLOCKED_PREFIXES):
            raise ModuleNotFoundError(f"blocked import for Hydra help probe: {fullname}")
        return None


sys.meta_path.insert(0, ImportBlocker())
""".lstrip(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    source_path = PROJECT_ROOT / "src"
    env["PYTHONPATH"] = os.pathsep.join(
        str(path)
        for path in (tmp_path, source_path, env.get("PYTHONPATH"))
        if path
    )

    result = subprocess.run(
        [sys.executable, "-m", "pulsefield_model.inference.hydra_entry", "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Config" in result.stdout


def test_ws_server_help_stays_import_light_without_runtime_endpoint(tmp_path: Path) -> None:
    sitecustomize = tmp_path / "sitecustomize.py"
    sitecustomize.write_text(
        """
import importlib.abc
import sys

BLOCKED_PREFIXES = (
    "torch",
    "pulsefield_model.inference.model_bundles",
    "pulsefield_model.inference.model_runtime",
    "pulsefield_model.inference.session_runtime",
    "pulsefield_model.inference.stream_with_cache",
    "pulsefield_model.inference.ws_endpoint",
)


class ImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in BLOCKED_PREFIXES):
            raise ModuleNotFoundError(f"blocked import for ws_server help probe: {fullname}")
        return None


sys.meta_path.insert(0, ImportBlocker())
""".lstrip(),
        encoding="utf-8",
    )
    env = os.environ.copy()
    source_path = PROJECT_ROOT / "src"
    env["PYTHONPATH"] = os.pathsep.join(
        str(path)
        for path in (tmp_path, source_path, env.get("PYTHONPATH"))
        if path
    )

    result = subprocess.run(
        [sys.executable, "-m", "pulsefield_model.inference.ws_server", "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Config" in result.stdout


def test_hydra_imports_stay_out_of_runtime_modules() -> None:
    restricted_paths = [
        PROJECT_ROOT / "src/pulsefield_model/inference/mapper_protocol.py",
        PROJECT_ROOT / "src/pulsefield_model/inference/protocol_adapter.py",
        PROJECT_ROOT / "src/pulsefield_model/inference/protobuf_transport.py",
        PROJECT_ROOT / "src/pulsefield_model/inference/session_runtime.py",
        PROJECT_ROOT / "src/pulsefield_model/inference/ws_endpoint.py",
        PROJECT_ROOT / "src/pulsefield_model/inference/mapper_v2_tuple_rollout.py",
        PROJECT_ROOT / "src/pulsefield_model/inference/mapper_v2_1_rollout.py",
        *sorted((PROJECT_ROOT / "src/pulsefield_model/inference/model_bundles").glob("*.py")),
    ]

    for path in restricted_paths:
        text = path.read_text()
        assert "import hydra" not in text, path
        assert "from hydra" not in text, path
        assert "omegaconf" not in text.lower(), path


def _compose(overrides: Sequence[str] = ()) -> InferenceServiceConfig:
    from pulsefield_model.inference.hydra_entry import compose_inference_service_config

    return compose_inference_service_config(overrides)
