from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from pulsefield_model.inference import model_runtime as model_runtime_module
from pulsefield_model.inference.model_runtime import ModelRuntimeConfig, load_model_runtime, release_torch_cache
from pulsefield_model.models.control import ControlDemoGlobalEncoder, ControlDemoGlobalEncoderConfig
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab
from pulsefield_model.models.mapper.v2 import MapperV2Config, MapperV2Model


class ModelRuntimeTests(unittest.TestCase):
    def test_config_defaults_to_auto_and_beatthis_cpu(self) -> None:
        config = ModelRuntimeConfig(
            mapper_checkpoint_path="mapper.pt",
            control_checkpoint_path="control.pt",
        )

        self.assertEqual(config.device, "auto")
        self.assertEqual(config.beatthis_device, "cpu")
        self.assertEqual(model_runtime_module._resolve_beatthis_device(config.beatthis_device), "cpu")
        self.assertTrue(config.eager_load_beatthis)

    def test_none_beatthis_device_resolves_to_cpu_for_legacy_callers(self) -> None:
        self.assertEqual(model_runtime_module._resolve_beatthis_device(None), "cpu")

    def test_explicit_beatthis_device_is_preserved(self) -> None:
        self.assertEqual(model_runtime_module._resolve_beatthis_device("cuda"), "cuda")

    def test_loads_models_for_cpu_test_freezes_eval_and_filters_embedded_control_encoder(self) -> None:
        mapper_path = Path("mapper.pt")
        control_path = Path("control.pt")
        payloads, source_states = _runtime_payloads(mapper_path=mapper_path, control_path=control_path)

        with (
            patch.object(model_runtime_module.torch, "load", side_effect=_fake_torch_load(payloads)) as torch_load,
            patch.object(model_runtime_module, "BeatThisTimingProvider", _FakeBeatThisTimingProvider),
        ):
            _FakeBeatThisTimingProvider.instances.clear()
            runtime = load_model_runtime(
                ModelRuntimeConfig(
                    mapper_checkpoint_path=mapper_path,
                    control_checkpoint_path=control_path,
                    device="cpu",
                )
            )

        self.assertEqual(runtime.device, torch.device("cpu"))
        self.assertEqual(runtime.beatthis_provider.device, "cpu")
        self.assertTrue(runtime.beatthis_provider.eager_loaded)
        self.assertIs(runtime.mapper_model.control_encoder, None)
        self.assertFalse(runtime.mapper_model.training)
        self.assertFalse(runtime.control_model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in runtime.mapper_model.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in runtime.control_model.parameters()))
        self.assertEqual(
            runtime.checkpoint_metadata["mapper"]["filtered_control_encoder_keys"],
            ("control_encoder.embedded.weight",),
        )
        self.assertFalse(runtime.checkpoint_metadata["mapper"]["optimizer_state_loaded"])
        self.assertFalse(runtime.checkpoint_metadata["control"]["optimizer_state_loaded"])

        for key, expected in source_states["mapper"].items():
            self.assertTrue(torch.equal(runtime.mapper_model.state_dict()[key], expected), key)
        for key, expected in source_states["control"].items():
            self.assertTrue(torch.equal(runtime.control_model.state_dict()[key], expected), key)

        self.assertEqual(torch_load.call_count, 2)
        for call in torch_load.call_args_list:
            self.assertEqual(call.kwargs["map_location"], "cpu")
            self.assertIs(call.kwargs["weights_only"], True)
        self.assertEqual(len(_FakeBeatThisTimingProvider.instances), 1)

    def test_mapper_control_config_mismatch_raises(self) -> None:
        mapper_path = Path("mapper.pt")
        control_path = Path("control.pt")
        payloads, _ = _runtime_payloads(mapper_path=mapper_path, control_path=control_path)
        payloads[control_path] = _GuardedCheckpoint(
            {
                **payloads[control_path],
                "model_config": {
                    **payloads[control_path]["model_config"],
                    "global_stride": 16,
                },
            }
        )

        with (
            patch.object(model_runtime_module.torch, "load", side_effect=_fake_torch_load(payloads)),
            patch.object(model_runtime_module, "BeatThisTimingProvider", _FakeBeatThisTimingProvider),
        ):
            with self.assertRaisesRegex(ValueError, "control_model_config"):
                load_model_runtime(
                    ModelRuntimeConfig(
                        mapper_checkpoint_path=mapper_path,
                        control_checkpoint_path=control_path,
                        device="cpu",
                    )
                )

    def test_release_torch_cache_accepts_cpu(self) -> None:
        release_torch_cache("cpu")


def _runtime_payloads(
    *,
    mapper_path: Path,
    control_path: Path,
) -> tuple[dict[Path, _GuardedCheckpoint], dict[str, dict[str, torch.Tensor]]]:
    vocab = MapperTupleVocab()
    control_config = _small_control_config()
    mapper_config = _small_mapper_config(vocab=vocab, control_dim=control_config.d_model)

    control_model = ControlDemoGlobalEncoder(control_config)
    mapper_model = MapperV2Model(mapper_config, vocab=vocab)
    _fill_parameters(control_model, start=0.01)
    _fill_parameters(mapper_model, start=0.02)

    control_state = _clone_state_dict(control_model.state_dict())
    mapper_state = _clone_state_dict(mapper_model.state_dict())
    mapper_state_with_embedded_control = dict(mapper_state)
    mapper_state_with_embedded_control["control_encoder.embedded.weight"] = torch.ones(1)

    control_payload = _GuardedCheckpoint(
        {
            "checkpoint_schema_version": 1,
            "model_config": asdict(control_config),
            "model_state_dict": control_state,
            "optimizer_state_dict": {"must": "not be read"},
            "history": [{"must": "not be read"}],
            "training_state": {"rng_state": {"must": "not be read"}},
        }
    )
    mapper_payload = _GuardedCheckpoint(
        {
            "checkpoint_schema_version": 1,
            "model_config": asdict(mapper_config),
            "control_model_config": asdict(control_config),
            "model_state_dict": mapper_state_with_embedded_control,
            "optimizer_state_dict": {"must": "not be read"},
            "history": [{"must": "not be read"}],
            "training_state": {"rng_state": {"must": "not be read"}},
        }
    )
    return {control_path: control_payload, mapper_path: mapper_payload}, {
        "control": control_state,
        "mapper": mapper_state,
    }


def _small_control_config() -> ControlDemoGlobalEncoderConfig:
    return ControlDemoGlobalEncoderConfig(
        mel_dim=8,
        timing_dim=2,
        d_model=8,
        heads=2,
        layers=1,
        ffn_dim=16,
        dropout=0.0,
        conv_blocks=0,
        conv_kernel_size=3,
        use_global_memory=True,
        global_stride=8,
        global_layers=1,
        global_ffn_dim=16,
        global_conv_blocks=0,
        global_fusion_start_layer=0,
    )


def _small_mapper_config(*, vocab: MapperTupleVocab, control_dim: int) -> MapperV2Config:
    return MapperV2Config(
        vocab_size=vocab.size,
        mel_dim=8,
        timing_dim=2,
        control_dim=control_dim,
        d_model=8,
        heads=2,
        layers=1,
        ffn_dim=16,
        dropout=0.0,
        max_seq_len=16,
        state_prior_hidden_dim=8,
        ln_close_hidden_dim=8,
        lane_embedding_dim=2,
        age_embedding_dim=2,
        num_age_buckets=4,
        age_cap_ms=1000,
        global_stride=8,
        global_layers=1,
        global_ffn_dim=16,
        global_conv_blocks=0,
        global_conv_kernel_size=3,
    )


def _fill_parameters(model: torch.nn.Module, *, start: float) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_(start + 0.001 * index)


def _clone_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state.items()}


def _fake_torch_load(payloads: dict[Path, _GuardedCheckpoint]):
    def fake_torch_load(path: str | Path, **kwargs: Any) -> _GuardedCheckpoint:
        del kwargs
        return payloads[Path(path)]

    return fake_torch_load


class _GuardedCheckpoint(dict):
    _forbidden_keys = {"optimizer_state_dict", "history", "training_state"}

    def get(self, key: Any, default: Any = None) -> Any:
        if key in self._forbidden_keys:
            raise AssertionError(f"runtime must not read training-only checkpoint key: {key}")
        return super().get(key, default)

    def __getitem__(self, key: Any) -> Any:
        if key in self._forbidden_keys:
            raise AssertionError(f"runtime must not read training-only checkpoint key: {key}")
        return super().__getitem__(key)


class _FakeBeatThisTimingProvider:
    instances: list[_FakeBeatThisTimingProvider] = []

    def __init__(self, *, checkpoint_path: str, device: str, float16: bool) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.float16 = float16
        self.eager_loaded = False
        self.instances.append(self)

    def _get_audio2frames(self) -> object:
        self.eager_loaded = True
        return object()


if __name__ == "__main__":
    unittest.main()
