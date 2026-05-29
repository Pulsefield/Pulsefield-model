import unittest

import importlib.util
from unittest.mock import patch

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

import torch

from pulsefield_model.models.mapper.v2_1 import (
    MapperTimepoint,
    MapperV21Config,
    MapperV21Model,
    MapperV21ModelLoss,
    MapperV21LossConfig,
    MapperV21Vocab,
    encode_mapper_window,
    ln_carry_state_tensors,
)
from pulsefield_model.models.mapper.v2_1.vocab import LaneAction


def _actions(*actions: LaneAction) -> tuple[LaneAction, ...]:
    padded = list(actions)
    while len(padded) < 4:
        padded.append(LaneAction.NONE)
    return tuple(padded)


def _batched_carry(carry: object) -> dict[str, torch.Tensor]:
    tensors = ln_carry_state_tensors(carry)  # type: ignore[arg-type]
    return {
        "current_ms": tensors["current_ms"].reshape(1),
        "open_mask": tensors["open_mask"].reshape(1, 4),
        "open_start_ms": tensors["open_start_ms"].reshape(1, 4),
        "open_age_ms": tensors["open_age_ms"].reshape(1, 4),
    }


def _batch_for_window(tokenized: object) -> dict[str, object]:
    return {
        "decoder_input_tokens": tokenized.decoder_input_tensor().unsqueeze(0),  # type: ignore[attr-defined]
        "target_fragment_tokens": tokenized.target_fragment_tensor().unsqueeze(0),  # type: ignore[attr-defined]
        "target_fragment_mask": torch.ones((1, tokenized.seq_len), dtype=torch.bool),  # type: ignore[attr-defined]
        "target_fragment_states": {
            "current_ms": tokenized.target_fragment_current_ms.unsqueeze(0),  # type: ignore[attr-defined]
            "open_mask": tokenized.target_fragment_open_mask.unsqueeze(0),  # type: ignore[attr-defined]
            "open_start_ms": tokenized.target_fragment_open_start_ms.unsqueeze(0),  # type: ignore[attr-defined]
            "open_age_ms": tokenized.target_fragment_open_age_ms.unsqueeze(0),  # type: ignore[attr-defined]
            "emitted_lane_mask": tokenized.target_fragment_emitted_lane_mask.unsqueeze(0),  # type: ignore[attr-defined]
            "last_lane_index": tokenized.target_fragment_last_lane_index.unsqueeze(0),  # type: ignore[attr-defined]
        },
        "ln_carry_in": _batched_carry(tokenized.ln_carry_in),  # type: ignore[attr-defined]
        "ln_carry_out": _batched_carry(tokenized.ln_carry_out),  # type: ignore[attr-defined]
        "close_labels": tokenized.close_labels.unsqueeze(0),  # type: ignore[attr-defined]
        "close_label_mask": tokenized.close_label_mask.unsqueeze(0),  # type: ignore[attr-defined]
        "write_start_ms": torch.tensor([tokenized.write_start_ms], dtype=torch.long),  # type: ignore[attr-defined]
        "write_end_ms": torch.tensor([tokenized.write_end_ms], dtype=torch.long),  # type: ignore[attr-defined]
        "chart_end_ms": torch.tensor([tokenized.chart_end_ms], dtype=torch.long),  # type: ignore[attr-defined]
        "is_full_chart_start": torch.tensor([tokenized.is_full_chart_start], dtype=torch.bool),  # type: ignore[attr-defined]
        "is_full_chart_end": torch.tensor([tokenized.is_full_chart_end], dtype=torch.bool),  # type: ignore[attr-defined]
        "difficulty": torch.tensor([[3.2]], dtype=torch.float32),
        "normalized_difficulty": torch.tensor([[0.1]], dtype=torch.float32),
        "density_target_8s": torch.zeros((1, 400, 1), dtype=torch.float32),
        "density_confidence_8s": torch.ones((1, 400, 1), dtype=torch.float32),
    }


def _small_config(**overrides: object) -> MapperV21Config:
    values = {
        "control_dim": 16,
        "d_model": 16,
        "heads": 4,
        "layers": 1,
        "ffn_dim": 32,
        "dropout": 0.0,
        "max_seq_len": 64,
        "state_prior_hidden_dim": 16,
        "ln_close_hidden_dim": 16,
        "lane_embedding_dim": 4,
        "age_embedding_dim": 4,
        "use_global_context": False,
        "global_conv_blocks": 0,
    }
    values.update(overrides)
    return MapperV21Config(**values)


class MapperV21ModelTests(unittest.TestCase):
    def test_forward_and_loss_use_sparse_same_time_state(self) -> None:
        torch.manual_seed(20260515)
        vocab = MapperV21Vocab()
        tokenized = encode_mapper_window(
            [MapperTimepoint(1000, _actions(LaneAction.TAP, LaneAction.NONE, LaneAction.TAP))],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=1000,
        )
        batch = _batch_for_window(tokenized)
        model = MapperV21Model(_small_config(), vocab=vocab)

        output = model(batch)

        lane_1_tap = vocab.lane_action_token_id(0, LaneAction.TAP)
        lane_3_tap = vocab.lane_action_token_id(2, LaneAction.TAP)
        self.assertTrue(torch.isneginf(output.grammar_mask[0, 2, lane_1_tap]))
        self.assertEqual(float(output.grammar_mask[0, 2, lane_3_tap].item()), 0.0)
        target = batch["target_fragment_tokens"]
        positions = torch.arange(target.shape[1])
        self.assertTrue(torch.isfinite(output.logits_final[0, positions, target[0]]).all().item())
        self.assertTrue(torch.equal(output.state_emitted_lane_mask, batch["target_fragment_states"]["emitted_lane_mask"]))
        self.assertTrue(torch.equal(output.state_last_lane_index, batch["target_fragment_states"]["last_lane_index"]))

        loss_fn = MapperV21ModelLoss(
            MapperV21LossConfig(lambda_density=0.0, lambda_ln_close=0.0, lambda_adapter_reg=1e-5),
            vocab=vocab,
        )
        loss = loss_fn(output, batch)
        loss.total_loss.backward()
        self.assertTrue(torch.isfinite(loss.total_loss).item())
        self.assertIsNotNone(model.token_embedding.weight.grad)
        self.assertGreater(float(model.token_embedding.weight.grad.abs().sum().item()), 0.0)

    def test_forward_can_skip_grammar_mask_without_changing_valid_logits(self) -> None:
        torch.manual_seed(20260520)
        vocab = MapperV21Vocab()
        tokenized = encode_mapper_window(
            [
                MapperTimepoint(100, _actions(LaneAction.TAP, LaneAction.NONE, LaneAction.NONE)),
                MapperTimepoint(200, _actions(LaneAction.NONE, LaneAction.TAP, LaneAction.NONE)),
            ],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=300,
        )
        batch = _batch_for_window(tokenized)
        model = MapperV21Model(_small_config(), vocab=vocab)
        model.eval()

        with torch.no_grad():
            masked = model(batch)
            unmasked = model({**batch, "apply_grammar_mask": False})
            with patch(
                "pulsefield_model.models.mapper.v2_1.model.build_grammar_mask",
                side_effect=AssertionError("build_grammar_mask should be skipped"),
            ):
                skipped = model({**batch, "apply_grammar_mask": False})

        valid = torch.isfinite(masked.grammar_mask)
        invalid = ~valid
        self.assertTrue(bool(invalid.any().item()))
        self.assertTrue(torch.equal(unmasked.grammar_mask, torch.zeros_like(unmasked.grammar_mask)))
        self.assertTrue(torch.equal(skipped.grammar_mask, torch.zeros_like(skipped.grammar_mask)))
        self.assertTrue(torch.allclose(masked.logits_final[valid], unmasked.logits_final[valid]))
        self.assertTrue(torch.isneginf(masked.logits_final[invalid]).all().item())

    def test_incremental_decode_matches_cached_full_forward_logits(self) -> None:
        torch.manual_seed(20260529)
        vocab = MapperV21Vocab()
        tokenized = encode_mapper_window(
            [
                MapperTimepoint(100, _actions(LaneAction.TAP, LaneAction.NONE, LaneAction.HOLD_START)),
                MapperTimepoint(300, _actions(LaneAction.NONE, LaneAction.TAP, LaneAction.HOLD_END)),
            ],
            vocab=vocab,
            write_start_ms=0,
            write_end_ms=8000,
            chart_end_ms=500,
        )
        batch = _batch_for_window(tokenized)
        batch["projected_control_memory_8s"] = torch.zeros((1, 400, 16), dtype=torch.float32)
        batch["density_teacher_8s"] = torch.zeros((1, 400, 1), dtype=torch.float32)
        batch["global_memory"] = torch.randn((1, 5, 16), dtype=torch.float32) * 0.05
        batch["global_memory_padding_mask"] = torch.tensor([[False, False, False, True, True]], dtype=torch.bool)
        batch["global_position_features"] = torch.tensor([[0.0, 0.25, 0.5, 0.75]], dtype=torch.float32)
        model = MapperV21Model(_small_config(layers=2, use_global_context=True), vocab=vocab)
        model.eval()

        with torch.no_grad():
            batch["control_attention_kv_cache"] = model.control_attention_kv_cache(
                batch["projected_control_memory_8s"],
            )
            batch["global_attention_kv_cache"] = model.global_attention_kv_cache(batch["global_memory"])
            full = model(batch)
            decode_state = model.create_empty_decode_state(
                batch_size=int(batch["decoder_input_tokens"].shape[0]),
                device=batch["decoder_input_tokens"].device,
            )
            states = batch["target_fragment_states"]
            for step in range(int(batch["decoder_input_tokens"].shape[1])):
                output = model.incremental_decode_next_token(
                    decode_state=decode_state,
                    decoder_input_token=batch["decoder_input_tokens"][:, step],
                    current_ms=states["current_ms"][:, step],
                    open_mask=states["open_mask"][:, step],
                    open_start_ms=states["open_start_ms"][:, step],
                    open_age_ms=states["open_age_ms"][:, step],
                    emitted_lane_mask=states["emitted_lane_mask"][:, step],
                    last_lane_index=states["last_lane_index"][:, step],
                    write_start_ms=batch["write_start_ms"],
                    write_end_ms=batch["write_end_ms"],
                    chart_end_ms=batch["chart_end_ms"],
                    is_full_chart_start=batch["is_full_chart_start"],
                    is_full_chart_end=batch["is_full_chart_end"],
                    ln_carry_in=batch["ln_carry_in"],
                    ln_carry_out=batch["ln_carry_out"],
                    density_teacher_8s=batch["density_teacher_8s"],
                    projected_control_memory_8s=batch["projected_control_memory_8s"],
                    control_attention_kv_cache=batch["control_attention_kv_cache"],
                    normalized_difficulty=batch["normalized_difficulty"],
                    global_memory=batch["global_memory"],
                    global_memory_padding_mask=batch["global_memory_padding_mask"],
                    global_position_features=batch["global_position_features"],
                    global_attention_kv_cache=batch["global_attention_kv_cache"],
                )

                self.assertTrue(
                    torch.allclose(output.logits_final, full.logits_final[:, step], atol=1e-5, rtol=1e-5),
                    msg=f"incremental logits mismatch at step {step}",
                )
                self.assertTrue(torch.allclose(output.base_logits, full.base_logits[:, step], atol=1e-5, rtol=1e-5))
                self.assertTrue(
                    torch.allclose(output.grammar_mask, full.grammar_mask[:, step], atol=0.0, rtol=0.0)
                )
                self.assertTrue(
                    torch.allclose(output.ln_close_event_bias, full.ln_close_event_bias[:, step], atol=1e-6, rtol=1e-6)
                )
                self.assertTrue(
                    torch.allclose(
                        output.ln_close_time_shift_bias,
                        full.ln_close_time_shift_bias[:, step],
                        atol=1e-6,
                        rtol=1e-6,
                    )
                )
                decode_state = output.decode_state
                self.assertEqual(decode_state.sequence_length, step + 1)
                self.assertIsNotNone(output.global_attention_gates)


if __name__ == "__main__":
    unittest.main()
