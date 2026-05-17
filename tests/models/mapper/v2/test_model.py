import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

import torch

from pulsefield_model.data.mapper_tuple_windows import collate_mapper_tuple_windows
from pulsefield_model.models.mapper.shared.replay import ln_carry_state_tensors
from pulsefield_model.models.mapper.shared.tokenizer import encode_mapper_window
from pulsefield_model.models.mapper.shared.vocab import MapperTupleVocab
from pulsefield_model.models.mapper.v2 import MapperV2Config, MapperV2Model
from pulsefield_model.training.mapper_common import MapperTuplePhaseBLossConfig, compute_phase_b_loss


class MapperV2ModelTests(unittest.TestCase):
    def test_forward_adds_gated_full_song_context_path(self) -> None:
        torch.manual_seed(11)
        vocab = MapperTupleVocab()
        model = MapperV2Model(_small_config(vocab), vocab=vocab)
        model.eval()
        batch = _batch(vocab)

        with torch.no_grad():
            output = model(**batch)

        self.assertEqual(output.logits_final.shape, (1, batch["target_fragment_tokens"].shape[1], vocab.size))
        self.assertIsNotNone(output.global_memory)
        self.assertIsNotNone(output.global_memory_padding_mask)
        self.assertIsNotNone(output.global_attention_gates)
        assert output.global_memory is not None
        assert output.global_memory_padding_mask is not None
        assert output.global_attention_gates is not None
        self.assertEqual(output.global_memory.shape, (1, 50, 16))
        self.assertEqual(output.global_memory_padding_mask.shape, (1, 50))
        self.assertEqual(output.global_attention_gates.shape, (2,))
        self.assertIsNotNone(output.global_position_features)
        assert output.global_position_features is not None
        self.assertEqual(output.global_position_features.shape, (1, 4))
        self.assertAlmostEqual(float(output.global_attention_gates[0].item()), 0.05, delta=0.005)

    def test_requires_full_song_context_when_global_path_enabled(self) -> None:
        vocab = MapperTupleVocab()
        model = MapperV2Model(_small_config(vocab), vocab=vocab)
        batch = _batch(vocab)
        batch.pop("full_mel")

        with self.assertRaisesRegex(ValueError, "full_mel"):
            model(**batch)

    def test_forward_reuses_precomputed_window_context(self) -> None:
        torch.manual_seed(12)
        vocab = MapperTupleVocab()
        model = MapperV2Model(_small_config(vocab), vocab=vocab)
        model.eval()
        batch = _batch(vocab)

        with torch.no_grad():
            base = model(**batch)
            cached_batch = _clone_batch(batch)
            cached_batch["projected_control_memory_8s"] = model.control_projection(
                cached_batch.pop("control_memory_8s"),
            )
            assert base.global_memory is not None
            assert base.global_memory_padding_mask is not None
            assert base.global_position_features is not None
            cached_batch["global_memory"] = base.global_memory
            cached_batch["global_memory_padding_mask"] = base.global_memory_padding_mask
            cached_batch["global_position_features"] = base.global_position_features
            for key in ("full_mel", "full_dense_timing_v2", "padding_mask", "frame_count", "source_frame_count"):
                cached_batch.pop(key)
            cached = model(**cached_batch)

        self.assertTrue(torch.allclose(base.logits_final, cached.logits_final, atol=1e-6, rtol=1e-6))
        assert cached.global_memory is not None
        self.assertTrue(torch.allclose(cached.global_memory, cached_batch["global_memory"], atol=0.0, rtol=0.0))

    def test_forward_reuses_precomputed_global_attention_kv_cache(self) -> None:
        torch.manual_seed(14)
        vocab = MapperTupleVocab()
        model = MapperV2Model(_small_config(vocab), vocab=vocab)
        model.eval()
        batch = _batch(vocab)

        with torch.no_grad():
            base = model(**batch)
            assert base.global_memory is not None
            assert base.global_memory_padding_mask is not None
            assert base.global_position_features is not None

            cached_batch = _clone_batch(batch)
            cached_batch["projected_control_memory_8s"] = model.control_projection(
                cached_batch.pop("control_memory_8s"),
            )
            cached_batch["control_attention_kv_cache"] = model.control_attention_kv_cache(
                cached_batch["projected_control_memory_8s"],
            )
            cached_batch["global_memory"] = base.global_memory
            cached_batch["global_memory_padding_mask"] = base.global_memory_padding_mask
            cached_batch["global_position_features"] = base.global_position_features
            cached_batch["global_attention_kv_cache"] = model.global_attention_kv_cache(base.global_memory)
            for key in ("full_mel", "full_dense_timing_v2", "padding_mask", "frame_count", "source_frame_count"):
                cached_batch.pop(key)
            cached = model(**cached_batch)

        first_key, first_value = cached_batch["global_attention_kv_cache"][0]
        self.assertEqual(
            tuple(first_key.shape),
            (1, model.config.heads, base.global_memory.shape[1], model.config.d_model // model.config.heads),
        )
        self.assertEqual(tuple(first_value.shape), tuple(first_key.shape))
        self.assertTrue(torch.allclose(base.logits_final, cached.logits_final, atol=1e-6, rtol=1e-6))

    def test_incremental_decode_matches_cached_full_forward_logits(self) -> None:
        torch.manual_seed(31)
        vocab = MapperTupleVocab()
        model = MapperV2Model(_small_config(vocab), vocab=vocab)
        model.eval()
        batch = _batch(vocab)
        state_dict_keys = tuple(model.state_dict().keys())

        with torch.no_grad():
            base = model(**batch)
            assert base.global_memory is not None
            assert base.global_memory_padding_mask is not None
            assert base.global_position_features is not None

            cached_batch = _clone_batch(batch)
            cached_batch["projected_control_memory_8s"] = model.control_projection(
                cached_batch.pop("control_memory_8s"),
            )
            cached_batch["control_attention_kv_cache"] = model.control_attention_kv_cache(
                cached_batch["projected_control_memory_8s"],
            )
            cached_batch["global_memory"] = base.global_memory
            cached_batch["global_memory_padding_mask"] = base.global_memory_padding_mask
            cached_batch["global_position_features"] = base.global_position_features
            cached_batch["global_attention_kv_cache"] = model.global_attention_kv_cache(base.global_memory)
            for key in ("full_mel", "full_dense_timing_v2", "padding_mask", "frame_count", "source_frame_count"):
                cached_batch.pop(key)
            full = model(**cached_batch)

            decode_state = model.create_empty_decode_state(
                batch_size=int(cached_batch["decoder_input_tokens"].shape[0]),
                device=cached_batch["decoder_input_tokens"].device,
            )
            for step in range(int(cached_batch["decoder_input_tokens"].shape[1])):
                output = model.incremental_decode_next_token(
                    decode_state=decode_state,
                    decoder_input_token=cached_batch["decoder_input_tokens"][:, step],
                    current_ms=cached_batch["target_fragment_states"]["current_ms"][:, step],
                    open_mask=cached_batch["target_fragment_states"]["open_mask"][:, step],
                    open_start_ms=cached_batch["target_fragment_states"]["open_start_ms"][:, step],
                    open_age_ms=cached_batch["target_fragment_states"]["open_age_ms"][:, step],
                    write_start_ms=cached_batch["write_start_ms"],
                    write_end_ms=cached_batch["write_end_ms"],
                    is_full_chart_start=cached_batch["is_full_chart_start"],
                    is_full_chart_end=cached_batch["is_full_chart_end"],
                    ln_carry_in=cached_batch["ln_carry_in"],
                    ln_carry_out=cached_batch["ln_carry_out"],
                    density_teacher_8s=cached_batch["density_teacher_8s"],
                    projected_control_memory_8s=cached_batch["projected_control_memory_8s"],
                    control_attention_kv_cache=cached_batch["control_attention_kv_cache"],
                    normalized_difficulty=cached_batch["normalized_difficulty"],
                    global_memory=cached_batch["global_memory"],
                    global_memory_padding_mask=cached_batch["global_memory_padding_mask"],
                    global_position_features=cached_batch["global_position_features"],
                    global_attention_kv_cache=cached_batch["global_attention_kv_cache"],
                )

                self.assertTrue(
                    torch.allclose(output.logits_final, full.logits_final[:, step], atol=1e-5, rtol=1e-5),
                    msg=f"incremental logits mismatch at step {step}",
                )
                self.assertTrue(torch.allclose(output.base_logits, full.base_logits[:, step], atol=1e-5, rtol=1e-5))
                self.assertTrue(
                    torch.allclose(output.state_prior_bias, full.state_prior_bias[:, step], atol=1e-6, rtol=1e-6)
                )
                self.assertTrue(
                    torch.allclose(
                        output.ln_close_event_bias,
                        full.ln_close_event_bias[:, step],
                        atol=1e-6,
                        rtol=1e-6,
                    )
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
                expected_cache_shape = (
                    1,
                    model.config.heads,
                    step + 1,
                    model.config.d_model // model.config.heads,
                )
                self.assertEqual(decode_state.sequence_length, step + 1)
                for layer_cache in decode_state.self_attention_kv_cache:
                    self.assertEqual(tuple(layer_cache.key.shape), expected_cache_shape)
                    self.assertEqual(tuple(layer_cache.value.shape), expected_cache_shape)

        self.assertEqual(tuple(model.state_dict().keys()), state_dict_keys)
        control_key, control_value = cached_batch["control_attention_kv_cache"][0]
        self.assertEqual(
            tuple(control_key.shape),
            (1, model.config.heads, 400, model.config.d_model // model.config.heads),
        )
        self.assertEqual(tuple(control_value.shape), tuple(control_key.shape))

    def test_reuses_v1_teacher_forced_loss_targets(self) -> None:
        torch.manual_seed(13)
        vocab = MapperTupleVocab()
        model = MapperV2Model(_small_config(vocab), vocab=vocab)
        batch = _batch(vocab)

        output = model(**batch)
        loss_output = compute_phase_b_loss(
            output,
            target_fragment_tokens=batch["target_fragment_tokens"],
            target_fragment_mask=batch["target_fragment_mask"],
            target_fragment_states=batch["target_fragment_states"],
            close_labels=batch["close_labels"],
            close_label_mask=batch["close_label_mask"],
            density_target_8s=batch["density_target_8s"],
            density_confidence_8s=batch["density_confidence_8s"],
            write_start_ms=batch["write_start_ms"],
            vocab=vocab,
            loss_config=MapperTuplePhaseBLossConfig(),
        )

        self.assertTrue(torch.isfinite(loss_output.total_loss))
        self.assertGreater(loss_output.metrics["target/token_count"], 0.0)

    def test_masked_full_song_tail_does_not_affect_output(self) -> None:
        torch.manual_seed(17)
        vocab = MapperTupleVocab()
        model = MapperV2Model(_small_config(vocab), vocab=vocab)
        model.eval()
        batch = _batch(vocab, write_start_ms=8000, frame_count=800, source_frame_count=450)
        changed = _clone_batch(batch)
        changed["full_mel"][:, 450:] = torch.randn_like(changed["full_mel"][:, 450:]) * 100.0
        changed["full_dense_timing_v2"][:, 450:] = (
            torch.randn_like(changed["full_dense_timing_v2"][:, 450:]) * 100.0
        )

        with torch.no_grad():
            base = model(**batch).base_logits
            perturbed = model(**changed).base_logits

        self.assertTrue(torch.allclose(base, perturbed, atol=1e-6, rtol=1e-6))

    def test_global_song_position_uses_source_frame_count(self) -> None:
        torch.manual_seed(19)
        vocab = MapperTupleVocab()
        model = MapperV2Model(_small_config(vocab), vocab=vocab)
        model.eval()
        batch = _batch(vocab, write_start_ms=8000, frame_count=800, source_frame_count=450)
        changed = _clone_batch(batch)
        changed["source_frame_count"] = torch.tensor([800], dtype=torch.long)

        with torch.no_grad():
            base = model(**batch).global_memory
            changed_memory = model(**changed).global_memory

        assert base is not None
        assert changed_memory is not None
        self.assertFalse(torch.allclose(base, changed_memory, atol=1e-6, rtol=1e-6))

    def test_direct_global_position_condition_reaches_decoder(self) -> None:
        torch.manual_seed(23)
        vocab = MapperTupleVocab()
        model = MapperV2Model(_small_config(vocab), vocab=vocab)
        model.eval()
        with torch.no_grad():
            assert model.global_position_projection is not None
            model.global_position_projection.weight.zero_()
            model.global_position_projection.bias.zero_()
            model.global_position_projection.weight[0, 0] = 2.0
            for block in model.global_cross_attention_layers:
                block.gate_logit.fill_(-100.0)
        batch = _batch(vocab, write_start_ms=8000, frame_count=800, source_frame_count=450)
        changed = _clone_batch(batch)
        changed["source_frame_count"] = torch.tensor([800], dtype=torch.long)
        changed["padding_mask"][:, 450:800] = False

        with torch.no_grad():
            base = model(**batch)
            changed_output = model(**changed)

        self.assertIsNotNone(base.global_position_features)
        assert base.global_position_features is not None
        assert changed_output.global_position_features is not None
        self.assertFalse(
            torch.allclose(
                base.global_position_features,
                changed_output.global_position_features,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertFalse(torch.allclose(base.base_logits, changed_output.base_logits, atol=1e-6, rtol=1e-6))


def _small_config(vocab: MapperTupleVocab) -> MapperV2Config:
    return MapperV2Config(
        vocab_size=vocab.size,
        control_dim=16,
        d_model=16,
        heads=4,
        layers=2,
        ffn_dim=32,
        dropout=0.0,
        max_seq_len=16,
        state_hidden_dim=16,
        ln_close_hidden_dim=16,
        global_stride=16,
        global_layers=1,
        global_ffn_dim=32,
        global_conv_blocks=1,
    )


def _batch(
    vocab: MapperTupleVocab,
    *,
    write_start_ms: int = 0,
    frame_count: int = 800,
    source_frame_count: int | None = None,
) -> dict[str, torch.Tensor]:
    source_frame_count = frame_count if source_frame_count is None else source_frame_count
    tokenized = encode_mapper_window(
        [],
        vocab=vocab,
        write_start_ms=write_start_ms,
        write_end_ms=write_start_ms + 8000,
        chart_start_ms=0,
        chart_end_ms=write_start_ms + 8000,
    )
    write_start_frame = write_start_ms // 20
    full_mel = torch.randn(frame_count, 160, dtype=torch.float32) * 0.05
    full_dense_timing_v2 = torch.randn(frame_count, 4, dtype=torch.float32) * 0.05
    sample = {
        "decoder_input_tokens": tokenized.decoder_input_tensor(),
        "target_fragment_tokens": tokenized.target_fragment_tensor(),
        "target_fragment_states": {
            "current_ms": tokenized.target_fragment_current_ms,
            "open_mask": tokenized.target_fragment_open_mask,
            "open_start_ms": tokenized.target_fragment_open_start_ms,
            "open_age_ms": tokenized.target_fragment_open_age_ms,
        },
        "ln_carry_in": ln_carry_state_tensors(tokenized.ln_carry_in),
        "ln_carry_out": ln_carry_state_tensors(tokenized.ln_carry_out),
        "close_labels": tokenized.close_labels,
        "close_label_mask": tokenized.close_label_mask,
        "control_memory_8s": torch.randn(400, 16, dtype=torch.float32) * 0.05,
        "density_teacher_8s": torch.zeros(400, 1, dtype=torch.float32),
        "density_target_8s": torch.zeros(400, 1, dtype=torch.float32),
        "density_confidence_8s": torch.ones(400, 1, dtype=torch.float32),
        "write_start_ms": torch.tensor(tokenized.write_start_ms, dtype=torch.long),
        "write_end_ms": torch.tensor(tokenized.write_end_ms, dtype=torch.long),
        "is_full_chart_start": torch.tensor(tokenized.is_full_chart_start, dtype=torch.bool),
        "is_full_chart_end": torch.tensor(tokenized.is_full_chart_end, dtype=torch.bool),
        "difficulty": torch.tensor([4.0], dtype=torch.float32),
        "normalized_difficulty": torch.tensor([0.0], dtype=torch.float32),
        "full_mel": full_mel,
        "full_dense_timing_v2": full_dense_timing_v2,
        "frame_count": torch.tensor(frame_count, dtype=torch.long),
        "source_frame_count": torch.tensor(source_frame_count, dtype=torch.long),
        "target_start_frame": torch.tensor(write_start_frame, dtype=torch.long),
        "mel_context": full_mel[write_start_frame : write_start_frame + 400],
        "timing_context": full_dense_timing_v2[write_start_frame : write_start_frame + 400],
        "context_padding_mask": torch.arange(400, dtype=torch.long) + write_start_frame >= source_frame_count,
        "control_slice_start_frames": torch.tensor(
            [write_start_frame + offset for offset in range(0, 400, 100)],
            dtype=torch.long,
        ),
    }
    return collate_mapper_tuple_windows([sample], pad_id=vocab.pad_id)


def _clone_batch(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_batch(item) for key, item in value.items()}
    if isinstance(value, list):
        return list(value)
    return value


if __name__ == "__main__":
    unittest.main()
