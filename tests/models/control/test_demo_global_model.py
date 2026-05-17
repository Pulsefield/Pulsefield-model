import unittest

import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("requires torch")

import torch

from pulsefield_model.models.control.demo_global import (
    ControlDemoGlobalEncoder,
    ControlDemoGlobalEncoderConfig,
    _GlobalFiLM,
    _GlobalFusionBlock,
    _masked_chunk_mean,
)


class ControlDemoGlobalModelTests(unittest.TestCase):
    def test_forward_shape_with_global_memory(self) -> None:
        torch.manual_seed(11)
        config = _small_config()
        model = ControlDemoGlobalEncoder(config)
        model.eval()
        padding_mask = torch.zeros(2, 1000, dtype=torch.bool)
        padding_mask[1, 700:] = True

        with torch.no_grad():
            output = model(
                context_mel=torch.randn(2, 600, 160),
                context_dense_timing_v2=torch.randn(2, 600, 4),
                normalized_difficulty=torch.tensor([-0.5, 0.5]),
                context_padding_mask=torch.zeros(2, 600, dtype=torch.bool),
                full_mel=torch.randn(2, 1000, 160),
                full_dense_timing_v2=torch.randn(2, 1000, 4),
                padding_mask=padding_mask,
                frame_count=torch.tensor([1000, 700], dtype=torch.long),
                target_start_frame=torch.tensor([300, 300], dtype=torch.long),
            )

        self.assertEqual(output.value_pred.shape, (2, 100, 1))
        self.assertEqual(output.control_memory.shape, (2, 600, 32))
        self.assertEqual(output.memory_padding_mask.shape, (2, 600))
        self.assertIsNotNone(output.global_memory)
        self.assertIsNotNone(output.global_memory_padding_mask)
        assert output.global_memory is not None
        assert output.global_memory_padding_mask is not None
        self.assertEqual(output.global_memory.shape[0], 2)
        self.assertEqual(output.global_memory.shape[-1], 32)
        self.assertEqual(output.global_memory_padding_mask.shape, output.global_memory.shape[:2])

    def test_masked_chunk_mean_masks_fully_padded_chunks(self) -> None:
        values = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
        padding_mask = torch.zeros(1, 10, dtype=torch.bool)
        padding_mask[:, 6:] = True

        pooled, pooled_mask = _masked_chunk_mean(values, padding_mask, chunk_size=4)

        self.assertEqual(pooled.shape, (1, 3, 1))
        self.assertEqual(pooled_mask.tolist(), [[False, False, True]])
        self.assertTrue(torch.allclose(pooled[0, 0, 0], torch.tensor(1.5)))
        self.assertTrue(torch.allclose(pooled[0, 1, 0], torch.tensor(4.5)))
        self.assertEqual(float(pooled[0, 2, 0].item()), 0.0)

    def test_global_film_initializes_as_identity(self) -> None:
        torch.manual_seed(17)
        film = _GlobalFiLM(16)
        hidden = torch.randn(2, 7, 16)
        condition = torch.randn(2, 16)

        output = film(hidden, condition)

        self.assertTrue(torch.allclose(output, hidden))

    def test_global_fusion_gate_starts_weak(self) -> None:
        block = _GlobalFusionBlock(_small_config())

        gate = float(torch.sigmoid(block.gate_logit).detach().cpu())

        self.assertAlmostEqual(gate, 0.05, delta=0.005)

    def test_rejects_full_resolution_or_near_full_resolution_global_stride(self) -> None:
        for bad_stride in (1, 2, 4, 7):
            with self.subTest(global_stride=bad_stride):
                with self.assertRaisesRegex(ValueError, "global_stride"):
                    ControlDemoGlobalEncoder(
                        ControlDemoGlobalEncoderConfig(
                            d_model=32,
                            heads=4,
                            layers=3,
                            ffn_dim=64,
                            global_stride=bad_stride,
                        )
                    )

        ControlDemoGlobalEncoder(
            ControlDemoGlobalEncoderConfig(
                d_model=32,
                heads=4,
                layers=3,
                ffn_dim=64,
                global_stride=8,
                global_layers=1,
                global_ffn_dim=64,
            )
        )

    def test_rejects_padding_mask_that_leaves_frame_count_tail_unmasked(self) -> None:
        model = ControlDemoGlobalEncoder(_global_behavior_config())
        batch_size = 2
        full_frames = 1000
        bad_padding_mask = torch.zeros(batch_size, full_frames, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "padding_mask.*frame_count tail"):
            model(
                context_mel=torch.zeros(batch_size, 600, 160),
                context_dense_timing_v2=torch.zeros(batch_size, 600, 4),
                normalized_difficulty=torch.tensor([0.0, 1.0]),
                context_padding_mask=torch.zeros(batch_size, 600, dtype=torch.bool),
                full_mel=torch.zeros(batch_size, full_frames, 160),
                full_dense_timing_v2=torch.zeros(batch_size, full_frames, 4),
                padding_mask=bad_padding_mask,
                frame_count=torch.tensor([1000, 700], dtype=torch.long),
                target_start_frame=torch.tensor([400, 300], dtype=torch.long),
            )

    def test_allows_masked_valid_window_before_padded_frame_count(self) -> None:
        model = ControlDemoGlobalEncoder(_global_behavior_config())
        model.eval()
        full_frames = 800
        padding_mask = torch.zeros(1, full_frames, dtype=torch.bool)
        padding_mask[:, 450:800] = True

        with torch.no_grad():
            output = model(
                context_mel=torch.zeros(1, 600, 160),
                context_dense_timing_v2=torch.zeros(1, 600, 4),
                normalized_difficulty=torch.tensor([0.0]),
                context_padding_mask=torch.zeros(1, 600, dtype=torch.bool),
                full_mel=torch.zeros(1, full_frames, 160),
                full_dense_timing_v2=torch.zeros(1, full_frames, 4),
                padding_mask=padding_mask,
                frame_count=torch.tensor([800], dtype=torch.long),
                target_start_frame=torch.tensor([400], dtype=torch.long),
            )

        self.assertEqual(output.memory_padding_mask.shape, (1, 600))

    def test_masked_full_song_tail_does_not_affect_output(self) -> None:
        torch.manual_seed(19)
        model = ControlDemoGlobalEncoder(_global_behavior_config())
        model.eval()
        batch_size = 1
        full_frames = 1000
        frame_count = torch.tensor([700], dtype=torch.long)
        padding_mask = torch.arange(full_frames).unsqueeze(0) >= frame_count.unsqueeze(1)
        full_mel_a = torch.randn(batch_size, full_frames, 160)
        full_timing_a = torch.randn(batch_size, full_frames, 4)
        full_mel_b = full_mel_a.clone()
        full_timing_b = full_timing_a.clone()
        full_mel_b[:, 700:] = torch.randn_like(full_mel_b[:, 700:]) * 100.0
        full_timing_b[:, 700:] = torch.randn_like(full_timing_b[:, 700:]) * 100.0
        common_kwargs = dict(
            padding_mask=padding_mask,
            frame_count=frame_count,
            target_start_frame=torch.tensor([300], dtype=torch.long),
            context_mel=torch.randn(batch_size, 600, 160),
            context_dense_timing_v2=torch.randn(batch_size, 600, 4),
            normalized_difficulty=torch.tensor([0.5]),
            context_padding_mask=torch.zeros(batch_size, 600, dtype=torch.bool),
        )

        with torch.no_grad():
            out_a = model(
                full_mel=full_mel_a,
                full_dense_timing_v2=full_timing_a,
                **common_kwargs,
            ).value_pred
            out_b = model(
                full_mel=full_mel_b,
                full_dense_timing_v2=full_timing_b,
                **common_kwargs,
            ).value_pred

        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6))

    def test_unmasked_full_song_inputs_affect_output(self) -> None:
        torch.manual_seed(23)
        model = ControlDemoGlobalEncoder(_global_behavior_config())
        model.eval()
        batch_size = 1
        full_frames = 1000
        frame_count = torch.tensor([1000], dtype=torch.long)
        padding_mask = torch.zeros(batch_size, full_frames, dtype=torch.bool)
        full_mel_a = torch.randn(batch_size, full_frames, 160)
        full_timing_a = torch.randn(batch_size, full_frames, 4)
        full_mel_b = full_mel_a.clone()
        full_mel_b[:, 100:300] += 5.0
        common_kwargs = dict(
            padding_mask=padding_mask,
            frame_count=frame_count,
            target_start_frame=torch.tensor([400], dtype=torch.long),
            context_mel=torch.randn(batch_size, 600, 160),
            context_dense_timing_v2=torch.randn(batch_size, 600, 4),
            normalized_difficulty=torch.tensor([0.5]),
            context_padding_mask=torch.zeros(batch_size, 600, dtype=torch.bool),
        )

        with torch.no_grad():
            out_a = model(
                full_mel=full_mel_a,
                full_dense_timing_v2=full_timing_a,
                **common_kwargs,
            ).value_pred
            out_b = model(
                full_mel=full_mel_b,
                full_dense_timing_v2=full_timing_a,
                **common_kwargs,
            ).value_pred

        self.assertFalse(torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6))


def _small_config() -> ControlDemoGlobalEncoderConfig:
    return ControlDemoGlobalEncoderConfig(
        d_model=32,
        heads=4,
        layers=3,
        ffn_dim=64,
        dropout=0.0,
        conv_blocks=1,
        use_global_memory=True,
        global_stride=16,
        global_layers=1,
        global_ffn_dim=64,
        global_conv_blocks=1,
        global_fusion_start_layer=1,
    )


def _global_behavior_config() -> ControlDemoGlobalEncoderConfig:
    return ControlDemoGlobalEncoderConfig(
        d_model=32,
        heads=4,
        layers=3,
        ffn_dim=64,
        dropout=0.0,
        conv_blocks=1,
        use_global_memory=True,
        global_stride=8,
        global_layers=1,
        global_ffn_dim=64,
        global_conv_blocks=1,
        global_fusion_start_layer=1,
    )


if __name__ == "__main__":
    unittest.main()
