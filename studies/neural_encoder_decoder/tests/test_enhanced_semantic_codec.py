from __future__ import annotations

import json
import math
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import torch


CODE_DIRECTORY = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CODE_DIRECTORY))

import train_enhanced_semantic_codec as codec  # noqa: E402

REPRODUCTION_CODE_DIRECTORY = Path(__file__).resolve().parents[2] / "airtalking_reproduction" / "code"
if str(REPRODUCTION_CODE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(REPRODUCTION_CODE_DIRECTORY))
import airtalking_reproduction as reproduction  # noqa: E402


class EightBitQuantizationTests(unittest.TestCase):
    def test_ste_is_on_uint8_grid_and_passes_gradient(self) -> None:
        quantizer = codec.EightBitSTEQuantizer()
        values = torch.tensor([0.0, 0.1, 0.501, 1.0], dtype=torch.float32, requires_grad=True)
        quantized = quantizer(values)
        expected = torch.round(values.detach() * 255.0) / 255.0
        torch.testing.assert_close(quantized.detach(), expected, rtol=0.0, atol=0.0)
        quantized.sum().backward()
        torch.testing.assert_close(values.grad, torch.ones_like(values), rtol=0.0, atol=0.0)

    def test_byte_transport_round_trip_is_exact(self) -> None:
        codes = torch.arange(40, dtype=torch.uint8).reshape(1, 5, 2, 4)
        received, raw_bytes, compressed_bytes = codec.transmit_latent_batch(
            codes, device=torch.device("cpu"), zlib_level=6
        )
        self.assertTrue(torch.equal(codes, received))
        self.assertEqual(raw_bytes, codes.numel())
        self.assertGreater(compressed_bytes, 0)


class PayloadRatioTests(unittest.TestCase):
    def test_all_scalable_stride16_uint8_ratios(self) -> None:
        expected = {
            20: 0.026041666666666668,
            40: 0.052083333333333336,
            60: 0.078125,
            80: 0.10416666666666667,
            120: 0.15625,
        }
        for channels, ratio in expected.items():
            with self.subTest(channels=channels):
                self.assertAlmostEqual(codec.theoretical_payload_ratio(channels), ratio, places=14)

    def test_exact_payload_bytes_match_theoretical_ratio(self) -> None:
        width, height, channels = 64, 32, 80
        codes = torch.zeros((channels, height // 16, width // 16), dtype=torch.uint8)
        payload = codec.latent_uint8_to_bytes(codes)
        raw_rgb = bytes(3 * width * height)
        self.assertEqual(len(payload), codes.numel())
        self.assertAlmostEqual(len(payload) / len(raw_rgb), codec.theoretical_payload_ratio(channels))


class ModelContractTests(unittest.TestCase):
    def test_forward_shapes_bounds_quantization_and_inactive_zeroes(self) -> None:
        torch.manual_seed(7)
        model = codec.ScalableSemanticCodec(
            max_latent_channels=120,
            active_channels=(20, 40, 60, 80, 120),
            encoder_name="residual",
            base_width=4,
            decoder_residual_blocks=1,
        ).eval()
        images = torch.rand((1, 3, 32, 64), dtype=torch.float32)
        with torch.no_grad():
            output = model(images, active_channels=80)
        latent = output["latent"]
        self.assertEqual(tuple(latent.shape), (1, 120, 2, 4))
        self.assertEqual(tuple(output["reconstructed_rgb"].shape), (1, 3, 32, 64))
        self.assertEqual(tuple(output["segmentation_logits"].shape), (1, 19, 32, 64))
        self.assertGreaterEqual(float(latent.min()), 0.0)
        self.assertLessEqual(float(latent.max()), 1.0)
        self.assertTrue(torch.equal(latent[:, 80:], torch.zeros_like(latent[:, 80:])))
        active_grid_error = (latent[:, :80] * 255.0 - torch.round(latent[:, :80] * 255.0)).abs().max()
        self.assertEqual(float(active_grid_error), 0.0)

    def test_decoder_rejects_non_transmitted_channel_shape(self) -> None:
        model = codec.ScalableSemanticCodec(base_width=4, decoder_residual_blocks=1)
        invalid = torch.zeros((1, 79, 2, 4))
        with self.assertRaises(ValueError):
            model.decode(invalid, (32, 64), active_channels=80)


class LossAndRateTrainingTests(unittest.TestCase):
    def test_ignore_only_crop_has_finite_zero_semantic_losses(self) -> None:
        loss_function = codec.CompositeSemanticCodecLoss(
            class_weights=torch.ones(codec.NUM_CLASSES),
            rgb_l1_weight=0.0,
            rgb_ssim_weight=0.0,
        )
        reconstructed = torch.zeros((1, 3, 8, 16), requires_grad=True)
        logits = torch.randn((1, codec.NUM_CLASSES, 8, 16), requires_grad=True)
        target_rgb = torch.zeros_like(reconstructed)
        target = torch.full((1, 8, 16), codec.IGNORE_INDEX, dtype=torch.long)
        total, components = loss_function(reconstructed, logits, target_rgb, target)
        self.assertTrue(bool(torch.isfinite(total)))
        self.assertEqual(float(components["ce"].detach()), 0.0)
        self.assertEqual(float(components["dice"].detach()), 0.0)
        total.backward()

    def test_training_batch_always_covers_minimum_maximum_and_middle_rate(self) -> None:
        class RecordingModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))
                self.rates: list[int] = []

            def encode_quantized_full(self, images: torch.Tensor) -> torch.Tensor:
                return images.mean(dim=1, keepdim=True) * self.scale

            def decode(
                self, latent: torch.Tensor, output_size: tuple[int, int], active_channels: int
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.rates.append(active_channels)
                batch = latent.shape[0]
                reconstructed = latent.expand(batch, 3, *output_size)
                logits = latent.expand(batch, codec.NUM_CLASSES, *output_size)
                return reconstructed, logits

        class SimpleLoss(torch.nn.Module):
            def forward(self, reconstructed, logits, target_rgb, target_segmentation):
                loss = reconstructed.mean() + logits.mean()
                component = {name: loss for name in ("ce", "dice", "rgb_l1", "rgb_ssim")}
                return loss, component

        model = RecordingModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        images = torch.rand((1, 3, 8, 16))
        labels = torch.zeros((1, 8, 16), dtype=torch.long)
        codec.train_one_epoch(
            model=model,
            loader=[(images, labels)],
            optimizer=optimizer,
            scaler=codec._make_grad_scaler(False),
            loss_function=SimpleLoss(),
            device=torch.device("cpu"),
            active_channels=(20, 40, 60, 80, 120),
            rate_rng=codec.random.Random(123),
            accumulation_steps=1,
            gradient_clip_norm=1.0,
            amp_enabled=False,
            global_step=0,
        )
        self.assertIn(20, model.rates)
        self.assertIn(120, model.rates)
        self.assertEqual(len(set(model.rates).intersection({40, 60, 80})), 1)


class StrictJsonTests(unittest.TestCase):
    def test_writer_outputs_parseable_standard_json(self) -> None:
        payload = {"finite": 1.25, "nested": [True, None, {"text": "한글"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            codec.write_strict_json(path, payload)
            parsed = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(parsed, payload)

    def test_writer_rejects_nan_and_infinity(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    codec.write_strict_json(Path(directory) / "bad.json", {"value": value})


class ProvenanceAndSummaryContractTests(unittest.TestCase):
    def test_airtalking_feature_contract_uses_latent_restore_bits(self) -> None:
        profiles = []
        for channels in codec.DEFAULT_ACTIVE_CHANNELS:
            profiles.append(
                {
                    "active_channels": channels,
                    "operating_point": "paper_like" if channels == 80 else f"rate_{channels}",
                    "measured_rho_uint8_over_raw_rgb": codec.theoretical_payload_ratio(channels),
                    "measured_rho_zlib_over_raw_rgb": codec.theoretical_payload_ratio(channels) * 0.9,
                    "mean_iou": 0.2,
                    "pixel_accuracy": 0.7,
                    "psnr_db": 18.0,
                    "ssim": 0.5,
                }
            )
        result = {
            "multi_rate_profiles": profiles,
            "timing": {
                "encode_including_8bit_fake_quantization": {"median_ms": 1.0},
                "decode_from_latent_only": {"median_ms": 2.0},
            },
            "dataset": {"image_size": {"width": 256, "height": 128}, "val_samples": 5},
            "scientific_scope": "test",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "airtalking.json"
            codec.write_airtalking_summary(path, result)
            summary = json.loads(path.read_text(encoding="utf-8"))
            raw_bits = 256 * 128 * 3 * 8
            expected_decode = raw_bits * codec.theoretical_payload_ratio(80) * 3.0 / 0.002 / 1e6
            self.assertAlmostEqual(summary["feature_decode_bitrate_mbps_median"], expected_decode)
            self.assertAlmostEqual(
                summary["rho_c_uncompressed_mean"], codec.theoretical_payload_ratio(80) * 0.9
            )
            self.assertNotIn("rho_c_png_mean", summary)
            self.assertNotIn("encode_bitrate_mbps_median", summary)
            updated, metadata = reproduction.apply_semantic_summary(
                reproduction.PaperParams(),
                path,
                raw_basis="uncompressed",
                encoder_mode="measured",
                decoder_mode="measured",
                profile_kind="feature",
            )
            self.assertAlmostEqual(updated.rho_c, codec.theoretical_payload_ratio(80))
            self.assertAlmostEqual(updated.dec_bitrate / 1e6, expected_decode)
            self.assertTrue(metadata["applied"])
            with self.assertRaisesRegex(ValueError, "feature"):
                reproduction.apply_semantic_summary(
                    reproduction.PaperParams(),
                    path,
                    raw_basis="uncompressed",
                    encoder_mode="measured",
                    decoder_mode="measured",
                    profile_kind="zlib",
                )

    def test_resume_configuration_mismatch_fails_before_loading_training_state(self) -> None:
        args = codec.build_argument_parser().parse_args([])
        codec.validate_arguments(args, codec.build_argument_parser())
        checkpoint_args = vars(args).copy()
        checkpoint_args["image_width"] = args.image_width // 2
        with self.assertRaisesRegex(ValueError, "image_width"):
            codec.validate_resume_configuration(checkpoint_args, args)

    def test_atomic_checkpoint_round_trip_and_class_weight_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            codec.atomic_torch_save({"model": {"weight": torch.tensor([1.0])}}, path)
            payload = codec.load_torch_checkpoint(path, torch.device("cpu"))
            self.assertEqual(float(payload["model"]["weight"][0]), 1.0)
            self.assertFalse(path.with_name(f"{path.name}.tmp").exists())

        labels_common = torch.zeros((8, 8), dtype=torch.long)
        labels_rare = labels_common.clone()
        labels_rare[0, 0] = 18
        dataset = [(torch.zeros((3, 8, 8)), labels_common), (torch.zeros((3, 8, 8)), labels_rare)]
        weights, _ = codec.compute_class_weights(dataset, clip_min=0.25, clip_max=4.0)
        self.assertGreaterEqual(float(weights.min()), 0.25)
        self.assertLessEqual(float(weights.max()), 4.0)

    def test_checkpoint_model_state_does_not_share_live_module_storage(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        args = codec.build_argument_parser().parse_args([])
        rate_rng = codec.random.Random(7)
        loader_generator = torch.Generator().manual_seed(7)
        checkpoint = codec.make_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=codec._make_grad_scaler(False),
            args=args,
            epoch=1,
            global_step=1,
            best_mean_iou=0.1,
            best_epoch=1,
            best_model_state=codec.cpu_state_dict(model),
            history=[],
            rate_rng=rate_rng,
            train_generator=loader_generator,
        )
        with torch.no_grad():
            model.weight.fill_(2.0)
        self.assertTrue(torch.equal(checkpoint["model"]["weight"], torch.ones((1, 2))))
        self.assertTrue(checkpoint["resume_supported"])


class CliFailureMessageTests(unittest.TestCase):
    def test_cuda_oom_has_actionable_message_and_nonzero_exit(self) -> None:
        error_output = io.StringIO()
        with (
            mock.patch.object(codec, "main", side_effect=torch.OutOfMemoryError("CUDA out of memory")),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            redirect_stderr(error_output),
        ):
            exit_code = codec.cli_entry()
        message = error_output.getvalue()
        self.assertEqual(exit_code, 2)
        self.assertIn("CUDA out of memory", message)
        self.assertIn("--batch-size 1", message)
        self.assertIn("--gradient-accumulation 16", message)


if __name__ == "__main__":
    unittest.main()
