"""Tests for PTQ W8A8 calibrator."""

import pytest
import torch
import torch.nn as nn

from scripts.ptq.calibrator_w8a8 import (
    CalibrationSample,
    FlashVSRTQDataset,
    ActivationCollector,
    run_calibration,
    save_calibration_cache,
)


class TestCalibrationSampleDataclass:
    """Test CalibrationSample dataclass."""

    def test_creates_calibration_sample(self):
        """Creates CalibrationSample, checks shapes."""
        latents = torch.randn(16, 24, 24, dtype=torch.bfloat16)
        timesteps = torch.tensor([500], dtype=torch.int64)
        contexts = torch.randn(10, 4096, dtype=torch.bfloat16)

        sample = CalibrationSample(
            latents=latents,
            timesteps=timesteps,
            contexts=contexts,
        )

        assert sample.latents.shape == (16, 24, 24)
        assert sample.timesteps.shape == (1,)
        assert sample.contexts.shape == (10, 4096)
        assert sample.latents.dtype == torch.bfloat16
        assert sample.contexts.dtype == torch.bfloat16
        assert sample.timesteps.dtype == torch.int64


class TestFlashVSRTQDataset:
    """Test FlashVSRTQDataset."""

    def test_dataset_returns_calibration_sample(self):
        """Creates dataset, gets sample, checks shapes."""
        dataset = FlashVSRTQDataset(
            root="datasets",
            num_samples=10,
            frame_size=(24, 24),
        )

        assert len(dataset) == 10

        sample = dataset[0]

        assert isinstance(sample, CalibrationSample)
        assert sample.latents.shape == (16, 24, 24)
        assert sample.timesteps.shape == (1,)
        assert sample.contexts.shape == (10, 4096)


class TestActivationCollector:
    """Test ActivationCollector."""

    def test_activation_collector_hooks_linear(self):
        """Creates Sequential of 2 Linear layers, hooks, forward, checks stats."""
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.Linear(256, 128),
        )

        collector = ActivationCollector(model)
        collector.register_hooks()

        # Forward pass
        x = torch.randn(4, 128)
        model(x)

        collector.remove_hooks()

        # Check stats collected
        assert "0" in collector.act_stats
        assert "1" in collector.act_stats

        stats_0 = collector.act_stats["0"]
        stats_1 = collector.act_stats["1"]

        assert "min" in stats_0
        assert "max" in stats_0
        assert "min" in stats_1
        assert "max" in stats_1

        # Each should have one min/max from the single forward pass
        assert len(stats_0["min"]) == 1
        assert len(stats_0["max"]) == 1

    def test_activation_collector_conv3d(self):
        """Creates Sequential of 2 Conv3d layers, hooks, forward (input (2,16,4,24,24)), checks stats."""
        model = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
        )

        collector = ActivationCollector(model)
        collector.register_hooks()

        # Forward pass: (batch, channels, depth, height, width)
        x = torch.randn(2, 16, 4, 24, 24)
        model(x)

        collector.remove_hooks()

        # Check stats collected
        assert "0" in collector.act_stats
        assert "1" in collector.act_stats

        stats_0 = collector.act_stats["0"]
        stats_1 = collector.act_stats["1"]

        assert "min" in stats_0
        assert "max" in stats_0

        # Conv3d output shape: (2, 32, 4, 24, 24) after first layer
        # amax/amin computed over spatial dims (dim 1 onward)
        assert len(stats_0["min"]) == 1
        assert len(stats_0["max"]) == 1

    def test_compute_scales(self):
        """Test scale computation from collected stats."""
        model = nn.Sequential(nn.Linear(64, 128))
        collector = ActivationCollector(model)
        collector.register_hooks()

        x = torch.randn(2, 64)
        model(x)

        scales = collector.compute_scales()

        collector.remove_hooks()

        assert "0" in scales
        assert "act_min" in scales["0"]
        assert "act_max" in scales["0"]
        assert "act_scale" in scales["0"]
        assert "zero_point" in scales["0"]

        # act_scale should be (max - min) / 255
        act_min = scales["0"]["act_min"]
        act_max = scales["0"]["act_max"]
        act_scale = scales["0"]["act_scale"]

        expected_scale = (act_max - act_min) / 255.0
        assert abs(act_scale - expected_scale) < 1e-5


class TestSaveCalibrationCache:
    """Test calibration cache saving."""

    def test_save_calibration_cache(self, tmp_path):
        """Saves cache to JSON, verifies content."""
        scales = {
            "layer0": {
                "act_min": 0.0,
                "act_max": 1.0,
                "act_scale": 0.00392156862745098,
                "zero_point": 0,
            },
            "layer1": {
                "act_min": -0.5,
                "act_max": 0.5,
                "act_scale": 0.00392156862745098,
                "zero_point": 128,
            },
        }

        output_path = tmp_path / "calibration.json"
        save_calibration_cache(scales, str(output_path))

        import json

        with open(output_path) as f:
            loaded = json.load(f)

        assert "layer0" in loaded
        assert "layer1" in loaded
        assert loaded["layer0"]["act_scale"] == pytest.approx(0.00392156862745098, abs=1e-9)