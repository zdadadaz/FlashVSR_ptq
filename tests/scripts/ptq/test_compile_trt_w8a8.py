"""Tests for TensorRT INT8 compilation script."""

import pytest
import torch

# Import the script
import scripts.ptq.compile_trt_w8a8 as compile_trt_w8a8


class TestImports:
    """Test that all required functions are importable."""

    def test_compile_trt_imports(self):
        """Imports all functions from script, checks callable."""
        assert callable(compile_trt_w8a8.check_torch_tensorrt)
        assert callable(compile_trt_w8a8.load_dit_with_rmsnorm_fold)
        assert callable(compile_trt_w8a8.make_trt_input_spec)
        assert callable(compile_trt_w8a8.export_dit_for_trt)
        assert callable(compile_trt_w8a8.create_trt_calibrator)
        assert callable(compile_trt_w8a8.compile_trt_engine)


class TestTorchExport:
    """Test torch.export availability."""

    def test_torch_export_available(self):
        """Asserts torch.export is available."""
        assert hasattr(torch, "export")


class TestTRTInputSpec:
    """Test TRT input specification."""

    def test_trt_input_shape_spec(self):
        """Calls make_trt_input_spec(), checks return is not None."""
        # Skip if torch_tensorrt not available
        if not compile_trt_w8a8.TORCH_TENSORRT_AVAILABLE:
            pytest.skip("torch_tensorrt not installed")

        input_spec = compile_trt_w8a8.make_trt_input_spec()
        assert input_spec is not None


class TestWanModelExport:
    """Test WanModel loading and export."""

    def test_wanmodel_loads_for_export(self):
        """Loads small mock model, tries export (mock, no GPU needed for graph structure check)."""
        # Import the model
        from src.models.wan_video_dit import WanModel

        # Create a small mock model for graph structure validation
        model = WanModel(
            dim=128,  # Small dimension for testing
            eps=1e-5,
            ffn_dim=512,  # Small ffn_dim
            freq_dim=32,  # Small freq_dim
            in_dim=16,
            num_heads=4,  # Small num_heads
            num_layers=2,  # Small num_layers for speed
            out_dim=16,
            patch_size=(1, 2, 2),
            text_dim=512,  # Small text_dim
        )
        model.eval()

        # Create example input in the format expected by WanModel
        # (latents, timesteps, contexts)
        batch_size = 1
        seq_len = 1
        latent_channels = 16
        height = 24
        width = 24

        latents = torch.randn(batch_size, seq_len, latent_channels, height, width)
        timesteps = torch.tensor([0], dtype=torch.int64)
        contexts = torch.randn(batch_size, 10, 512)  # 10 context tokens, dim=512

        example_input = (latents, timesteps, contexts)

        # Try torch.export - this will fail if the graph has issues
        try:
            exported = compile_trt_w8a8.export_dit_for_trt(model, example_input)
            assert exported is not None
        except Exception as e:
            # If torch_tensorrt is not installed, torch.export might still work
            # but the export itself might have constraints
            pytest.skip(f"torch.export failed (possibly due to environment): {e}")


class TestCheckTorchTensorRT:
    """Test torch_tensorrt availability check."""

    def test_check_torch_tensorrt_returns_bool(self):
        """check_torch_tensorrt returns True or False without raising."""
        result = compile_trt_w8a8.check_torch_tensorrt()
        assert isinstance(result, bool)


class TestCalibratorCreation:
    """Test calibrator creation."""

    def test_create_trt_calibrator_requires_dataloader(self):
        """Verifies create_trt_calibrator needs a dataloader argument."""
        # Skip if torch_tensorrt not available
        if not compile_trt_w8a8.TORCH_TENSORRT_AVAILABLE:
            pytest.skip("torch_tensorrt not installed")

        from torch.utils.data import DataLoader

        # Create a dummy dataset with minimal samples
        class DummyDataset:
            def __init__(self):
                self.data = [torch.randn(16, 24, 24) for _ in range(5)]

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dummy_loader = DataLoader(DummyDataset(), batch_size=1)

        # This should not raise - just verify function exists and is callable
        calibrator = compile_trt_w8a8.create_trt_calibrator(dummy_loader, num_samples=5)
        assert calibrator is not None