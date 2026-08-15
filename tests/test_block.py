"""Unit tests for TransformerBlock module and weight-sharing verification."""

import pytest
import torch
from models.block import TransformerBlock
from utils.param_count import count_parameters


def test_block_shape_preservation():
    """Verify input shape [B, N, D] matches output shape [B, N, D]."""
    batch_size, seq_len, d_model, num_heads, d_ff = 2, 16, 128, 4, 512
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    x = torch.randn(batch_size, seq_len, d_model)

    out = block(x)

    assert out.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected shape {(batch_size, seq_len, d_model)}, got {out.shape}"


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [8, 16, 32, 64])
def test_block_various_batch_and_seq_lengths(batch_size, seq_len):
    """Test TransformerBlock across multiple batch sizes and sequence lengths."""
    d_model, num_heads, d_ff = 64, 4, 256
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    x = torch.randn(batch_size, seq_len, d_model)

    out = block(x)

    assert out.shape == (batch_size, seq_len, d_model)


def test_block_forward_no_nans_or_infs():
    """Verify forward pass output contains no NaNs or Infinities."""
    block = TransformerBlock(d_model=128, num_heads=4, d_ff=512)
    x = torch.randn(2, 16, 128)

    out = block(x)

    assert torch.isfinite(out).all(), "Output tensor contains NaNs or Infinities"


def test_block_single_backward_pass():
    """Verify single backward pass populates valid gradients for all parameters."""
    block = TransformerBlock(d_model=128, num_heads=4, d_ff=512)
    x = torch.randn(2, 16, 128, requires_grad=True)

    out = block(x)
    loss = out.mean()
    loss.backward()

    for name, param in block.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(
            param.grad
        ).all(), f"Gradient for {name} contains NaN/Inf"
        assert (
            param.grad != 0
        ).any(), f"Gradient for {name} is all zeroes"


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_block_k_simulation(k):
    """Test repeated execution of the SAME block over K iterations."""
    batch_size, seq_len, d_model = 2, 16, 128
    block = TransformerBlock(d_model=d_model, num_heads=4, d_ff=512)
    x = torch.randn(batch_size, seq_len, d_model)

    # Store initial parameter data pointers
    initial_ptrs = [p.data_ptr() for p in block.parameters()]
    initial_param_count = sum(p.numel() for p in block.parameters())

    # Repeated block application
    for step in range(k):
        x = block(x)

    # Post-execution verification
    current_ptrs = [p.data_ptr() for p in block.parameters()]
    current_param_count = sum(p.numel() for p in block.parameters())

    assert x.shape == (batch_size, seq_len, d_model)
    assert torch.isfinite(x).all(), f"K={k} simulation produced NaNs or Infinities"
    assert (
        initial_param_count == current_param_count
    ), f"Parameter count changed during K={k} simulation"
    assert (
        initial_ptrs == current_ptrs
    ), f"Parameter memory storage pointers changed during K={k} simulation"


def test_weight_sharing_and_storage_identity():
    """Explicit test proving identical parameter storage pointers during repeated calls."""
    block = TransformerBlock(d_model=128, num_heads=4, d_ff=512)
    x = torch.randn(2, 8, 128)

    # 1. Record parameter identities and data pointers before execution
    param_instances_before = [id(p) for p in block.parameters()]
    data_ptrs_before = [p.data_ptr() for p in block.parameters()]
    total_params_before = sum(p.numel() for p in block.parameters())

    # 2. Execute block repeatedly over K=4 steps
    for _ in range(4):
        x = block(x)

    # 3. Record parameter identities and data pointers after execution
    param_instances_after = [id(p) for p in block.parameters()]
    data_ptrs_after = [p.data_ptr() for p in block.parameters()]
    total_params_after = sum(p.numel() for p in block.parameters())

    # 4. Assert strict weight sharing & storage identity
    assert (
        param_instances_before == param_instances_after
    ), "Module parameter object IDs changed"
    assert (
        data_ptrs_before == data_ptrs_after
    ), "Parameter tensor storage memory pointers (data_ptr) changed"
    assert (
        total_params_before == total_params_after
    ), "Parameter count changed after repeated execution"


def test_gradient_flow_through_repeated_applications():
    """Verify autograd backpropagates gradients through K=4 repeated block applications."""
    block = TransformerBlock(d_model=128, num_heads=4, d_ff=512)
    x = torch.randn(2, 16, 128, requires_grad=True)

    # Unroll 4 steps with the same block instance
    out = x
    for _ in range(4):
        out = block(out)

    loss = out.mean()
    loss.backward()

    # Verify input tensor gradients
    assert x.grad is not None, "Input tensor gradient is None after K=4 BPTT"
    assert torch.isfinite(x.grad).all(), "Input tensor gradient has NaNs/Infs"

    # Verify shared block parameter gradients
    for name, param in block.named_parameters():
        assert (
            param.grad is not None
        ), f"Parameter {name} gradient is None after K=4 BPTT"
        assert torch.isfinite(
            param.grad
        ).all(), f"Parameter {name} gradient has NaNs/Infs after K=4 BPTT"
        assert (
            param.grad != 0
        ).any(), f"Parameter {name} gradient is all zeroes after K=4 BPTT"


def test_block_parameter_count_measurement():
    """Verify expected parameter count (~198K) for prototype small block configuration."""
    block = TransformerBlock(d_model=128, num_heads=4, d_ff=512)
    counts = count_parameters(block)

    # LN1 (256) + MHA (66,048) + LN2 (256) + FFN (131,712) = 198,272
    expected_count = 198272

    assert (
        counts["trainable"] == expected_count
    ), f"Expected {expected_count} trainable parameters, got {counts['trainable']}"
    assert (
        counts["total"] == expected_count
    ), f"Expected {expected_count} total parameters, got {counts['total']}"


def test_invalid_input_dimensions():
    """Verify passing invalid input tensor shapes raises ValueError."""
    block = TransformerBlock(d_model=128, num_heads=4, d_ff=512)

    # Wrong feature dimension D
    with pytest.raises(ValueError):
        block(torch.randn(2, 16, 64))

    # Wrong tensor rank (2D instead of 3D)
    with pytest.raises(ValueError):
        block(torch.randn(16, 128))

