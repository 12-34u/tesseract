"""Unit tests for MultiHeadAttention module."""

import pytest
import torch
from models.attention import MultiHeadAttention


def test_attention_shape_preservation():
    """Verify input shape [B, N, D] matches output shape [B, N, D]."""
    batch_size, seq_len, d_model, num_heads = 2, 16, 128, 4
    attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    out = attn(x)

    assert out.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected shape {(batch_size, seq_len, d_model)}, got {out.shape}"


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [8, 16, 32])
def test_attention_various_batch_and_seq_lengths(batch_size, seq_len):
    """Test attention module across multiple batch sizes and sequence lengths."""
    d_model, num_heads = 64, 4
    attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    out = attn(x)

    assert out.shape == (batch_size, seq_len, d_model)


def test_head_configuration_validation():
    """Verify that d_model % num_heads == 0 is enforced during initialization."""
    # Valid setup
    attn = MultiHeadAttention(d_model=128, num_heads=4)
    assert attn.head_dim == 32

    # Invalid setup
    with pytest.raises(ValueError) as exc_info:
        MultiHeadAttention(d_model=128, num_heads=5)

    assert "divisible" in str(exc_info.value)


def test_attention_backward_pass_and_gradients():
    """Verify backward pass populates non-None, finite, non-zero gradients."""
    batch_size, seq_len, d_model, num_heads = 2, 16, 128, 4
    attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

    out = attn(x)
    loss = out.mean()
    loss.backward()

    for name, param in attn.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"
        assert torch.isfinite(
            param.grad
        ).all(), f"Gradient for {name} contains NaN/Inf values"
        assert (
            param.grad != 0
        ).any(), f"Gradient for {name} is all zeroes"


def test_invalid_input_dimensions():
    """Verify that invalid input dimensions raise ValueError."""
    attn = MultiHeadAttention(d_model=128, num_heads=4)

    # Wrong feature dimension D
    x_wrong_dim = torch.randn(2, 16, 64)
    with pytest.raises(ValueError):
        attn(x_wrong_dim)

    # Wrong tensor rank (2D instead of 3D)
    x_2d = torch.randn(16, 128)
    with pytest.raises(ValueError):
        attn(x_2d)
