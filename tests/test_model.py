import math

import pytest
import torch
from torch.nn import functional as F

from minigpt.model import CausalSelfAttention, GPT, GPTConfig, LayerNorm, next_token_loss


def small_config():
    return GPTConfig(vocab_size=13, context_length=6, n_layer=2, n_head=2, n_embd=8)


def test_output_shapes_backward_and_default_parameter_count():
    model = GPT(small_config())
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    targets = torch.tensor([[2, 3, 4], [5, 6, 7]])
    logits = model(ids)
    assert logits.shape == (2, 3, 13)
    loss = next_token_loss(logits, targets)
    assert loss.ndim == 0
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert sum(p.numel() for p in GPT(GPTConfig()).parameters()) == 1_298_880


def test_mask_gives_exact_prefix_averages_in_each_head():
    config = small_config()
    attention = CausalSelfAttention(config)
    width = config.n_embd
    with torch.no_grad():
        # Zero Q and K make the softmax uniform over just the visible positions.
        attention.qkv.weight.zero_()
        attention.qkv.weight[2 * width :].copy_(torch.eye(width))
        attention.qkv.bias.zero_()
        attention.projection.weight.copy_(torch.eye(width))
        attention.projection.bias.zero_()
    x = torch.arange(4 * width).float().reshape(1, 4, width)
    expected = x.cumsum(dim=1) / torch.arange(1, 5).reshape(1, 4, 1)
    torch.testing.assert_close(attention(x), expected)
    assert "causal_mask" not in attention.state_dict()


def test_future_tokens_cannot_change_prefix_logits():
    model = GPT(small_config()).eval()
    x = torch.tensor([[1, 2, 3, 4, 5, 6]])
    y = torch.tensor([[1, 2, 3, 8, 9, 10]])
    torch.testing.assert_close(model(x)[:, :3], model(y)[:, :3])


def test_layer_norm_matches_reference_with_learned_affine_parameters():
    norm = LayerNorm(4)
    x = torch.tensor([[[1.0, 2.0, 0.0, 4.0], [3.0, 3.0, 3.0, 3.0]]])
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([2.0, 1.0, 0.5, -1.0]))
        norm.bias.fill_(0.2)
    expected = F.layer_norm(x, (4,), norm.weight, norm.bias, norm.eps)
    torch.testing.assert_close(norm(x), expected)


def test_loss_matches_hand_calculated_next_token_probabilities():
    probabilities = torch.tensor([[[0.8, 0.2], [0.3, 0.7]]])
    targets = torch.tensor([[0, 1]])
    loss = next_token_loss(probabilities.log(), targets)
    assert loss.item() == pytest.approx(-(math.log(0.8) + math.log(0.7)) / 2)


def test_invalid_context_and_dimensions_fail_early():
    with pytest.raises(ValueError, match="divide"):
        GPTConfig(n_embd=10, n_head=4)
    model = GPT(small_config())
    with pytest.raises(ValueError, match="context_length"):
        model(torch.ones(1, 7, dtype=torch.long))
    with pytest.raises(ValueError, match="non-empty"):
        model(torch.empty(1, 0, dtype=torch.long))
