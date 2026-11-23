import pytest
import torch

from minigpt.generation import generate
from minigpt.model import GPT, GPTConfig


@pytest.fixture
def model():
    return GPT(GPTConfig(vocab_size=11, context_length=4, n_layer=1, n_head=2, n_embd=8, dropout=0.2))


def test_sampling_is_seeded_and_preserves_prompt_mode_and_global_rng(model):
    ids = torch.tensor([[1, 2], [4, 3]])
    rng_before = torch.random.get_rng_state()
    result = generate(model, ids, 12, temperature=0.9, top_k=4, seed=17)
    assert model.training
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
    torch.testing.assert_close(generate(model, ids, 12, temperature=0.9, top_k=4, seed=17), result)
    assert result.shape == (2, 14)
    torch.testing.assert_close(result[:, :2], ids)
    with torch.no_grad():
        model.eval()
        for t in range(2, result.shape[1]):
            prefix = result[:, max(0, t - model.config.context_length) : t]
            options = model(prefix)[:, -1].topk(4).indices
            assert (options == result[:, t : t + 1]).any(dim=1).all()


def test_long_prompt_uses_last_window_and_retains_entire_input(model):
    model.eval()
    prompt = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
    with torch.no_grad():
        expected = model(prompt[:, -4:])[:, -1].argmax(dim=-1)
    output = generate(model, prompt, 1, temperature=0.0)
    torch.testing.assert_close(output[:, :-1], prompt)
    torch.testing.assert_close(output[:, -1], expected)
    torch.testing.assert_close(generate(model, prompt, 1, top_k=1), output)
    assert not model.training
    torch.testing.assert_close(generate(model, prompt, 0), prompt)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires an Apple GPU")
def test_sampling_on_mps_returns_ids_on_mps(model):
    model.to("mps")
    ids = torch.tensor([[1, 2]], device="mps")
    result = generate(model, ids, 3, top_k=2)
    assert result.device == ids.device
    assert result.shape == (1, 5)


@pytest.mark.parametrize("options", [
    {"max_new_tokens": -1}, {"max_new_tokens": 1, "temperature": float("nan")},
    {"max_new_tokens": 1, "top_k": 12}, {"max_new_tokens": 1, "top_k": 0},
])
def test_invalid_controls(model, options):
    with pytest.raises(ValueError):
        generate(model, torch.tensor([[1]]), **options)
