import math
import random

import pytest
import torch

from minigpt.engine import (
    evaluate,
    load_checkpoint,
    make_optimizer,
    restore_training,
    save_checkpoint,
    seed_all,
    update,
)
from minigpt.model import GPT, GPTConfig, next_token_loss


def tiny_model(dropout=0.0):
    return GPT(GPTConfig(vocab_size=16, context_length=4, n_layer=1, n_head=2,
                         n_embd=12, dropout=dropout))


def draw(sampler):
    ids = torch.randint(16, (3, 5), generator=sampler)
    return ids[:, :-1], ids[:, 1:]


def test_evaluation_counts_chapter_local_targets_and_tails():
    seed_all(7, "cpu")
    model = tiny_model(dropout=0.3)
    # Three windows of lengths 4, 2 and 1, plus one chapter with no transitions.
    documents = [[0, 1, 2, 3, 4, 0, 2], [1, 3], [5]]
    windows = [[0, 1, 2, 3, 4], [4, 0, 2], [1, 3]]
    model.eval()
    with torch.no_grad():
        nll = 0.0
        for window in windows:
            row = torch.tensor([window])
            nll += next_token_loss(model(row[:, :-1]), row[:, 1:]).item() * (len(window) - 1)
    model.train()
    metrics = evaluate(model, documents, batch_size=2)
    assert model.training
    assert metrics["targets"] == 7
    assert metrics["loss"] == pytest.approx(nll / 7)
    assert metrics["perplexity"] == pytest.approx(math.exp(nll / 7))
    with pytest.raises(ValueError, match="transition"):
        evaluate(model, [[1], []], batch_size=1)


def test_cpu_checkpoint_replays_next_adamw_step_with_dropout(tmp_path):
    seed_all(18, "cpu")
    model = tiny_model(dropout=0.2)
    sampler = torch.Generator().manual_seed(19)
    lr = 0.001
    optimizer = make_optimizer(model, lr)
    before = model.token_embedding.weight.detach().clone()
    update(model, optimizer, *draw(sampler))
    assert not torch.equal(before, model.token_embedding.weight)
    checkpoint = tmp_path / "run" / "last.pt"
    save_checkpoint(checkpoint, model, optimizer, 1, sampler, "abc", {"lr": lr})

    expected_random = random.random()
    expected_batch = draw(sampler)
    expected_loss = update(model, optimizer, *expected_batch)
    loaded, payload = load_checkpoint(str(checkpoint), "cpu")
    assert not loaded.training
    assert payload["step"] == 1
    with pytest.raises(ValueError, match="fingerprint"):
        restore_training(payload, loaded, sampler, "wrong")
    with pytest.raises(ValueError, match="original device"):
        restore_training(dict(payload, device="mps"), loaded, sampler, "abc")
    resumed_sampler = torch.Generator()
    resumed_optimizer = restore_training(payload, loaded, resumed_sampler, "abc")
    assert random.random() == expected_random
    resumed_batch = draw(resumed_sampler)
    assert torch.equal(resumed_batch[0], expected_batch[0])
    assert torch.equal(resumed_batch[1], expected_batch[1])
    assert update(loaded, resumed_optimizer, *resumed_batch) == expected_loss
    for name, parameter in model.named_parameters():
        assert torch.equal(loaded.get_parameter(name), parameter), name


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple GPU")
def test_mps_checkpoint_moves_to_cpu_and_restores_rng(tmp_path):
    seed_all(29, "mps")
    model = tiny_model(dropout=0.2).to("mps")
    sampler = torch.Generator().manual_seed(30)
    optimizer = make_optimizer(model, 0.001)
    x, y = draw(sampler)
    assert math.isfinite(update(model, optimizer, x, y))
    checkpoint = tmp_path / "gpu.pt"
    save_checkpoint(checkpoint, model, optimizer, 1, sampler, "abc", {"lr": 0.001})
    expected_draw = torch.randn(8, device="mps").cpu()
    cpu_model, _ = load_checkpoint(checkpoint, "cpu")
    assert cpu_model(x).shape == (3, 4, 16)
    restored, payload = load_checkpoint(checkpoint, "mps")
    optimizer = restore_training(payload, restored, sampler, "abc")
    assert torch.equal(expected_draw, torch.randn(8, device="mps").cpu())
    assert math.isfinite(update(restored, optimizer, *draw(sampler)))
