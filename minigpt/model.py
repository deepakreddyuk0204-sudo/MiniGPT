"""A decoder-only Transformer. All feature dimensions are explicit in the code."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 256
    context_length: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 160
    dropout: float = 0.0

    def __post_init__(self):
        for name in ("vocab_size", "context_length", "n_layer", "n_head", "n_embd"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must divide evenly into n_head heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class LayerNorm(nn.Module):
    """Normalize the features of each token independently, then scale and shift."""

    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True)
        return self.weight * (centered * torch.rsqrt(variance + self.eps)) + self.bias


class CausalSelfAttention(nn.Module):
    """Each head attends only to the current token and its predecessors."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_width = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.projection = nn.Linear(config.n_embd, config.n_embd)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        mask = torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        self.register_buffer("causal_mask", mask.tril(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        # [B, T, 3*C] -> three tensors of shape [B, heads, T, C/heads].
        projected = self.qkv(x).reshape(batch, length, 3, self.n_head, self.head_width)
        query, key, value = projected.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_width)
        scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        weights = self.attention_dropout(F.softmax(scores, dim=-1))
        heads = weights @ value
        joined = heads.transpose(1, 2).reshape(batch, length, width)
        return self.output_dropout(self.projection(joined))


class DecoderBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        width = config.n_embd
        self.attention_norm = LayerNorm(width)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    """Map token IDs [batch, time] to next-token logits [batch, time, vocab]."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.context_length, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(DecoderBlock(config) for _ in range(config.n_layer)))
        self.final_norm = LayerNorm(config.n_embd)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or ids.size(0) == 0 or ids.size(1) == 0:
            raise ValueError("ids must be a non-empty [batch, time] tensor")
        length = ids.size(1)
        if length > self.config.context_length:
            raise ValueError("input length exceeds the configured context_length")
        positions = torch.arange(length, device=ids.device)
        x = self.embedding_dropout(self.token_embedding(ids) + self.position_embedding(positions))
        x = self.final_norm(self.blocks(x))
        # The same token vectors used on input serve as the output classifier.
        return F.linear(x, self.token_embedding.weight)


def next_token_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average cross-entropy in nats per token. The caller supplies shifted targets."""
    if logits.ndim != 3 or logits.shape[:2] != targets.shape:
        raise ValueError("expected logits [batch, time, vocab] and targets [batch, time]")
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
