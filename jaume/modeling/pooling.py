import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    def forward(self, hidden_states, attention_mask):
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class CLSPooling(nn.Module):
    def forward(self, hidden_states, attention_mask):
        return hidden_states[:, 0, :]


class AttentionPooling(nn.Module):
    """Single learned query attending over all (non-pad) tokens."""

    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.scale = hidden_size ** -0.5

    def forward(self, hidden_states, attention_mask):
        scores = (hidden_states @ self.query.transpose(-2, -1)).squeeze(-1) * self.scale
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


def build_pooling(name, hidden_size):
    if name == "mean":
        return MeanPooling()
    if name == "cls":
        return CLSPooling()
    if name == "attention":
        return AttentionPooling(hidden_size)
    if name == "last":
        return LastPooling()
    raise ValueError(f"unknown pooling: {name}")


class LastPooling(nn.Module):
    def forward(self, hidden_states, attention_mask):
        if attention_mask is None:
            return hidden_states[:, -1, :]
        lengths = (attention_mask.sum(dim=1) - 1).long()
        return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), lengths]
