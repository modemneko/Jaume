import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import JaumeBackbone
from .pooling import build_pooling


class ChoiceHead(nn.Module):
    """Fixed-size option classification head (v0.3)."""

    def __init__(self, hidden_size, max_options, num_options):
        super().__init__()
        self.max_options = max_options
        self.num_options = num_options
        self.proj = nn.Linear(hidden_size, max_options, bias=False)

    def forward(self, pooled):
        logits = self.proj(pooled)
        logits[..., self.num_options:] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        probs = probs[..., : self.num_options]
        choice = probs.argmax(dim=-1)
        return {"choice": choice, "probabilities": probs}


class ScoreHead(nn.Module):
    """Bounded score prediction: sigmoid in [0, 1], multiplied by levels."""

    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, pooled, levels=5):
        value = torch.sigmoid(self.proj(pooled)).squeeze(-1)
        return {"score": value * levels, "confidence": value}


class NoulHead(nn.Module):
    """Binary/continuous probability judgment."""

    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, pooled):
        probability = torch.sigmoid(self.proj(pooled)).squeeze(-1)
        return {"probability": probability, "confidence": probability}


class JaumeDecisionModel(nn.Module):
    """Encoder + shared representation + Choice / Score / Noul heads.

    All heads read the same pooled state, so any number of decisions shares
    one encoder forward pass.
    """

    def __init__(self, config, decisions: dict):
        super().__init__()
        self.config = config
        self.backbone = JaumeBackbone(config)
        self.pooling = build_pooling(config.pooling, config.hidden_size)

        self.decision_types = dict(decisions)
        self.heads = nn.ModuleDict()
        self.head_levels = {}
        for name, spec in decisions.items():
            kind = spec["type"]
            if kind == "choice":
                options = spec["options"]
                self.head_levels[name] = options
                self.heads[name.replace("-", "_")] = ChoiceHead(config.hidden_size, len(options), len(options))
            elif kind == "score":
                self.head_levels[name] = spec.get("levels", 5)
                self.heads[name.replace("-", "_")] = ScoreHead(config.hidden_size)
            elif kind == "noul":
                self.heads[name.replace("-", "_")] = NoulHead(config.hidden_size)
            else:
                raise ValueError(f"unknown decision type: {kind}")

    def forward(self, input_ids, attention_mask=None):
        hidden_states, aux_loss = self.backbone(input_ids, attention_mask)
        pooled = self.pooling(hidden_states, attention_mask)

        outputs = {}
        for name, head_key in zip(self.decision_types, self.heads):
            kind = self.decision_types[name]["type"]
            if kind == "choice":
                result = self.heads[head_key](pooled)
                result["choice"] = [
                    self.head_levels[name][i] for i in result["choice"].tolist()
                ]
                result["probabilities"] = {
                    opt: p for opt, p in zip(
                        self.head_levels[name],
                        result["probabilities"][0].tolist(),
                    )
                }
            elif kind == "score":
                result = self.heads[head_key](pooled, levels=self.head_levels[name])
                result["score"] = result["score"].tolist()
                result["confidence"] = result["confidence"].tolist()
            else:
                result = self.heads[head_key](pooled)
                result["probability"] = result["probability"].tolist()
                result["confidence"] = result["confidence"].tolist()
            outputs[name] = result
        return outputs, aux_loss
