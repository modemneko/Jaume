import torch
import torch.nn as nn

from .attention import BidirectionalGQA
from .moe import MoEFeedForward, FeedForward


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = BidirectionalGQA(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = MoEFeedForward(config) if config.use_moe else FeedForward(config)
        self.use_moe = config.use_moe

    def forward(self, x, attention_mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        if self.use_moe:
            x, aux_loss = self.mlp(x)
        else:
            x, aux_loss = self.mlp(x), None
        x = residual + x
        return x, aux_loss


class JaumeBackbone(nn.Module):
    """Bidirectional Transformer Encoder.

    Returns final hidden states plus the summed MoE load-balance aux loss.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def _run_layer(self, layer, x, attention_mask):
        if self._gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                layer, x, attention_mask, use_reentrant=False
            )
        return layer(x, attention_mask)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embed_tokens(input_ids)

        aux_total = torch.tensor(0.0, device=input_ids.device)
        for layer in self.layers:
            hidden_states, aux = self._run_layer(layer, hidden_states, attention_mask)
            if aux is not None:
                aux_total = aux_total + aux

        return self.norm(hidden_states), aux_total
