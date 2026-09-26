import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MoEFeedForward(nn.Module):
    """Top-K MoE with switch-style load balancing loss.

    The aux loss is returned separately so the embedding loss can add it
    scaled by config.load_balance_weight.
    """

    def __init__(self, config):
        super().__init__()
        self.expert_num = config.expert_num
        self.topk = config.topk
        self.gate = nn.Linear(config.hidden_size, self.expert_num, bias=False)
        self.experts = nn.ModuleList([FeedForward(config) for _ in range(self.expert_num)])

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)

        router_logits = self.gate(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, self.topk, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        for i in range(self.expert_num):
            expert_mask = (topk_indices == i).any(dim=-1)
            if expert_mask.sum() > 0:
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[i](expert_input)
                expert_positions = (topk_indices[expert_mask] == i).nonzero(as_tuple=True)[1]
                expert_weights = topk_probs[expert_mask, expert_positions]
                output[expert_mask] += expert_output * expert_weights.unsqueeze(-1)

        # switch aux loss: encourage uniform token assignment across experts
        mean_probs = router_probs.mean(dim=0)
        aux_loss = self.expert_num * (mean_probs * mean_probs).sum()

        return output.view(batch_size, seq_len, hidden_size), aux_loss
