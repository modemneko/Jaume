import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rope(head_dim, seq_len, theta, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    # x: [B, H, T, D]; cos/sin: [T, D]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


class BidirectionalGQA(nn.Module):
    """GQA attention with no causal mask: every token sees the full sequence."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def repeat_kv(self, x, n_rep):
        batch_size, n_kv_h, seq_len, head_dim = x.shape
        if n_rep == 1:
            return x
        return x[:, :, None, :, :].expand(batch_size, n_kv_h, n_rep, seq_len, head_dim).reshape(
            batch_size, n_kv_h * n_rep, seq_len, head_dim
        )

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope(self.head_dim, seq_len, 1000000.0, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = self.repeat_kv(k, self.num_heads // self.num_key_value_heads)
        v = self.repeat_kv(v, self.num_heads // self.num_key_value_heads)

        # padding mask: [B, T] -> [B, 1, 1, T]; no causal component
        attn_mask = None
        if attention_mask is not None:
            attn_mask = (1.0 - attention_mask[:, None, None, :]) * torch.finfo(x.dtype).min

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(output)
