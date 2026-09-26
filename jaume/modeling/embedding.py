import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import JaumeBackbone
from .pooling import build_pooling


class JaumeEmbeddingModel(nn.Module):
    """Backbone + Pooling + Projection + L2 normalize.

    forward() returns (embeddings [B, embedding_dim], aux_loss).
    Embeddings are unit-normalized so cosine similarity == dot product.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = JaumeBackbone(config)
        self.pooling = build_pooling(config.pooling, config.hidden_size)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.embedding_dim, bias=False),
            nn.LayerNorm(config.embedding_dim),
        )

    def forward(self, input_ids, attention_mask=None):
        hidden_states, aux_loss = self.backbone(input_ids, attention_mask)
        pooled = self.pooling(hidden_states, attention_mask)
        embeddings = self.projection(pooled)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings, aux_loss

    def forward_matryoshka(self, input_ids, attention_mask=None):
        """Returns {dim: embeddings} for every configured Matryoshka dim."""
        embeddings, aux_loss = self.forward(input_ids, attention_mask)
        out = {self.config.embedding_dim: embeddings}
        for d in self.config.matryoshka_dims:
            if d < self.config.embedding_dim:
                out[d] = F.normalize(embeddings[..., :d], p=2, dim=-1)
        return out, aux_loss
