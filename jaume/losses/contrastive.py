import torch
import torch.nn as nn
import torch.nn.functional as F


class MultipleNegativesRankingLoss(nn.Module):
    """InfoNCE over in-batch negatives.

    For each query Qi, all passages Pj (j != i) act as negatives.
    Temperature and scale follow sentence-transformers defaults.
    """

    def __init__(self, scale: float = 20.0, similarity_fct=torch.nn.functional.cosine_similarity):
        super().__init__()
        self.scale = scale
        self.similarity_fct = similarity_fct

    def forward(self, query_embeddings, passage_embeddings):
        scores = self.similarity_fct(
            query_embeddings.unsqueeze(1),
            passage_embeddings.unsqueeze(0),
            dim=-1,
        ) * self.scale
        labels = torch.arange(scores.size(0), device=scores.device)
        return F.cross_entropy(scores, labels)


class MatryoshkaLoss(nn.Module):
    """Wraps any contrastive loss with Matryoshka truncated-dim training."""

    def __init__(self, base_loss, dims):
        super().__init__()
        self.base_loss = base_loss
        self.dims = list(dims)

    def forward(self, query_embeddings, passage_embeddings):
        total = self.base_loss(query_embeddings, passage_embeddings)
        for d in self.dims[1:-1]:
            total = total + self.base_loss(
                query_embeddings[..., :d], passage_embeddings[..., :d]
            )
        return total / len(self.dims)
