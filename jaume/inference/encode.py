import os
import json

import torch
import torch.nn.functional as F

from ..modeling.config import JaumeConfig
from ..modeling.embedding import JaumeEmbeddingModel
from ..modeling.decision import JaumeDecisionModel


def _resolve_tokenizer_path(model_path):
    for candidate in (
        os.path.join(model_path, "tokenizer"),
        os.path.join(os.path.dirname(model_path), "jaume_tokenizer"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "jaume_tokenizer"),
    ):
        if os.path.isdir(candidate):
            return candidate
    return None


class JaumeEmbedding:
    """Public inference API: encode() / similarity() / decide()."""

    def __init__(self, model, tokenizer, device="cpu", matryoshka_dims=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.matryoshka_dims = matryoshka_dims

    @classmethod
    def from_pretrained(cls, model_path, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        config = JaumeConfig.load(os.path.join(model_path, "config.json"))
        model = JaumeEmbeddingModel(config)
        state = torch.load(
            os.path.join(model_path, "pytorch_model.bin"),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
        from transformers import AutoTokenizer

        tok_path = _resolve_tokenizer_path(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
        return cls(model, tokenizer, device=device, matryoshka_dims=config.matryoshka_dims)

    def _encode_batch(self, texts, dimensions, batch_size=32):
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)
                if dimensions is None:
                    embeddings, _ = self.model(enc["input_ids"], enc["attention_mask"])
                else:
                    out, _ = self.model.forward_matryoshka(enc["input_ids"], enc["attention_mask"])
                    embeddings = out[dimensions]
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)

    def encode(self, texts, dimensions=None, batch_size=32):
        if isinstance(texts, str):
            return self._encode_batch([texts], dimensions, batch_size)[0]
        return self._encode_batch(list(texts), dimensions, batch_size)

    def similarity(self, queries, documents, batch_size=32):
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(documents, str):
            documents = [documents]
        q = self.encode(queries, batch_size=batch_size)
        d = self.encode(documents, batch_size=batch_size)
        return q @ d.T

    def decide(self, state, questions, decision_model):
        """Runs shared-state multi-question inference.

        `decision_model` is a JaumeDecisionModel configured with matching
        question names; `questions` selects which registered decisions to run.
        """
        self.model.eval()
        enc = self.tokenizer(
            [state], padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(self.device)
        selected = {k: decision_model.decision_types[k] for k in questions}
        decision_model.eval()
        outputs, _ = decision_model(enc["input_ids"], enc["attention_mask"])
        return {k: outputs[k] for k in questions}


def save_pretrained(model, tokenizer, save_path):
    """Saves config + weights + tokenizer into a from_pretrained-ready folder."""
    os.makedirs(save_path, exist_ok=True)
    config: JaumeConfig = model.config
    config.save(os.path.join(save_path, "config.json"))
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    if tokenizer is not None:
        tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))
