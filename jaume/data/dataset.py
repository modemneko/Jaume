import json

import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """Reads {query, positive} (optionally {hard_negative}) pairs from jsonl."""

    def __init__(self, path, max_samples=None):
        self.pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                query, positive = None, None
                if "query" in row and "positive" in row:
                    query, positive = row["query"], row["positive"]
                elif "sentence1" in row and "sentence2" in row:
                    # STS-style rows: high similarity = positive pair
                    if float(row.get("score", 1.0)) >= 0.75:
                        query, positive = row["sentence1"], row["sentence2"]
                elif "anchor" in row and "positive" in row:
                    query, positive = row["anchor"], row["positive"]
                elif "premise" in row and "hypothesis" in row:
                    if int(row.get("label", 1)) in (0, 1):  # entailment / neutral->skip
                        if int(row["label"]) == 0:
                            query, positive = row["premise"], row["hypothesis"]
                if query and positive and query.strip() and positive.strip():
                    self.pairs.append({"query": query, "positive": positive})
                if max_samples and len(self.pairs) >= max_samples:
                    break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def load_tokenizer(path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
