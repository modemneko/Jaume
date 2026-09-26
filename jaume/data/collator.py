import torch


class PairCollator:
    """Tokenizes query/positive pairs into two padded batches."""

    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _encode(self, texts):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"]

    def __call__(self, batch):
        queries = [b["query"] for b in batch]
        positives = [b["positive"] for b in batch]
        q_ids, q_mask = self._encode(queries)
        p_ids, p_mask = self._encode(positives)
        return {
            "query_input_ids": q_ids,
            "query_attention_mask": q_mask,
            "passage_input_ids": p_ids,
            "passage_attention_mask": p_mask,
        }


class EvalPairCollator:
    """Tokenizes (sentence1, sentence2, score) rows for STS evaluation."""

    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts1 = [b["sentence1"] for b in batch]
        texts2 = [b["sentence2"] for b in batch]
        scores = torch.tensor([float(b["score"]) for b in batch], dtype=torch.float32)
        out = {"scores": scores}
        for name, texts in (("sent1", texts1), ("sent2", texts2)):
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            out[f"{name}_input_ids"] = enc["input_ids"]
            out[f"{name}_attention_mask"] = enc["attention_mask"]
        return out
