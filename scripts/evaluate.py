"""STS evaluation: Spearman correlation between cosine similarity and gold score.

Usage:
    python scripts/evaluate.py --model output/350m-stage2/final \
        --data /root/autodl-tmp/data/STSB.jsonl
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaume.inference.encode import JaumeEmbedding


def spearman(x, y):
    def rank(v):
        order = torch.argsort(v)
        ranks = torch.empty_like(v)
        ranks[order] = torch.arange(len(v), dtype=torch.float32)
        return ranks

    rx, ry = rank(x), rank(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    return (rx * ry).sum() / (rx.norm() * ry.norm() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dimensions", type=int, default=None, help="Matryoshka truncation dim")
    ap.add_argument("--baseline", action="store_true", help="also report untrained-model baseline")
    args = ap.parse_args()

    import json
    with open(args.data, "r", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]

    s1 = [r["sentence1"] for r in rows]
    s2 = [r["sentence2"] for r in rows]
    gold = torch.tensor([float(r["score"]) for r in rows])

    model = JaumeEmbedding.from_pretrained(args.model)
    print(f"[eval] {args.model} on {args.data} ({len(rows)} pairs, device={model.device})")

    def run(m, dims=None):
        import torch as t

        e1, e2 = [], []
        with t.no_grad():
            for i in range(0, len(rows), args.batch_size):
                for chunk_list, store in ((s1, e1), (s2, e2)):
                    chunk = chunk_list[i:i + args.batch_size]
                    enc = model.tokenizer(
                        chunk, padding=True, truncation=True,
                        max_length=256, return_tensors="pt",
                    ).to(model.device)
                    if dims is None:
                        emb, _ = m(enc["input_ids"], enc["attention_mask"])
                    else:
                        out, _ = m.forward_matryoshka(enc["input_ids"], enc["attention_mask"])
                        emb = out[dims]
                    store.append(emb.cpu())
        e1 = t.cat(e1)
        e2 = t.cat(e2)
        sims = (e1 * e2).sum(dim=-1)
        return spearman(sims, gold).item()

    result = run(model.model, args.dimensions)
    print(f"[eval] spearman = {result:.4f}"
          + (f" @ {args.dimensions}D" if args.dimensions else f" @ {model.model.config.embedding_dim}D"))

    if args.baseline:
        from jaume.modeling.config import JaumeConfig
        from jaume.modeling.embedding import JaumeEmbeddingModel
        config = JaumeConfig.load(os.path.join(args.model, "config.json"))
        base = JaumeEmbeddingModel(config).to(model.device).eval()
        base_result = run(base, args.dimensions)
        print(f"[eval] untrained baseline spearman = {base_result:.4f}  (delta {result - base_result:+.4f})")


if __name__ == "__main__":
    main()
