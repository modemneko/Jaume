"""Stage 2 embedding training: contrastive (MNRL) with Matryoshka support.

Usage:
    python scripts/train_embedding.py \
        --config configs/jaume-smoke.json \
        --train-data /root/autodl-tmp/data/stsb.jsonl \
        --output output/smoke \
        --epochs 1 --batch-size 32 --grad-accum 4
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaume.modeling.config import JaumeConfig
from jaume.modeling.embedding import JaumeEmbeddingModel
from jaume.losses.contrastive import MultipleNegativesRankingLoss, MatryoshkaLoss
from jaume.data.dataset import PairDataset, load_tokenizer
from jaume.data.collator import PairCollator
from jaume.inference.encode import save_pretrained


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", default=None, help="defaults to <repo>/jaume_tokenizer")
    p.add_argument("--train-data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--use-matryoshka", action="store_true")
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    config = JaumeConfig.load(args.config)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tokenizer_path = args.tokenizer or os.path.join(repo_root, "jaume_tokenizer")
    tokenizer = load_tokenizer(tokenizer_path)

    model = JaumeEmbeddingModel(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[jaume] params: {n_params/1e6:.1f}M  device: {device}")
    print(f"[jaume] config: hidden={config.hidden_size} layers={config.num_layers} "
          f"experts={config.expert_num}x top{config.topk} emb={config.embedding_dim}D "
          f"pooling={config.pooling}")

    base_loss = MultipleNegativesRankingLoss()
    if args.use_matryoshka:
        loss_fn = MatryoshkaLoss(base_loss, config.matryoshka_dims)
        print(f"[jaume] matryoshka dims: {config.matryoshka_dims}")
    else:
        loss_fn = base_loss

    dataset = PairDataset(args.train_data, max_samples=args.max_samples)
    collator = PairCollator(tokenizer, max_length=args.max_length)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=args.num_workers, drop_last=True,
    )
    print(f"[jaume] training pairs: {len(dataset)}  steps/epoch: {len(loader)//args.grad_accum}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.output, exist_ok=True)
    global_step = 0
    accum = 0
    t0 = time.time()
    running = 0.0
    if args.grad_ckpt:
        model.backbone.gradient_checkpointing_enable()
        print("[jaume] gradient checkpointing enabled")
    model.train()

    for epoch in range(args.epochs):
        for batch in loader:
            input_ids = batch["query_input_ids"].to(device)
            mask = batch["query_attention_mask"].to(device)
            p_ids = batch["passage_input_ids"].to(device)
            p_mask = batch["passage_attention_mask"].to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                if args.use_matryoshka:
                    q_emb, q_aux = model.forward_matryoshka(input_ids, mask)
                    p_emb, _ = model.forward_matryoshka(p_ids, p_mask)
                    loss = loss_fn(q_emb[config.embedding_dim], p_emb[config.embedding_dim])
                else:
                    q_emb, q_aux = model(input_ids, mask)
                    p_emb, _ = model(p_ids, p_mask)
                    loss = loss_fn(q_emb, p_emb)
                loss = loss + config.load_balance_weight * q_aux

            scaler.scale(loss / args.grad_accum).backward()
            running += loss.item()
            accum += 1

            if accum == args.grad_accum:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                accum = 0
                global_step += 1

                if global_step % args.log_every == 0:
                    speed = global_step * args.grad_accum * args.batch_size / (time.time() - t0)
                    print(f"[jaume] epoch {epoch} step {global_step} "
                          f"loss {running / (args.log_every * args.grad_accum):.4f} "
                          f"lr {scheduler.get_last_lr()[0]:.2e} pairs/s {speed:.0f}")
                    running = 0.0

                if args.save_every and global_step % args.save_every == 0:
                    save_pretrained(model, tokenizer, os.path.join(args.output, f"step-{global_step}"))

    save_pretrained(model, tokenizer, os.path.join(args.output, "final"))
    print(f"[jaume] done in {time.time()-t0:.0f}s, saved to {args.output}/final")


if __name__ == "__main__":
    main()
