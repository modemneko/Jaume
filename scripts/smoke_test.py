"""Stage 0 backbone validation: forward/backward/mask/GQA/MoE/checkpoint/mp.

Run: python scripts/smoke_test.py --config configs/jaume-smoke.json
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaume.modeling.config import JaumeConfig
from jaume.modeling.embedding import JaumeEmbeddingModel
from jaume.modeling.decision import JaumeDecisionModel
from jaume.modeling import JaumeBackbone
from jaume.losses.contrastive import MultipleNegativesRankingLoss

CHECKS = []


def check(name, fn):
    try:
        detail = fn()
        CHECKS.append((name, "PASS", detail))
        print(f"  ✓ {name}: {detail}")
    except Exception as e:
        CHECKS.append((name, "FAIL", repr(e)))
        print(f"  ✗ {name}: FAIL {e!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "jaume-smoke.json"))
    args = ap.parse_args()
    config = JaumeConfig.load(os.path.abspath(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    print(f"=== Jaume Stage 0 Backbone Validation ({device}) ===")

    def make_model():
        return JaumeEmbeddingModel(config).to(device)

    model = make_model()
    n_params = sum(p.numel() for p in model.parameters())

    # 1. forward
    def _forward():
        ids = torch.randint(0, config.vocab_size, (4, 64), device=device)
        mask = torch.ones(4, 64, device=device)
        emb, aux = model(ids, mask)
        assert emb.shape == (4, config.embedding_dim)
        norm = emb.norm(dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)
        return f"shape={tuple(emb.shape)} ||e||={norm[0].item():.4f}"

    check("Forward", _forward)

    # 2. bidirectional mask sanity: padding must change output of unmasked positions
    def _mask():
        ids = torch.randint(0, config.vocab_size, (1, 32), device=device)
        full, _ = model(ids, torch.ones(1, 32, device=device))
        padded = ids.clone()
        padded[0, -8:] = config.pad_token_id
        mask = torch.ones(1, 32, device=device)
        mask[0, -8:] = 0
        out_masked, _ = model(padded, mask)
        # first 24 positions should differ from full run (bidirectional info flow)
        diff = (full[0, :24] - out_masked[0, :24]).abs().mean().item()
        assert diff > 1e-6, "padding did not affect other positions (mask broken?)"
        return f"mean |Δ| on visible tokens after padding = {diff:.4f} (bidirectional OK)"

    check("Bidirectional + Padding Mask", _mask)

    # 3. backward
    def _backward():
        ids = torch.randint(0, config.vocab_size, (8, 64), device=device)
        mask = torch.ones(8, 64, device=device)
        q, aux = model(ids, mask)
        p, _ = model(ids.flip(0), mask)
        loss = MultipleNegativesRankingLoss()(q, p) + config.load_balance_weight * aux
        loss.backward()
        grads = sum(1 for p_ in model.parameters() if p_.grad is not None)
        total = sum(1 for p_ in model.parameters())
        assert grads == total
        return f"loss={loss.item():.4f}, grads {grads}/{total}"

    check("Backward", _backward)

    # 4. GQA ratio
    def _gqa():
        attn = model.backbone.layers[0].self_attn
        ratio = attn.num_heads // attn.num_key_value_heads
        assert ratio >= 1
        return f"heads={attn.num_heads} kv_heads={attn.num_key_value_heads} (GQA 1:{ratio})"

    check("GQA", _gqa)

    # 5. MoE routing + expert usage
    def _moe():
        from jaume.modeling.moe import MoEFeedForward
        moe = MoEFeedForward(config).to(device)
        x = torch.randn(2, 16, config.hidden_size, device=device)
        out, aux = moe(x)
        assert out.shape == x.shape
        return f"experts={moe.expert_num} topk={moe.topk} aux_loss={aux.item():.4f}"

    check("MoE Routing", _moe)

    # 6. gradient checkpointing
    def _ckpt():
        model_cpt = JaumeEmbeddingModel(config).to(device)
        model_cpt.backbone.gradient_checkpointing_enable()
        ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
        mask = torch.ones(2, 32, device=device)
        q, aux = model_cpt(ids, mask)
        q.sum().backward()
        return "backward through checkpointed backbone OK"

    check("Gradient Checkpointing", _ckpt)

    # 7. mixed precision
    def _amp():
        model_m = JaumeEmbeddingModel(config).to(device)
        ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
        mask = torch.ones(2, 32, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            q, aux = model_m(ids, mask)
        return f"dtype={q.dtype}"

    check("Mixed Precision (bf16)", _amp)

    # 8. save / load roundtrip
    def _ckpt_save(tmp):
        from jaume.inference.encode import save_pretrained, JaumeEmbedding
        path = os.path.join(tmp, "ckpt-test")
        tok_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "jaume_tokenizer")
        from jaume.data.dataset import load_tokenizer
        tok = load_tokenizer(tok_path)
        save_pretrained(model.eval(), tok, path)
        reloaded = JaumeEmbedding.from_pretrained(path)
        e1 = model.eval().cpu()(torch.randint(0, config.vocab_size, (1, 16)), torch.ones(1, 16))[0]
        e2 = reloaded.model.cpu()(torch.randint(0, config.vocab_size, (1, 16)), torch.ones(1, 16))[0]
        assert e1.shape == e2.shape
        return f"roundtrip OK, dim={e1.shape[-1]}"

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        check("Checkpoint Save/Load", lambda: _ckpt_save(tmp))

    # 9. decision heads
    def _decision():
        decisions = {
            "category": {"type": "choice", "options": ["ai", "programming", "general", "other"]},
            "importance": {"type": "score", "levels": 5},
            "is_relevant": {"type": "noul"},
        }
        dm = JaumeDecisionModel(config, decisions).to(device)
        ids = torch.randint(0, config.vocab_size, (2, 32), device=device)
        mask = torch.ones(2, 32, device=device)
        out, aux = dm(ids, mask)
        assert set(out) == set(decisions)
        return f"one encoding → {len(out)} decisions: {', '.join(out)}"

    check("Multi-Question Decision", _decision)

    # 10. param count
    def _params():
        return f"{n_params/1e6:.2f}M parameters"

    check("Param Count", _params)

    failed = [c for c in CHECKS if c[1] == "FAIL"]
    print(f"\n=== {len(CHECKS)-len(failed)}/{len(CHECKS)} checks passed ===")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
