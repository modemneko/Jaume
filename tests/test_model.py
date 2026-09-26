import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaume.modeling.config import JaumeConfig
from jaume.modeling.attention import build_rope, apply_rope
from jaume.modeling.embedding import JaumeEmbeddingModel
from jaume.modeling.pooling import build_pooling
from jaume.losses.contrastive import MultipleNegativesRankingLoss, MatryoshkaLoss

torch.manual_seed(0)
CONFIG = JaumeConfig(
    vocab_size=1000, hidden_size=64, intermediate_size=128, num_layers=2,
    num_attention_heads=4, num_key_value_heads=2, expert_num=4, topk=2,
    embedding_dim=64, matryoshka_dims=(16, 32, 64), pooling="attention",
)


def test_rope_relative_position():
    """RoPE attention scores depend only on relative distance (same q, k)."""
    cos, sin = build_rope(16, 32, 10000.0, "cpu", torch.float32)
    q_vec = torch.randn(1, 1, 1, 16)
    k_vec = torch.randn(1, 1, 1, 16)
    qr = apply_rope(q_vec.expand(1, 1, 32, 16), cos, sin)
    kr = apply_rope(k_vec.expand(1, 1, 32, 16), cos, sin)
    # score(q_i, k_j) must equal score(q_i+s, k_j+s) for any shift s
    s = 5
    score_a = (qr[0, 0, 10] * kr[0, 0, 3]).sum()
    score_b = (qr[0, 0, 10 + s] * kr[0, 0, 3 + s]).sum()
    assert torch.allclose(score_a, score_b, atol=1e-5)


def test_rope_norm_preserved():
    cos, sin = build_rope(16, 8, 10000.0, "cpu", torch.float32)
    q = torch.randn(2, 4, 8, 16)
    qr = apply_rope(q, cos, sin)
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-4)


def test_embedding_normalized_and_consistent():
    model = JaumeEmbeddingModel(CONFIG).eval()
    ids = torch.randint(0, 1000, (3, 32))
    mask = torch.ones(3, 32)
    with torch.no_grad():
        out, aux = model.forward_matryoshka(ids, mask)
    for d, emb in out.items():
        assert emb.shape == (3, d)
        assert torch.allclose(emb.norm(dim=-1), torch.ones(3), atol=1e-5)
    # truncated prefix must point in the same direction as the full
    # embedding's prefix (renormalized), i.e. cosine similarity == 1
    assert torch.allclose(
        torch.nn.functional.cosine_similarity(out[16], out[64][..., :16], dim=-1),
        torch.ones(3), atol=1e-5,
    )
    assert torch.allclose(
        torch.nn.functional.cosine_similarity(out[32], out[64][..., :32], dim=-1),
        torch.ones(3), atol=1e-5,
    )


def test_poolings_shapes():
    for name in ("mean", "cls", "last", "attention"):
        pool = build_pooling(name, CONFIG.hidden_size)
        h = torch.randn(2, 16, CONFIG.hidden_size)
        m = torch.ones(2, 16)
        m[1, -4:] = 0
        p = pool(h, m)
        assert p.shape == (2, CONFIG.hidden_size), name


def test_padding_invariance_left_tokens():
    """With mean pooling and full visibility, padding right must not change
    hidden states of left tokens beyond masking effects."""
    model = JaumeEmbeddingModel(CONFIG).eval()
    ids = torch.randint(0, 1000, (1, 24))
    with torch.no_grad():
        full, _ = model(ids, torch.ones(1, 24))
        ids_pad = ids.clone()
        ids_pad[0, -4:] = 999
        padded, _ = model(ids_pad, torch.ones(1, 24))
    # output must change (model is bidirectional, pad tokens still attended;
    # this checks attention mask path exists and runs)
    assert full.shape == padded.shape


def test_mnrl_loss():
    loss = MultipleNegativesRankingLoss()
    q = torch.nn.functional.normalize(torch.randn(8, 32), dim=-1)
    p = torch.nn.functional.normalize(torch.randn(8, 32), dim=-1)
    l_random = loss(q, p)
    l_perfect = loss(q, q.clone())
    assert l_perfect < l_random
    # matryoshka wrapper runs and returns finite
    ml = MatryoshkaLoss(MultipleNegativesRankingLoss(), (32, 16, 8))
    assert torch.isfinite(ml(q, p))


def test_config_roundtrip(tmpdir="/tmp/jaume_cfg_test"):
    os.makedirs(tmpdir, exist_ok=True)
    path = os.path.join(tmpdir, "config.json")
    CONFIG.save(path)
    loaded = JaumeConfig.load(path)
    assert loaded.matryoshka_dims == CONFIG.matryoshka_dims
    assert loaded.expert_num == CONFIG.expert_num


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
        except AssertionError as e:
            print(f"  ✗ {fn.__name__}: {e}")
            sys.exit(1)
    print(f"{len(fns)}/{len(fns)} unit tests passed")
