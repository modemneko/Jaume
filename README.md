# Jaume-Embedding v0.1

轻量级语义表示 + 结构化决策模型。规范见 [SPEC.md](SPEC.md)，训练计划见 [PLAN.md](PLAN.md)。

架构：Bidirectional Transformer Encoder（GQA + RoPE + RMSNorm + MoE），Pooling → Projection → L2 Normalize 输出单位向量；同一 encoder 表征上挂 Choice / Score / Noul 决策头，支持一次编码多路决策。

## 模型

| 模型 | 参数量 | Hidden | Embedding | Context | Experts |
|---|---|---|---|---|---|
| Jaume-Embedding-350M | 291.9M | 512 | 512D (128/256/384 可截断) | 8K | 4 experts, top-2 |
| Jaume-Embedding-700M | 699.1M | 768 | 768D (128–768 可截断) | 16K | 8 experts, top-2 |

Tokenizer 复用 [Loftey](https://github.com/modemneko/loftey) 的 151646 词表 BPE。

## 快速开始

```bash
# Stage 0 骨干验证（10 项检查）
python scripts/smoke_test.py --config configs/jaume-smoke.json

# 训练
python scripts/train_embedding.py \
    --config configs/jaume-350m.json \
    --train-data /root/autodl-tmp/data/all_nli.jsonl \
    --output output/350m-nli \
    --epochs 1 --batch-size 16 --grad-accum 4 --grad-ckpt \
    --lr 1e-4 --use-matryoshka

# 评测（Spearman vs gold score）
python scripts/evaluate.py --model output/350m-nli/final \
    --data /root/autodl-tmp/data/stsbenchmark_sts.jsonl --baseline
```

## Python API

```python
from jaume import JaumeEmbedding

model = JaumeEmbedding.from_pretrained("output/350m-nli/final")
emb = model.encode(["HakusAgent 是一个 AI Agent。"])          # [1, 512]
scores = model.similarity(["如何训练模型？"], ["训练方法介绍"])  # cosine
emb128 = model.encode(text, dimensions=128)                   # Matryoshka 截断
```

## 硬件实测（RTX 3080 Ti 12GB）

- 350M + bf16 + gradient checkpointing + batch 8：约 6.3GB 显存，12 pairs/s（max_length 128）
- 单卡对比学习需要梯度累积凑有效 batch（`--grad-accum`），大规模训练需 GradCache / 多卡
