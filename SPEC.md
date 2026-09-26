# Jaume-Embedding v0.1

> A compact semantic representation and decision model for intelligent software.

**Project:** Jaume
**Component:** Jaume-Embedding
**Version:** v0.1
**Status:** Architecture Specification
**Base:** Loftey / Transformer Encoder / BERT-style representation learning
**Primary Tasks:** Embedding / Retrieval / Similarity / Classification / Structured Decision

---

# 1. Overview

Jaume-Embedding 是一个面向 Agent、RAG、Memory、Semantic Search 和结构化决策的轻量级语义模型。

Jaume 不以文本生成作为主要目标，而是将输入文本编码为高质量语义表示，并进一步提供低延迟、结构化的判断能力。

核心设计：

```text
                         Jaume
                           │
              ┌────────────┴────────────┐
              │                         │
        Representation             Decision
              │                         │
        Dense Embedding          Choice / Score / Noul
              │                         │
       ┌──────┼──────┐                  │
       ↓      ↓      ↓                  ↓
      RAG   Memory  Search          Agent Runtime
```

Jaume 的设计吸收以下方向：

* BERT：Bidirectional Transformer Encoder
* Sentence-BERT：Sentence-level Representation Learning
* Loftey：Transformer / GQA / MoE 等已有技术基础
* Modern Embedding Models：Contrastive Learning、Hard Negative、Matryoshka Representation
* Jev / System One：Structured Decision、Choice / Score / Noul、多问题共享状态推理

Jaume 不试图复刻任何闭源模型的内部实现。

---

# 2. Design Goals

## 2.1 Primary Goals

### G1. 高质量语义表示

输入：

```text
Text
```

输出：

```text
Dense Vector
```

要求：

* 语义相似文本距离更近
* 语义无关文本距离更远
* 支持 Query / Passage / Document / Memory
* 支持中文
* 支持英文
* 逐步扩展多语言

---

### G2. Agent-first

Jaume 应该能够作为 Agent 的快速认知层：

```text
User Message
      │
      ↓
    Jaume
      │
 ┌────┼───────────────┐
 ↓    ↓               ↓
RAG Memory       Intent / Routing
```

减少简单判断对大型 LLM 的依赖。

---

### G3. 低延迟

Jaume 应适合：

* 本地运行
* Desktop Agent
* Mobile Agent
* RAG Pipeline
* Memory Pipeline
* Semantic Cache

---

### G4. Structured Decision

在 Embedding 之外提供：

```text
Choice
Score
Noul
```

三种基础 Decision Primitive。

---

### G5. 可扩展

未来支持：

* Multimodal Embedding
* Code Embedding
* Cross-modal Retrieval
* Agent Decision
* Knowledge Graph
* Memory Ranking

---

# 3. Non-Goals

Jaume v0.1 不追求：

* 长文本生成
* Chatbot
* 通用 LLM
* 直接替代大型生成模型
* 复刻 Jev 内部架构
* 第一版本追求 MTEB 第一名

Jaume 的核心定位是：

> Semantic Intelligence Layer

---

# 4. Model Family

Jaume v0.1 计划提供两个模型。

## 4.1 Jaume-Embedding-350M

定位：

> Lightweight Agent / Memory / RAG Embedding Model

目标：

```text
~235M–350M Parameters
Hidden Size: 512
Embedding: 512D
Context: 8K
```

主要用途：

* Agent Memory
* RAG
* Semantic Search
* Semantic Cache
* Intent Classification
* Local AI

---

## 4.2 Jaume-Embedding-700M

定位：

> General-purpose Semantic Embedding Model

目标：

```text
~600M–700M Parameters
Hidden Size: 768
Embedding: 768D
Context: 16K–32K
```

主要用途：

* General Retrieval
* Document Retrieval
* Code Retrieval
* Semantic Similarity
* Classification
* Clustering
* Long-context Retrieval

---

# 5. Architecture

## 5.1 Core Architecture

Jaume 不直接使用 Loftey 原本的 decoder-only causal architecture。

Jaume 使用：

> Bidirectional Transformer Encoder

整体：

```text
Input Text
    │
    ↓
Tokenizer
    │
    ↓
Token Embedding
    │
    ↓
Position Encoding
    │
    ↓
Bidirectional Transformer Encoder
    │
    ├── GQA
    ├── RMSNorm
    ├── RoPE
    └── MoE FFN
    │
    ↓
Hidden States
    │
    ├───────────────┐
    ↓               ↓
Pooling         Decision Heads
    │               │
    ↓               ├── Choice
Projection          ├── Score
    │               └── Noul
    ↓
L2 Normalize
    │
    ↓
Embedding
```

---

# 6. Relationship with Loftey

Loftey 与 Jaume 应共享尽可能多的基础代码。

```text
                 Hakus Transformer Core
                          │
             ┌────────────┴────────────┐
             ↓                         ↓
          Loftey                     Jaume
             │                         │
       Causal Attention        Bidirectional Attention
             │                         │
          Causal LM                Embedding
```

共享：

* Tokenizer
* Embedding Layer
* RMSNorm
* RoPE
* GQA
* MoE
* Expert Layer
* Initialization
* Checkpoint Infrastructure
* Configuration System

区别：

| Component     | Loftey     | Jaume          |
| ------------- | ---------- | -------------- |
| Attention     | Causal     | Bidirectional  |
| Primary Task  | Generation | Representation |
| Output        | Logits     | Embedding      |
| Training      | Causal LM  | Contrastive    |
| Pooling       | N/A        | Required       |
| Decision Head | Optional   | Native         |

---

# 7. Encoder Design

## 7.1 Attention

默认：

```text
Bidirectional GQA
```

所有 token 可以访问完整序列：

```text
Token A ←→ Token B ←→ Token C
```

而不是：

```text
Token A → Token B → Token C
```

这样更适合 sentence-level semantic representation。

---

# 8. Position Encoding

优先使用：

```text
RoPE
```

而不是原始 BERT 的 absolute positional embedding。

原因：

* 与 Loftey 基础设施兼容
* 更现代
* 更容易扩展上下文
* 减少两套 Position Encoding 实现

---

# 9. MoE

Jaume-Embedding-350M 初始设计：

```text
Experts: 4
Top-K: 2
```

结构：

```text
Hidden
  │
Router
  │
  ├── Expert 1
  ├── Expert 2
  ├── Expert 3
  └── Expert 4
       │
       ↓
Weighted Combination
```

MoE 是否最终提升 Embedding 能力必须通过实验验证。

需要记录：

* Expert Usage
* Router Entropy
* Load Balance
* Expert Collapse
* Retrieval Performance

不能预设不同 Expert 会自动形成领域分工。

---

# 10. Pooling

Jaume 第一版必须进行 Pooling Ablation。

候选方案：

```text
A. CLS Pooling
B. Mean Pooling
C. Last Token Pooling
D. Attention Pooling
E. CLS + Mean
```

推荐首先实现：

```text
Mean
Attention
CLS
```

最终默认方案由 benchmark 决定。

---

# 11. Attention Pooling

候选实现：

```text
scores = Linear(hidden_states)

weights = Softmax(scores)

pooled = Σ(
    weights_i * hidden_i
)
```

然后：

```text
pooled
 ↓
Projection
 ↓
LayerNorm
 ↓
L2 Normalize
```

---

# 12. Projection Head

350M：

```text
Hidden 512
   ↓
Linear
   ↓
512
   ↓
LayerNorm
   ↓
L2 Normalize
```

700M：

```text
Hidden 768
   ↓
Linear
   ↓
768
   ↓
LayerNorm
   ↓
L2 Normalize
```

最终：

```text
||embedding||₂ ≈ 1
```

这样可以直接使用 cosine similarity / dot product。

---

# 13. Matryoshka Representation Learning

Jaume-Embedding-700M 默认目标：

```text
768D
```

同时训练：

```text
128D
256D
384D
512D
768D
```

使同一个 embedding 可以根据应用场景截断。

例如：

```text
Mobile:
128D

Desktop:
384D

RAG:
512D

High-quality Retrieval:
768D
```

API：

```python
model.encode(text, dimensions=128)
model.encode(text, dimensions=256)
model.encode(text, dimensions=384)
model.encode(text, dimensions=512)
model.encode(text, dimensions=768)
```

350M 默认：

```text
128 / 256 / 384 / 512
```

---

# 14. Embedding Training

## 14.1 Basic Training Sample

数据：

```json
{
  "query": "如何训练一个 Embedding 模型？",
  "positive": "Embedding 模型通常通过对比学习训练语义表示。",
  "negative": "Python 中如何读取 CSV 文件？"
}
```

目标：

```text
Similarity(query, positive) ↑

Similarity(query, negative) ↓
```

---

# 15. Contrastive Learning

第一版采用：

> Multiple Negatives Ranking Loss / InfoNCE

Batch：

```text
Q1 P1
Q2 P2
Q3 P3
Q4 P4
```

构造：

```text
Q1 ↔ P1  Positive
Q1 ↔ P2  Negative
Q1 ↔ P3  Negative
Q1 ↔ P4  Negative
```

从而提高训练效率。

---

# 16. Hard Negative Mining

Hard Negative 是 Jaume 训练的重点。

普通 Negative：

```text
Query:
如何训练 Embedding？

Negative:
今天深圳天气怎么样？
```

过于简单。

Hard Negative：

```text
Query:
如何训练 Embedding？

Positive:
Embedding 模型使用对比学习训练语义表示。

Hard Negative:
如何训练一个 Reranker？
```

两者语义接近，但任务不同。

训练 Pipeline：

```text
Dataset
   ↓
Initial Encoder
   ↓
Retrieve Top-K
   ↓
Find Hard Negatives
   ↓
Human / Teacher Filtering
   ↓
Retrain
   ↓
Repeat
```

---

# 17. Teacher Distillation

可选 Teacher：

* BGE
* E5
* Jina Embeddings
* Qwen Embedding
* 其他经过许可的数据/模型

Student：

```text
Jaume-Embedding-350M
```

训练目标：

```text
Teacher Similarity Matrix
          ↓
        Student
          ↓
Similarity Matrix
```

使用：

```text
L_distillation
```

帮助小模型学习成熟 embedding 空间。

Teacher 的具体选择需要根据许可证、语言覆盖和实际 benchmark 决定。

---

# 18. Multi-Objective Training

最终：

```text
L_total =
    L_contrastive
  + λ1 L_matryoshka
  + λ2 L_hard_negative
  + λ3 L_distillation
```

初始 λ 只作为实验配置。

不要把初始权重视为最终最佳参数。

---

# 19. Decision Engine

Jaume 的第二个核心能力：

> Structured Decision

架构：

```text
Encoder
   │
   ↓
Shared Representation
   │
   ├──────────┬──────────┐
   ↓          ↓          ↓
Choice      Score       Noul
```

---

# 20. Choice

适合：

```text
Intent
Category
Routing
Classification
```

例如：

```json
{
  "type": "choice",
  "options": [
    "technical",
    "general",
    "social",
    "other"
  ]
}
```

输出：

```json
{
  "choice": "technical",
  "probabilities": {
    "technical": 0.91,
    "general": 0.05,
    "social": 0.02,
    "other": 0.02
  }
}
```

---

# 21. Score

用于：

```text
Importance
Relevance
Quality
Priority
Confidence
```

例如：

```json
{
  "type": "score",
  "levels": 5
}
```

输出：

```json
{
  "score": 4.2,
  "confidence": 0.89
}
```

---

# 22. Noul

Noul 用于二元或连续概率判断。

例如：

```json
{
  "type": "noul"
}
```

输出：

```json
{
  "probability": 0.94,
  "confidence": 0.91
}
```

典型任务：

```text
是否相关？
是否值得写入 Memory？
是否应该调用工具？
是否属于某领域？
是否需要 RAG？
```

---

# 23. Multi-Question Inference

多个 Decision 可以共享一次 Encoder Forward。

```text
Input
 │
 ↓
Jaume Encoder
 │
 ↓
Shared State
 │
 ├── Memory Relevance
 ├── Intent
 ├── Importance
 ├── Category
 └── Tool Routing
```

而不是：

```text
Input
 ↓
LLM
 ↓
Question 1

Input
 ↓
LLM
 ↓
Question 2

Input
 ↓
LLM
 ↓
Question 3
```

目标：

> One Encoding → Multiple Decisions

这也是 Jaume 与普通 Embedding 模型的重要区别之一。

---

# 24. Calibration

Decision Engine 不只评价 Accuracy。

必须评价：

```text
Accuracy
F1
AUROC
AUPRC
NLL
Brier Score
ECE
```

特别关注：

> Calibration

例如模型输出：

```text
confidence = 0.90
```

应该验证这一概率是否具有实际统计意义。

---

# 25. Agent Integration

Jaume 最重要的实际应用之一是 Hakus Agent。

推荐 Pipeline：

```text
User Message
      │
      ↓
    Jaume
      │
 ┌────┼───────────────┐
 ↓    ↓               ↓
Embed Intent       Importance
 │    │               │
 ↓    ↓               ↓
Memory RAG Router   Memory Manager
```

---

# 26. Memory Pipeline

例如：

```text
User:
“我正在开发 HakusAgent 手机版。”
```

Jaume：

```json
{
  "memory": 0.96,
  "importance": 0.89,
  "category": "project",
  "embedding": [...]
}
```

Memory Manager 再决定：

```text
Long-term Memory
```

而不是让大型 LLM 负责所有简单判断。

---

# 27. RAG Pipeline

```text
Query
 │
 ↓
Jaume Embedding
 │
 ↓
Vector Database
 │
 ↓
Top-K
 │
 ↓
Optional Jaume Decision
 │
 ↓
Relevant Documents
 │
 ↓
LLM
```

Jaume Decision 可以进一步判断：

```text
Document relevance
```

从而减少无关上下文进入 LLM。

---

# 28. Semantic Cache

```text
New Query
   │
   ↓
Jaume Embedding
   │
   ↓
Cache Search
   │
   ↓
Similarity
   │
   ├── High → Reuse
   │
   └── Low → LLM
```

可以降低 Agent 的 LLM 调用成本。

---

# 29. API

## Encode

```python
from jaume import JaumeEmbedding

model = JaumeEmbedding.from_pretrained(
    "HakusAI/Jaume-Embedding-350M"
)

embeddings = model.encode([
    "HakusAgent 是一个 AI Agent。",
    "这是一个 Transformer 模型。"
])
```

返回：

```text
[B, D]
```

---

## Similarity

```python
scores = model.similarity(
    queries,
    documents
)
```

---

## Decision

```python
result = model.decide(
    state="用户正在询问 Embedding 模型训练",
    questions={
        "technical": {
            "type": "noul"
        },
        "category": {
            "type": "choice",
            "options": [
                "ai",
                "programming",
                "general",
                "other"
            ]
        },
        "importance": {
            "type": "score",
            "levels": 5
        }
    }
)
```

---

# 30. Evaluation

## Embedding Benchmark

至少：

```text
MTEB
MTEB Chinese
CMTEB
STS
Retrieval
Classification
Clustering
```

需要分别记录：

```text
STS
Retrieval
Reranking
Classification
Clustering
```

不能只报告一个总分。

---

# 31. Agent Benchmark

建立：

```text
Jaume-AgentBench
```

第一版包含：

```text
Memory Decision
Tool Selection
Intent Classification
RAG Relevance
Document Selection
Task Routing
Semantic Cache
```

评价：

```text
Accuracy
F1
ECE
Brier
Latency
Throughput
Memory Usage
```

---

# 32. Ablation Study

Jaume 必须进行以下 Ablation：

```text
Pooling
├── CLS
├── Mean
├── Last
├── Attention
└── CLS + Mean

MoE
├── Dense
├── 4 Expert Top-1
└── 4 Expert Top-2

Position
├── RoPE
└── Absolute

Training
├── Contrastive
├── Contrastive + Hard Negative
├── + Distillation
└── + Matryoshka
```

最终结果决定默认架构。

---

# 33. Training Stages

## Stage 0 — Backbone Validation

确认：

* Forward
* Backward
* Mask
* GQA
* MoE
* Checkpoint
* Mixed Precision

---

## Stage 1 — Encoder Pretraining

可使用：

```text
MLM
Denoising
Contrastive Pretraining
```

目标：

> 建立双向语义表示能力。

---

## Stage 2 — Sentence Embedding

加入：

```text
Positive / Negative
Contrastive Learning
```

目标：

> 获得 sentence-level semantic representation。

---

## Stage 3 — Hard Negative

加入：

```text
In-batch Negative
Hard Negative
Teacher Filtering
```

目标：

> 提升 Retrieval 和 Similarity。

---

## Stage 4 — Matryoshka

训练：

```text
128
256
384
512
768
```

目标：

> 单模型多维度部署。

---

## Stage 5 — Decision

训练：

```text
Choice
Score
Noul
```

目标：

> 让 Jaume 成为 Semantic Intelligence Layer。

---

## Stage 6 — Calibration

优化：

```text
ECE
Brier
NLL
```

目标：

> 让 Decision confidence 具有实际可用性。

---

# 34. Repository Structure

```text
Jaume/
│
├── configs/
│   ├── jaume-350m.yaml
│   └── jaume-700m.yaml
│
├── jaume/
│   ├── modeling/
│   │   ├── backbone.py
│   │   ├── encoder.py
│   │   ├── attention.py
│   │   ├── moe.py
│   │   ├── pooling.py
│   │   ├── embedding.py
│   │   └── decision.py
│   │
│   ├── losses/
│   │   ├── contrastive.py
│   │   ├── matryoshka.py
│   │   ├── distillation.py
│   │   └── calibration.py
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   ├── collator.py
│   │   └── negative_mining.py
│   │
│   └── inference/
│       ├── encode.py
│       ├── similarity.py
│       └── decide.py
│
├── scripts/
│   ├── pretrain.py
│   ├── train_embedding.py
│   ├── train_decision.py
│   └── evaluate.py
│
├── benchmarks/
│
├── tests/
│
├── README.md
├── MODEL_CARD.md
├── SPEC.md
└── PLAN.md
```

---

# 35. Version Roadmap

## v0.1

```text
Encoder
+
Embedding
+
Contrastive Learning
```

---

## v0.2

```text
Hard Negative
+
Matryoshka
+
Distillation
```

---

## v0.3

```text
Choice
+
Score
+
Noul
```

---

## v0.4

```text
Agent Integration
+
Memory
+
RAG
+
Semantic Cache
```

---

## v0.5

```text
Multilingual
+
Code Embedding
```

---

## v1.0

目标：

```text
Jaume-Embedding-350M
Jaume-Embedding-700M
```

形成稳定 API：

```text
encode()
similarity()
decide()
```

并发布：

```text
Model Weights
Tokenizer
Config
Training Code
Evaluation
Model Card
```

---

# 36. Core Design Principle

Jaume 的核心不是：

> 更大的语言模型。

而是：

> **把语义表示和高速结构化判断从大型生成模型中独立出来。**

最终架构：

```text
                         Hakus Agent
                              │
                    ┌─────────┴─────────┐
                    │                   │
                 LLM Brain           Jaume
                    │                   │
              Reasoning / Tool      Fast Cognition
                    │                   │
                    │          ┌────────┴────────┐
                    │          │                 │
                    │      Embedding          Decision
                    │          │                 │
                    │       Memory/RAG     Choice/Score/Noul
                    │          │                 │
                    └──────────┴─────────────────┘
                               │
                         Agent Runtime
```

Jaume 的目标不是替代 LLM。

Jaume 的目标是：

> **让 Agent 不需要每一个判断都调用 LLM。**

---

# 37. Final Architecture

```text
                         Jaume
                           │
                    Bidirectional
                  Transformer Encoder
                           │
              ┌────────────┴────────────┐
              │                         │
             GQA                       MoE
              │                         │
              └────────────┬────────────┘
                           │
                     Hidden States
                           │
                 ┌─────────┴─────────┐
                 │                   │
              Pooling             Decision
                 │                   │
           Projection          ┌──────┼──────┐
                 │             │      │      │
           Matryoshka       Choice  Score  Noul
                 │
           L2 Normalize
                 │
              Vector
```

最终形成：

```text
Jaume-Embedding-350M
        +
Jaume-Embedding-700M
```

两个模型共享技术路线，但分别针对本地 Agent 和通用语义检索进行优化。

**Jaume 的第一原则：Representation First, Decision Fast.**
