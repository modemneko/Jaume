# Jaume Training Plan (v0.1)

## Stage 0 — Backbone Validation ✅
`scripts/smoke_test.py`：10 项检查全部通过（Forward / Padding Mask / Backward / GQA / MoE Routing / Gradient Checkpointing / bf16 / Checkpoint Save-Load / Multi-Question Decision / Param Count）。

## Stage 1 — Encoder Pretraining（待做）
- MLM / Denoising / Contrastive Pretraining，数据：中英文语料
- 训练量：350M 建议单卡分阶段，或迁移多卡

## Stage 2 — Sentence Embedding（进行中）
- MNRL / InfoNCE，`scripts/train_embedding.py`
- 已验证：smoke 配置端到端收敛；350M 单卡可训（6.3GB @ batch 8 + grad ckpt）
- 注意：训练集与评测集必须分离（STS-B 直接自训自测会过拟合，Spearman 反而下降）

## Stage 3 — Hard Negative
- Pipeline：initial encoder 检索 top-K → 挖掘 hard negative → 过滤 → 重训
- 数据管线已预留 `jaume/data/negative_mining.py`（待实现）

## Stage 4 — Matryoshka
- `--use-matryoshka` + `MatryoshkaLoss` 已实现，需在 Stage 2/3 数据量级上验证各截断维度的性能衰减

## Stage 5 — Decision（Choice / Score / Noul）
- `JaumeDecisionModel` 已实现，待训练数据与标定

## Stage 6 — Calibration
- ECE / Brier / NLL 评测，`jaume/losses/calibration.py`（待实现）

## 已知问题 / 记录
1. STS-B 自训自测导致评测指标下降（过拟合 + 各向异性），任何评测必须 hold-out
2. 单卡 12GB 限制有效 batch，in-batch negative 数量受限；扩 batch 需 GradCache
3. MoE load-balance aux loss（switch 风格）系数 0.01 为初始值，需实验调整

## CPU（无 GPU）环境测试记录
- cgroup 内存限额 2GB 且与 agent 进程共享（基线约 0.9GB），350M 权重（1.2GB）无法在 CPU 加载；21M 模型训练也超限（词表嵌入占 92% 参数）
- 新增 `configs/jaume-cpu-tiny.json`（hidden 64, 2 层, 9.9M 参数）：CPU 训练 46s 跑通（loss 1.77→1.18），推理 API / decide() / Matryoshka 截断全部可用，峰值 RSS 963MB
- 需要 `OMP_NUM_THREADS` 限制 + `MALLOC_ARENA_MAX=2` + `num_workers 0`，否则内存超限被 OOM-kill
- 单元测试 `tests/test_model.py` 7/7 通过（RoPE 相对位置性/范数保持、Matryoshka 截断一致性、pooling、MNRL、config 往返）
