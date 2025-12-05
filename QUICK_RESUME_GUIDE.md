# 快速续训练指南

## 🎯 场景：训练中断后恢复

### 第一步：找到最新 Checkpoint

```bash
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v18_embrag

ls -lht rag_bert.model.ep*
```

**示例输出**:
```
-rw-r--r-- 1 user group 123M Dec  5 10:30 rag_bert.model.ep2
-rw-r--r-- 1 user group 123M Dec  5 09:15 rag_bert.model.ep1
-rw-r--r-- 1 user group 123M Dec  5 08:00 rag_bert.model.ep0
```

假设最新是 `rag_bert.model.ep2`（从 Epoch 2 恢复，下一个 Epoch 是 3）

---

### 第二步：编辑训练脚本

```bash
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup

vim run_v18_embedding_rag.sh
```

**找到这一段**（约 Line 75-82）:
```bash
# === Checkpoint恢复配置 (可选) ===
# 如果需要从checkpoint恢复训练，请取消注释以下两行并修改路径
# RESUME_PATH="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v18_embrag/rag_bert.model.ep2"
# RESUME_EPOCH=2
```

**取消注释并修改**:
```bash
# === Checkpoint恢复配置 (可选) ===
# 如果需要从checkpoint恢复训练，请取消注释以下两行并修改路径
RESUME_PATH="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v18_embrag/rag_bert.model.ep2"
RESUME_EPOCH=2
```

**然后在 python 命令中添加参数**（约 Line 79-82 的注释）:
```bash
# 然后在下方python命令中添加:
#     --resume_path ${RESUME_PATH} \
#     --resume_epoch ${RESUME_EPOCH} \
```

**找到 python 命令**（约 Line 84）:
```bash
python -m src.train_embedding_rag \
    --train_dataset /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/train_split.h5 \
    --train_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/train_panel.txt \
    \
    --val_dataset /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/val_split.h5 \
    --val_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/val_panel.txt \
    \
    --resume_path ${RESUME_PATH} \
    --resume_epoch ${RESUME_EPOCH} \
    \
    --freq_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/Freq.npy \
    --window_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/segments_chr21.maf.csv \
    ...
```

**保存并退出**: `:wq`

---

### 第三步：启动训练

```bash
bash run_v18_embedding_rag.sh
```

**预期输出**:
```
================================================================================
Resuming from Checkpoint...
================================================================================
Loading weights from: .../rag_bert.model.ep2
✓ Weights loaded successfully
✓ Resuming from Epoch: 2
✓ Curriculum Learning Level restored to: 1 (Mask Rate: 30%)

================================================================================
▣ 构建Embedding-based RAG索引 (内存优化版)
================================================================================
✓ FAISS索引目录: .../faiss_indexes_train
✓ use_dynamic_mask: False
✓ Encoding Reference Panel: 100%|██████| 331/331 [40:00<00:00]

================================================================================
Setting Validation Mask Level to 50%...
================================================================================
✓ FAISS索引目录: .../faiss_indexes_val
✓ Encoding Reference Panel: 100%|██████| 331/331 [40:00<00:00]

✓ WindowGroupedSampler initialized:
  - Total samples: 30000+
  - Total windows: 331
  - Shuffle enabled: True

================================================================================
Starting Epoch 3 (continuing from Epoch 2)
================================================================================
```

---

## ⏰ 时间预算

**总耗时**: ~80 分钟 + 训练时间

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 训练集预编码 | 40 分钟 | 必须重新编码（Embedding 权重已更新） |
| 验证集预编码 | 40 分钟 | 必须重新编码（Embedding 权重已更新） |
| Sampler 初始化 | < 1 秒 | 已优化 |
| Epoch 3+ 训练 | 变动 | 取决于剩余 Epoch 数量 |

---

## ⚠️ 重要提醒

### 1. 每次续训练都需要预编码（80 分钟）

**无法避免！** 原因：
- Embedding Layer 权重每个 Epoch 都在更新
- FAISS 索引必须与最新权重匹配
- 旧索引的 Embedding 语义已过时

### 2. Checkpoint 必须是修复后的版本

**检查方法**:
```bash
git log --oneline --all | grep "修复致命类型错误"
```

如果你的 checkpoint 在 commit `04376e3` 之前创建，则**不可用**（dtype 错误）。

### 3. 续训练的课程学习 Level 会自动恢复

根据 `--resume_epoch` 参数，系统会自动计算正确的 Mask Level：
- Epoch 0-1: Level 0 (10% mask)
- Epoch 2-4: Level 1 (30% mask)
- Epoch 5-7: Level 2 (50% mask)
- Epoch 8-11: Level 3 (70% mask)
- Epoch 12+: Level 4 (80% mask)

**无需手动设置！**

---

## 🔧 常见问题

### Q1: 如果我想从 Epoch 5 开始训练怎么办？

**A**: 修改脚本中的参数：
```bash
RESUME_PATH=".../rag_bert.model.ep4"  # 注意：从 ep4 恢复，下一个是 ep5
RESUME_EPOCH=4
```

### Q2: 如果脚本已经有 `--resume_path` 参数怎么办？

**A**: 直接修改路径即可：
```bash
# 找到 python 命令中的这两行，直接修改
--resume_path /path/to/your/checkpoint \
--resume_epoch 2 \
```

### Q3: 如果我不想续训练，想从头开始怎么办？

**A**: 注释掉或删除这两个参数：
```bash
# 注释掉（推荐）:
# --resume_path ${RESUME_PATH} \
# --resume_epoch ${RESUME_EPOCH} \

# 或者直接删除这两行
```

然后：
```bash
# 清理旧索引（可选）
rm -rf maf_data/faiss_indexes_train maf_data/faiss_indexes_val

# 启动从头训练
bash run_v18_embedding_rag.sh
```

### Q4: 我能跳过预编码直接训练吗？

**A**: **不能！** 因为：
1. Embedding Layer 权重已更新（即使是同一个 Epoch）
2. 旧索引的 Embedding 与当前权重不匹配
3. 会导致 RAG 检索语义错误

**必须重新预编码（80 分钟）！**

---

## 📊 完整示例

### 场景：从 Epoch 2 恢复训练

**步骤 1**: 找到 checkpoint
```bash
ls -lht output_v18_embrag/rag_bert.model.ep*
# 找到: rag_bert.model.ep2
```

**步骤 2**: 编辑 `run_v18_embedding_rag.sh`
```bash
# 取消注释并修改:
RESUME_PATH="/cpfs01/.../output_v18_embrag/rag_bert.model.ep2"
RESUME_EPOCH=2

# 在 python 命令中添加:
python -m src.train_embedding_rag \
    --resume_path ${RESUME_PATH} \
    --resume_epoch ${RESUME_EPOCH} \
    ...
```

**步骤 3**: 启动训练
```bash
bash run_v18_embedding_rag.sh
```

**步骤 4**: 观察日志
```
✓ Resuming from Epoch: 2
✓ Curriculum Learning Level restored to: 1 (Mask Rate: 30%)
...
Starting Epoch 3
```

**完成！** 训练将从 Epoch 3 继续。

---

## ✅ 检查清单

续训练前确认：

- [ ] 已找到最新 checkpoint (ep*)
- [ ] 已修改 `run_v18_embedding_rag.sh` 中的 RESUME_PATH
- [ ] 已修改 `run_v18_embedding_rag.sh` 中的 RESUME_EPOCH
- [ ] 已在 python 命令中添加 `--resume_path` 和 `--resume_epoch`
- [ ] 已确认 checkpoint 是修复后的版本（commit 04376e3 之后）
- [ ] 已预留 80 分钟预编码时间

续训练后观察：

- [ ] 日志显示 "✓ Resuming from Epoch: X"
- [ ] 日志显示正确的 Curriculum Learning Level
- [ ] 训练从 Epoch X+1 开始
- [ ] Loss 和 F1 延续之前的趋势

---

## 🎯 总结

**续训练步骤**（3 步）:
1. 找到 checkpoint: `ls -lht output_v18_embrag/rag_bert.model.ep*`
2. 修改脚本: `RESUME_PATH` 和 `RESUME_EPOCH`
3. 启动训练: `bash run_v18_embedding_rag.sh`

**关键点**:
- ✅ 每次续训练需要 80 分钟预编码（无法避免）
- ✅ Checkpoint 必须是修复后的版本
- ✅ Level 会自动恢复，无需手动设置
- ✅ 训练从 `RESUME_EPOCH + 1` 开始

**现在可以随时续训练了！🚀**
