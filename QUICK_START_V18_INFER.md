# ⚡ V18 推理快速开始（5 分钟上手）

## 📦 最新代码

```bash
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup

git pull origin main
```

**新增文件**：
- `src/dataset/embedding_rag_infer_dataset.py` - V18 推理数据集
- `src/infer_embedding_rag.py` - V18 推理脚本
- `run_infer_embedding_rag.sh` - 一键启动脚本
- `V18_INFERENCE_GUIDE.md` - 详细文档

---

## 🚀 三步启动推理

### 第一步：修改配置（2 分钟）

```bash
vim run_infer_embedding_rag.sh
```

**必须修改的 5 个参数**：

```bash
# 1. 模型 Checkpoint（训练好的最佳模型）
CHECK_POINT="/cpfs01/.../rag_bert.model.ep11"  # ← 修改这里

# 2. 模型架构参数（必须与训练一致！）
DIMS=384         # ← 确认训练时的值
LAYERS=6         # ← 确认训练时的值
HEADS=8          # ← 确认训练时的值

# 3. Target Dataset（待填补的数据）
TARGET_DATASET="/path/to/your/target.h5"        # ← 修改这里
TARGET_PANEL="/path/to/your/target_panel.txt"  # ← 修改这里

# 4. Reference Panel（已提供，通常无需修改）
REF_PANEL="/cpfs01/.../train_split.h5"

# 5. 输出路径
OUTPUT_DIR="/cpfs01/.../infer_output_v18"       # ← 修改这里
```

**如何确认训练参数？**

```bash
# 方法 1: 查看训练脚本
grep "python -m src.train_embedding_rag" run_v18_embedding_rag.sh | grep -E "dims|layers|attn_heads"

# 方法 2: 查看训练日志
grep "Architecture" logs/v18_embedding_rag/training_*.log | head -1
```

### 第二步：启动推理（1 分钟）

```bash
# 给脚本执行权限
chmod +x run_infer_embedding_rag.sh

# 启动推理
bash run_infer_embedding_rag.sh
```

### 第三步：等待完成（15-25 分钟）

```
时间分配:
  - 索引构建: 15-20 分钟（首次，一次性）
  - 推理: 5-10 分钟（1000 samples）

总计: 20-30 分钟（首次）
     5-10 ���钟（后续，复用索引）
```

---

## 📊 预期输出

### 日志示例

```
================================================================================
▣ V18 Embedding RAG Inference
================================================================================
Device: cuda
Model: dims=384, layers=6, heads=8
Checkpoint: /path/to/rag_bert.model.ep11
...

▣ Step 1: Loading Vocabulary
✓ Vocab size: 2519

▣ Step 2: Loading V18 Model
✓ Model loaded successfully

▣ Step 3: Creating EmbeddingRAGInferDataset
预编码推理窗口: 100%|████████████████| 50/50 [15:00<00:00]
✓ 推理索引构建完成!

▣ Step 4: Creating DataLoader
✓ DataLoader created: 63 batches

▣ Step 5: Starting Inference
Imputing: 100%|████████████████████████| 63/63 [05:30<00:00]
✓ Inference completed in 330.45s

▣ Step 6: Generating Imputed VCF
✓ VCF file generated: /path/to/output/imputed.vcf

▣ V18 Inference Completed Successfully!
Total time: 1230.57s
```

### 输出文件

```
OUTPUT_DIR/
├── imputed.vcf                    # 填补后的 VCF 文件 ← 这是你要的
├── inference_log.txt              # 推理日志
└── faiss_indexes_infer/           # FAISS 索引（临时文件）
```

---

## ⚠️ 常见问题（90% 的问题在这里）

### ❌ 错误 1: 加载 checkpoint 失败

```
RuntimeError: size mismatch for embedding.token.weight
```

**原因**: 模型架构参数与训练不一致

**解决**:
1. 查看训练日志确认 `--dims`, `--layers`, `--attn_heads`
2. 在 `run_infer_embedding_rag.sh` 中使用相同的值

### ❌ 错误 2: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**解决**: 降低 Batch Size

```bash
# 在 run_infer_embedding_rag.sh 中修改
BATCH_SIZE=8  # 从 16 降到 8
```

### ❌ 错误 3: 找不到文件

```
FileNotFoundError: [Errno 2] No such file or directory: '/path/to/target.h5'
```

**解决**: 检查 `TARGET_DATASET` 和 `TARGET_PANEL` 路径是否正确

```bash
# 验证文件存在
ls -lh /path/to/target.h5
ls -lh /path/to/target_panel.txt
```

---

## 🎯 核心概念（2 分钟理解）

### Imputation Masking

**训练时**:
- 随机 Mask（数据增强）
- 每 Epoch 刷新

**推理时**:
- Imputation Mask（真实缺失）
- `Mask = All_Ref_Positions - Target_Known_Positions`
- Mask=1: 需要填补
- Mask=0: 已知（Context）

### 对称 Masking

**问题**: Reference 完整 + Query 残缺 → 检索失效

**解决**: Reference 和 Query 在相同位置 Mask

```
Query:    [0, MASK, 1, MASK, 0]  (残缺)
Reference: [0, MASK, 1, MASK, 0]  (也残缺) ← 关键!
→ Embedding 在相同语义空间 → 检索有效
```

---

## 📚 详细文档

完整文档: [V18_INFERENCE_GUIDE.md](V18_INFERENCE_GUIDE.md)

包含:
- 详细的推理流程
- 参数配置说明
- 性能优化建议
- 调试技巧
- 常见问题完整列表

---

## ✅ 检查清单

启动前:
- [ ] 已拉取最新代码 (`git pull`)
- [ ] 已修改 `CHECK_POINT` 路径
- [ ] 已确认 `DIMS`, `LAYERS`, `HEADS` 与训练一致
- [ ] 已修改 `TARGET_DATASET` 和 `TARGET_PANEL`
- [ ] 已修改 `OUTPUT_DIR`
- [ ] 已确认 GPU 可用 (`nvidia-smi`)

完成后:
- [ ] `imputed.vcf` 文件已生成
- [ ] 文件大小合理（非空）
- [ ] 日志无错误
- [ ] 填补位点数符合预期

---

## 🎉 总结

**快速启动**:
```bash
# 1. 拉取代码
git pull origin main

# 2. 修改配置（5 个参数）
vim run_infer_embedding_rag.sh

# 3. 启动推理
bash run_infer_embedding_rag.sh
```

**时间**: 20-30 分钟（首次）/ 5-10 分钟（后续）

**输出**: `imputed.vcf`（完整的填补后 VCF）

**现在可以开始 V18 推理了！🚀**
