# V18 从 Epoch 2 恢复训练 - 快速指南

## 🎯 核心改进

| 项目 | 修复前 | 修复后 |
|------|--------|--------|
| **训练集 Mask** | 每个 epoch 增加 | 每 **2** 个 epoch 增加 |
| **验证集 Mask** | 每个 epoch 增加 ❌ | 固定 **50%** ✅ |
| **Val Loss 可比性** | 无法比较 ❌ | 可以比较 ✅ |
| **Checkpoint 恢复** | 不支持 ❌ | 完全支持 ✅ |

---

## 📋 立即开始恢复训练

### 一键运行（推荐）

```bash
cd /path/to/VCF-Bert
bash run_v18_embedding_rag_resume_ep2.sh
```

**这个脚本会自动**:
- ✅ 检查 checkpoint 是否存在
- ✅ 从 Epoch 2 恢复训练
- ✅ 设置正确的 Mask Level
- ✅ 固定验证集难度为 50%
- ✅ 保存新的日志和 CSV

---

### 手动配置（灵活）

如果需要从其他 Epoch 恢复，修改 `run_v18_embedding_rag.sh`:

1. **取消注释并修改路径**:
```bash
# 在第 77-78 行
RESUME_PATH="/your/path/to/rag_bert.model.ep2"
RESUME_EPOCH=2
```

2. **添加参数到 python 命令**:
```bash
python -m src.train_embedding_rag \
    ... (其他参数) ...
    --resume_path ${RESUME_PATH} \
    --resume_epoch ${RESUME_EPOCH} \
    2>&1 | tee ${LOG_FILE}
```

3. **运行**:
```bash
bash run_v18_embedding_rag.sh
```

---

## 📊 预期训练行为

### Mask Rate 时间表

| Epoch | 训练集 Mask | 验证集 Mask | 说明 |
|-------|-------------|-------------|------|
| 0-1   | 10%         | 50% (固定)  | 初始难度 |
| 2-3   | 10%→20%     | 50% (固定)  | Epoch 2 结束时训练难度提升 |
| 4-5   | 20%→30%     | 50% (固定)  | Epoch 4 结束时再次提升 |
| 6-7   | 30%→40%     | 50% (固定)  | 逐步增加 |
| ...   | ...         | 50% (固定)  | 最高 80% |

### Loss 曲线预期

```
从 Epoch 2 恢复后:

Epoch 2: Train Loss ~75  | Val Loss ~350 (首次评估，50% mask 基准)
Epoch 3: Train Loss ~72  | Val Loss ~345 (同样 10% mask，性能改善)
         ↓ 训练难度提升到 20% ↓
Epoch 4: Train Loss ~150 | Val Loss ~340 (训练 Loss 跳跃是正常的!)
Epoch 5: Train Loss ~145 | Val Loss ~335 (性能继续改善)
         ↓ 训练难度提升到 30% ↓
Epoch 6: Train Loss ~230 | Val Loss ~330 (训练 Loss 再次跳跃)
Epoch 7: Train Loss ~220 | Val Loss ~325 (性能继续改善)
```

**关键点**:
- ✅ **Val Loss 持续下降** = 模型性能真正改善
- ✅ **Train Loss 跳跃** = 正常现象（训练难度提升）
- ✅ **F1/Accuracy 提升** = 最终目标

---

## 🔍 如何验证恢复成功

### 1. 检查日志输出

训练开始时应该看到:
```
================================================================================
Resuming from Checkpoint...
================================================================================
Loading weights from: /path/to/rag_bert.model.ep2
✓ Weights loaded successfully
✓ Resuming from epoch 2
================================================================================

================================================================================
Setting Validation Mask Level to 50%...
================================================================================
✓ Validation mask level set to 50%
✓ Validation difficulty is now FIXED for all epochs
================================================================================

================================================================================
Restoring Training Mask Level for Epoch 2...
================================================================================
✓ Training mask level restored to: 10%
================================================================================
```

### 2. 监控验证 Loss

```bash
# 实时查看最新日志
tail -f logs/v18_embedding_rag/latest_resume.log

# 查看 CSV 指标
cat metrics/v18_embedding_rag/latest_resume.csv | column -t -s,
```

**预期**: Val Loss 应该从 Epoch 3 开始稳定下降（Epoch 2 可能跳跃，因为验证难度从 20%→50%）

### 3. 对比 F1 分数

```bash
# 提取 F1 分数
grep "overall_f1" metrics/v18_embedding_rag/latest_resume.csv
```

**预期**: Overall F1, Rare F1, Common F1 都应该逐步提升

---

## ⚠️ 常见问题

### Q1: 恢复后 Val Loss 突然很高？

**A**: 这是**正常的**！原因:
- Epoch 1-2 使用的验证 mask 是 10%/20%
- 恢复后验证 mask 固定为 50%
- 更高的 mask 比例 = 更高的 Loss（但更准确反映泛化能力）
- **关键是看 Epoch 3+ 的 Loss 是否持续下降**

### Q2: Train Loss 在 Epoch 2/4/6 跳跃？

**A**: 这是**正常的**！原因:
- 每 2 个 epoch，训练 mask 增加（10%→20%→30%...）
- 更多的 masked 位点 = 更高的 Loss（但模型学到更多）
- **关键是看同样 mask rate 下，Loss 是否下降**

例如:
```
Epoch 2 (10% mask): Train Loss = 75
Epoch 3 (10% mask): Train Loss = 72  ✅ 改善!
Epoch 4 (20% mask): Train Loss = 150 ⚠️ 跳跃 (难度提升)
Epoch 5 (20% mask): Train Loss = 145 ✅ 改善!
```

### Q3: Checkpoint 文件找不到？

**A**: 检查以下路径:
```bash
# 检查 Epoch 2 checkpoint 是否存在
ls -lh /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v18_embrag/rag_bert.model.ep2

# 列出所有 checkpoint
ls -lh /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v18_embrag/*.ep*
```

如果不存在，可能文件名不同。检查:
```bash
ls -lh /path/to/output_v18_embrag/
```

### Q4: 如何从其他 Epoch 恢复？

**A**: 修改 `RESUME_EPOCH` 变量:
```bash
# 例如从 Epoch 5 恢复
RESUME_PATH="/path/to/rag_bert.model.ep5"
RESUME_EPOCH=5
```

训练 Mask 会自动恢复到正确的 level:
- Epoch 5: level = 5 // 2 = 2 → 30% mask

---

## 📈 性能监控脚本

### 实时监控 Val Loss

```bash
# 持续显示 Val Loss 变化
watch -n 10 "grep 'val,' metrics/v18_embedding_rag/latest_resume.csv | tail -5 | column -t -s,"
```

### 绘制 Loss 曲线

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV
df = pd.read_csv('metrics/v18_embedding_rag/latest_resume.csv')

# 分离训练和验证
train = df[df['mode'] == 'train']
val = df[df['mode'] == 'val']

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss 曲线
ax1.plot(train['epoch'], train['loss'], label='Train Loss', marker='o')
ax1.plot(val['epoch'], val['loss'], label='Val Loss (50% mask)', marker='s')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.set_title('Loss Curves (Fixed Val Mask)')

# F1 曲线
ax2.plot(train['epoch'], train['overall_f1'], label='Train F1', marker='o')
ax2.plot(val['epoch'], val['overall_f1'], label='Val F1', marker='s')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('F1 Score')
ax2.legend()
ax2.set_title('F1 Scores')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("✓ Saved to training_curves.png")
```

---

## 🚀 最佳实践

1. **使用专用脚本**: `run_v18_embedding_rag_resume_ep2.sh` 已预配置好所有参数
2. **定期检查 GPU**: `watch -n 60 nvidia-smi`
3. **备份 Checkpoint**: 每隔几个 epoch 备份一次 best model
4. **观察 Val Loss**: 只关注验证 Loss 是否持续下降（训练 Loss 跳跃是正常的）
5. **关注 F1 分数**: 这是最终目标，比 Loss 更重要

---

## 📞 需要帮助？

如果遇到问题:
1. 检查日志文件: `logs/v18_embedding_rag/latest_resume.log`
2. 查看 GPU 状态: `nvidia-smi`
3. 验证 checkpoint 路径: `ls -lh /path/to/checkpoint`
4. 检查 CSV 输出: `tail metrics/v18_embedding_rag/latest_resume.csv`

---

**现在可以开始恢复训练了！祝训练顺利 🚀**
