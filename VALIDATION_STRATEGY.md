# Validation策略完整指南

## 📊 你的测试数据结构

```
/cpfs01/.../New_VCF/Test/
├── TestData/          # 多个测试集（不同样本分组）
│   ├── Test1/         # 测试集1（不同mask比例10%-90%）✓
│   ├── Test2/         # 测试集2（不同mask比例10%-90%）✓
│   ├── Test3/         # 空
│   ├── Test4/         # 空
│   └── Test5/         # 空
├── Masked_VCFs/       # 主测试集（不同mask比例10%-90%）✓
└── Truth/             # 真实标签
    └── KGP.chr21.TestTruth.vcf.gz  # 完整基因型真值
```

---

## 🎯 推荐策略

### 策略：使用 `Masked_VCFs/` 作为Validation集 ⭐

**理由**：
1. ✅ 有完整的真实标签（`Truth/KGP.chr21.TestTruth.vcf.gz`）
2. ✅ 多个mask比例（10%-90%）可评估不同难度
3. ✅ 格式与训练数据一致
4. ✅ `TestData/Test1-Test2` 保留作为最终测试集

**数据划分**：
```
训练集：maf_data/KGP.chr21.Train.maf01.vcf.h5  (你现有的训练数据)
验证集：New_VCF/Test/Masked_VCFs/KGP.chr21.TestMask30.vcf.gz  (30% masked)
测试集：New_VCF/Test/TestData/Test1/  (最终评估，不用于训练)
```

**为什么选择Mask30作为标准验证集？**
- Mask10/20太简单，不能有效区分模型好坏
- Mask50/70太难，可能导致过早停止训练
- **Mask30是适中的难度**，最接近真实应用场景

---

## 🛠️ 完整实施步骤

### 步骤1：准备验证数据（转换VCF → H5）

在服务器上运行：

```bash
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup

python scripts/prepare_val_data.py \
    --test_dir /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/New_VCF/Test \
    --output_dir data/validation \
    --mask_ratios 30 \
    --truth_vcf /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/New_VCF/Test/Truth/KGP.chr21.TestTruth.vcf.gz
```

**输出**：
```
data/validation/
├── val_mask30.h5         # 验证集（30% masked）
├── val_truth.h5          # 真实标签
└── val_config.txt        # 配置文件
```

**如果想评估多个难度**（可选）：
```bash
python scripts/prepare_val_data.py \
    --test_dir /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/New_VCF/Test \
    --output_dir data/validation_multi \
    --mask_ratios 10 30 50 70
```

---

### 步骤2：准备验证集的Panel文件

验证集需要对应的panel文件。你有几个选择：

**选项A：使用训练集的panel**（如果验证集样本与训练集相同）
```bash
# 直接使用
--val_panel /cpfs01/.../train.980.sample.panel
```

**选项B：从TestTruth.vcf.gz提取样本列表创建panel**
```bash
# 提取样本ID
bcftools query -l /cpfs01/.../KGP.chr21.TestTruth.vcf.gz > data/validation/test_samples.list

# 创建panel（假设所有样本属于同一群体，如EUR）
awk '{print $1"\tEUR"}' data/validation/test_samples.list > data/validation/val.panel
```

**选项C：如果有KGP_INFO文件**
```bash
# 使用现有的样本信息文件
# 从 KGP_INFO_with_balanced_Subset.csv 中提取测试集样本
```

---

### 步骤3：创建训练脚本

创建 `run_v11_with_val.sh`：

```bash
#!/bin/bash

# ==========================================
# RAG-SNVBERT训练脚本 - 带Validation支持
# Version: v11 (2025-04-XX)
# ==========================================

python -m src.train_with_val \
    --train_dataset /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/KGP.chr21.Train.maf01.vcf.h5 \
    --train_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/VCF/Train/train.980.sample.panel \
    \
    --val_dataset /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/validation/val_mask30.h5 \
    --val_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/VCF/Train/train.980.sample.panel \
    \
    --refpanel_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/KGP.chr21.Panel.maf01.vcf.gz \
    --freq_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/Freq.npy \
    --window_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/segments_chr21.maf.csv \
    --type_path data/type_to_idx.bin \
    --pop_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/pop_to_idx.bin \
    --pos_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/pos_to_idx.bin \
    \
    --output_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_with_val/rag_bert.model \
    \
    --dims 128 \
    --layers 8 \
    --attn_heads 4 \
    --train_batch_size 64 \
    --val_batch_size 128 \
    --epochs 20 \
    --cuda_devices 0 \
    --log_freq 1000 \
    \
    --rag_k 1 \
    --grad_accum_steps 1 \
    \
    --patience 5 \
    --val_metric f1 \
    --min_delta 0.001
```

**关键参数说明**：

```bash
# === 数据参数 ===
--train_dataset         # 训练H5文件
--train_panel           # 训练panel
--val_dataset           # 验证H5文件（新增）
--val_panel             # 验证panel（新增）

# === 模型参数 ===
--dims 128              # 你之前成功的配置
--layers 8
--attn_heads 4

# === 训练参数 ===
--train_batch_size 64   # 训练batch size
--val_batch_size 128    # 验证batch size（可以更大，不需要反向传播）
--epochs 20

# === 显存优化参数 ===
--rag_k 1               # ⭐ 关键！从3降到1，节省60-70%显存
--grad_accum_steps 1    # 如果显存还不够，可以设为2或4

# === Validation & Early Stopping ===
--patience 5            # 5个epoch不改进就停止
--val_metric f1         # 监控F1分数
--min_delta 0.001       # 最小改进阈值
```

---

### 步骤4：开始训练

```bash
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup

chmod +x run_v11_with_val.sh
bash run_v11_with_val.sh
```

---

## 📈 训练输出示例

### 每个Epoch你将看到：

```
============================================================
Epoch 1 - TRAINING
============================================================
EP_Train:0: 100%|████████| 1000/1000 [15:23<00:00]
============================================================
Epoch 1 TRAIN Summary
============================================================
Avg Loss:      0.6234
Avg Accuracy:  0.7123
Haplotype Metrics:
  - F1:        0.7045
  - Precision: 0.7189
  - Recall:    0.6905
Genotype Metrics:
  - Class 0 F1: 0.8234
  - Class 1 F1: 0.6543
  - Class 2 F1: 0.6789
  - Class 3 F1: 0.7012
  - Avg F1:    0.7145
============================================================

============================================================
Epoch 1 - VALIDATION
============================================================
EP_Val:0: 100%|████████| 100/100 [02:15<00:00]
============================================================
Epoch 1 VAL Summary
============================================================
Avg Loss:      0.7123
Avg Accuracy:  0.6834
Haplotype Metrics:
  - F1:        0.6712  ← 每个epoch都能看到validation效果！
  - Precision: 0.6890
  - Recall:    0.6542
Genotype Metrics:
  - Class 0 F1: 0.7834
  - Class 1 F1: 0.6123
  - Class 2 F1: 0.6345
  - Class 3 F1: 0.6712
  - Avg F1:    0.6754
============================================================

✓ New best f1: 0.6712
EP:1 Model Saved: .../rag_bert.model.best.pth
```

### Early Stopping示例：

```
Epoch 5 VAL Summary
...
⚠ No improvement for 1 epochs (best f1: 0.7456)

Epoch 6 VAL Summary
...
⚠ No improvement for 2 epochs (best f1: 0.7456)

...

Epoch 10 VAL Summary
...
⚠ No improvement for 5 epochs (best f1: 0.7456)

============================================================
⛔ Early stopping triggered! No improvement for 5 epochs.
Training stopped early at epoch 10
Best f1: 0.7456
Best model saved: .../rag_bert.model.best.pth
============================================================
```

---

## 🔧 高级配置

### 配置1：多难度Validation

如果你想同时评估多个mask比例：

```python
# 修改 src/train_with_val.py，添加多个验证集
val_loaders = {
    'easy': val_loader_10,
    'medium': val_loader_30,
    'hard': val_loader_50
}

for name, val_loader in val_loaders.items():
    print(f"\nValidation on {name}:")
    trainer.validate(epoch, val_loader)
```

### 配置2：显存不足怎么办？

如果训练时显存OOM，尝试以下组合：

```bash
# 组合1：降低RAG K值
--rag_k 1                    # 从3降到1，节省最多显存

# 组合2：减小batch size + 梯度累积
--train_batch_size 32        # batch减半
--grad_accum_steps 2         # 梯度累积，等效batch=64

# 组合3：验证时用更大batch（验证不需要反向传播）
--val_batch_size 256         # 验证batch可以很大
```

### 配置3：更激进的Early Stopping

```bash
# 更早停止（适合快速实验）
--patience 3
--min_delta 0.005

# 更宽容的Early Stopping（适合长时间训练）
--patience 10
--min_delta 0.0001
```

---

## 📊 如何使用TestData/Test1-Test2？

`TestData/Test1` 和 `Test2` 应该保留作为**最终测试集**，不用于训练调参：

```bash
# 在训练完成后，用Test1评估最终性能
python infer.py \
    --test_dataset data/validation_test1/test1_mask30.h5 \
    --model_path output_with_val/rag_bert.model.best.pth \
    ...
```

**用途区分**：
- **Validation（Masked_VCFs/）**：训练时每个epoch评估，用于Early Stopping和选择最佳模型
- **Test（TestData/Test1-Test2）**：训练完成后最终评估，用于报告性能

---

## ❓ 常见问题

### Q1: 我的验证集样本和训练集重复吗？

**A**: 需要检查。如果你的`TestTruth.vcf.gz`包含的样本与`Train.vcf.h5`不同，那就是独立的验证集（最好）。如果重复，建议：
- 从训练集中移除这些样本
- 或使用Cross-Validation

### Q2: 为什么不直接从训练集划分validation？

**A**: 你已经有专门的测试数据（TestData），利用它们更好：
- ✅ 独立评估，避免数据泄露
- ✅ 不浪费训练数据
- ✅ 测试数据已经做好了mask

### Q3: Mask30太简单或太难怎么办？

**A**: 先用Mask30训练一轮，观察验证F1：
- 如果F1 > 0.9，说明太简单 → 换Mask50
- 如果F1 < 0.5，说明太难 → 换Mask10或Mask20

### Q4: 我能在本地Windows上准备数据吗？

**A**: 可以，但建议在服务器上：
- `prepare_val_data.py` 需要`allel`库读取VCF
- VCF文件通常很大，服务器处理更快
- H5文件生成后可以在训练中直接使用

---

## ✅ 检查清单

开始训练前确认：

- [ ] 已运行`prepare_val_data.py`转换验证集
- [ ] 已准备好验证集的panel文件
- [ ] 已修改训练脚本添加`--val_dataset`和`--val_panel`
- [ ] 已设置合理的`--rag_k`值（建议1-2）
- [ ] 已设置Early Stopping参数（`--patience`, `--val_metric`）
- [ ] 已确认输出目录有足够空间
- [ ] 已检查GPU显存是否充足

---

## 🎉 预期效果

使用Validation后，你将能够：

1. ✅ **每个epoch看到验证效果**，不再盲目训练
2. ✅ **自动选择最佳模型**，不需要手动挑选checkpoint
3. ✅ **防止过拟合**，通过Early Stopping及时停止
4. ✅ **节省训练时间**，不需要训练完整20个epochs
5. ✅ **可靠的性能评估**，通过独立验证集

**训练时间估计**（相比之前）：
- 每个epoch增加约15-20%时间（验证时间）
- 但总训练时间可能减少30-50%（Early Stopping）

祝训练顺利！🚀
