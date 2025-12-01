# 增强版输出使用指南

## 🎯 新增功能

增强版训练脚本 **不修改任何模型架构**，只增强输出信息：

### ✅ 新增输出

1. **Rare vs Common F1分解**
   - Rare: MAF < 0.05 (可调整)
   - Common: MAF >= 0.05
   - 每个epoch输出两者的F1/Precision/Recall

2. **CSV指标保存**
   - 每个epoch自动保存到CSV
   - 包含所有指标（overall, rare, common）
   - 便于后续分析和绘图

3. **详细日志**
   - 保留原有的所有输出
   - 添加Rare/Common分解到summary
   - 同时输出到终端和文件

---

## 📝 快速开始

### 1. Pull代码到服务器

```bash
# 本地 (Windows)
git add .
git commit -m "Add enhanced output with Rare/Common F1"
git push origin main

# 服务器
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup
git pull origin main
```

### 2. 运行增强版训练

```bash
# 确保脚本可执行
chmod +x run_v12_enhanced_with_log.sh

# 运行训练
bash run_v12_enhanced_with_log.sh
```

---

## 📊 输出示例

### 终端/日志输出

```
============================================================
Epoch 1 VAL Summary
============================================================
Avg Loss:      0.6512
Avg Accuracy:  0.6987

Haplotype Metrics (Overall):
  - F1:        0.6823
  - Precision: 0.6945
  - Recall:    0.6705

Rare Variants (MAF<0.05):         ← 新增!
  - F1:        0.6234
  - Precision: 0.6456
  - Recall:    0.6023

Common Variants (MAF>=0.05):      ← 新增!
  - F1:        0.7123
  - Precision: 0.7245
  - Recall:    0.7005

Genotype Metrics:
  - Class 0 F1: 0.7234
  - Class 1 F1: 0.6512
  - Class 2 F1: 0.6834
  - Class 3 F1: 0.5923
  - Avg F1:    0.6626
============================================================
```

### CSV文件格式

```csv
epoch,mode,loss,accuracy,overall_f1,overall_precision,overall_recall,rare_f1,rare_precision,rare_recall,common_f1,common_precision,common_recall
1,train,0.6234,0.7123,0.7045,0.7189,0.6905,0.6512,0.6734,0.6298,0.7234,0.7398,0.7076
1,val,0.6512,0.6987,0.6823,0.6945,0.6705,0.6234,0.6456,0.6023,0.7123,0.7245,0.7005
2,train,0.5987,0.7345,0.7234,0.7389,0.7083,0.6734,0.6956,0.6521,0.7456,0.7598,0.7318
2,val,0.6289,0.7123,0.7012,0.7156,0.6871,0.6456,0.6678,0.6245,0.7312,0.7445,0.7183
...
```

---

## 🔍 实时监控

### 监控Overall F1

```bash
tail -f logs/baseline_gamma5_recon30/latest.log | \
    grep 'Haplotype Metrics (Overall)' -A 3
```

### 监控Rare vs Common F1

```bash
# Rare变异F1
tail -f logs/baseline_gamma5_recon30/latest.log | \
    grep 'Rare Variants' -A 3

# Common变异F1
tail -f logs/baseline_gamma5_recon30/latest.log | \
    grep 'Common Variants' -A 3
```

### 同时监控两者对比

```bash
tail -f logs/baseline_gamma5_recon30/latest.log | \
    grep -E '(Rare|Common) Variants' -A 3
```

---

## 📈 数据分析

### 1. 从CSV生成图表

```bash
# 绘制所有指标
python scripts/plot_metrics_csv.py \
    metrics/baseline_gamma5_recon30/latest.csv

# 指定输出目录
python scripts/plot_metrics_csv.py \
    metrics/baseline_gamma5_recon30/metrics_20250101_120000.csv \
    --output plots/
```

**生成的图表包含**:
1. Overall F1曲线 (Train vs Val)
2. Rare vs Common F1对比 (Val)
3. Loss曲线 (Train vs Val)
4. Validation P/R/F1
5. Rare变异详细指标
6. Common变异详细指标

### 2. 提取特定指标

```bash
# 提取validation的rare和common F1
cat metrics/baseline_gamma5_recon30/latest.csv | \
    awk -F',' 'NR==1 || $2=="val" {print $1","$8","$11}' | \
    column -t -s','

# 输出示例:
# epoch  rare_f1  common_f1
# 1      0.6234   0.7123
# 2      0.6456   0.7312
# 3      0.6678   0.7456
# ...
```

### 3. 计算Rare vs Common Gap

```bash
# 使用Python快速计算
python << 'EOF'
import pandas as pd
df = pd.read_csv('metrics/baseline_gamma5_recon30/latest.csv')
val_df = df[df['mode'] == 'val']
val_df['gap'] = val_df['common_f1'] - val_df['rare_f1']
print(val_df[['epoch', 'rare_f1', 'common_f1', 'gap']])
EOF
```

---

## 🎯 关键观察指标

### 1. Rare vs Common性能差距

```bash
# 查看每个epoch的差距
grep -E '(Rare|Common) Variants' logs/baseline_gamma5_recon30/latest.log | \
    grep 'F1:' | \
    paste - - | \
    awk '{print "Epoch", NR, "Rare:", $3, "Common:", $6, "Gap:", $6-$3}'
```

**健康信号**:
```
Epoch 1: Rare: 0.6234, Common: 0.7123, Gap: 0.0889
Epoch 2: Rare: 0.6456, Common: 0.7312, Gap: 0.0856
Epoch 3: Rare: 0.6678, Common: 0.7456, Gap: 0.0778  ← Gap逐渐缩小
```

**问题信号**:
```
Epoch 1: Rare: 0.6234, Common: 0.7123, Gap: 0.0889
Epoch 2: Rare: 0.6123, Common: 0.7345, Gap: 0.1222  ← Gap扩大
Epoch 3: Rare: 0.6012, Common: 0.7512, Gap: 0.1500  ← Rare下降
```

### 2. Rare变异学习速度

```bash
# 提取rare F1趋势
grep 'Rare Variants' -A 1 logs/baseline_gamma5_recon30/latest.log | \
    grep 'F1:' | \
    awk '{print NR, $3}'
```

**期望**: Rare F1应该稳定增长

### 3. 整体性能

```bash
# 对比三个指标
grep -E 'Haplotype Metrics \(Overall\)|Rare Variants|Common Variants' \
    logs/baseline_gamma5_recon30/latest.log | \
    grep 'F1:' | \
    paste - - -
```

---

## 📊 对比不同配置

### 场景: 对比baseline和优化版本

```bash
# 假设有两个CSV文件
BASELINE_CSV="metrics/baseline_gamma5_recon30/latest.csv"
OPTIMIZED_CSV="metrics/optimized_gamma25_norecon/latest.csv"

# 对比rare F1
python << EOF
import pandas as pd

baseline = pd.read_csv('$BASELINE_CSV')
optimized = pd.read_csv('$OPTIMIZED_CSV')

baseline_val = baseline[baseline['mode'] == 'val']
optimized_val = optimized[optimized['mode'] == 'val']

print("Rare F1 Comparison:")
print("Epoch | Baseline | Optimized | Improvement")
print("------|----------|-----------|------------")
for idx in range(min(len(baseline_val), len(optimized_val))):
    b = baseline_val.iloc[idx]
    o = optimized_val.iloc[idx]
    imp = o['rare_f1'] - b['rare_f1']
    print(f"{b['epoch']:.0f}     | {b['rare_f1']:.4f}   | {o['rare_f1']:.4f}    | {imp:+.4f}")
EOF
```

---

## 🔧 调整Rare阈值

如果想调整Rare/Common的MAF阈值:

```bash
# 修改run_v12_enhanced_with_log.sh
--rare_threshold 0.05  # 默认

# 改为更严格的定义 (只有MAF<0.01算rare)
--rare_threshold 0.01

# 或更宽松 (MAF<0.1算rare)
--rare_threshold 0.1
```

重新训练后对比效果。

---

## 📁 文件组织

```
VCF-Bert/
├── logs/
│   └── baseline_gamma5_recon30/
│       ├── training_20250101_120000.log
│       └── latest.log -> training_20250101_120000.log
│
├── metrics/
│   └── baseline_gamma5_recon30/
│       ├── metrics_20250101_120000.csv     ← 新增!
│       ├── metrics_20250101_120000_plots.png  ← 新增!
│       └── latest.csv -> metrics_20250101_120000.csv
│
└── plots/  (可选)
    └── ...
```

---

## 🚀 完整工作流

### Day 1: 启动训练

```bash
# 1. Pull代码
git pull origin main

# 2. 运行增强版训练
bash run_v12_enhanced_with_log.sh

# 3. 新开终端监控
tail -f logs/baseline_gamma5_recon30/latest.log | \
    grep -E '(Overall|Rare|Common)' -A 1
```

### Day 1-2: 训练中 (每1-2小时检查)

```bash
# 查看最新epoch的rare vs common
tail -50 logs/baseline_gamma5_recon30/latest.log | \
    grep -E 'Epoch.*Summary|Rare|Common' -A 1

# 生成图表
python scripts/plot_metrics_csv.py \
    metrics/baseline_gamma5_recon30/latest.csv
```

### Day 2-3: 训练完成后

```bash
# 1. 生成完整分析
python scripts/plot_metrics_csv.py \
    metrics/baseline_gamma5_recon30/latest.csv \
    --output analysis/

# 2. 查看rare vs common趋势
cat metrics/baseline_gamma5_recon30/latest.csv | \
    awk -F',' 'NR==1 || $2=="val"' | \
    cut -d',' -f1,8,11 | \
    column -t -s','

# 3. 如果rare F1显著低于common → 考虑优化
```

---

## ❓ 常见问题

### Q: Rare F1很低怎么办?

**观察**:
```
Rare F1: 0.45
Common F1: 0.78
Gap: 0.33  ← 差距很大
```

**可能原因**:
1. Focal gamma=5忽略了简单的rare变异
2. 训练数据中rare样本少
3. 模型偏向common变异

**解决方案**:
1. 降低focal gamma (5 → 2.5)
2. 使用rare variant weighted sampling
3. 调整loss权重

---

### Q: Rare和Common F1都很低?

**观察**:
```
Overall F1: 0.55
Rare F1: 0.52
Common F1: 0.56
```

**可能原因**:
1. 整体训练有问题
2. 数据质量问题
3. 模型配置不当

**解决方案**:
1. 先优化overall F1
2. 检查数据和模型配置
3. 参考MODEL_ARCHITECTURE_ANALYSIS.md

---

### Q: Common F1高但Rare F1低?

**观察**:
```
Rare F1: 0.58
Common F1: 0.82
Gap: 0.24
```

**这是预期的!** Rare变异本身就更难预测。

**判断标准**:
- Gap < 0.15: ✅ 良好
- Gap 0.15-0.25: ⚠️ 可接受
- Gap > 0.25: 🔴 需要优化rare性能

---

## 📚 相关文档

- [TRAINING_COMPARISON_GUIDE.md](TRAINING_COMPARISON_GUIDE.md) - 完整训练对比指南
- [LOG_GUIDE.md](LOG_GUIDE.md) - 日志使用详细说明
- [MODEL_ARCHITECTURE_ANALYSIS.md](MODEL_ARCHITECTURE_ANALYSIS.md) - 架构优化建议
- [FOCAL_LOSS_ANALYSIS.md](FOCAL_LOSS_ANALYSIS.md) - 理解gamma对rare的影响

---

**现在可以开始增强版训练了!** 🚀

你将看到完整的Rare vs Common F1分解，帮助你更好地理解模型在不同频率变异上的表现！
