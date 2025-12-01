# ✅ Baseline训练检查清单

## 🎯 训练前检查

### 环境检查
```bash
# 1. 确认在正确目录
pwd
# 应该输出: /cpfs01/.../00_RAG-SNVBERT-packup

# 2. 检查数据文件
ls -lh data/train_val_split/
# 应该看到:
#   train_split.h5
#   train_panel.txt
#   val_split.h5
#   val_panel.txt

# 3. 检查GPU可用
nvidia-smi
# 确保至少一块GPU空闲

# 4. 创建日志目录
mkdir -p logs/baseline_gamma5_recon30
ls -ld logs/baseline_gamma5_recon30/
```

### 脚本检查
```bash
# 5. 确认训练脚本存在
ls -lh run_v12_split_val_with_log.sh
chmod +x run_v12_split_val_with_log.sh

# 6. 快速检查脚本内容
head -20 run_v12_split_val_with_log.sh
# 应该看到 LOG_DIR="logs/baseline_gamma5_recon30"
```

---

## 🚀 启动训练

### 终端1: 运行训练
```bash
bash run_v12_split_val_with_log.sh
```

**预期初始输出**:
```
================================================
Starting training with logging
================================================
Log directory: logs/baseline_gamma5_recon30
Log file: logs/baseline_gamma5_recon30/training_20250101_120534.log
================================================

============================================================
Loading Data...
============================================================
✓ Panel loaded
Initializing Vocab...
✓ Vocab size: 9

Loading Training Dataset...
▣ 开始构建FAISS索引
...
```

### 终端2: 监控训练 (可选)
```bash
# 等待第一个日志文件出现 (约1-2分钟)
ls -lh logs/baseline_gamma5_recon30/

# 实时查看日志
tail -f logs/baseline_gamma5_recon30/latest.log
```

---

## 📊 第一个Epoch检查 (~15-20分钟后)

### 检查点1: 训练是否正常启动

```bash
# 查看最新日志
tail -50 logs/baseline_gamma5_recon30/latest.log
```

**健康信号**: 应该看到类似
```
EP_Train:0:  15%|███▌              | 645/4305 [02:15<13:02, 4.68it/s]
```

**问题信号**:
- ❌ 没有进度条更新 → 训练卡住
- ❌ 出现ERROR/Exception → 代码错误
- ❌ NaN/Inf → 数值不稳定

### 检查点2: 第一个Epoch Summary

```bash
# 等待第一个epoch完成 (~15-20分钟)
# 查看summary
grep 'Epoch 1.*Summary' -A 15 logs/baseline_gamma5_recon30/latest.log
```

**预期输出**:
```
============================================================
Epoch 1 TRAIN Summary
============================================================
Avg Loss:      0.6234  ← 应该在0.5-0.8之间
Avg Accuracy:  0.7123  ← 应该>0.5

Haplotype Metrics:
  - F1:        0.6823  ← 第一个epoch通常0.6-0.75
  - Precision: 0.6945
  - Recall:    0.6705

============================================================
Epoch 1 VAL Summary
============================================================
Avg Loss:      0.6512
Avg Accuracy:  0.6987

Haplotype Metrics:
  - F1:        0.6512  ← Val F1通常略低于Train
  - Precision: 0.6634
  - Recall:    0.6395
```

**健康判断**:
- ✅ Loss在合理范围 (0.5-0.8)
- ✅ Accuracy > 0.5 (否则比随机猜测还差)
- ✅ Train F1 > Val F1 (正常)
- ✅ Train F1 - Val F1 < 0.1 (没有严重过拟合)

**问题信号**:
- ❌ Loss > 1.5 → 可能学习率过大
- ❌ Loss < 0.1 → 可能数据/代码问题
- ❌ Accuracy < 0.5 → 模型没有学习
- ❌ Val F1 > Train F1 → 异常 (数据泄漏?)

---

## 🔍 前5个Epoch观察 (~1.5小时后)

### 检查点3: F1趋势

```bash
# 提取前5个epoch的val F1
grep 'VAL Summary' -A 10 logs/baseline_gamma5_recon30/latest.log | \
    grep 'F1:' | head -5
```

**健康信号** (稳定增长):
```
Epoch 1: F1: 0.6512
Epoch 2: F1: 0.6734  ← +0.0222
Epoch 3: F1: 0.6912  ← +0.0178
Epoch 4: F1: 0.7023  ← +0.0111
Epoch 5: F1: 0.7145  ← +0.0122
```
判断: ✅ 每个epoch稳定提升

**问题信号** (震荡):
```
Epoch 1: F1: 0.6512
Epoch 2: F1: 0.6234  ← 下降!
Epoch 3: F1: 0.6756  ← 大幅波动
Epoch 4: 0.6423  ← 继续波动
Epoch 5: F1: 0.6834
```
判断: ⚠️ 训练不稳定 (可能是gamma=5的问题)

### 检查点4: 过拟合检查

```bash
# 对比train和val F1
grep -E 'Epoch [1-5] (TRAIN|VAL) Summary' -A 5 logs/baseline_gamma5_recon30/latest.log | \
    grep 'F1:' | paste - -
```

**预期输出**:
```
Epoch 1 TRAIN F1: 0.7045    Epoch 1 VAL F1: 0.6512    Gap: 0.0533
Epoch 2 TRAIN F1: 0.7234    Epoch 2 VAL F1: 0.6734    Gap: 0.0500
Epoch 3 TRAIN F1: 0.7412    Epoch 3 VAL F1: 0.6912    Gap: 0.0500
Epoch 4 TRAIN F1: 0.7545    Epoch 4 VAL F1: 0.7023    Gap: 0.0522
Epoch 5 TRAIN F1: 0.7678    Epoch 5 VAL F1: 0.7145    Gap: 0.0533
```

**健康判断**:
- ✅ Gap稳定在0.03-0.06 → 轻微过拟合,可接受
- ✅ Gap没有明显增大 → 未恶化

**问题信号**:
- ⚠️ Gap > 0.10 → 明显过拟合
- 🔴 Gap持续增大 (0.05 → 0.10 → 0.15) → 过拟合加剧

### 检查点5: 生成中期分析

```bash
# 5个epoch后生成分析
python scripts/analyze_training_log.py \
    logs/baseline_gamma5_recon30/latest.log \
    --output logs/analysis/

# 查看生成的图表
ls -lh logs/analysis/*.png
```

**查看图表** (如果有图形界面):
```bash
# 查看F1/Loss曲线
xdg-open logs/analysis/*_analysis.png  # Linux
# 或
open logs/analysis/*_analysis.png      # macOS
```

---

## 🎯 决策点 (5个Epoch后)

### 场景A: 一切正常 ✅

**信号**:
- Val F1稳定增长
- Gap < 0.08
- Loss平滑下降

**行动**: 继续训练到early stopping或20 epochs

```bash
# 继续监控
tail -f logs/baseline_gamma5_recon30/latest.log | \
    grep 'Summary' -A 10
```

---

### 场景B: 训练不稳定 ⚠️

**信号**:
- Val F1震荡
- Loss波动大
- Gap不稳定

**分析**:
```bash
# 查看loss曲线
grep 'Avg Loss:' logs/baseline_gamma5_recon30/latest.log | head -10
```

**行动**:
1. 继续观察2-3个epochs
2. 如果持续震荡 → 考虑提前停止并优化
3. 记录问题 → 为优化版本提供对比

---

### 场景C: 严重过拟合 🔴

**信号**:
- Train F1 - Val F1 > 0.15
- Val F1不增长,Train F1持续增长

**示例**:
```
Epoch 3 TRAIN: 0.7856, VAL: 0.6234  Gap=0.1622
Epoch 4 TRAIN: 0.8123, VAL: 0.6189  Gap=0.1934  ← 恶化
Epoch 5 TRAIN: 0.8345, VAL: 0.6156  Gap=0.2189  ← 继续恶化
```

**行动**:
1. 提前停止训练 (已经有baseline数据)
2. 立即应用优化 (gamma=2.5, no recon)
3. 对比优化效果

---

## 📝 训练完成后分析

### 完整分析

```bash
# 1. 生成完整分析报告
python scripts/analyze_training_log.py \
    logs/baseline_gamma5_recon30/latest.log \
    --output logs/analysis/

# 2. 查看关键指标
cat << 'EOF'
============================================================
Baseline Training Summary
============================================================
EOF

# 最佳validation F1
echo "Best Val F1:"
grep 'New best f1:' logs/baseline_gamma5_recon30/latest.log | tail -1

# 收敛速度
echo -e "\nConvergence Speed:"
grep 'VAL Summary' -A 10 logs/baseline_gamma5_recon30/latest.log | \
    grep 'F1:' | \
    awk '{print NR, $3}' | \
    awk '$2 > 0.7 {print "Epochs to F1>0.7:", NR; exit}'

# 是否触发early stopping
echo -e "\nEarly Stopping:"
grep 'Early stopping' logs/baseline_gamma5_recon30/latest.log || \
    echo "Completed all epochs (no early stopping)"

# 总epochs
echo -e "\nTotal Epochs Trained:"
grep 'Epoch.*Summary' logs/baseline_gamma5_recon30/latest.log | \
    tail -1 | grep -oP 'Epoch \K\d+'
```

---

## 🗂️ 保存Baseline结果

```bash
# 创建baseline总结文件
cat > logs/baseline_gamma5_recon30/SUMMARY.txt << 'EOF'
Baseline Training Configuration
================================

Model Config:
- Focal gamma: 5
- Recon loss weight: 30% (0.15+0.15)
- RAG K: 1
- Batch size: 64 (train), 128 (val)

Training Config:
- Epochs: 20
- Patience: 5
- Learning rate: 1e-5

Results:
- Best Val F1: [FILL]
- @Epoch: [FILL]
- Final Val F1: [FILL]
- Overfitting Gap: [FILL]
- Epochs to F1>0.7: [FILL]
- Training time: [FILL]

Issues Observed:
- [ ] Val F1震荡
- [ ] 严重过拟合
- [ ] Loss不稳定
- [ ] 收敛慢

Next Steps:
- [ ] 应用优化 (gamma=2.5, no recon)
- [ ] 对比分析
EOF

# 手动填写结果
nano logs/baseline_gamma5_recon30/SUMMARY.txt
```

---

## 🚀 准备优化版本

### 如果决定优化

```bash
# 1. 备份当前代码
cp src/main/pretrain_with_val.py src/main/pretrain_with_val.py.baseline

# 2. 创建优化版本的issue tracking
cat > logs/OPTIMIZATION_PLAN.md << 'EOF'
# Optimization Plan

## Baseline Issues
- [ ] Issue 1: ...
- [ ] Issue 2: ...

## Optimization Config
- Focal gamma: 5 → 2.5
- Recon loss: 30% → 0%

## Expected Improvements
- Val F1: +5-10%
- Convergence: 2-3x faster
- Stability: Loss曲线更平滑

## Timeline
- Day 1: Baseline完成
- Day 2: 修改代码,启动优化训练
- Day 3: 对比分析
EOF
```

---

## 📞 遇到问题检查

### 问题: 训练卡住不动

```bash
# 检查进程是否存活
ps aux | grep python | grep train_with_val

# 检查GPU使用
nvidia-smi

# 查看最后几行日志
tail -20 logs/baseline_gamma5_recon30/latest.log
```

### 问题: 显存不足 (OOM)

```bash
# 查看错误
grep -i 'out of memory\|oom' logs/baseline_gamma5_recon30/latest.log

# 解决: 降低batch size
# 编辑 run_v12_split_val_with_log.sh
--train_batch_size 32
--val_batch_size 64
```

### 问题: 日志文件没有创建

```bash
# 检查目录权限
ls -ld logs/baseline_gamma5_recon30/

# 手动创建
mkdir -p logs/baseline_gamma5_recon30

# 检查tee命令
which tee
```

---

## ✅ 最终检查清单

### 训练前
- [ ] GPU可用 (`nvidia-smi`)
- [ ] 数据文件存在 (`ls data/train_val_split/`)
- [ ] 日志目录创建 (`mkdir -p logs/baseline_gamma5_recon30`)
- [ ] 脚本可执行 (`chmod +x run_v12_split_val_with_log.sh`)

### 训练中 (每1-2小时)
- [ ] 检查训练进度 (`tail logs/baseline_gamma5_recon30/latest.log`)
- [ ] 验证F1趋势 (`grep 'VAL.*F1:'`)
- [ ] 监控过拟合 (`对比train和val F1`)

### 5个Epoch后
- [ ] 生成中期分析 (`python scripts/analyze_training_log.py`)
- [ ] 决定继续或优化

### 训练完成后
- [ ] 完整分析 (`analyze_training_log.py --output`)
- [ ] 填写SUMMARY.txt
- [ ] 保存baseline checkpoint
- [ ] 准备优化版本

---

**现在开始baseline训练!** 🚀

```bash
bash run_v12_split_val_with_log.sh
```
