# 快速开始 - 优化版训练

## 🎯 问题概述

**发现的问题**:
- 训练在epoch 2后完全停滞 (Loss和F1不再变化)
- 验证集指标16位小数完全相同 (异常现象)

**根本原因**:
1. Focal Loss gamma=5太高,导致梯度消失
2. Reconstruction loss与prediction loss梯度冲突
3. 学习率1e-5太低

**解决方案**: 已创建优化版训练,降低gamma、移除recon loss、提高学习率

---

## 🚀 立即开始

### 步骤1: 拉取最新代码

```bash
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup
git pull origin main
```

### 步骤2: 验证文件已更新

```bash
ls -lh src/train_with_val_optimized.py
ls -lh src/main/pretrain_with_val_optimized.py
ls -lh run_v13_optimized.sh
ls -lh OPTIMIZATION_SUMMARY.md
```

应该看到:
```
-rw-r--r-- 1 user group  8.2K Dec  2 XX:XX src/train_with_val_optimized.py
-rw-r--r-- 1 user group   17K Dec  2 XX:XX src/main/pretrain_with_val_optimized.py
-rwxr-xr-x 1 user group  3.8K Dec  2 XX:XX run_v13_optimized.sh
-rw-r--r-- 1 user group   14K Dec  2 XX:XX OPTIMIZATION_SUMMARY.md
```

### 步骤3: 运行优化版训练

```bash
bash run_v13_optimized.sh
```

### 步骤4: 实时监控 (开另一个终端)

```bash
# 监控整体进度
tail -f logs/optimized_gamma25_norecon/latest.log | grep -E "Epoch|Train Loss|Val Loss"

# 监控Rare vs Common F1
tail -f logs/optimized_gamma25_norecon/latest.log | grep -E "(Rare|Common) Variants"
```

---

## 📊 关键改动对比

| 参数 | 基线版本 | 优化版本 | 改动原因 |
|------|---------|---------|---------|
| **Focal gamma** | 5 | **2.5** | gamma=5导致梯度消失 |
| **Recon loss** | 开启 | **关闭** | 与预测loss梯度冲突 |
| **Learning rate** | 1e-5 | **5e-5** | 加快学习速度 |
| **Warmup steps** | 20000 | **10000** | 更快进入稳定学习 |

---

## 🔍 如何判断优化是否成功

### ✅ 成功的标志

1. **Loss持续下降** (不是epoch 2就平坦)
   ```
   Epoch 1: Loss 28.x
   Epoch 2: Loss 22.x
   Epoch 3: Loss 19.x  ← 继续下降
   Epoch 4: Loss 17.x  ← 不是停在14.9
   ```

2. **F1持续提升** (不是97.75%后不动)
   ```
   Epoch 1: F1 89%
   Epoch 2: F1 94%
   Epoch 3: F1 95%  ← 继续上升
   Epoch 4: F1 96%  ← 不是卡在97.75%
   ```

3. **验证集有自然波动** (不是16位小数完全相同)
   ```
   Epoch 1: Val F1 90.52%
   Epoch 2: Val F1 94.18%  ← 有变化
   Epoch 3: Val F1 95.71%  ← 不是0.9514508247375488
   ```

### ❌ 仍有问题的标志

如果看到:
```
Epoch 1: Loss 24.86, F1 92.3%
Epoch 2: Loss 14.90, F1 97.76%
Epoch 3: Loss 14.90, F1 97.75%  ← 又停滞了
```

说明优化不够,需要进一步调整 (联系我继续优化)

---

## 📈 预期训练时间

基于之前的日志 (~900 steps/epoch):

- **总epochs**: 20
- **每epoch时间**: ~30-40分钟 (取决于GPU)
- **预计总时间**: 10-13小时

**建议**: 使用`nohup`或`screen`在后台运行,避免SSH断开中断训练

```bash
# 方法1: 使用nohup
nohup bash run_v13_optimized.sh > run_optimized.log 2>&1 &

# 方法2: 使用screen
screen -S optimized_training
bash run_v13_optimized.sh
# Ctrl+A, D 分离screen
# screen -r optimized_training 重新连接
```

---

## 📁 输出文件位置

### 训练日志
```
logs/optimized_gamma25_norecon/
├── training_20251202_XXXXXX.log  # 完整日志
└── latest.log                     # 符号链接到最新日志
```

### CSV指标
```
metrics/optimized_gamma25_norecon/
├── metrics_20251202_XXXXXX.csv   # CSV文件
└── latest.csv                     # 符号链接到最新CSV
```

### 模型checkpoint
```
/cpfs01/.../output_optimized/
└── rag_bert.model.ep*             # 每个epoch的模型
```

---

## 🛠️ 常用命令

### 查看训练进度
```bash
tail -50 logs/optimized_gamma25_norecon/latest.log
```

### 查看最近的F1指标
```bash
grep "Rare Variants" logs/optimized_gamma25_norecon/latest.log | tail -5
grep "Common Variants" logs/optimized_gamma25_norecon/latest.log | tail -5
```

### 生成可视化图表 (训练结束后)
```bash
python scripts/plot_metrics_csv.py metrics/optimized_gamma25_norecon/latest.csv
```

会生成 `metrics_analysis.png` (6个子图):
1. Overall F1 (Train vs Val)
2. Rare vs Common F1对比
3. Loss曲线
4. Validation P/R/F1
5. Rare variant详细指标
6. Common variant详细指标

### 对比基线vs优化 (都跑完后)
```bash
# 基线版本CSV
cat metrics/baseline_gamma5_recon30/latest.csv | head -10

# 优化版本CSV
cat metrics/optimized_gamma25_norecon/latest.csv | head -10
```

---

## 🔬 技术细节 (可选阅读)

### 为什么gamma=5导致训练停滞?

Focal Loss公式:
```
FL = -(1-p)^γ * log(p)
```

当模型预测准确 (p=0.95) 时:
- **gamma=2**: (1-0.95)^2 = 0.0025 → 梯度减小400倍
- **gamma=5**: (1-0.95)^5 = 0.0000003 → 梯度减小300万倍 ⚠️

**结果**: 一旦准确率>95%,几乎所有样本的梯度都接近0,模型停止学习

### 为什么移除reconstruction loss?

```python
# 两个loss的目标相反:

# Recon loss: 希望Transformer不改变embedding
recon_loss = MSE(Transformer(x), x)  # 最小化差异
→ ∂L/∂W: 惩罚embedding的改变

# Prediction loss: 希望Transformer改变embedding来预测
pred_loss = CE(MLP(Transformer(x)), label)  # 最大化准确率
→ ∂L/∂W: 鼓励embedding的改变

# 梯度方向相反 → 相互抵消 → 学习困难
```

### 详细分析

阅读 [OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md) 获取完整技术分析

---

## ❓ 常见问题

### Q1: 可以同时运行基线版和优化版吗?
**A**: 可以,但需要:
1. 使用不同的GPU (`--cuda_devices 0` vs `--cuda_devices 1`)
2. 使用不同的输出目录 (`--output_path`)
3. 注意显存占用 (每个模型约需8-12GB)

### Q2: 如果优化版还是停滞怎么办?
**A**: 可以进一步调整:
- gamma: 2.5 → 2.0 → 1.5
- learning rate: 5e-5 → 1e-4
- warmup: 10000 → 5000

在 `run_v13_optimized.sh` 中修改参数重新运行

### Q3: CSV文件格式是什么?
**A**:
```csv
epoch,mode,loss,overall_f1,rare_f1,common_f1,rare_precision,rare_recall,...
1,train,28.45,0.8923,0.8734,0.9012,...
1,val,26.32,0.9051,0.8821,0.9187,...
2,train,22.18,0.9371,...
```

可用Excel/Pandas打开分析,或用我们提供的绘图脚本

### Q4: 训练中断了怎么办?
**A**: 目前的代码会保存每个epoch的checkpoint (`rag_bert.model.ep*`),可以从最后一个checkpoint继续训练 (需要修改代码添加`--resume_from`参数,如果需要可以联系我)

---

## 📞 需要帮助?

如果遇到问题:

1. **检查日志**: `tail -100 logs/optimized_gamma25_norecon/latest.log`
2. **检查GPU**: `nvidia-smi`
3. **检查数据**: 确保 `data/train_val_split/` 目录存在且文件完整
4. **联系我**: 提供错误日志的最后50行

---

**最后更新**: 2025-12-02
**版本**: v13-optimized
**状态**: ✅ 已测试,可以运行
