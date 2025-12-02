# 训练优化总结 - 修复训练停滞问题

## 🔍 问题诊断

### 观察到的症状
1. **训练在Epoch 2后停滞**
   - Epoch 1: Train F1 = 92.3%, Loss = 24.86
   - Epoch 2: Train F1 = 97.76%, Loss = 14.90 (巨大跳跃)
   - Epoch 3-6: Train F1 = ~97.75%, Loss = ~14.9 (完全平坦)

2. **验证集指标完全不变**
   - Rare F1: 0.9514508247375488 (16位小数完全一致)
   - Common F1: 0.9807226061820984 (每个epoch完全相同)

### 根本原因分析

#### 1. Focal Loss Gamma过高 (gamma=5)
```python
# 原代码
FocalLoss(gamma=5)  # 过度关注难样本
```

**问题**:
- Focal Loss公式: `FL = -(1-p)^gamma * log(p)`
- gamma=5时,对于p=0.9的样本: `(1-0.9)^5 = 0.00001` (梯度几乎为0)
- 一旦模型达到~98%准确率,几乎所有样本的梯度都接近0
- 模型停止学习

**标准值对比**:
- **论文推荐**: gamma=2 (Focal Loss原论文)
- **目前使用**: gamma=5 (过于激进)
- **优化值**: gamma=2.5 (折中方案)

#### 2. Reconstruction Loss梯度冲突

```python
# 原代码 - 两个相反的目标
recon_loss = MSE(transformed_emb, original_emb)  # 希望embedding不变
prediction_loss = CrossEntropy(...)              # 希望embedding改变以更好预测
```

**问题**:
- **Recon loss**: 希望Transformer不改变embedding (MSE最小化)
- **Prediction loss**: 希望Transformer改变embedding来预测mask (准确率最大化)
- 两个loss的梯度方向相反,导致优化困难

**实验证据**:
```python
# 原loss权重
total_loss = 0.2*hap1 + 0.2*hap2 + 0.3*gt + 0.15*recon1 + 0.15*recon2
# recon占比 30%, 阻碍了预测任务的学习
```

#### 3. 学习率过低

```python
lr = 1e-5  # 原值
warmup_steps = 20000  # 原值
```

**问题**:
- 与Focal gamma=5结合时,梯度已经很小
- 学习率再很小,权重更新幅度微乎其微
- 导致模型快速收敛到局部最优

---

## ✅ 优化方案

### 核心改动

| 参数 | 原值 | 优化值 | 理由 |
|------|------|--------|------|
| `focal_gamma` | 5 | 2.5 | 减轻难样本过度关注,保持梯度流动 |
| `use_recon_loss` | true | false | 避免梯度冲突 |
| `learning_rate` | 1e-5 | 5e-5 | 加快学习,配合降低的gamma |
| `warmup_steps` | 20000 | 10000 | 更快进入稳定学习阶段 |

### 代码实现

#### 1. 新增可配置参数 ([train_with_val_optimized.py](src/train_with_val_optimized.py))

```python
parser.add_argument("--focal_gamma", type=float, default=2.5,
                   help="Focal Loss gamma (默认2.5, 原版5)")
parser.add_argument("--use_recon_loss", type=str, default="false",
                   choices=["true", "false"],
                   help="是否使用reconstruction loss (默认false)")
parser.add_argument("--lr", type=float, default=5e-5,
                   help="学习率 (优化: 5e-5)")
```

#### 2. 修改Trainer类 ([pretrain_with_val_optimized.py](src/main/pretrain_with_val_optimized.py))

```python
class BERTTrainerWithValidationOptimized:
    def __init__(
        self,
        # ... 其他参数
        focal_gamma: float = 2.5,      # 新增
        use_recon_loss: bool = False,  # 新增
        lr: float = 5e-5,              # 修改默认值
        warmup_steps=10000,            # 修改默认值
    ):
        # 使用可配置的focal gamma
        self.hap_criterion = FocalLoss(gamma=focal_gamma, reduction='sum')
        self.gt_criterion = FocalLoss(gamma=focal_gamma, reduction='sum')

        # 根据配置决定是否使用recon loss
        if self.use_recon_loss:
            # ... 计算recon loss
        else:
            total_loss = 3 * hap_1_loss + 3 * hap_2_loss + 4 * gt_loss
```

#### 3. 新增运行脚本 ([run_v13_optimized.sh](run_v13_optimized.sh))

```bash
python -m src.train_with_val_optimized \
    --focal_gamma 2.5 \
    --use_recon_loss false \
    --lr 5e-5 \
    --warmup_steps 10000 \
    --rare_threshold 0.05 \
    --metrics_csv ${METRICS_CSV} \
    # ... 其他参数
```

---

## 🧪 预期效果

### 训练行为改变

**之前 (gamma=5 + recon loss)**:
```
Epoch 1: Loss 24.86 → F1 92.3%
Epoch 2: Loss 14.90 → F1 97.76%  ← 巨大跳跃后停滞
Epoch 3-6: Loss ~14.9 → F1 ~97.75%  ← 完全不动
```

**优化后 (gamma=2.5, no recon)**:
```
Epoch 1: Loss 应该更高 (更多梯度流动)
Epoch 2-6: Loss 应该逐渐下降 (持续学习)
Epoch 10+: 可能达到更高F1 (98%+)
```

### 验证集行为改变

**之前**:
- Rare F1: 0.9514508... (完全不变,16位小数一致)
- Common F1: 0.9807226... (完全不变)

**优化后**:
- 应该看到轻微波动 (±0.001-0.01)
- 可能整体趋势上升
- 不应该16位小数完全相同

### Loss曲线改变

**之前**:
```
Loss: 24.86 → 14.90 → 14.9 → 14.9 → ... (L形曲线)
```

**优化后**:
```
Loss: 应该平滑下降 (逐渐收敛曲线)
```

---

## 📊 如何验证优化效果

### 1. 运行优化版训练

```bash
# 在服务器上
cd /cpfs01/projects-HDD/.../00_RAG-SNVBERT-packup
git pull origin main
bash run_v13_optimized.sh
```

### 2. 监控关键指标

#### 实时监控 (训练中)
```bash
# 查看实时日志
tail -f logs/optimized_gamma25_norecon/latest.log | grep "Epoch"

# 查看Rare vs Common F1
tail -f logs/optimized_gamma25_norecon/latest.log | grep -E "(Rare|Common) Variants"
```

#### 训练后分析
```bash
# 生成图表
python scripts/plot_metrics_csv.py metrics/optimized_gamma25_norecon/latest.csv

# 查看CSV
head -20 metrics/optimized_gamma25_norecon/latest.csv
```

### 3. 对比基线vs优化

| 指标 | 基线 (gamma=5) | 优化 (gamma=2.5) | 期望改进 |
|------|---------------|------------------|----------|
| Epoch 6 Train F1 | 97.75% | ? | 持续上升 |
| Epoch 6 Val F1 | 97.8% (不变) | ? | 有波动 |
| Loss曲线 | L形 (快速停滞) | ? | 平滑下降 |
| Rare F1 | 95.1% (不变) | ? | 可能更高 |

---

## 🚀 运行步骤

### 服务器端操作

```bash
# 1. 拉取最新代码
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup
git pull origin main

# 2. 验证文件存在
ls -lh src/train_with_val_optimized.py
ls -lh src/main/pretrain_with_val_optimized.py
ls -lh run_v13_optimized.sh

# 3. 运行优化版训练
bash run_v13_optimized.sh

# 4. 实时监控
# 在另一个终端中:
tail -f logs/optimized_gamma25_norecon/latest.log
```

### 输出文件位置

```
logs/optimized_gamma25_norecon/
├── training_YYYYMMDD_HHMMSS.log  # 完整训练日志
└── latest.log                     # 符号链接到最新日志

metrics/optimized_gamma25_norecon/
├── metrics_YYYYMMDD_HHMMSS.csv   # CSV指标
└── latest.csv                     # 符号链接到最新CSV

/cpfs01/.../output_optimized/
└── rag_bert.model.ep*             # 模型checkpoint
```

---

## 🔬 技术细节

### Focal Loss数学推导

```
标准CE Loss: L_CE = -log(p)

Focal Loss: L_FL = -(1-p)^γ * log(p)

当p接近1时 (模型预测准确):
  γ=0:  (1-0.9)^0 = 1.0     → 梯度正常
  γ=2:  (1-0.9)^2 = 0.01    → 梯度减小100倍
  γ=5:  (1-0.9)^5 = 0.00001 → 梯度减小10万倍 ⚠️

结论: γ=5时,一旦准确率>95%,梯度几乎消失
```

### Reconstruction Loss冲突证据

```python
# Forward pass
x = original_embedding
x_transformed = Transformer(x)
prediction = MLP(x_transformed)

# Loss 1: Prediction loss (希望x_transformed变化)
L_pred = CrossEntropy(prediction, label)
∂L_pred/∂Transformer_weights → 鼓励改变embedding

# Loss 2: Reconstruction loss (希望x_transformed不变)
L_recon = MSE(x_transformed, x)
∂L_recon/∂Transformer_weights → 惩罚改变embedding

# 总loss
L_total = 0.6*L_pred + 0.3*L_recon
# 梯度方向相反,相互抵消!
```

### Mask机制总结 (用户之前的问题)

1. **Mask位置**: 在dataset初始化时生成,每个window固定mask位置
2. **Mask比例**: 从10%开始,每个epoch增长10% (add_level)
3. **Mask token**: vocab中的index=4 (`<mask>`)
4. **Mask范围**: 只mask基因型,不mask metadata (POS/CHROM等)

```python
# Dataset.__init__
self.window_masks = [generate_mask(len) for window in windows]
self.__mask_rate = [0.1, 0.2, 0.3, ..., 0.8]

# Dataset.__getitem__
mask = self.window_masks[window_idx]  # 固定位置
hap_masked = tokenize(hap_original, mask)  # 替换为<mask>

# add_level()每个epoch调用
self.__level = min(self.__level + 1, 7)  # 0→1→2... (10%→20%→30%...)
```

---

## 📈 成功标准

### ✅ 训练成功的标志

1. **Loss持续下降**
   - 不应该在epoch 2就停滞
   - 应该看到平滑的下降曲线

2. **F1持续提升**
   - Train F1应该从epoch 2继续上升
   - 不应该97.75%后完全不动

3. **验证集有波动**
   - Val F1不应该16位小数完全相同
   - 应该看到±0.001-0.01的自然波动

4. **Rare F1改进**
   - 目前Rare F1=95.1%低于Common F1=98.1%
   - 优化后Rare F1可能提升到96-97%

### ⚠️ 仍需警惕的问题

1. **过拟合**: 如果Train F1>>Val F1,需要调整正则化
2. **欠拟合**: 如果Train和Val F1都很低,需要增加模型容量
3. **训练太快**: 如果loss下降过快,可能gamma还是太大

---

## 📝 日志示例

### 正常训练日志 (期望看到)

```
Epoch 1: Train Loss: 28.45, Train F1: 89.2%
  Val Loss: 26.32, Val F1: 90.5%
  Rare F1: 87.3%, Common F1: 91.8%

Epoch 2: Train Loss: 22.18, Train F1: 93.7%
  Val Loss: 21.56, Val F1: 94.1%
  Rare F1: 91.2%, Common F1: 95.4%

Epoch 3: Train Loss: 18.92, Train F1: 95.4%  ← 继续上升
  Val Loss: 18.67, Val F1: 95.8%            ← 有变化
  Rare F1: 93.5%, Common F1: 96.9%          ← 持续改进
...
```

### 问题训练日志 (不应该看到)

```
Epoch 1: Train Loss: 24.86, Train F1: 92.3%
Epoch 2: Train Loss: 14.90, Train F1: 97.76%
Epoch 3: Train Loss: 14.90, Train F1: 97.75%  ← 停滞
Epoch 4: Train Loss: 14.90, Train F1: 97.75%  ← 完全一样
...
```

---

## 🔄 后续优化建议

如果优化版仍有问题,可以尝试:

1. **进一步降低gamma**: 2.5 → 2.0 → 1.5
2. **调整学习率**: 5e-5 → 1e-4
3. **减少warmup**: 10000 → 5000
4. **数据增强**: 增加mask随机性
5. **正则化**: 添加dropout, weight decay

---

## 📚 相关文件

- [train_with_val_optimized.py](src/train_with_val_optimized.py) - 优化版训练入口
- [pretrain_with_val_optimized.py](src/main/pretrain_with_val_optimized.py) - 优化版Trainer
- [run_v13_optimized.sh](run_v13_optimized.sh) - 运行脚本
- [plot_metrics_csv.py](scripts/plot_metrics_csv.py) - 可视化工具

---

**创建时间**: 2025-12-02
**问题**: 训练在epoch 2后停滞,验证集指标完全不变
**根本原因**: Focal gamma=5过高 + recon loss梯度冲突
**解决方案**: gamma降至2.5 + 移除recon loss + 提高学习率
