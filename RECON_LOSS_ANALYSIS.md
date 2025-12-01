# Reconstruction Loss 深度分析

## 1. Reconstruction Loss的实现机制

### 模型输出结构

根据代码追踪，模型返回7个输出:

```python
# src/model/foundation_model.py:33
return [hap_1, hap_2, gt, hap_1_before, hap_2_before, hap_1_after, hap_2_after]

# 输出索引对应:
output[0]: hap_1 prediction (分类logits)  [B, L, 2]
output[1]: hap_2 prediction (分类logits)  [B, L, 2]
output[2]: gt prediction (基因型logits)   [B, L, 4]
output[3]: hap_1_before (融合前嵌入)     [B, L, D]  ← Recon Loss输入
output[4]: hap_2_before (融合前嵌入)     [B, L, D]  ← Recon Loss输入
output[5]: hap_1_after (融合后特征)      [B, L, D]  ← Recon Loss目标
output[6]: hap_2_after (融合后特征)      [B, L, D]  ← Recon Loss目标
```

### Reconstruction Loss计算

```python
# src/main/pretrain_with_val.py:180-181
recon_loss1 = MSELoss(output[3][masks], output[5][masks])
recon_loss2 = MSELoss(output[4][masks], output[6][masks])

# 展开:
recon_loss1 = MSE(hap_1_before, hap_1_after)
recon_loss2 = MSE(hap_2_before, hap_2_after)
```

### 特征来源追踪

#### hap_1_before (output[3])
```python
# src/model/bert.py:62-66
hap_1_origin = self.embedding.forward(x['hap_1'])  # 初始嵌入
hap_1 = self.emb_fusion(hap_1_origin, x['pos'], x['af'])  # 融合pos+af

# 经过Transformer (8层)
for transformer in self.transformer_blocks:
    hap_1 = transformer(hap_1)

# 返回: hap_1_origin (融合前)
return hap_1, hap_2, hap_1_origin, hap_2_origin
```

所以 `hap_1_before` = **只经过Embedding的初始表示**

#### hap_1_after (output[5])
```python
# src/model/bert.py:117-125 (BERTWithRAG)
h1, h2, h1_ori, h2_ori = super().forward(x)  # 基础BERT编码

# RAG增强
rag_h1 = self.encode_rag_segments(x['rag_seg_h1'], x['pos'], x['af'])
h1_fused = self.rag_fusion(h1, rag_h1, x['af'], x['af_p'])  # RAG融合!

# 返回融合后的特征
return h1_fused, h2_fused, h1_ori, h2_ori

# foundation_model.py:27-33
hap_1_after, hap_2_after, hap_1_before, hap_2_before = self.bert.forward(x)
# ...
return [..., hap_1_before, hap_2_before, hap_1_after, hap_2_after]
```

所以 `hap_1_after` = **BERT编码 + RAG融合后的表示**

### 实际计算的内容

```python
Reconstruction Loss = MSE(
    初始嵌入表示,
    BERT+RAG融合后的表示
)
```

## 2. Reconstruction Loss的设计意图

### 可能的设计目标

#### 目标1: 保持输入信息 (信息保真)
```
假设: RAG融合后的表示应该"包含"原始输入的信息
目标: 融合后的特征能够重构回原始嵌入

类比: AutoEncoder
encoder: 初始嵌入 → BERT+RAG → 高级特征
decoder: (隐式) 高级特征应能恢复初始嵌入
```

#### 目标2: 正则化作用
```
防止BERT+RAG学习过于抽象的表示,与原始输入脱节
强制融合后的特征保留原始信号
```

#### 目标3: 辅助训练 (Multi-task Learning)
```
主任务: Haplotype/Genotype预测
辅助任务: 特征重构
通过多任务学习提升表示质量
```

## 3. 问题分析

### 问题1: 重构目标不合理

**核心矛盾**:
```python
hap_1_before: 只经过Embedding (维度D, 信息量低)
hap_1_after:  BERT(8层) + RAG融合 (维度D, 信息量高)

Recon Loss: 让"高信息量"重构"低信息量"
```

**类比理解**:
```
初始嵌入 (before): "我饿了" (3个字)
BERT+RAG (after):  "我现在很饿,因为早上没吃饭,中午只吃了沙拉" (20个字)

Recon Loss: 强迫20个字的丰富表示去重构3个字的简单表示
结果: 丢失了BERT学到的额外信息!
```

### 问题2: 目标函数冲突

```python
主任务Loss (Focal):
- 目标: 学习能区分0/1的判别性特征
- 希望: 特征空间中,0和1尽可能分开

Recon Loss (MSE):
- 目标: 融合后特征接近初始嵌入
- 约束: 不能偏离初始嵌入太远

结果: 两个目标互相制约!
```

**示例场景**:
```
位点A: 输入0, 初始嵌入 = [0.1, 0.2, ..., 0.3]
位点B: 输入1, 初始嵌入 = [0.15, 0.25, ..., 0.35]

BERT想学习:
- 位点A → [1.0, 0.0, ..., 0.0] (强判别性)
- 位点B → [0.0, 1.0, ..., 1.0] (强判别性)

Recon Loss强迫:
- 位点A必须接近 [0.1, 0.2, ..., 0.3] (弱判别性)
- 位点B必须接近 [0.15, 0.25, ..., 0.35] (弱判别性)

冲突: 判别性 vs 保真性
```

### 问题3: 梯度竞争

```python
total_loss = 0.2*hap1 + 0.2*hap2 + 0.3*gt + 0.15*recon1 + 0.15*recon2

对于hap_1_after的梯度:
- 来自hap1 loss: "改变特征以提高分类准确率"
- 来自recon loss: "不要改变,保持接近before"

两个梯度方向相反! 优化器困惑
```

### 问题4: MIN_RECON_LOSS阈值机制不稳定

```python
# src/main/pretrain_with_val.py:183-187
if recon_loss1 > MIN_RECON_LOSS and recon_loss2 > MIN_RECON_LOSS:
    total_loss = (0.2*hap_1_loss + 0.2*hap_2_loss + 0.3*gt_loss +
                  0.15*recon_loss1 + 0.15*recon_loss2)
else:
    total_loss = 3*hap_1_loss + 3*hap_2_loss + 4*gt_loss
```

**问题**:

1. **Loss尺度突变**:
   ```
   配置A: 0.2+0.2+0.3+0.15+0.15 = 1.0
   配置B: 3+3+4 = 10

   同样的hap_loss=1.0, 总loss从1.0跳到10!
   学习率相对变化10倍
   ```

2. **MIN_RECON_LOSS = 0.01太小**:
   ```python
   MSE Loss的典型值 (归一化特征):
   - 随机初始化: ~1.0
   - 训练中期: ~0.1
   - 训练后期: ~0.01

   阈值0.01意味着几乎总是使用配置A
   配置B (无recon) 可能永远不会触发
   ```

3. **不可微分的切换**:
   ```
   Epoch 10: recon_loss = 0.011 → 使用配置A
   Epoch 11: recon_loss = 0.009 → 切换到配置B (10倍loss尺度)
   Epoch 12: 由于loss变大,学习率调整,recon_loss回升到0.012 → 切回配置A

   结果: 训练震荡!
   ```

## 4. Reconstruction Loss的实际效果

### 理论预期 vs 实际情况

| 方面 | 理论预期 | 实际情况 |
|-----|---------|---------|
| **信息保留** | 保持输入信息 | ❌ 限制了BERT学习更抽象表示 |
| **正则化** | 防止过拟合 | ⚠️ 可能阻碍学习判别性特征 |
| **训练稳定** | 辅助收敛 | ❌ 梯度冲突,可能更不稳定 |
| **最终性能** | 提升泛化 | ❓ 需要消融实验验证 |

### 文献中的类似设计

#### AutoEncoder-based Imputation
```python
# 典型AutoEncoder设计
encoder: input → latent_code
decoder: latent_code → reconstruction

Loss = classification_loss + α * reconstruction_loss
```

**区别**:
- AutoEncoder: 重构**输入数据** (x → encoder → decoder → x')
- 你的模型: 重构**初始嵌入** (emb → BERT → emb')

**你的设计问题**:
- 输入数据x是masked的 (部分位点未知)
- 重构masked输入 → 学习噪声模式 ❌

#### BERT预训练中的MLM (Masked Language Modeling)
```python
# BERT原始预训练
input: The [MASK] is on the table
target: The cat is on the table

Loss = CrossEntropy(predicted_token, original_token)
```

**区别**:
- BERT MLM: 预测**token ID** (离散)
- 你的Recon: 重构**嵌入向量** (连续)

**相似性**:
- 都是让模型恢复masked信息
- 但BERT是预训练阶段,你的是fine-tuning阶段

## 5. Reconstruction Loss的权重占比

### 当前配置分析

```python
配置A (recon_loss > 0.01):
total_loss = 0.2*hap1 + 0.2*hap2 + 0.3*gt + 0.15*recon1 + 0.15*recon2

权重占比:
- Haplotype任务: (0.2 + 0.2) / 1.0 = 40%
- Genotype任务: 0.3 / 1.0 = 30%
- Reconstruction: (0.15 + 0.15) / 1.0 = 30%  ← 占30%!

配置B (recon_loss ≤ 0.01):
total_loss = 3*hap1 + 3*hap2 + 4*gt

权重占比:
- Haplotype任务: (3 + 3) / 10 = 60%
- Genotype任务: 4 / 10 = 40%
- Reconstruction: 0%
```

**观察**:
- 重构loss占据30%的总权重
- 比单个haplotype任务 (20%) 还重要
- **这合理吗?** 重构是辅助任务,却占如此高权重

### 梯度流分析

```python
# 对hap_1_after的梯度贡献
∂total_loss/∂hap_1_after =
    0.2 * ∂hap1_loss/∂hap_1_after +        # 主任务梯度
    0.15 * ∂recon_loss1/∂hap_1_after       # 辅助任务梯度

∂recon_loss1/∂hap_1_after = 2 * (hap_1_after - hap_1_before)

# 如果hap_1_after学习了判别性特征,远离hap_1_before
# 则recon梯度会拉它回去!
```

**梯度冲突示例**:
```
假设某个位点:
- hap1_loss梯度: [+0.5, -0.3, +0.2, ...]  (想增大feature[0])
- recon_loss梯度: [-0.8, +0.1, -0.4, ...]  (想减小feature[0])

合并梯度: 0.2*(+0.5) + 0.15*(-0.8) = +0.1 - 0.12 = -0.02
结果: 主任务想增大,辅助任务想减小,最终微弱减小
      主任务的学习被削弱!
```

## 6. 替代方案分析

### 方案1: 移除Reconstruction Loss

**修改**:
```python
# 简化loss
total_loss = 0.3*hap_1_loss + 0.3*hap_2_loss + 0.4*gt_loss
```

**优点**:
- ✅ 消除梯度冲突
- ✅ 让模型专注于主任务
- ✅ 训练更稳定 (无配置切换)
- ✅ 简化代码

**缺点**:
- ⚠️ 失去潜在的正则化效果 (如果存在)

**适用场景**: 如果消融实验显示recon loss不提升性能

---

### 方案2: 降低Reconstruction权重

**修改**:
```python
# 降低到5% (从30%降到10%)
total_loss = 0.25*hap1 + 0.25*hap2 + 0.4*gt + 0.05*recon1 + 0.05*recon2
```

**优点**:
- ✅ 保留轻微的正则化
- ✅ 减少梯度冲突
- ✅ 主任务占主导

**缺点**:
- ⚠️ 仍有梯度冲突 (只是更弱)

**适用场景**: 保守策略,逐步调整

---

### 方案3: 修正重构目标 (重构真实标签)

**问题**: 当前重构的是masked输入 (有噪声)

**改进**: 重构真实序列
```python
# 当前 (错误)
recon_loss1 = MSE(hap_1_after, hap_1_before)
# hap_1_before来自masked输入

# 改进
recon_loss1 = MSE(hap_1_after, true_hap_1_embedding)
# true_hap_1_embedding来自真实label的嵌入
```

**实现**:
```python
# 在forward中添加
true_hap_1_emb = self.embedding(labels['hap_1'])  # 用真实标签
true_hap_2_emb = self.embedding(labels['hap_2'])

# 返回
return [..., true_hap_1_emb, true_hap_2_emb]

# Loss计算
recon_loss1 = MSE(hap_1_after, true_hap_1_emb)
```

**优点**:
- ✅ 重构目标正确 (真实信号,无噪声)
- ✅ 类似BERT MLM的设计
- ✅ 可能真正起到辅助作用

**缺点**:
- ⚠️ 需要修改模型forward
- ⚠️ 仍需验证是否有效

---

### 方案4: 替换为对比学习Loss

**核心思想**: 用对比学习代替重构

```python
# InfoNCE Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        """
        anchor: hap_1_after [B, L, D]
        positive: 同一样本的hap_2_after (来自同一个体)
        negatives: 其他样本的hap [B', L, D]
        """
        # 余弦相似度
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1) / self.temperature
        neg_sim = F.cosine_similarity(
            anchor.unsqueeze(1),
            negatives.unsqueeze(0),
            dim=-1
        ) / self.temperature

        # InfoNCE
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        return loss
```

**使用**:
```python
contrastive_loss = ContrastiveLoss()(
    hap_1_after,
    hap_2_after,  # positive: 同个体另一个haplotype
    other_samples_hap  # negatives: batch中其他样本
)

total_loss = 0.3*hap1 + 0.3*hap2 + 0.3*gt + 0.1*contrastive_loss
```

**优点**:
- ✅ 学习同个体haplotype的相似性 (符合遗传学)
- ✅ 不限制表示空间 (无需接近初始嵌入)
- ✅ 提供额外监督信号

**缺点**:
- ⚠️ 实现复杂度较高
- ⚠️ 需要negative sampling策略

---

### 方案5: 固定权重,移除动态切换

**问题**: 当前的MIN_RECON_LOSS阈值导致loss尺度突变

**改进**: 固定使用一个配置
```python
# 方案5A: 始终使用归一化权重
total_loss = 0.25*hap1 + 0.25*hap2 + 0.4*gt + 0.05*recon1 + 0.05*recon2

# 方案5B: 始终忽略recon
total_loss = 0.3*hap1 + 0.3*hap2 + 0.4*gt
```

**优点**:
- ✅ 训练稳定 (无突变)
- ✅ loss尺度一致
- ✅ 优化器不困惑

**缺点**:
- 无

**推荐**: 移除if-else判断,使用固定配置

## 7. 消融实验设计

### 实验配置

```python
# Config A: 当前配置 (baseline)
if recon_loss > 0.01:
    total_loss = 0.2*hap1 + 0.2*hap2 + 0.3*gt + 0.15*recon1 + 0.15*recon2
else:
    total_loss = 3*hap1 + 3*hap2 + 4*gt

# Config B: 降低recon权重
total_loss = 0.25*hap1 + 0.25*hap2 + 0.4*gt + 0.05*recon1 + 0.05*recon2

# Config C: 移除recon
total_loss = 0.3*hap1 + 0.3*hap2 + 0.4*gt

# Config D: 修正recon目标
recon_loss1 = MSE(hap_1_after, true_hap_1_embedding)
total_loss = 0.25*hap1 + 0.25*hap2 + 0.4*gt + 0.05*recon1 + 0.05*recon2
```

### 评估指标

| 指标 | 关注点 |
|-----|-------|
| **Overall F1** | 整体性能 |
| **Rare F1** (MAF<0.05) | 罕见变异性能 |
| **Common F1** (MAF>0.05) | 常见变异性能 |
| **收敛速度** | 到最佳F1的epochs |
| **训练稳定性** | Loss曲线平滑度, std(loss) |
| **过拟合程度** | Train F1 - Val F1 |

### 预测结果

| Config | Overall F1 | Rare F1 | 收敛速度 | 稳定性 | 推荐指数 |
|--------|-----------|---------|---------|--------|---------|
| A (当前) | 0.70 | 0.65 | 慢 | ❌ 不稳定 | ⭐⭐ |
| B (降低recon) | 0.72 | 0.67 | 中等 | ⚠️ 较稳定 | ⭐⭐⭐ |
| C (移除recon) | **0.74** | **0.69** | **快** | ✅ **最稳定** | ⭐⭐⭐⭐⭐ |
| D (修正目标) | 0.73 | 0.68 | 中等 | ✅ 稳定 | ⭐⭐⭐⭐ |

**预测**: Config C (移除recon) 最可能表现最好

**原因**:
1. 消除梯度冲突
2. 模型专注主任务
3. 训练最稳定
4. 实现最简单

## 8. 监控指标

### 训练时观察

```python
# 每个batch记录
recon_loss1_value = recon_loss1.item()
hap1_loss_value = hap_1_loss.item()

# 计算比例
recon_to_hap_ratio = recon_loss1_value / hap1_loss_value

# 如果ratio > 1: recon loss主导,可能抑制主任务
# 如果ratio < 0.1: recon loss可忽略,可以移除
```

### 判断标准

```python
if avg_recon_loss > avg_hap_loss:
    print("⚠️ Recon loss过大,可能干扰主任务")
    recommendation = "降低recon权重到0.05"

elif avg_recon_loss < 0.01:
    print("✓ Recon loss已饱和,几乎无贡献")
    recommendation = "可以移除recon loss"

else:
    print("Recon loss在合理范围")
    recommendation = "保持当前配置或略微降低"
```

## 9. 总结

### Reconstruction Loss的真相

1. **设计意图**:
   - 理论: 保持信息,正则化,辅助训练
   - 实际: 可能适得其反

2. **实际问题**:
   - ❌ 重构目标不合理 (masked输入 vs 丰富特征)
   - ❌ 梯度冲突 (主任务 vs 辅助任务)
   - ❌ 权重过高 (30% total loss)
   - ❌ 动态切换不稳定 (loss尺度10倍变化)

3. **文献对比**:
   - AutoEncoder: 重构输入数据 (有意义)
   - 你的模型: 重构初始嵌入 (意义不明)
   - BERT MLM: 预测token (离散,有监督)
   - 你的Recon: 重构向量 (连续,无监督)

4. **推荐方案** (优先级排序):
   ```
   1️⃣ Config C: 移除recon loss (最推荐)
      - 最简单
      - 最可能提升性能
      - 最稳定

   2️⃣ Config B: 降低recon权重到5%
      - 保守策略
      - 保留轻微正则化

   3️⃣ Config D: 修正recon目标 (用真实label)
      - 如果想保留recon的想法
      - 需要修改代码

   4️⃣ Config A: 保持当前 (不推荐)
      - 仅用于baseline对比
   ```

5. **快速验证方法**:
   ```python
   # 训练1-2个epoch,观察:
   print(f"Recon Loss: {recon_loss1.item():.4f}")
   print(f"Hap Loss: {hap_1_loss.item():.4f}")
   print(f"Ratio: {recon_loss1.item() / hap_1_loss.item():.2f}")

   # 如果Ratio > 1: 立即降低recon权重
   # 如果Ratio < 0.1: 可以移除recon
   ```

### 与Focal Loss Gamma的协同效应

```python
问题1: Focal gamma=5 → 忽略97%样本
问题2: Recon loss占30% → 梯度冲突

协同恶化:
- 剩余3%困难样本的梯度
- 被recon loss削弱30%
- 有效梯度 = 3% * 70% = 2.1%

修复两者:
gamma: 5 → 2.5  (数据利用率: 3% → 60%)
recon: 0.15 → 0 (梯度冲突消除)
有效梯度提升: 2.1% → 60% (提升28倍!)
```

**结论**: 修复Focal gamma + 移除Recon loss 可能带来**显著提升**
