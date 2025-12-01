# 实际训练代码分析报告

## 📋 概述

已成功pull你的实际训练代码。这是你在服务器上真正使用的版本。

---

## 🎯 当前使用的训练脚本

### 主要训练命令：[run_v10_20250411_mafData.sh](run_v10_20250411_mafData.sh)

```bash
python run.py \
    --train_dataset /cpfs01/.../maf_data/KGP.chr21.Train.maf01.vcf.h5 \
    --train_panel /cpfs01/.../train.980.sample.panel \
    --refpanel_path /cpfs01/.../maf_data/KGP.chr21.Panel.maf01.vcf.gz \
    --freq_path /cpfs01/.../maf_data/Freq.npy \
    --window_path /cpfs01/.../maf_data/segments_chr21.maf.csv \
    --type_path data/type_to_idx.bin \
    --pop_path /cpfs01/.../maf_data/pop_to_idx.bin \
    --pos_path /cpfs01/.../maf_data/pos_to_idx.bin \
    --output_path /cpfs01/.../output_rag_20250411_mafData/rag_bert.model \
    --dims 128 \
    --layers 8 \
    --attn_heads 4 \
    --train_batch_size 64 \
    --epochs 20 \
    --cuda_devices 0 \
    --log_freq 1000
```

**关键参数**：
- 模型维度：`dims=128`, `layers=8`, `attn_heads=4`
- 批次大小：`batch_size=64`
- 训练轮数：`epochs=20`
- FAISS TopK：`K=3`（在代码中硬编码，见 [src/train.py:105](src/train.py#L105)）

---

## 🏗️ 实际代码架构

### 1. 训练入口流程

```
run.py
  ↓
src/train.py::train()
  ↓
创建数据集：
  - TrainDataset (基础数据集，未实际使用)
  - RAGTrainDataset (实际使用的RAG数据集)
  ↓
创建模型：BERTWithRAG
  ↓
创建Trainer：BERTTrainer
  ↓
训练循环：
  for epoch in range(epochs):
      trainer.train(epoch)
      trainer.save(epoch)
      rag_train_data_loader.dataset.add_level()  # 增加难度
```

### 2. 实际使用的模型：`BERTWithRAG`

定义在 [src/model/bert.py:76-127](src/model/bert.py#L76-L127)

**关键组件**：
```python
class BERTWithRAG(BERT):
    def __init__(self, vocab_size, dims=512, n_layers=12, attn_heads=16, dropout=0.1):
        super().__init__(...)

        # 核心：使用 EnhancedRareVariantFusion
        self.rag_fusion = EnhancedRareVariantFusion(dims)

    def encode_rag_segments(self, rag_segs, pos, af):
        # 显存优化版参考编码
        # 使用梯度检查点 (torch.utils.checkpoint.checkpoint)
        # 分块处理参考序列 (chunk_size = max(1, 512 // L))

    def forward(self, x):
        # 1. BERT编码 haplotypes
        h1, h2, h1_ori, h2_ori = super().forward(x)

        # 2. 编码RAG参考序列
        rag_h1 = self.encode_rag_segments(x['rag_seg_h1'], ...)
        rag_h2 = self.encode_rag_segments(x['rag_seg_h2'], ...)

        # 3. 融合增强（核心！）
        h1_fused = self.rag_fusion(h1, rag_h1, x['af'], x['af_p'])
        h2_fused = self.rag_fusion(h2, rag_h2, x['af'], x['af_p'])

        return h1_fused, h2_fused, h1_ori, h2_ori
```

### 3. 实际使用的Fusion模块：`EnhancedRareVariantFusion`

定义在 [src/model/fusion.py:89-158](src/model/fusion.py#L89-L158)

**核心机制**：
```python
class EnhancedRareVariantFusion(nn.Module):
    def __init__(self, dims):
        # 1. 跨层次AF交互模块
        self.af_interaction = CrossAFInteraction(dims)

        # 2. AF适配器（权重生成）
        self.af_adapter = nn.Sequential(...)

        # 3. 动态聚合层（注意力池化）
        self.pooling = nn.Sequential(...)

        # 4. 特征融合层
        self.fusion = nn.Sequential(...)

    def forward(self, orig_feat, rag_feat, global_af, pop_af):
        # Step 1: 融合全局AF和群体AF
        fused_af = self.af_interaction(global_af, pop_af)

        # Step 2: 生成AF权重
        af_weight = self.af_adapter(fused_af)

        # Step 3: 加权参考特征
        weighted_ref = rag_feat * af_weight.unsqueeze(1)

        # Step 4: 动态注意力聚合（K个参考序列 → 1个）
        pool_weights = self.pooling(weighted_ref)
        pooled_ref = torch.sum(weighted_ref * pool_weights, dim=2)

        # Step 5: 特征融合
        fused = self.fusion(torch.cat([orig_feat, pooled_ref], dim=-1))

        # Step 6: MAF逆向加权（强调罕见变异）
        maf = torch.min(global_af, 1 - global_af)
        maf_weight = (1.0 / (maf + 1e-6)).clamp(max=10.0)

        return orig_feat + res_scale * (fused * maf_weight)
```

**设计亮点**：
1. ✅ **跨层次AF融合**：结合全局AF（Global）和群体AF（Population-specific）
2. ✅ **动态注意力聚合**：不是简单平均，而是学习权重聚合K个参考
3. ✅ **罕见变异强调**：通过MAF逆向加权（1/MAF）增强罕见变异信号
4. ✅ **残差连接**：`res_scale=0.1` 保证训练稳定性

---

## 🔧 Trainer配置：`BERTTrainer`

定义在 [src/main/pretrain.py:38-](src/main/pretrain.py#L38)

**关键特性**：

### 1. 损失函数
```python
# Focal Loss with gamma=5（强调难样本）
self.hap_criterion = FocalLoss(gamma=5, reduction='sum')
self.gt_criterion = FocalLoss(gamma=5, reduction='sum')
self.recon_critetion = nn.MSELoss()  # 重构损失
```

### 2. 优化器配置
```python
self.optim = Adam(
    lr=1e-5,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    fused=True  # CUDA优化的Adam
)
self.optim_schedule = ScheduledOptim(warmup_steps=20000)
```

### 3. 混合精度训练
```python
self.scaler = GradScaler(enabled=with_cuda)  # AMP
self.grad_accum_steps = grad_accum_steps     # 梯度累积
```

### 4. 显存优化技术
- **梯度检查点**（Gradient Checkpointing）：在 `encode_rag_segments` 中
- **分块编码**：`chunk_size = max(1, 512 // L)`
- **混合精度**：使用AMP自动混合精度
- **TF32加速**：`torch.backends.cuda.matmul.allow_tf32 = True`

---

## 📊 数据流程

### RAGTrainDataset 关键特性

定义在 [src/dataset/rag_train_dataset.py](src/dataset/rag_train_dataset.py)

**数据字段**：
```python
{
    # Long类型字段（基因型数据）
    'hap_1': torch.long,        # 单倍型1
    'hap_2': torch.long,        # 单倍型2
    'rag_seg_h1': torch.long,   # RAG参考序列（K个）
    'rag_seg_h2': torch.long,

    # Float类型字段（频率和位置）
    'pos': torch.float,         # 位点位置
    'af': torch.float,          # 全局等位基因频率
    'af_p': torch.float,        # 群体特异性频率
    'ref': torch.float,         # REF基因型频率
    'het': torch.float,         # HET基因型频率
    'hom': torch.float,         # HOM基因型频率

    # 标签
    'hap_1_label': torch.long,
    'hap_2_label': torch.long,
    'gt_label': torch.long,
    'mask': torch.bool
}
```

### FAISS检索流程
```python
def rag_collate_fn_with_dataset(batch_list, dataset, k=3):
    # K=3: 为每个样本检索3个最相似的参考序列
    # 使用FAISS IVF索引进行高效检索
    # 基于L2距离计算相似度
```

---

## 🚀 训练流程细节

### 每个Epoch的操作

1. **数据加载**：RAGTrainDataset加载h5文件，动态FAISS检索
2. **前向传播**：
   - BERT编码单倍型 → `h1, h2`
   - 编码RAG参考 → `rag_h1, rag_h2`（分块+梯度检查点）
   - Fusion增强 → `h1_fused, h2_fused`
3. **损失计算**：
   - Haplotype Loss（单倍型预测）
   - Genotype Loss（基因型预测）
   - Reconstruction Loss（重构损失）
4. **优化更新**：AMP混合精度 + 梯度累积
5. **难度递增**：`dataset.add_level()` 增加mask难度

---

## 🗂️ 其他发现的训练脚本版本

你的代码库包含多个训练版本的历史记录：

| 脚本名称 | 日期 | 主要改动 |
|---------|------|---------|
| `run.sh` | 原始版本 | 基础训练 |
| `run_v2_20250320.sh` | 2025-03-20 | 更新数据准备 |
| `run_v3_20250325_rareVariantAware.sh` | 2025-03-25 | 引入罕见变异感知 |
| `run_v4_20250325_rareVariantAware_v2.sh` | 2025-03-25 | 罕见变异v2 |
| `run_v5_20250325_rareVariantAware_v3.sh` | 2025-03-25 | 罕见变异v3 |
| `run_v6_20250401_DynamicGateFusion.sh` | 2025-04-01 | 动态门控融合 |
| `run_v7_20250401_newMask.sh` | 2025-04-01 | 新mask策略 |
| `run_v8_20250402_LD_newMask_newPar.sh` | 2025-04-02 | LD引导+新参数 |
| `run_v9_20250403_newMask.sh` | 2025-04-03 | mask优化 |
| **`run_v10_20250411_mafData.sh`** ⭐ | **2025-04-11** | **当前使用版本** |

---

## 🔍 Fusion模块演化历史

在你的代码库中发现了多个fusion模块的备份版本：

### 1. `EnhancedRareVariantFusion` ⭐ 当前使用
- 文件：[src/model/fusion.py:89-158](src/model/fusion.py#L89-L158)
- 特性：跨层次AF交互 + 动态注意力聚合 + MAF逆向加权

### 2. `RareVariantAwareFusion`
- 文件：[src/model/fusion.py:161-195](src/model/fusion.py#L161-L195)
- 特性：AF敏感转换 + 混合池化（0.7均值+0.3最大值）

### 3. `ConcatFusion` / `FixedConcatFusion`
- 文件：[src/model/fusion.py:198-249](src/model/fusion.py#L198-L249)
- 特性：简单拼接融合（基线方法）

### 备份的fusion版本
- `fusion_bk20250401.py` - 2025-04-01备份
- `fusion_bk20250402_right_beforeUpdate.py` - 2025-04-02备份（更新前正确版本）
- `fusion_update20250401error.py` - 错误的更新版本
- `fusion_update20250402error.py` - 另一个错误版本

**结论**：你在开发过程中尝试了多种fusion方法，最终选择了`EnhancedRareVariantFusion`，并且谨慎地保留了多个备份。

---

## 📈 代码质量评估

### ✅ 优点

1. **架构清晰**：
   - 模块化设计良好（model/dataset/main分离）
   - 代码职责明确

2. **显存优化充分**：
   - 梯度检查点
   - 分块编码
   - 混合精度训练
   - Fused Adam优化器

3. **实验管理规范**：
   - 保留了历史版本脚本
   - 关键文件有备份（*_bk*.py）
   - 版本命名清晰（v1-v10带日期）

4. **技术栈先进**：
   - FAISS高效检索
   - Focal Loss处理不平衡
   - 动态注意力机制
   - 罕见变异特殊处理

### ⚠️ 当前缺失的功能

1. **没有Validation**：
   - 每个epoch只有训练，没有验证集评估
   - 容易过拟合（你之前提到的问题）
   - 没有Early Stopping

2. **Hard-coded参数**：
   - FAISS TopK=3 硬编码在 [src/train.py:105](src/train.py#L105)
   - Focal Loss gamma=5 硬编码
   - 没有配置文件统一管理

3. **日志不完整**：
   - 只有loss，没有F1/Precision/Recall
   - 没有TensorBoard集成
   - 没有实验跟踪（如Wandb）

4. **代码冗余**：
   - 大量备份文件（*_bk*.py）混在源码目录
   - 测试文件（test_*.py）未整理到单独目录
   - 缓存文件（__pycache__/）被提交到git

---

## 🎯 下一步建议

### 紧急改进（解决你的训练问题）

1. **添加Validation支持** ⭐⭐⭐
   - 修改 `src/train.py` 支持验证集
   - 修改 `BERTTrainer` 添加 `validate()` 方法
   - 每个epoch输出验证F1/Precision/Recall
   - 添加Early Stopping

2. **优化显存使用** ⭐⭐
   - 降低FAISS K值（从3改为1-2）
   - 增加梯度累积步数
   - 考虑使用更小的validation batch size

3. **改进日志记录** ⭐
   - 集成TensorBoard
   - 记录详细的验证指标
   - 可视化训练曲线

### 长期优化

1. **代码整理**：
   - 将备份文件移到 `backups/` 目录
   - 将测试文件移到 `tests/` 目录
   - 添加 `.gitignore` 排除 `__pycache__/`

2. **配置管理**：
   - 创建统一的配置文件（YAML/JSON）
   - 参数外部化（不硬编码）

3. **实验管理**：
   - 集成Weights & Biases或MLflow
   - 自动记录超参数和结果

---

## 📝 总结

你的实际训练代码是一个**技术含量很高**的实现，特别是：

1. ✅ **EnhancedRareVariantFusion** 设计巧妙（跨层次AF + 动态聚合 + MAF加权）
2. ✅ **显存优化**做得很好（梯度检查点+分块+AMP）
3. ✅ **实验版本管理**规范（v1-v10清晰记录）

但确实缺少**Validation机制**，这就是你遇到的核心问题：
> "每个Epoch我都只能看到训练的结果，容易过拟合，val中的F1得分并不理想"

**建议优先级**：
1. 🔴 **立即添加Validation** - 解决过拟合问题
2. 🟡 **降低RAG K值** - 解决显存问题
3. 🟢 **代码整理** - 提升可维护性

如果你需要，我可以帮你：
- 修改现有代码添加Validation支持
- 创建带验证的训练脚本
- 整理代码结构

需要我继续帮你做哪方面的工作？
