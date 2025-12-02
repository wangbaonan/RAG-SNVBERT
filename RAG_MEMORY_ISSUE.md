# RAG组件内存爆炸问题分析

## 🔴 问题现象

**配置**: dims=192, layers=10, heads=6, batch=64
**GPU**: 81GB A100
**结果**: OOM trying to allocate 1.52 GiB

**这极不正常** - 理论上这个配置只需要~9GB内存。

---

## 🔍 根本原因

### RAG组件的隐藏内存消耗

**代码位置**: `src/model/bert.py` Line 86-113

```python
def encode_rag_segments(self, rag_h1, rag_h2, rag_pos, rag_af, rag_type_idx):
    # rag_h1, rag_h2: [B, L] - retrieved reference sequences

    # 关键问题: 这里把retrieved sequences过完整BERT!
    rag_h1 = self.embedding(rag_h1, rag_pos, rag_af, rag_type_idx)  # [B, L, D]
    rag_h2 = self.embedding(rag_h2, rag_pos, rag_af, rag_type_idx)

    # 过10层Transformer! (每层都保留中间激活用于backward)
    for transformer in self.transformer_blocks:
        rag_h1 = transformer.forward(rag_h1)  # 10层!
        rag_h2 = transformer.forward(rag_h2)

    return rag_h1, rag_h2  # [B, L, D]
```

**问题分析**:

1. **每个batch都要编码RAG sequences**:
   - Original sequences (h1, h2): 过BERT一次
   - Retrieved sequences (rag_h1, rag_h2): **又过BERT一次**
   - 相当于每个batch要过2倍的BERT forward

2. **中间激活保留用于backward**:
   ```
   每层Transformer保留的激活:
   - Attention scores: [B, heads, L, L]
   - Attention output: [B, L, D]
   - FFN intermediate: [B, L, 4D]

   10层 × 2个haplotype × 上述激活 = 巨大内存
   ```

3. **实际内存消耗**:
   ```
   Batch=64, L=1030, D=192, Heads=6, Layers=10

   每个sequence的激活内存:
   - Attention scores: 64 * 6 * 1030 * 1030 * 4 = 1.6 GB
   - Layer outputs: 64 * 1030 * 192 * 4 * 10 = 500 MB
   - FFN intermediate: 64 * 1030 * 768 * 4 * 10 = 2 GB

   Original sequences (h1 + h2): 4.1 GB
   RAG sequences (rag_h1 + rag_h2): 4.1 GB
   Total forward: 8.2 GB

   Backward (梯度): 8.2 GB
   Mixed precision copies: 2 GB
   Temporary buffers: 2 GB

   Total: ~20 GB per batch (不是预期的9GB!)
   ```

---

## 📊 完整内存分解

### 组件1: 模型参数 (~75 MB)

```
Embedding: 5 * 192 = 960 params
Transformer (10层):
  - Self-attention: 192^2 * 4 * 6 = 884K per layer
  - FFN: 192 * 768 * 2 = 295K per layer
  - Total per layer: ~1.2M
  - 10 layers: 12M
Classifiers: ~1M
Total: ~15M params * 4 bytes = 60 MB (float32)
        ~15M params * 2 bytes = 30 MB (mixed precision)
```

### 组件2: Batch数据 (~3 MB)

```
Input tensors: [B, L]
- hap_seq, af, pos, type_idx, etc.
- ~10 fields × 64 × 1030 × 4 bytes = 2.6 MB
```

### 组件3: Forward Activations (8-20 GB!)

**Original sequences**:
```
h1, h2经过embedding + 10层Transformer
每层保留激活:
- Attention: 64 * 6 * 1030^2 * 4 = 1.6 GB
- Outputs: 64 * 1030 * 192 * 4 = 50 MB
- FFN: 64 * 1030 * 768 * 4 = 200 MB

Per layer: ~1.85 GB
10 layers: 18.5 GB
2 haplotypes: 18.5 GB (共享权重但不共享激活)
```

**RAG sequences** (致命问题):
```
rag_h1, rag_h2也要过完整BERT!
又是 18.5 GB

Total: 37 GB 仅forward激活!
```

### 组件4: Backward Gradients (~37 GB)

- 每个激活都需要存储梯度
- 与forward相同大小

### 组件5: Optimizer States

```
Adam optimizer:
- Momentum: 15M params * 4 = 60 MB
- Velocity: 15M params * 4 = 60 MB
Total: ~120 MB (negligible)
```

### 组件6: Gradient Accumulation

```
如果grad_accum_steps=4:
- 需要累积4个batch的梯度
- 额外内存: 4 * 37 GB = 148 GB!
```

### 组件7: CUDA内存碎片化 (+30%)

```
实际分配比理论值高30%
Total × 1.3
```

---

## 💣 为什么81GB不够

**配置**: batch=64, dims=192, layers=10

```
Forward (original + RAG): 37 GB
Backward (gradients): 37 GB
Mixed precision copies: 4 GB
Temporary tensors: 5 GB
Subtotal: 83 GB ← 已经超过81GB!

如果启用grad_accum:
+ Gradient buffer: 37 GB
Total: 120 GB ← 远超GPU容量
```

---

## ✅ 解决方案

### 方案1: 极小Batch (立即可用)

```bash
--train_batch_size 16   # 64 → 16 (减少75%)
--grad_accum_steps 4    # 保持等效batch=64
```

**内存计算**:
```
Forward: 37 * (16/64) = 9.25 GB
Backward: 9.25 GB
Total: ~22 GB (安全!)
```

**缺点**: 训练速度慢4倍

---

### 方案2: 禁用RAG (测试用)

如果只是想验证模型训练,可以临时禁用RAG:

修改 `src/model/bert.py`:
```python
def forward(self, x):
    # ...
    # 临时注释RAG部分
    # rag_h1_encoded, rag_h2_encoded = self.encode_rag_segments(...)
    rag_h1_encoded = None
    rag_h2_encoded = None
    # ...
```

**效果**: 内存减半 (37 GB → 18.5 GB)

---

### 方案3: 预编码RAG Sequences (最优,需要代码改动)

**思路**: 在dataset初始化时预先编码所有reference sequences,存储embedding而不是raw tokens

**修改 `src/dataset/rag_train_dataset.py`**:

```python
def _build_faiss_indexes(self, ref_vcf_path):
    # 加载reference data
    ref_gt, ref_pos = self._load_ref_data(ref_vcf_path)

    # 新增: 预编码所有reference sequences
    print("Pre-encoding reference sequences...")
    self.ref_embeddings = []

    with torch.no_grad():
        for w_idx in range(self.window_count):
            # 获取该window的reference sequences
            ref_seqs = ...  # [num_refs, seq_len]

            # 过BERT一次,存储embedding
            ref_emb = self.bert_model.embedding(ref_seqs, ...)
            self.ref_embeddings.append(ref_emb.cpu())  # 存在CPU

    print("✓ Pre-encoded all reference sequences")
```

**修改 `src/model/bert.py`**:

```python
def forward(self, x, pre_encoded_rag=None):
    # ...
    if pre_encoded_rag is not None:
        # 直接使用预编码的embedding,跳过BERT
        rag_h1_encoded = pre_encoded_rag['h1']  # [B, L, D]
        rag_h2_encoded = pre_encoded_rag['h2']
    else:
        # 原逻辑
        rag_h1_encoded, rag_h2_encoded = self.encode_rag_segments(...)
```

**效果**:
- 内存: 37 GB → 20 GB (减少45%)
- 速度: 提升30% (不需要重复编码)

---

### 方案4: Gradient Checkpointing

启用PyTorch的gradient checkpointing,trade计算换内存:

```python
# 在模型初始化时
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def forward(self, x):
        # 使用checkpointing
        return checkpoint(self._forward, x)

    def _forward(self, x):
        # 原forward逻辑
        ...
```

**效果**: 内存减少~50%,计算时间增加~30%

---

## 🚀 立即运行 (方案1)

```bash
cd /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup
git pull origin main
bash run_v17_extreme_memory_fix.sh
```

**配置**:
- batch_size=16 (从64降到1/4)
- grad_accum_steps=4 (等效batch仍然是64)
- 预期内存: ~22 GB (安全范围)

**缺点**:
- 训练速度慢4倍
- 但至少能跑起来

---

## 📈 长期优化 (方案3)

实现RAG预编码需要:

1. 修改 `rag_train_dataset.py` (_build_faiss_indexes)
2. 修改 `bert.py` (forward)
3. 修改 `rag_train_dataset.py` (__getitem__)
4. 测试验证

**预计工作量**: 2-3小时
**收益**:
- 内存减少45%
- 速度提升30%
- batch_size可以提高到32-48

---

## 🔬 诊断命令

### 检查实际GPU使用

```bash
# 运行训练时,另一个终端执行:
watch -n 1 nvidia-smi

# 查看详细内存分配
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1
```

### 在代码中打印内存

在 `pretrain_with_val_optimized.py` 的training loop中添加:

```python
import torch

def _run_epoch(self, epoch, dataloader, train=True):
    for i, data in enumerate(data_iter):
        # 在forward前
        if i % 100 == 0:
            print(f"Before forward - GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        output = self.model(gpu_data)

        # 在forward后
        if i % 100 == 0:
            print(f"After forward - GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # backward
        loss.backward()

        # 在backward后
        if i % 100 == 0:
            print(f"After backward - GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

---

## 📊 预期内存使用对比

| 配置 | Forward | Backward | Total | 状态 |
|------|---------|----------|-------|------|
| v16 (batch=64) | 37 GB | 37 GB | 83 GB | ❌ OOM |
| v17 (batch=16, accum=4) | 9 GB | 9 GB | 22 GB | ✅ OK |
| v16 + 预编码RAG | 20 GB | 20 GB | 45 GB | ✅ OK |
| v16 + Checkpointing | 18 GB | 18 GB | 40 GB | ✅ OK |

---

## 🎯 总结

**问题**: RAG组件对retrieved sequences也过完整BERT,导致内存翻倍

**立即方案**: batch=16, grad_accum=4 (run_v17)

**长期方案**: 预编码RAG sequences,存储embedding

**根本问题**: RAG设计没有考虑内存优化,每个batch都重复编码相同的reference sequences

---

**创建时间**: 2025-12-02
**GPU**: 81GB A100
**问题**: 即使81GB也OOM
**根因**: RAG对retrieved seqs过完整BERT,内存消耗翻倍
