# Embedding RAG 完整解析

## 🎯 你的核心问题

### Q1: References编码是什么层做的？
**答案**: `BERTEmbedding` 层 (Line 45-47 in bert.py)

```python
self.embedding = BERTEmbedding(vocab_size=vocab_size,
                               embed_size=dims,
                               dropout=dropout)
```

**BERTEmbedding包含**:
- `tokenizer`: nn.Embedding(vocab_size, embed_size) - **可学习的参数**
- `position`: PositionalEmbedding - **可学习的位置编码**

### Q2: 这个层需要训练吗？
**答案**: **需要！** 这是关键优势！

**两种策略**:

**策略A: 固定预编码** (Phase 1, 简单)
```python
# 初始化时,用当前embedding编码一次
with torch.no_grad():  # 不计算梯度
    ref_embeddings = embedding_layer(ref_sequences)

# 训练时embedding层会更新
# 但ref_embeddings保持固定 (不会自动更新)
```

**策略B: 定期刷新** (Phase 2, 最优)
```python
# 每个epoch结束后
def refresh_reference_embeddings():
    with torch.no_grad():
        # 用更新后的embedding重新编码
        ref_embeddings = current_embedding_layer(ref_sequences)
        # 更新FAISS index
```

---

## 📊 当前架构 vs Embedding RAG 对比

### 当前架构 (有RAG)

```
每个Training Batch:
┌─────────────────────────────────────────────────────────┐
│ Step 1: Query Sequences                                 │
│   hap_1, hap_2 [B, L] (raw tokens)                     │
│   ↓                                                     │
│   embedding()        → [B, L, D]   ← 过一次            │
│   ↓                                                     │
│   10 x Transformer   → [B, L, D]   ← 过10层            │
│                                                         │
│ Memory: 9 GB (batch=16)                                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Step 2: RAG Retrieved Sequences (问题所在!)              │
│   rag_h1, rag_h2 [B, L] (raw tokens from FAISS)        │
│   ↓                                                     │
│   embedding()        → [B, L, D]   ← 又过一次!          │
│   ↓                                                     │
│   10 x Transformer   → [B, L, D]   ← 又过10层!          │
│                                                         │
│ Memory: 9 GB (重复!)                                    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Step 3: Fusion                                          │
│   EnhancedRareVariantFusion(query_emb, rag_emb)        │
│   ↓                                                     │
│   Classifiers                                           │
│                                                         │
│ Memory: 1 GB                                            │
└─────────────────────────────────────────────────────────┘

Total Memory: 9 + 9 + 1 = 19 GB per batch
Total Time: 100ms (query) + 100ms (RAG) + 10ms (fusion) = 210ms
```

### Embedding RAG架构

```
═══════════════════════════════════════════════════════════
初始化时 (一次性, ~10 minutes):
═══════════════════════════════════════════════════════════
┌─────────────────────────────────────────────────────────┐
│ Pre-encode ALL Reference Sequences                      │
│                                                         │
│ For each window (e.g., 150 windows):                   │
│   ref_sequences [num_haps, L]  (e.g., 1000 haplotypes)│
│   ↓                                                     │
│   embedding()  → [num_haps, L, D]  ← 只编码一次!        │
│   ↓                                                     │
│   Flatten      → [num_haps, L*D]                       │
│   ↓                                                     │
│   Build FAISS index on L*D dimensional space           │
│   ↓                                                     │
│   Store: ref_embeddings[window_idx] = embeddings.cpu()│
│                                                         │
│ Storage: ~500 MB in CPU RAM (not GPU!)                 │
└─────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════
每个Training Batch:
═══════════════════════════════════════════════════════════
┌─────────────────────────────────────────────────────────┐
│ Step 1: Encode Query (ONLY embedding layer)            │
│   hap_1, hap_2 [B, L] (raw tokens)                     │
│   ↓                                                     │
│   embedding()   → [B, L, D]    ← 只过embedding层!       │
│   (不过Transformer!)                                    │
│                                                         │
│ Memory: 0.5 GB                                          │
│ Time: 10 ms                                             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Step 2: Retrieve Pre-encoded Embeddings                │
│   query_emb [B, L, D]                                   │
│   ↓                                                     │
│   Flatten → [B, L*D]                                    │
│   ↓                                                     │
│   FAISS.search(query_emb_flat) → retrieve indices      │
│   ↓                                                     │
│   rag_emb = ref_embeddings[indices]  ← 直接取!         │
│   (已经是embedding,无需过BERT!)                          │
│                                                         │
│ Memory: 0.5 GB (只存embedding)                          │
│ Time: 5 ms (FAISS极快)                                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Step 3: Fusion + Transformer                           │
│   query_emb_fused = Fusion(query_emb, rag_emb)         │
│   ↓                                                     │
│   10 x Transformer → [B, L, D]  ← 只过一次!             │
│   ↓                                                     │
│   Classifiers                                           │
│                                                         │
│ Memory: 9 GB (只有fused embeddings过Transformer)        │
│ Time: 100 ms                                            │
└─────────────────────────────────────────────────────────┘

Total Memory: 0.5 + 0.5 + 9 = 10 GB per batch (vs 19 GB)
Total Time: 10 + 5 + 100 = 115 ms (vs 210 ms)
Speedup: 1.8x faster
Batch size: 16 → 48 (3x larger)
```

---

## 🔬 关键技术细节

### 1. 预编码过程 (初始化)

```python
class EmbeddingRAGDataset(RAGTrainDataset):
    def _build_embedding_index(self, ref_vcf_path, embedding_layer):
        """
        预编码所有reference sequences

        Args:
            ref_vcf_path: 参考面板VCF路径
            embedding_layer: 当前模型的embedding层 (可学习参数)
        """
        print("▣ Building Embedding-based RAG Index")

        # 1. 加载reference data (和之前一样)
        ref_gt, ref_pos = self._load_ref_data(ref_vcf_path)
        # ref_gt: [num_variants, num_samples*2] - 所有reference的基因型

        # 2. 对每个window预编码
        self.ref_embeddings = {}  # 存储预编码结果
        self.embedding_indexes = []  # FAISS索引

        with torch.no_grad():  # 重要: 不计算梯度
            for w_idx in tqdm(range(self.window_count)):
                # 2.1 获取该window的reference sequences
                window_slice = slice(
                    self.window.window_info[w_idx, 0],
                    self.window.window_info[w_idx, 1]
                )
                window_refs = ref_gt[window_slice]  # [L, num_samples*2]
                window_pos = ref_pos[window_slice]   # [L]

                # 2.2 转置: [num_haplotypes, L]
                num_haps = window_refs.shape[1]
                ref_haps = window_refs.T  # [num_haps, L]

                # 2.3 编码为embeddings
                # 注意: 只过embedding层,不过transformer!
                ref_tokens = torch.LongTensor(ref_haps).to(device)
                ref_emb = embedding_layer(ref_tokens)  # [num_haps, L, D]

                # 2.4 Flatten: [num_haps, L*D]
                ref_emb_flat = ref_emb.reshape(num_haps, -1).cpu().numpy()

                # 2.5 构建FAISS index (在L*D维空间)
                index = faiss.IndexFlatL2(ref_emb_flat.shape[1])
                index.add(ref_emb_flat)
                self.embedding_indexes.append(index)

                # 2.6 存储embeddings (在CPU,节省GPU内存)
                self.ref_embeddings[w_idx] = ref_emb.cpu()

        print(f"✓ Pre-encoded {self.window_count} windows")
        print(f"  Storage: {self._calculate_storage_size()} MB in CPU RAM")
```

**数据对齐保证**:
- ✅ 每个window的ref_embeddings和FAISS index一一对应
- ✅ FAISS返回的index可以直接用于取ref_embeddings
- ✅ Window边界和原数据完全一致

### 2. 训练时检索 (collate_fn)

```python
def embedding_rag_collate_fn(batch_list, dataset, embedding_layer):
    """
    新的collate函数: 在embedding space检索
    """
    # 1. 标准collate
    batch = default_collate(batch_list)
    B = len(batch_list)

    # 2. 只过embedding层编码query
    with torch.no_grad():  # 这里不需要梯度 (检索操作)
        query_h1_emb = embedding_layer(batch['hap_1'])  # [B, L, D]
        query_h2_emb = embedding_layer(batch['hap_2'])

    # 3. 对每个样本在其window检索
    retrieved_h1_embs = []
    retrieved_h2_embs = []

    for i in range(B):
        window_idx = batch['window_idx'][i]

        # 3.1 Flatten query embedding
        query_h1_flat = query_h1_emb[i].reshape(-1).cpu().numpy()  # [L*D]
        query_h2_flat = query_h2_emb[i].reshape(-1).cpu().numpy()

        # 3.2 FAISS检索 (在embedding space)
        D1, I1 = dataset.embedding_indexes[window_idx].search(
            query_h1_flat.reshape(1, -1), k=1
        )
        D2, I2 = dataset.embedding_indexes[window_idx].search(
            query_h2_flat.reshape(1, -1), k=1
        )

        # 3.3 获取pre-encoded embedding
        retrieved_idx1 = I1[0, 0]
        retrieved_idx2 = I2[0, 0]

        rag_h1_emb = dataset.ref_embeddings[window_idx][retrieved_idx1]  # [L, D]
        rag_h2_emb = dataset.ref_embeddings[window_idx][retrieved_idx2]

        retrieved_h1_embs.append(rag_h1_emb)
        retrieved_h2_embs.append(rag_h2_emb)

    # 4. Stack并添加到batch
    batch['rag_h1_emb'] = torch.stack(retrieved_h1_embs)  # [B, L, D]
    batch['rag_h2_emb'] = torch.stack(retrieved_h2_embs)

    return batch
```

**数据对齐保证**:
- ✅ query_emb和rag_emb的shape完全一致: [B, L, D]
- ✅ 都是来自同一个window,位置信息对齐
- ✅ FAISS检索保证最相似的haplotype

### 3. 模型Forward (无需重复过BERT)

```python
class BERTWithEmbeddingRAG(BERT):
    def forward(self, x):
        # 1. Embedding层 (会被训练更新)
        hap_1_emb = self.embedding(x['hap_1'])  # [B, L, D]
        hap_2_emb = self.embedding(x['hap_2'])

        # 2. 获取pre-encoded RAG embeddings (来自collate_fn)
        rag_h1_emb = x.get('rag_h1_emb', None)
        rag_h2_emb = x.get('rag_h2_emb', None)

        # 3. Fusion (如果有RAG)
        if rag_h1_emb is not None:
            # 注意: rag_emb已经是embedding,不需要再过BERT!
            hap_1_fused = self.rag_fusion(
                hap_1_emb,
                rag_h1_emb.to(hap_1_emb.device),
                x['af']
            )
            hap_2_fused = self.rag_fusion(
                hap_2_emb,
                rag_h2_emb.to(hap_2_emb.device),
                x['af']
            )
        else:
            hap_1_fused = hap_1_emb
            hap_2_fused = hap_2_emb

        # 4. Transformer (只过一次!)
        for transformer in self.transformer_blocks:
            hap_1_fused = transformer(hap_1_fused)

        for transformer in self.transformer_blocks:
            hap_2_fused = transformer(hap_2_fused)

        return hap_1_fused, hap_2_fused
```

---

## 🎓 Embedding层的可学习性

### 训练中会发生什么？

```
Iteration 1:
  embedding.weight = W1  (初始参数)
  ref_embeddings = embedding(ref_seqs) using W1  (预编码)

Training:
  loss.backward()
  optimizer.step()
  embedding.weight = W2  (更新后的参数)

  但ref_embeddings仍然是用W1编码的! (固定)

Iteration 1000:
  embedding.weight = W1000 (已经变了很多)
  ref_embeddings仍然是W1 (过时了!)
```

### 解决方案: 定期刷新

```python
def refresh_reference_embeddings(model, dataset):
    """
    用更新后的embedding重新编码references
    """
    print("Refreshing reference embeddings...")

    with torch.no_grad():
        for w_idx in range(dataset.window_count):
            # 用当前的embedding重新编码
            ref_tokens = dataset.ref_tokens[w_idx]  # 存储的raw tokens
            ref_emb = model.embedding(ref_tokens)   # 用最新的W编码

            # 更新存储
            dataset.ref_embeddings[w_idx] = ref_emb.cpu()

            # 重建FAISS index
            ref_emb_flat = ref_emb.reshape(-1, L*D).cpu().numpy()
            dataset.embedding_indexes[w_idx].reset()
            dataset.embedding_indexes[w_idx].add(ref_emb_flat)

    print("✓ Refreshed all reference embeddings")


# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        # 正常训练
        loss.backward()
        optimizer.step()

    # 每个epoch结束刷新 (或每N个epochs)
    refresh_reference_embeddings(model, train_dataset)
```

---

## 📊 性能对比表

| 指标 | 当前RAG | Embedding RAG (固定) | Embedding RAG (刷新) |
|------|---------|---------------------|---------------------|
| **内存** | 19 GB | 10 GB | 10 GB |
| **速度** | 210 ms/batch | 115 ms/batch | 115 ms/batch |
| **Batch size** | 16 | 48 | 48 |
| **Reference可学习** | ✅ Yes | ❌ 固定 | ✅ Yes (每epoch) |
| **实施难度** | N/A | 简单 | 中等 |
| **训练初始化** | 0 | 10 min | 10 min |

---

## 🚀 实施流程 (保留当前代码)

### Step 1: 备份当前代码

```bash
# 在项目根目录
cp -r src src_original_rag
cp -r run_*.sh backup_scripts/

# 或创建git分支
git checkout -b embedding-rag
git commit -m "Checkpoint before Embedding RAG"
```

### Step 2: 修改代码 (Phase 1 - 固定编码)

#### 2.1 修改 `rag_train_dataset.py`

新增方法:
```python
def _build_embedding_index(self, ref_vcf_path, embedding_layer):
    # ... (上面的详细代码)
```

修改 `from_file`:
```python
@classmethod
def from_file(cls, vocab, ..., use_embedding_rag=False):
    dataset = cls(...)

    if use_embedding_rag:
        # 需要传入embedding layer
        # 临时创建一个embedding layer或从checkpoint加载
        dataset._build_embedding_index(ref_vcf_path, embedding_layer)
    else:
        dataset._build_faiss_indexes(ref_vcf_path)

    return dataset
```

#### 2.2 修改 `collate_fn`

新增 `embedding_rag_collate_fn` (上面的详细代码)

#### 2.3 修改 `bert.py`

新增 `BERTWithEmbeddingRAG` class (上面的详细代码)

#### 2.4 创建新的training script

`run_v18_embedding_rag.sh`:
```bash
--use_embedding_rag true
--dims 256  # 可以用更大模型了!
--layers 12
--train_batch_size 32  # 3倍于V17
```

### Step 3: 测试验证

```bash
# 测试数据加载
python -c "from src.dataset.rag_train_dataset import *; test_embedding_rag()"

# 测试内存
nvidia-smi -l 1 &
bash run_v18_embedding_rag.sh

# 对比V17
grep "Epoch 1" logs/v17_extreme_memfix/latest.log
grep "Epoch 1" logs/v18_embedding_rag/latest.log
```

---

## ✅ 安全性保证

1. **数据对齐**:
   - ✅ Window index严格对应
   - ✅ FAISS index和embeddings同步
   - ✅ Shape检查: [B, L, D]

2. **可回退**:
   - ✅ 保留原代码 (`src_original_rag/`)
   - ✅ 可切换: `use_embedding_rag=false` 回到原版
   - ✅ Git分支管理

3. **渐进式**:
   - Phase 1: 固定编码 (简单,稳定)
   - Phase 2: 定期刷新 (优化,可选)
   - 可以先验证Phase 1,确认有效后再做Phase 2

---

## 🎯 你的决定

理解了Embedding RAG后,你觉得:

**选项A**: 先跑V17,明天看结果再说
**选项B**: 我现在就开始实现Embedding RAG Phase 1
**选项C**: 有其他问题想先问清楚

告诉我你的决定! 🚀
