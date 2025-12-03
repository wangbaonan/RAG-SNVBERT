# V18 Embedding RAG - 内存OOM问题分析

## 问题现象

```
系统内存: 256GB
预编码进度: 54/331 (16%)
状态: OOM (Out of Memory) 中断
```

---

## 内存消耗分析

### 预期内存使用 (按设计文档)

```
331个窗口 × 2008条单倍型 × 1030位点 × 192维 × 4字节 × 2套
= 331 × 2008 × 1030 × 192 × 4 × 2 bytes
= 1,048,567,910,400 bytes
≈ 1000 GB  ← 远超256GB!!!
```

**根本问题**: 设计时假设每个窗口只有几百MB，但实际上：

```
单个窗口内存:
  ref_tokens_complete:     2008 × 1030 × 8 bytes (int64) = 16.5 MB
  ref_tokens_masked:       2008 × 1030 × 8 bytes         = 16.5 MB
  ref_embeddings_complete: 2008 × 1030 × 192 × 4 bytes  = 1586 MB
  ref_embeddings_masked:   2008 × 1030 × 192 × 4 bytes  = 1586 MB
  FAISS index:             2008 × 197760 × 4 bytes      = 1586 MB
  Total per window:        ≈ 4.8 GB

331个窗口:
  4.8 GB × 331 = 1,588 GB  ← 远超256GB!!!
```

### 实际观察 (54/331窗口就OOM)

```
54个窗口 × 4.8 GB = 259 GB
→ 接近256GB系统内存上限
→ OOM!
```

---

## 问题根源

### 设计缺陷

**原设计假设**:
- 存储embeddings到CPU RAM可以节省GPU显存
- CPU RAM足够大，可以存储所有窗口的embeddings

**实际情况**:
- 331个窗口 × 每窗口5GB = 1.6TB
- 即使256GB RAM也远远不够
- **根本不可行!**

### 为什么V17没有这个问题？

**V17设计**:
- 只存储tokens，不存储embeddings
- 每次batch动态编码
- 内存消耗: 331 × 33MB ≈ 11GB ✅

**V18设计**:
- 存储两套embeddings (masked + complete)
- 预编码所有窗口
- 内存消耗: 331 × 4.8GB ≈ 1.6TB ❌

---

## 解决方案

### 方案1: 流式编码 (推荐) ⭐

**核心思路**: 不预存储embeddings，按需编码

**实现**:
```python
# 初始化时: 只存储tokens和FAISS索引
self.ref_tokens_complete[w_idx]  # [2008, 1030] 33MB
self.ref_tokens_masked[w_idx]    # [2008, 1030] 33MB
self.embedding_indexes[w_idx]    # FAISS 1.6GB
# Total per window: 1.6GB

# 训练时: 动态编码complete embeddings
def get_complete_embeddings(self, w_idx):
    ref_tokens = self.ref_tokens_complete[w_idx]
    ref_af = self.ref_af_windows[w_idx]

    with torch.no_grad():
        ref_emb = self.embedding_layer(
            torch.LongTensor(ref_tokens).to(device),
            af=...,
            pos=True
        )
    return ref_emb  # 不存储，直接返回

# 内存节省:
# 331 × 1.6GB = 530GB → 仍然太大!
```

**问题**: FAISS索引本身就占用530GB!

### 方案2: 窗口级别的延迟加载 (推荐) ⭐⭐

**核心思路**: 只在内存中保留当前batch用到的窗口

**实现**:
```python
# 初始化时: 只构建FAISS索引，不保存到内存
# 将FAISS索引保存到磁盘

class EmbeddingRAGDataset:
    def __init__(self, ...):
        self.index_dir = "faiss_indexes/"
        self.emb_cache = {}  # LRU cache
        self.cache_size = 5  # 只缓存5个窗口

    def _build_embedding_indexes(self, ...):
        for w_idx in range(self.window_count):
            # ... tokenize and encode ...

            # 构建FAISS索引
            index = faiss.IndexFlatL2(L * D)
            index.add(ref_emb_masked_flat_np)

            # 保存索引到磁盘
            index_path = f"{self.index_dir}/index_{w_idx}.faiss"
            faiss.write_index(index, index_path)

            # 保存tokens到磁盘 (pickle)
            tokens_path = f"{self.index_dir}/tokens_{w_idx}.pkl"
            with open(tokens_path, 'wb') as f:
                pickle.dump({
                    'complete': ref_tokens_complete,
                    'af': ref_af
                }, f)

            # 不保存到内存!

    def get_window_data(self, w_idx):
        # LRU缓存
        if w_idx in self.emb_cache:
            return self.emb_cache[w_idx]

        # 从磁盘加载
        index = faiss.read_index(f"{self.index_dir}/index_{w_idx}.faiss")
        with open(f"{self.index_dir}/tokens_{w_idx}.pkl", 'rb') as f:
            tokens_data = pickle.load(f)

        # 缓存
        if len(self.emb_cache) >= self.cache_size:
            # 移除最旧的
            self.emb_cache.pop(next(iter(self.emb_cache)))

        self.emb_cache[w_idx] = {
            'index': index,
            'tokens': tokens_data
        }

        return self.emb_cache[w_idx]

# 内存消耗:
# 缓存5个窗口: 5 × 1.6GB = 8GB ✅
```

**优点**:
- 内存消耗可控 (仅缓存窗口)
- 支持任意大的数据集
- 磁盘占用: 331 × 1.6GB ≈ 530GB (可接受)

**缺点**:
- 磁盘I/O开销
- 首次访问慢

### 方案3: 量化存储 (中等效果)

**核心思路**: 用float16代替float32

```python
# 存储时量化
self.ref_embeddings_complete.append(ref_emb_complete.half().cpu())

# 使用时恢复
ref_emb = self.ref_embeddings_complete[w_idx].float().to(device)

# 内存节省:
# 1.6TB → 0.8TB (仍然太大!)
```

**问题**: 节省50%仍然不够

### 方案4: 混合方案 (最优) ⭐⭐⭐

**核心思路**:
1. FAISS索引保存到磁盘
2. Tokens保存到内存 (小)
3. Embeddings按需编码 (不存储)

**实现**:
```python
class EmbeddingRAGDataset:
    def __init__(self, ...):
        self.index_dir = "faiss_indexes/"
        os.makedirs(self.index_dir, exist_ok=True)

        # 只在内存中保存:
        self.ref_tokens_complete = []  # 小
        self.ref_af_windows = []       # 小
        self.window_actual_lens = []   # 小

        # FAISS索引路径
        self.index_paths = []

    def _build_embedding_indexes(self, ref_vcf_path, embedding_layer):
        for w_idx in tqdm(range(self.window_count)):
            # ... 前面的过滤、tokenize、AF计算 ...

            # 编码masked版本 (构建索引)
            ref_emb_masked = embedding_layer(
                ref_tokens_masked_tensor,
                af=ref_af_tensor,
                pos=True
            )

            # 构建FAISS索引
            index = faiss.IndexFlatL2(L * D)
            index.add(ref_emb_masked_flat_np)

            # 保存到磁盘
            index_path = f"{self.index_dir}/index_{w_idx}.faiss"
            faiss.write_index(index, index_path)
            self.index_paths.append(index_path)

            # 只保存tokens和AF (小数据)
            self.ref_tokens_complete.append(ref_tokens_complete)
            self.ref_af_windows.append(ref_af)

            # 不保存embeddings! (节省内存)
            # self.ref_embeddings_complete.append(...)  ← 删除
            # self.ref_embeddings_masked.append(...)    ← 删除

    def load_index(self, w_idx):
        """延迟加载FAISS索引"""
        return faiss.read_index(self.index_paths[w_idx])

    def encode_complete_embeddings(self, w_idx, device='cuda'):
        """按需编码complete embeddings"""
        ref_tokens = self.ref_tokens_complete[w_idx]
        ref_af = self.ref_af_windows[w_idx]

        num_haps = ref_tokens.shape[0]
        ref_af_expanded = np.tile(ref_af, (num_haps, 1))

        with torch.no_grad():
            ref_emb = self.embedding_layer(
                torch.LongTensor(ref_tokens).to(device),
                af=torch.FloatTensor(ref_af_expanded).to(device),
                pos=True
            )
        return ref_emb  # [num_haps, L, D] GPU tensor

# 内存消耗:
# Tokens: 331 × 33MB = 11GB
# AF: 331 × 0.004MB = 1.3MB
# Total: ≈ 11GB ✅✅✅

# 磁盘占用:
# FAISS: 331 × 1.6GB = 530GB
```

**修改collate_fn**:
```python
def embedding_rag_collate_fn(batch_list, dataset, embedding_layer, k_retrieve=1):
    for win_idx, group in window_groups.items():
        # 1. 加载FAISS索引 (从磁盘)
        index = dataset.load_index(win_idx)

        # 2. 编码Query
        h1_emb = embedding_layer(h1_tokens, af=af_batch, pos=True)

        # 3. 检索
        D1, I1 = index.search(h1_emb_flat, k=k_retrieve)

        # 4. 按需编码complete embeddings (不从内存读取)
        ref_emb_complete = dataset.encode_complete_embeddings(
            win_idx,
            device=h1_emb.device
        )

        # 5. 获取retrieved
        for i, sample in enumerate(group):
            topk_h1 = []
            for k in range(k_retrieve):
                ref_idx = I1[i, k]
                topk_h1.append(ref_emb_complete[ref_idx])
            sample['rag_emb_h1'] = torch.stack(topk_h1)
```

**优点**:
- ✅ 内存消耗: 11GB (可接受)
- ✅ 支持大规模数据
- ✅ 训练时只需编码一次per batch

**缺点**:
- ⚠️ 磁盘I/O (FAISS索引加载)
- ⚠️ 每个batch需要编码complete embeddings

---

## 推荐实施方案

### 最终推荐: 方案4 (混合方案)

**理由**:
1. 内存消耗降低: 1.6TB → 11GB (降低99%)
2. 实现简单: 主要修改存储和加载逻辑
3. 性能可接受: FAISS加载很快，编码一次per batch

**实施步骤**:
1. 修改`_build_embedding_indexes`: 不保存embeddings到内存
2. 添加`load_index`方法: 从磁盘加载FAISS
3. 添加`encode_complete_embeddings`方法: 按需编码
4. 修改`collate_fn`: 使用新方法
5. 修改`rebuild_indexes`: 重新保存到磁盘
6. 修改`refresh_complete_embeddings`: 删除此方法 (不需要了)

---

## 进一步优化 (可选)

### 优化1: FAISS索引缓存

```python
class IndexCache:
    def __init__(self, max_size=5):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, w_idx, index_path):
        if w_idx in self.cache:
            # 移到最后 (LRU)
            self.cache.move_to_end(w_idx)
            return self.cache[w_idx]

        # 加载
        index = faiss.read_index(index_path)

        # 缓存
        self.cache[w_idx] = index
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

        return index
```

### 优化2: 预加载下一个batch的索引

```python
# 在collate_fn中
# 预测下一个batch可能用到的窗口
next_windows = predict_next_windows()
for w_idx in next_windows:
    threading.Thread(
        target=lambda: dataset.load_index(w_idx)
    ).start()
```

---

## 性能对比

| 方案 | 内存 | 磁盘 | 训练速度 | 复杂度 |
|-----|-----|-----|---------|--------|
| 原设计 | 1.6TB ❌ | 0 | 快 | 简单 |
| 方案1 (流式) | 530GB ❌ | 0 | 慢 | 中等 |
| 方案2 (延迟加载) | 8GB ✅ | 530GB | 中等 | 复杂 |
| 方案3 (量化) | 800GB ❌ | 0 | 快 | 简单 |
| 方案4 (混合) | 11GB ✅ | 530GB | 中等 | 中等 |

**结论**: 方案4是最佳平衡点

---

## 立即行动

我现在将实施方案4的代码修改。

**修改文件**:
1. `src/dataset/embedding_rag_dataset.py` - 主要修改
2. `src/train_embedding_rag.py` - 删除refresh_complete调用

**预期效果**:
- 内存消耗: 1.6TB → 11GB
- 可以完整运行331个窗口
- 训练速度略慢 (每batch需编码一次)
