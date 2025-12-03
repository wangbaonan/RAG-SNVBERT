# AF编码策略深度分析

## 🎯 核心问题

**当前问题**: AF (1维) vs Embedding (192维) → AF信息被严重稀释

**你的建议**: 把AF加入到Embedding中

**我的分析**: 这是正确的方向，但需要仔细设计

---

## 📊 方案对比

### 方案1: 当前方式 (Late Fusion)

```python
emb = embedding_layer(tokens)  # [B, L, 192]
af_feat = af.unsqueeze(-1)      # [B, L, 1]
fused = concat([emb, af_feat])  # [B, L, 193]
output = Linear(193 → 192)
```

**优点**:
- 简单
- Embedding和AF独立学习

**缺点**:
- ❌ AF只占0.5%维度
- ❌ Linear层难以捕捉AF的非线性影响
- ❌ AF信息容易被淹没

---

### 方案2: AF Encoding + Concat (Improved Late Fusion)

```python
emb = embedding_layer(tokens)   # [B, L, 192]
af_encoded = af_encoder(af)     # [B, L, 192]  ← 编码到同等维度!
fused = concat([emb, af_encoded])  # [B, L, 384]
output = Linear(384 → 192)
```

**优点**:
- ✅ AF占50%维度，不被稀释
- ✅ AF可以有复杂的非线性编码
- ✅ 保持模块独立性

**缺点**:
- 参数量增加
- 仍然是late fusion

---

### 方案3: AF-Conditioned Embedding (Early Fusion) ⭐

```python
# Embedding层在生成时就考虑AF
token_emb = token_embedding(tokens)  # [B, L, 192]
af_emb = af_embedding(af)            # [B, L, 192]
emb = token_emb + af_emb             # [B, L, 192]  ← 直接相加!
```

**优点**:
- ✅ AF在最早阶段融入
- ✅ 不增加最终维度
- ✅ AF信息贯穿整个模型
- ✅ 类似BERT的positional embedding

**缺点**:
- 需要设计AF embedding方式

---

### 方案4: AF作为Continuous Embedding (推荐!) ⭐⭐

```python
class AFEmbedding(nn.Module):
    """将连续的AF值编码为高维向量"""
    def __init__(self, embed_size=192, num_basis=32):
        super().__init__()
        # 使用可学习的basis functions
        self.basis_freqs = nn.Parameter(torch.randn(num_basis))
        self.basis_weights = nn.Linear(num_basis * 2, embed_size)  # sin + cos

    def forward(self, af):
        # af: [B, L] - 连续值 0-1
        # 使用Fourier features (类似NeRF)
        af_expanded = af.unsqueeze(-1) * self.basis_freqs  # [B, L, num_basis]
        af_sin = torch.sin(2 * π * af_expanded)
        af_cos = torch.cos(2 * π * af_expanded)
        af_features = torch.cat([af_sin, af_cos], dim=-1)  # [B, L, 2*num_basis]

        return self.basis_weights(af_features)  # [B, L, embed_size]

# 使用
token_emb = token_embedding(tokens)  # [B, L, 192]
af_emb = af_embedding(af)            # [B, L, 192]
final_emb = token_emb + af_emb       # [B, L, 192]
```

**优点**:
- ✅ AF被编码为与token embedding等权的向量
- ✅ Fourier features能捕捉AF的周期性和非线性模式
- ✅ 可学习basis让模型自适应学习AF的重要模式
- ✅ 不增加维度
- ✅ 数学上优雅 (类似NeRF的position encoding)

**原理**:
```
AF=0.02 → [sin(2πf₁*0.02), cos(2πf₁*0.02), ..., sin(2πf₃₂*0.02), cos(2πf₃₂*0.02)]
       → Linear(64 → 192)
       → 192维embedding向量

不同AF值会产生完全不同的embedding pattern
```

---

### 方案5: Hybrid Approach (最全面) ⭐⭐⭐

```python
class HybridAFIntegration(nn.Module):
    def __init__(self, embed_size=192):
        super().__init__()
        # 1. AF Embedding (early fusion)
        self.af_embedding = AFEmbedding(embed_size)

        # 2. AF Conditioning (modulation)
        self.af_scale = nn.Linear(1, embed_size)
        self.af_shift = nn.Linear(1, embed_size)

    def forward(self, token_emb, af):
        # Early fusion: AF embedding
        af_emb = self.af_embedding(af)  # [B, L, D]
        emb = token_emb + af_emb

        # Modulation: AF-based scale and shift
        scale = torch.sigmoid(self.af_scale(af.unsqueeze(-1)))  # [B, L, D]
        shift = self.af_shift(af.unsqueeze(-1))

        return emb * scale + shift
```

**优点**:
- ✅ 结合了additive和multiplicative两种方式
- ✅ 最大限度利用AF信息
- ✅ 灵活性最高

**缺点**:
- 参数量略多
- 稍微复杂

---

## 🎯 推荐方案

### 主推荐: 方案4 (AF Continuous Embedding)

**理由**:
1. 类似BERT的positional embedding，理论成熟
2. Fourier features数学优雅，表达能力强
3. 不增加维度，不影响后续架构
4. 与token embedding地位平等

### 备选: 方案5 (Hybrid)

如果需要最大化AF的影响力

---

## 📐 具体实现

### 完整的AFEmbedding实现

```python
import torch
import torch.nn as nn
import math

class AFEmbedding(nn.Module):
    """
    Allele Frequency Embedding using Fourier Features

    将连续的AF值 (0-1) 编码为高维向量

    类似于:
    - BERT的PositionalEmbedding (但AF是数据驱动的)
    - NeRF的Positional Encoding (但这里是可学习的)
    """
    def __init__(self, embed_size=192, num_basis=32, learnable_basis=True):
        super().__init__()
        self.embed_size = embed_size
        self.num_basis = num_basis

        if learnable_basis:
            # 可学习的basis frequencies
            self.basis_freqs = nn.Parameter(
                torch.randn(num_basis) * 10.0  # 初始化为较大范围
            )
        else:
            # 固定的basis (类似NeRF)
            freqs = 2.0 ** torch.arange(num_basis, dtype=torch.float32)
            self.register_buffer('basis_freqs', freqs)

        # 将Fourier features投影到embed_size
        self.projection = nn.Sequential(
            nn.Linear(num_basis * 2, embed_size),  # sin + cos
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, af):
        """
        Args:
            af: [B, L] - Allele frequency (0-1)

        Returns:
            [B, L, embed_size] - AF embedding
        """
        # Fourier features
        af_expanded = af.unsqueeze(-1) * self.basis_freqs  # [B, L, num_basis]
        af_sin = torch.sin(2 * math.pi * af_expanded)
        af_cos = torch.cos(2 * math.pi * af_expanded)

        # Concat sin and cos
        af_features = torch.cat([af_sin, af_cos], dim=-1)  # [B, L, 2*num_basis]

        # Project to embed_size
        af_emb = self.projection(af_features)  # [B, L, embed_size]

        return af_emb

    def visualize_encoding(self, af_values):
        """可视化不同AF值的编码"""
        with torch.no_grad():
            af_tensor = torch.tensor(af_values).unsqueeze(0)  # [1, L]
            embeddings = self.forward(af_tensor)  # [1, L, D]
            return embeddings.squeeze(0).numpy()  # [L, D]
```

### 修改BERTEmbedding

```python
class BERTEmbedding(nn.Module):
    """
    BERT Embedding with AF integration

    组成:
        1. Token Embedding
        2. Position Embedding
        3. AF Embedding (新增!)
    """
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super().__init__()

        # Token embedding
        self.tokenizer = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # Position embedding
        self.position = PositionalEmbedding(embed_size)

        # AF embedding (新增!)
        self.af_embedding = AFEmbedding(embed_size)

        self.embed_size = embed_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, af=None, use_pos=True):
        """
        Args:
            seq: [B, L] - Token sequences
            af: [B, L] - Allele frequencies (可选)
            use_pos: bool - 是否使用position embedding

        Returns:
            [B, L, D] - Final embeddings
        """
        # Token embedding
        token_emb = self.tokenizer(seq)  # [B, L, D]

        # Position embedding
        if use_pos:
            token_emb = token_emb + self.position(seq)

        # AF embedding (如果提供)
        if af is not None:
            af_emb = self.af_embedding(af)  # [B, L, D]
            token_emb = token_emb + af_emb  # ← 关键: 加性融合!

        return self.dropout(token_emb)
```

---

## 🔍 为什么这样设计有效

### 1. 数学原理

**Fourier Features的表达能力**:

```python
f(af) = Σ[w_i * sin(2π * freq_i * af) + b_i * cos(2π * freq_i * af)]
```

- 可以逼近任意连续函数 (Universal approximation)
- 不同频率捕捉不同尺度的模式:
  - 低频: 捕捉common (0.3-0.5) vs rare (0-0.05) 的大趋势
  - 高频: 捕捉fine-grained差异 (0.01 vs 0.02)

### 2. 与其他embedding对齐

```python
Token embedding:   learned vector [D]
Position embedding: learned vector [D]
AF embedding:      learned vector [D]  ← 地位平等!

Final: token + position + af = [D]
```

所有信息都在同一个维度空间，没有稀释！

### 3. 可解释性

```python
AF = 0.02 (rare):
  → sin(2π*f*0.02) for various f
  → 产生一个"rare pattern"的embedding

AF = 0.45 (common):
  → sin(2π*f*0.45) for various f
  → 产生一个"common pattern"的embedding

模型可以学习:
  "rare pattern" → 特殊处理
  "common pattern" → 常规处理
```

---

## 📊 对比总结

| 方案 | AF维度占比 | 信息保留 | 复杂度 | 推荐度 |
|------|-----------|---------|--------|--------|
| 当前 (Late Fusion) | 0.5% | ❌ 低 | 低 | ❌ |
| Improved Late Fusion | 50% | ⚠️ 中 | 中 | ⚠️ |
| Early Fusion (Simple) | 100% | ✅ 高 | 低 | ✅ |
| **Continuous Embedding** | **100%** | **✅ 高** | **中** | **⭐⭐⭐** |
| Hybrid | 100%+ | ✅ 最高 | 高 | ✅✅ |

---

## 🎯 最终推荐

### 采用方案4: AF Continuous Embedding

**实施步骤**:
1. 实现`AFEmbedding` class
2. 修改`BERTEmbedding`加入AF
3. 在所有使用embedding的地方传入AF
4. 预编码时也使用AF

**优点**:
- ✅ AF信息不被稀释
- ✅ 理论成熟 (类似positional encoding)
- ✅ 实现相对简单
- ✅ 不破坏现有架构

**预期效果**:
- AF的影响力提升100-200倍 (从0.5%到100%)
- Rare variant识别能力显著提升
- 整体F1预期提升2-5%

---

## 📐 与现有架构的整合

### 移除EmbeddingFusionModule?

**建议**: 保留，但简化

```python
class SimplifiedEmbeddingFusionModule(nn.Module):
    """
    现在只需要处理POS
    AF已经在Embedding层处理了
    """
    def __init__(self, emb_size):
        super().__init__()
        self.pos_feat = PositionFeatModule()
        self.fusion = nn.Linear(emb_size + 1, emb_size)
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, emb, pos):
        # emb已经包含了AF信息!
        pos_feat = self.pos_feat(pos).unsqueeze(-1)
        fused = self.fusion(torch.cat([emb, pos_feat], dim=-1))
        return self.norm(emb + fused)
```

或者完全移除，只在Embedding层做：
```python
emb = BERTEmbedding(tokens, af=af, use_pos=True)  # 一步到位!
```

---

**创建时间**: 2025-12-02
**推荐方案**: AF Continuous Embedding (Fourier Features)
**预期提升**: F1 +2-5%, Rare F1 +5-10%
