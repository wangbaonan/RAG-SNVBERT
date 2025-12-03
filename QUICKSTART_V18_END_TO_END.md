# V18 End-to-End Learnable RAG - 快速开始指南

## 1. 在服务器上拉取最新代码

```bash
# 进入项目目录
cd /path/to/VCF-Bert

# 拉取最新代码
git pull origin main

# 应该看到以下文件更新:
# - src/dataset/embedding_rag_dataset.py
# - src/main/pretrain_with_val_optimized.py
# - src/model/fusion.py
# - src/train_embedding_rag.py
# - END_TO_END_LEARNABLE_RAG_FIX.md (新文件)
```

## 2. 验证关键修改

```bash
# 验证1: 检查 process_batch_retrieval 方法
grep -n "def process_batch_retrieval" src/dataset/embedding_rag_dataset.py
# 应该输出: 306:    def process_batch_retrieval(self, batch, embedding_layer, device, k_retrieve=1):

# 验证2: 检查梯度启用
grep -n "grad_enabled=True" src/dataset/embedding_rag_dataset.py
# 应该输出: 381:                win_idx, device=device, grad_enabled=True

# 验证3: 检查 num_workers 恢复
grep -n "num_workers.*4" src/train_embedding_rag.py
# 应该输出: 69:    parser.add_argument("--num_workers", type=int, default=4, ...

# 验证4: 检查模型维度更新
grep -n "dims.*384" src/train_embedding_rag.py
# 应该输出: 40:    parser.add_argument("--dims", type=int, default=384, ...

# 验证5: 检查 log1p 优化
grep -n "log1p" src/model/fusion.py
# 应该输出: 160:        maf_weight = torch.log1p(1.0 / (maf + 1e-6)).clamp(max=3.0)
```

## 3. 确认预编码已完成

```bash
# 检查 FAISS 索引是否存在
ls faiss_indexes/ | head -5
# 应该看到: index_0.faiss, index_1.faiss, index_2.faiss, ...

# 统计索引文件数量
ls faiss_indexes/*.faiss | wc -l
# 应该输出: 331

# 检查磁盘占用
du -sh faiss_indexes/
# 应该显示约 490GB
```

**如果 faiss_indexes/ 不存在或不完整**，需要重新运行预编码：
```bash
bash run_v18_embedding_rag.sh
# 预编码会自动执行，约需 20-30 分钟
```

## 4. 运行训练脚本

### 方式1: 使用默认参数（推荐）

```bash
# 直接运行训练脚本
bash run_v18_embedding_rag.sh

# 或者查看脚本内容后手动执行
cat run_v18_embedding_rag.sh
```

### 方式2: 自定义参数

```bash
python -m src.train_embedding_rag \
    --train_dataset data/train.h5 \
    --train_panel data/train_panel.txt \
    --val_dataset data/val.h5 \
    --val_panel data/val_panel.txt \
    --refpanel_path data/reference_panel.vcf.gz \
    --freq_path data/freq.pkl \
    --window_path data/windows.pkl \
    --type_path data/type_to_idx.pkl \
    --pop_path data/pop_to_idx.pkl \
    --pos_path data/pos_to_idx.pkl \
    --dims 384 \
    --layers 12 \
    --attn_heads 12 \
    --train_batch_size 24 \
    --val_batch_size 48 \
    --num_workers 4 \
    --epochs 20 \
    --lr 7.5e-5 \
    --warmup_steps 15000 \
    --grad_accum_steps 2 \
    --rag_k 1 \
    --output_path models/v18_embedding_rag.pt \
    --metrics_csv metrics/v18_embedding_rag.csv
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--dims` | **384** | 模型维度（从192增加） |
| `--layers` | **12** | Transformer层数（从10增加） |
| `--attn_heads` | **12** | 注意力头数（从6增加） |
| `--train_batch_size` | **24** | 训练batch size（从32减少，适应384维） |
| `--num_workers` | **4** | DataLoader worker数（从0恢复） |
| `--rag_k` | 1 | RAG检索K值 |

## 5. 监控训练过程

### 终端1: 主训练日志
```bash
tail -f logs/v18_embedding_rag/latest.log
```

**预期输出**:
```
Epoch 1/20
============================================================
Epoch 1 - TRAINING
============================================================
EP_Train:0:   0%|| 1/8617 [00:00<?, ?it/s]
  ↑ 第一个batch应该成功（无CUDA fork error）

EP_Train:0:   1%|| 100/8617 [00:30<50:15, 2.82it/s]
  Loss: 0.523, F1: 0.887
  ↑ 速度应该比之前快（num_workers=4的效果）
```

### 终端2: 系统内存监控
```bash
watch -n 5 "free -h | grep Mem"
```

**预期**:
```
              total        used        free      shared  buff/cache   available
Mem:          256Gi        25Gi       220Gi        1.0Gi        10Gi       228Gi
                           ↑ 应该稳定在 20-30GB
```

### 终端3: GPU 监控
```bash
watch -n 2 nvidia-smi
```

**预期**:
```
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  A100-SXM... On           | 00000000:00:1E.0 Off |                    0 |
|-------------------------------+----------------------+----------------------+
| 20GB / 80GB      |  85%        | 75°C                | ...                  |
  ↑ 384维模型会用更多显存，但80GB A100足够
```

### 终端4: 训练指标
```bash
watch -n 10 "tail -5 metrics/v18_embedding_rag/latest.csv"
```

## 6. 预期训练效果

### 第一个 Batch（关键检查点）
```
EP_Train:0:   0%|| 1/8617 [00:01<?, ?it/s]
✅ 如果成功完成 → process_batch_retrieval 工作正常
❌ 如果 CUDA fork error → 检查 num_workers 是否真的是4
```

### 前 100 个 Batch
```
EP_Train:0:   1%|| 100/8617 [00:35<49:30, 2.87it/s]
  Loss: 0.512
  Train F1: 0.892

✅ 速度: 2.5-3.0 it/s（比之前的num_workers=0快）
✅ Loss: 应该平稳下降
✅ F1: 应该逐渐提升
```

### Epoch 1 完成
```
Epoch 1 Summary:
  Train Loss: 0.405
  Train F1: 0.943
  Val F1: 0.955
  Rare F1: 0.928
  Time: 1.8-2.0h

✅ Train F1 > 0.92
✅ Val F1 > 0.95
✅ 比V17更好（端到端学习的效果）
```

### Epoch 2 开始（Mask 刷新）
```
Epoch 2/20
================================================================================
▣ Epoch 2: 刷新Mask和索引 (数据增强)
================================================================================
▣ 刷新Mask Pattern (版本 1, Seed=2)
✓ Mask刷新完成! 新版本: 1

▣ 重建FAISS索引 (基于新Mask)
重建索引: 100%|███████| 331/331 [08:15<00:00, 1.50s/it]
✓ 索引重建完成! 耗时: 495.32s
✓ Mask和索引刷新完成!
```

## 7. 验证梯度回传（可选）

在训练过程中，可以添加以下代码验证梯度是否正确回传：

```python
# 在第一个epoch的第一个batch后检查
# 修改 src/train_embedding_rag.py 的训练循环:

if epoch == 0:
    # 保存初始权重
    initial_weights = embedding_layer.token.weight.clone()

# ... 训练第一个epoch ...

if epoch == 0:
    # 检查权重是否更新
    final_weights = embedding_layer.token.weight
    weights_changed = not torch.allclose(initial_weights, final_weights, atol=1e-6)
    print(f"\n✅ Embedding层梯度回传验证: {weights_changed}")
    print(f"   权重变化范围: {(final_weights - initial_weights).abs().max():.6f}")
```

**预期输出**:
```
✅ Embedding层梯度回传验证: True
   权重变化范围: 0.003421

↑ True 表示梯度正确回传，权重已更新
```

## 8. 常见问题排查

### 问题1: CUDA fork error 仍然出现
```
RuntimeError: Cannot re-initialize CUDA in forked subprocess
```

**检查**:
```bash
# 确认 num_workers 是否为 4
python -c "import sys; sys.path.insert(0, 'src'); from train_embedding_rag import *; import argparse; parser = argparse.ArgumentParser(); args = parser.parse_args([]); print(args.num_workers if hasattr(args, 'num_workers') else 'not set')"

# 或直接查看
grep "num_workers" src/train_embedding_rag.py | grep "default"
# 应该输出包含 default=4
```

**解决**: 如果还是0，重新 `git pull`

### 问题2: 内存 OOM
```
Killed (OOM)
```

**检查内存使用**:
```bash
free -h
```

**可能原因**:
- FAISS 索引缓存过多 → 正常，设计如此
- 其他进程占用内存 → 清理其他进程

**解决**:
- 确保有 >100GB 空闲内存
- 减小 `--train_batch_size` 到 16 或 20

### 问题3: 训练速度没有提升

**检查 DataLoader 配置**:
```bash
grep -A 5 "train_dataloader = DataLoader" src/train_embedding_rag.py
# 应该看到 num_workers=args.num_workers 和 pin_memory=True
```

**可能原因**:
- 磁盘 I/O 慢（FAISS 索引加载）
- batch 处理时间主要在 GPU（正常）

### 问题4: Loss 不下降或震荡

**可能原因**:
- 学习率过大 → 降低 `--lr` 到 5e-5
- Warmup 不足 → 增加 `--warmup_steps` 到 20000
- AF 加权问题 → 已通过 log1p 修复，应该不会再出现

**验证 log1p 修复**:
```bash
grep "log1p" src/model/fusion.py
# 应该找到修复的代码
```

## 9. 完整训练时间预估

```
预编码:    20-30 分钟 (已完成)
Epoch 1:   1.8-2.0 小时
Epoch 2:   1.8-2.0 小时 (含 8 分钟 mask 刷新)
...
Epoch 20:  1.8-2.0 小时

总计: 30分钟 + 1.9h × 20 ≈ 38-40 小时
```

**加速因素**:
- ✅ `num_workers=4`: 数据加载快 ~4x
- ✅ `pin_memory=True`: CPU→GPU 传输快
- ✅ 更大batch size（如果显存允许）

**减速因素**:
- ⚠️ 384维模型: 计算量增加 ~2x
- ⚠️ FAISS 索引加载: 每batch约 50ms

**净效果**: 与之前相比可能快 10-20%

## 10. 成功标志

### ✅ 训练成功的标志

1. **第一个 batch 成功**
   ```
   EP_Train:0:   0%|| 1/8617 [00:01<?, ?it/s]
   ✓ 无 CUDA fork error
   ```

2. **稳定训练**
   ```
   EP_Train:0:  10%|| 861/8617 [05:30<50:15, 2.57it/s]
   Loss: 0.487, F1: 0.901
   ✓ Loss 平稳下降
   ✓ 速度稳定
   ```

3. **Epoch 1 完成**
   ```
   Train F1: 0.943, Val F1: 0.955
   ✓ 指标优秀
   ```

4. **Mask 刷新成功**
   ```
   ✓ Mask刷新完成! 新版本: 1
   ✓ 索引重建完成!
   ✓ Mask和索引刷新完成!
   ```

5. **梯度回传验证**（可选）
   ```
   ✅ Embedding层梯度回传验证: True
   ```

---

## 总结

**核心改进**:
- ✅ 真正的端到端学习（梯度正确回传）
- ✅ 高效数据加载（多worker，无CUDA fork）
- ✅ 更强模型容量（384维）
- ✅ 稳定训练（log1p 平滑）

**运行步骤**:
1. `git pull origin main`
2. 验证关键修改
3. `bash run_v18_embedding_rag.sh`
4. 监控训练过程

**预期效果**:
- 训练稳定、Loss 平稳下降
- F1 分数高于 V17（端到端学习的优势）
- 速度略快于之前（多worker）

**现在可以开始训练了！** 🚀

如有任何问题，参考 [END_TO_END_LEARNABLE_RAG_FIX.md](END_TO_END_LEARNABLE_RAG_FIX.md) 获取详细技术文档。
