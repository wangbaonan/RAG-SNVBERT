# 如何运行 V18 Embedding RAG

## 🎯 三步开始

### Step 1: 测试 (30分钟)

```bash
cd /e/AI4S/00_SNVBERT/VCF-Bert
python test_embedding_rag.py
```

**预期输出**: ✓ All tests passed!

---

### Step 2: 小规模验证 (2小时，可选但推荐)

创建测试脚本:
```bash
cat > run_v18_test_quick.sh << 'EOF'
#!/bin/bash
python -m src.train_embedding_rag \
    --train_dataset /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/train_split.h5 \
    --train_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/train_panel.txt \
    --val_dataset /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/val_split.h5 \
    --val_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/val_panel.txt \
    --refpanel_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/KGP.chr21.Panel.maf01.vcf.gz \
    --freq_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/Freq.npy \
    --window_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/segments_chr21.maf.csv \
    --type_path data/type_to_idx.bin \
    --pop_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/pop_to_idx.bin \
    --pos_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/pos_to_idx.bin \
    --output_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v18_test/rag_bert.model \
    --dims 192 --layers 10 --attn_heads 6 \
    --train_batch_size 8 --val_batch_size 16 \
    --epochs 1 --log_freq 10 \
    --rag_k 1 --grad_accum_steps 4 \
    --lr 7.5e-5 --warmup_steps 100 \
    --focal_gamma 2.0 --use_recon_loss false \
    --patience 5 --val_metric f1 --min_delta 0.001 \
    --rare_threshold 0.05 --metrics_csv metrics/v18_test.csv \
    --cuda_devices 0
EOF

chmod +x run_v18_test_quick.sh
./run_v18_test_quick.sh
```

**检查**: 无OOM，Loss下降，速度快

---

### Step 3: 完整训练 (26小时)

#### 配置选择

##### 选项A: V18-Current (保守，与当前V17类似)
```bash
bash run_v18_embedding_rag.sh
# 使用默认配置: dims=192, layers=10, heads=6, batch=32
```
- 参数: 8M
- 内存: 15 GB
- 适合: 快速验证

##### 选项B: V18-Large (推荐)
```bash
# 编辑 run_v18_embedding_rag.sh
vi run_v18_embedding_rag.sh

# 修改这几行:
--dims 256          # 192 → 256
--layers 12         # 10 → 12
--attn_heads 8      # 6 → 8

# 运行
bash run_v18_embedding_rag.sh
```
- 参数: 18M (2.25x V17)
- 内存: 25 GB
- 适合: 最佳性能

---

## 📊 监控训练

### 实时日志
```bash
tail -f logs/v18_embedding_rag/latest.log
```

### GPU监控
```bash
watch -n 1 nvidia-smi
```

### 预期输出
```
Epoch 1/20
================================================================================
▣ 构建Embedding-based RAG索引  (首次约15分钟)
...
✓ 预编码完成! 总耗时: 523s

[Training]
  Batch [100/500] | Loss: 2.134 | F1: 0.923 | Time: 120ms/batch
  ...
  ✓ Epoch 1 Train | Loss: 1.756 | F1: 0.956

[Validation]
  ✓ Epoch 1 Val | Loss: 1.834 | F1: 0.952

▣ 刷新Reference Embeddings (约8分钟)
...
✓ 刷新完成! 耗时: 495s
```

---

## ⚠️ 如果遇到问题

### 问题1: OOM
```bash
# 减小batch size
--train_batch_size 24  # 32 → 24
--grad_accum_steps 3   # 2 → 3 (保持等效batch=72)
```

### 问题2: 训练不收敛
```bash
# 调整学习率
--lr 5e-5  # 7.5e-5 → 5e-5
```

### 问题3: 速度太慢
```bash
# 检查GPU利用率
nvidia-smi
# 应该接近100%

# 如果低，增加num_workers
--num_workers 8  # 4 → 8
```

---

## 📈 性能对比 (V17 vs V18)

| 指标 | V17 | V18-Current | V18-Large |
|------|-----|-------------|-----------|
| 参数 | 8M | 8M | 18M |
| Batch | 16 | 32 | 32 |
| 内存 | 19 GB | 15 GB | 25 GB |
| 速度 | 210 ms/batch | 120 ms/batch | 125 ms/batch |
| Epoch | 4.2h | 1.3h | 1.3h |
| 总时间 | 84h | 26h | 26h |

---

## ✅ Checklist

运行前确认:
- [ ] V17代码已备份在 `src_v17_backup/`
- [ ] GPU至少20GB可用
- [ ] 已运行测试脚本通过
- [ ] 选择了合适的配置 (Current或Large)
- [ ] 设置了日志监控

**全部确认后**: `bash run_v18_embedding_rag.sh` 🚀

---

## 📚 详细文档

- **完整审计**: [COMPLETE_AUDIT_SUMMARY.md](COMPLETE_AUDIT_SUMMARY.md)
- **修复说明**: [FIXES_AND_DEPLOYMENT.md](FIXES_AND_DEPLOYMENT.md)
- **快速指南**: [V18_QUICK_START.md](V18_QUICK_START.md)

---

**最后更新**: 2025-12-02
**状态**: ✅ Ready to run
