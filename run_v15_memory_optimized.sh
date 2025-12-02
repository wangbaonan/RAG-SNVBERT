#!/bin/bash

# ==========================================
# v15: 内存优化版 - 更大模型 + 更小batch size
# Version: v15-memory-optimized
#
# 策略:
# 1. 保持大模型 (dims=256, layers=12, heads=8)
# 2. 减小batch size: 64→32 (减少50%内存)
# 3. 增加梯度累积: 1→2 (保持等效batch size=64)
# 4. 其他配置保持v14不变
# ==========================================

# 创建日志和数据目录
LOG_DIR="logs/v15_memory_optimized"
METRICS_DIR="metrics/v15_memory_optimized"
mkdir -p ${LOG_DIR}
mkdir -p ${METRICS_DIR}

# 生成带时间戳的文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
METRICS_CSV="${METRICS_DIR}/metrics_${TIMESTAMP}.csv"

echo "================================================"
echo "V15: Memory Optimized (dims=256, batch=32, accum=2)"
echo "================================================"
echo "Model Architecture:"
echo "  - Dims: 256"
echo "  - Layers: 12"
echo "  - Heads: 8"
echo "  - Est. Params: ~15M"
echo ""
echo "Memory Optimization:"
echo "  - Train batch size: 32 (reduced from 64)"
echo "  - Grad accum steps: 2 (increased from 1)"
echo "  - Effective batch size: 32*2 = 64 (same as before)"
echo "  - Memory usage: ~50% reduction"
echo ""
echo "Output:"
echo "  - Log: ${LOG_FILE}"
echo "  - CSV: ${METRICS_CSV}"
echo "================================================"
echo ""

# 运行训练
python -m src.train_with_val_optimized \
    --train_dataset /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/train_split.h5 \
    --train_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/train_panel.txt \
    \
    --val_dataset /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/val_split.h5 \
    --val_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split/val_panel.txt \
    \
    --refpanel_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/KGP.chr21.Panel.maf01.vcf.gz \
    --freq_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/Freq.npy \
    --window_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/segments_chr21.maf.csv \
    --type_path data/type_to_idx.bin \
    --pop_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/pop_to_idx.bin \
    --pos_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/pos_to_idx.bin \
    \
    --output_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v15_memopt/rag_bert.model \
    \
    --dims 256 \
    --layers 12 \
    --attn_heads 8 \
    --train_batch_size 32 \
    --val_batch_size 64 \
    --epochs 20 \
    --cuda_devices 0 \
    --log_freq 1000 \
    \
    --rag_k 1 \
    --grad_accum_steps 2 \
    \
    --lr 1e-4 \
    --warmup_steps 20000 \
    \
    --focal_gamma 2.0 \
    --use_recon_loss false \
    \
    --patience 5 \
    --val_metric f1 \
    --min_delta 0.001 \
    \
    --rare_threshold 0.05 \
    --metrics_csv ${METRICS_CSV} \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "================================================"
echo "Training finished"
echo "================================================"
echo "Log saved to: ${LOG_FILE}"
echo "Metrics CSV saved to: ${METRICS_CSV}"
echo ""

# 创建符号链接
ln -sf ${LOG_FILE} ${LOG_DIR}/latest.log
ln -sf ${METRICS_CSV} ${METRICS_DIR}/latest.csv

echo "Latest files:"
echo "  Log: ${LOG_DIR}/latest.log"
echo "  CSV: ${METRICS_DIR}/latest.csv"
