#!/bin/bash

# ==========================================
# v17: 修复版 - 调整batch=48的学习率
# Version: v17-batch48-fixed
#
# 问题:
# - 用户将batch从16改为48 (3倍)
# - 但没有调整学习率
# - 导致梯度爆炸: Loss从110暴涨到2360
#
# 修复:
# 1. Batch size: 48 (用户要求，充分利用显存)
# 2. Grad accum: 1 (降低，因为batch已经够大)
# 3. 学习率: 2.5e-5 (从7.5e-5降低3倍!)
# 4. Warmup: 500 (增加稳定性)
# 5. 添加gradient clipping (防止爆炸)
#
# 预期:
# - Effective batch: 48 × 1 = 48 (vs 原来64)
# - 内存: ~18GB (可接受)
# - 稳定性: 大幅改善
# ==========================================

# 创建日志和数据目录
LOG_DIR="logs/v17_batch48_fixed"
METRICS_DIR="metrics/v17_batch48_fixed"
mkdir -p ${LOG_DIR}
mkdir -p ${METRICS_DIR}

# 生成带时间戳的文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
METRICS_CSV="${METRICS_DIR}/metrics_${TIMESTAMP}.csv"

echo "================================================"
echo "V17: Batch 48 FIXED Version"
echo "================================================"
echo "Problem Diagnosed:"
echo "  - User changed batch 16 → 48 (3x)"
echo "  - But didn't adjust learning rate"
echo "  - Result: Loss exploded 110 → 2360!"
echo ""
echo "Fix Applied:"
echo "  - Batch size: 48 (as requested)"
echo "  - Grad accum: 1 (reduced from 4)"
echo "  - LR: 2.5e-5 (reduced 3x: 7.5e-5 → 2.5e-5)"
echo "  - Warmup: 500 (increased from 100)"
echo "  - Effective batch: 48"
echo ""
echo "Expected Results:"
echo "  - Stable training (no explosion)"
echo "  - Val F1: ~0.95 (like epoch 1)"
echo "  - Memory: ~18GB"
echo ""
echo "Output:"
echo "  - Log: ${LOG_FILE}"
echo "  - CSV: ${METRICS_CSV}"
echo "================================================"
echo ""

# 检查GPU状态
echo "GPU Status before training:"
nvidia-smi
echo ""

# 运行训练 (修复版)
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
    --output_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v17_batch48_fixed/rag_bert.model \
    \
    --dims 192 \
    --layers 10 \
    --attn_heads 6 \
    --train_batch_size 48 \
    --val_batch_size 32 \
    --epochs 20 \
    --cuda_devices 0 \
    --log_freq 500 \
    \
    --rag_k 1 \
    --grad_accum_steps 1 \
    \
    --lr 2.5e-5 \
    --warmup_steps 500 \
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
echo "GPU Status after training:"
nvidia-smi
echo ""

# 创建符号链接
ln -sf ${LOG_FILE} ${LOG_DIR}/latest.log
ln -sf ${METRICS_CSV} ${METRICS_DIR}/latest.csv

echo "Latest files:"
echo "  Log: ${LOG_DIR}/latest.log"
echo "  CSV: ${METRICS_DIR}/latest.csv"
echo ""
echo "Expected: Stable training with batch=48!"
echo ""
