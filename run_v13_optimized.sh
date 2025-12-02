#!/bin/bash

# ==========================================
# 优化版训练脚本 - 修复训练停滞问题
# Version: v13-optimized
#
# 主要优化:
# 1. Focal gamma: 5 → 2.5 (减轻对难样本的过度关注)
# 2. 移除reconstruction loss (避免梯度冲突)
# 3. 调整学习率: 1e-5 → 5e-5 (加快学习)
# 4. 保留Rare/Common F1分解和CSV日志
# ==========================================

# 创建日志和数据目录
LOG_DIR="logs/optimized_gamma25_norecon"
METRICS_DIR="metrics/optimized_gamma25_norecon"
mkdir -p ${LOG_DIR}
mkdir -p ${METRICS_DIR}

# 生成带时间戳的文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
METRICS_CSV="${METRICS_DIR}/metrics_${TIMESTAMP}.csv"

echo "================================================"
echo "Optimized Training (Focal gamma=2.5, No Recon)"
echo "================================================"
echo "Log directory: ${LOG_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Metrics CSV: ${METRICS_CSV}"
echo "================================================"
echo ""

# 运行优化版训练
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
    --output_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_optimized/rag_bert.model \
    \
    --dims 128 \
    --layers 8 \
    --attn_heads 4 \
    --train_batch_size 64 \
    --val_batch_size 128 \
    --epochs 20 \
    --cuda_devices 0 \
    --log_freq 1000 \
    \
    --rag_k 1 \
    --grad_accum_steps 1 \
    \
    --lr 5e-5 \
    --warmup_steps 10000 \
    \
    --focal_gamma 2.5 \
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
echo "Quick analysis:"
echo "  # View rare vs common F1"
echo "  grep 'Rare Variants' -A 3 ${LOG_FILE}"
echo "  grep 'Common Variants' -A 3 ${LOG_FILE}"
echo ""
echo "  # Plot metrics from CSV"
echo "  python scripts/plot_metrics_csv.py ${METRICS_CSV}"
echo "================================================"

# 创建符号链接到最新文件
ln -sf ${LOG_FILE} ${LOG_DIR}/latest.log
ln -sf ${METRICS_CSV} ${METRICS_DIR}/latest.csv

echo "Latest files also available at:"
echo "  Log: ${LOG_DIR}/latest.log"
echo "  CSV: ${METRICS_DIR}/latest.csv"
