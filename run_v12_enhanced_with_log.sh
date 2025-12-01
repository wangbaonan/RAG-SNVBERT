#!/bin/bash

# ==========================================
# 增强版训练脚本 - Rare/Common F1分解 + CSV日志
# Version: v12-enhanced
# 新增功能:
# 1. Rare (MAF<0.05) vs Common (MAF>=0.05) F1分解
# 2. 每个epoch指标保存到CSV
# 3. 完整日志保存
# ==========================================

# 创建日志和数据目录
LOG_DIR="logs/baseline_gamma5_recon30"
METRICS_DIR="metrics/baseline_gamma5_recon30"
mkdir -p ${LOG_DIR}
mkdir -p ${METRICS_DIR}

# 生成带时间戳的文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
METRICS_CSV="${METRICS_DIR}/metrics_${TIMESTAMP}.csv"

echo "================================================"
echo "Enhanced Training with Rare/Common F1 Breakdown"
echo "================================================"
echo "Log directory: ${LOG_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Metrics CSV: ${METRICS_CSV}"
echo "================================================"
echo ""

# 运行增强版训练
python -m src.train_with_val_enhanced \
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
    --output_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_split_val/rag_bert.model \
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

# ==========================================
# 说明:
#
# 新增输出示例:
# ============================================================
# Epoch 1 VAL Summary
# ============================================================
# Avg Loss:      0.6512
# Avg Accuracy:  0.6987
#
# Haplotype Metrics (Overall):
#   - F1:        0.6823
#   - Precision: 0.6945
#   - Recall:    0.6705
#
# Rare Variants (MAF<0.05):       ← 新增!
#   - F1:        0.6234
#   - Precision: 0.6456
#   - Recall:    0.6023
#
# Common Variants (MAF>=0.05):    ← 新增!
#   - F1:        0.7123
#   - Precision: 0.7245
#   - Recall:    0.7005
# ============================================================
#
# CSV格式:
# epoch,mode,loss,accuracy,overall_f1,overall_precision,overall_recall,
# rare_f1,rare_precision,rare_recall,common_f1,common_precision,common_recall
# 1,train,0.6234,0.7123,0.7045,0.7189,0.6905,0.6512,0.6734,0.6298,0.7234,0.7398,0.7076
# 1,val,0.6512,0.6987,0.6823,0.6945,0.6705,0.6234,0.6456,0.6023,0.7123,0.7245,0.7005
# ...
# ==========================================
