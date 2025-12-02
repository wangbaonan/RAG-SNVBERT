#!/bin/bash

# ==========================================
# v14: 更大模型 + 修复学习率调度器
# Version: v14-larger-model
#
# 主要改进:
# 1. 修复学习率调度器(之前被硬编码覆盖)
# 2. 增加模型容量: dims 128→256, layers 8→12, heads 4→8
# 3. 提高学习率: 5e-5 → 1e-4 (配合更大模型)
# 4. 更慢的warmup: 10k → 20k steps
# 5. 降低focal gamma: 2.5 → 2.0
# 6. 保持动态mask和CSV日志
# ==========================================

# 创建日志和数据目录
LOG_DIR="logs/v14_larger_model"
METRICS_DIR="metrics/v14_larger_model"
mkdir -p ${LOG_DIR}
mkdir -p ${METRICS_DIR}

# 生成带时间戳的文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
METRICS_CSV="${METRICS_DIR}/metrics_${TIMESTAMP}.csv"

echo "================================================"
echo "V14: Larger Model Training (dims=256, L=12, H=8)"
echo "================================================"
echo "Log directory: ${LOG_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Metrics CSV: ${METRICS_CSV}"
echo ""
echo "Model Architecture:"
echo "  - Dims: 256 (2x increase from 128)"
echo "  - Layers: 12 (1.5x increase from 8)"
echo "  - Heads: 8 (2x increase from 4)"
echo "  - Est. Params: ~15M (vs previous 2.1M)"
echo ""
echo "Training Config:"
echo "  - LR: 1e-4 (scheduler fixed, was hardcoded to 1e-4 before)"
echo "  - Warmup: 20k steps (slower)"
echo "  - Focal gamma: 2.0 (reduced from 2.5)"
echo "  - Dynamic val mask: True"
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
    --output_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v14_larger/rag_bert.model \
    \
    --dims 256 \
    --layers 12 \
    --attn_heads 8 \
    --train_batch_size 64 \
    --val_batch_size 128 \
    --epochs 20 \
    --cuda_devices 0 \
    --log_freq 1000 \
    \
    --rag_k 1 \
    --grad_accum_steps 1 \
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
echo "Quick analysis:"
echo "  # Compare with v13 (smaller model)"
echo "  diff <(grep 'TRAIN Summary' logs/optimized_gamma25_norecon/latest.log) <(grep 'TRAIN Summary' ${LOG_FILE})"
echo ""
echo "  # View training progress"
echo "  grep 'TRAIN Summary' -A 10 ${LOG_FILE}"
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
