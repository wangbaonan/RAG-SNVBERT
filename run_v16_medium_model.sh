#!/bin/bash

# ==========================================
# v16: 中等模型 - 平衡容量和内存
# Version: v16-medium-model
#
# 策略:
# 1. 中等大小: dims=192, layers=10, heads=6
# 2. 参数量: ~8M (介于2.1M和15M之间)
# 3. 内存消耗: 适中,batch_size可以保持64
# 4. 仍然比原模型大4倍
# ==========================================

# 创建日志和数据目录
LOG_DIR="logs/v16_medium_model"
METRICS_DIR="metrics/v16_medium_model"
mkdir -p ${LOG_DIR}
mkdir -p ${METRICS_DIR}

# 生成带时间戳的文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
METRICS_CSV="${METRICS_DIR}/metrics_${TIMESTAMP}.csv"

echo "================================================"
echo "V16: Medium Model (dims=192, L=10, H=6)"
echo "================================================"
echo "Model Architecture:"
echo "  - Dims: 192 (1.5x from 128)"
echo "  - Layers: 10 (1.25x from 8)"
echo "  - Heads: 6 (1.5x from 4)"
echo "  - Est. Params: ~8M (4x from original 2.1M)"
echo ""
echo "Memory:"
echo "  - Should fit in GPU with batch_size=64"
echo "  - Attention memory: ~1.4GB per layer (vs 2.1GB for dims=256)"
echo ""
echo "Training Config:"
echo "  - Batch size: 64 (same as v13)"
echo "  - LR: 7.5e-5 (between 5e-5 and 1e-4)"
echo "  - Warmup: 15k steps"
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
    --output_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v16_medium/rag_bert.model \
    \
    --dims 192 \
    --layers 10 \
    --attn_heads 6 \
    --train_batch_size 64 \
    --val_batch_size 128 \
    --epochs 20 \
    --cuda_devices 0 \
    --log_freq 1000 \
    \
    --rag_k 1 \
    --grad_accum_steps 1 \
    \
    --lr 7.5e-5 \
    --warmup_steps 15000 \
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
