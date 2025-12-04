#!/bin/bash

# ==========================================
# v18: Embedding RAG - Resume from Epoch 2
#
# 关键改进:
# 1. 从 Epoch 2 checkpoint 恢复训练
# 2. 验证集固定在 50% mask (不再增加)
# 3. 训练集每 2 个 epoch 增加难度 (课程学习)
# ==========================================

# 创建日志和数据目录
LOG_DIR="logs/v18_embedding_rag"
METRICS_DIR="metrics/v18_embedding_rag"
mkdir -p ${LOG_DIR}
mkdir -p ${METRICS_DIR}

# 生成带时间戳的文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_resume_ep2_${TIMESTAMP}.log"
METRICS_CSV="${METRICS_DIR}/metrics_resume_ep2_${TIMESTAMP}.csv"

# === Checkpoint路径配置 ===
RESUME_PATH="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v18_embrag/rag_bert.model.ep2"
RESUME_EPOCH=2

echo "================================================"
echo "V18: Embedding RAG - Resume from Epoch 2"
echo "================================================"
echo "Resume Configuration:"
echo "  - Checkpoint: ${RESUME_PATH}"
echo "  - Starting Epoch: ${RESUME_EPOCH}"
echo "  - Training Mask: 10% (will increase to 20% at epoch 4)"
echo "  - Validation Mask: 50% (FIXED)"
echo ""
echo "Key Changes:"
echo "  - Validation difficulty is now FIXED at 50%"
echo "  - Training difficulty increases every 2 epochs"
echo "  - Loss curves are now comparable across epochs"
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

# 检查checkpoint是否存在
if [ ! -f "${RESUME_PATH}" ]; then
    echo "❌ ERROR: Checkpoint not found at ${RESUME_PATH}"
    echo "Please check the path and try again."
    exit 1
fi

echo "✓ Checkpoint found, starting resume training..."
echo ""

# 运行训练 (从 Epoch 2 恢复)
python -m src.train_embedding_rag \
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
    --output_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v18_embrag/rag_bert.model \
    \
    --dims 384 \
    --layers 12 \
    --attn_heads 12 \
    --train_batch_size 24 \
    --val_batch_size 48 \
    --epochs 20 \
    --cuda_devices 0 \
    --log_freq 500 \
    \
    --rag_k 1 \
    --grad_accum_steps 2 \
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
    \
    --resume_path ${RESUME_PATH} \
    --resume_epoch ${RESUME_EPOCH} \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "================================================"
echo "Resume Training finished"
echo "================================================"
echo "Log saved to: ${LOG_FILE}"
echo "Metrics CSV saved to: ${METRICS_CSV}"
echo ""
echo "GPU Status after training:"
nvidia-smi
echo ""

# 创建符号链接
ln -sf ${LOG_FILE} ${LOG_DIR}/latest_resume.log
ln -sf ${METRICS_CSV} ${METRICS_DIR}/latest_resume.csv

echo "Latest files:"
echo "  Log: ${LOG_DIR}/latest_resume.log"
echo "  CSV: ${METRICS_DIR}/latest_resume.csv"
echo ""
echo "✓ Resume Training Complete!"
echo ""
echo "Key Improvements:"
echo "  - Validation loss is now comparable across epochs (fixed 50% mask)"
echo "  - Training difficulty increases gradually (every 2 epochs)"
echo "  - Model performance can be accurately tracked"
