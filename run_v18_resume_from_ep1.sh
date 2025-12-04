#!/bin/bash

# ==========================================
# V18: 从 Epoch 1 恢复训练
#
# 关键配置:
# - 从 rag_bert.model.ep1 恢复
# - 验证集固定 50% mask
# - 训练集每 2 个 epoch 增加难度
# ==========================================

# 创建日志和数据目录
LOG_DIR="logs/v18_embedding_rag"
METRICS_DIR="metrics/v18_embedding_rag"
mkdir -p ${LOG_DIR}
mkdir -p ${METRICS_DIR}

# 生成带时间戳的文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/resume_ep1_${TIMESTAMP}.log"
METRICS_CSV="${METRICS_DIR}/resume_ep1_${TIMESTAMP}.csv"

# === Checkpoint配置 ===
RESUME_PATH="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v18_embrag/rag_bert.model.ep1"
RESUME_EPOCH=1

echo "================================================"
echo "V18: 从 Epoch 1 恢复训练"
echo "================================================"
echo "配置:"
echo "  - Checkpoint: ${RESUME_PATH}"
echo "  - 起始 Epoch: ${RESUME_EPOCH}"
echo "  - 训练 Mask: 10% (Epoch 2结束后升至 20%)"
echo "  - 验证 Mask: 50% (固定)"
echo ""
echo "训练计划:"
echo "  Epoch 1-2: 训练10%, 验证50%"
echo "  Epoch 3-4: 训练20%, 验证50%"
echo "  Epoch 5-6: 训练30%, 验证50%"
echo "  ..."
echo ""
echo "输出:"
echo "  - 日志: ${LOG_FILE}"
echo "  - CSV: ${METRICS_CSV}"
echo "================================================"
echo ""

# 检查GPU
echo "GPU状态:"
nvidia-smi
echo ""

# 检查checkpoint
if [ ! -f "${RESUME_PATH}" ]; then
    echo "❌ 错误: Checkpoint不存在: ${RESUME_PATH}"
    echo "请检查路径后重试"
    exit 1
fi

echo "✓ Checkpoint已找到，开始恢复训练..."
echo ""

# 运行训练
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
echo "训练完成"
echo "================================================"
echo "日志: ${LOG_FILE}"
echo "CSV: ${METRICS_CSV}"
echo ""
echo "GPU状态:"
nvidia-smi
echo ""

# 创建符号链接
ln -sf ${LOG_FILE} ${LOG_DIR}/latest_resume.log
ln -sf ${METRICS_CSV} ${METRICS_DIR}/latest_resume.csv

echo "快捷访问:"
echo "  日志: ${LOG_DIR}/latest_resume.log"
echo "  CSV: ${METRICS_DIR}/latest_resume.csv"
echo ""
echo "✓ 恢复训练完成!"
