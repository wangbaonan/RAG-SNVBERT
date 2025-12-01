#!/bin/bash

# ==========================================
# RAG-SNVBERT训练脚本 - 带日志保存
# Version: v12-log (Split Validation with Logging)
# 改进: 保存训练日志到文件,方便后续对比
# ==========================================

# 创建日志目录
LOG_DIR="logs/baseline_gamma5_recon30"
mkdir -p ${LOG_DIR}

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

echo "================================================"
echo "Starting training with logging"
echo "================================================"
echo "Log directory: ${LOG_DIR}"
echo "Log file: ${LOG_FILE}"
echo "================================================"
echo ""

# 同时输出到终端和日志文件
python -m src.train_with_val \
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
    2>&1 | tee ${LOG_FILE}

echo ""
echo "================================================"
echo "Training finished"
echo "================================================"
echo "Log saved to: ${LOG_FILE}"
echo ""
echo "View log summary:"
echo "  grep 'Epoch.*Summary' ${LOG_FILE}"
echo ""
echo "Extract validation F1:"
echo "  grep 'VAL Summary' -A 10 ${LOG_FILE} | grep 'F1:'"
echo ""
echo "Compare train vs val F1:"
echo "  grep -E '(TRAIN|VAL) Summary' -A 5 ${LOG_FILE} | grep 'F1:'"
echo "================================================"

# 同时保存一份到当前配置的符号链接 (方便查看最新日志)
ln -sf ${LOG_FILE} ${LOG_DIR}/latest.log
echo "Latest log also available at: ${LOG_DIR}/latest.log"

# ==========================================
# 说明：
# 1. 日志保存在 logs/baseline_gamma5_recon30/ 目录下
# 2. 文件名包含时间戳，每次运行都会创建新文件
# 3. 使用 tee 命令同时输出到终端和文件
# 4. 2>&1 确保stderr也被记录
#
# 查看实时日志:
#   tail -f logs/baseline_gamma5_recon30/latest.log
#
# 提取关键指标:
#   grep 'F1:' logs/baseline_gamma5_recon30/latest.log
#
# 对比不同版本:
#   diff logs/baseline_gamma5_recon30/training_*.log logs/optimized_gamma25_norecon/training_*.log
# ==========================================
