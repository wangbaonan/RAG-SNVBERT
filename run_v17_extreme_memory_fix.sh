#!/bin/bash

# ==========================================
# v17: 极限内存修复 - 超小batch + 梯度累积
# Version: v17-extreme-memory-fix
#
# 问题诊断:
# - 81GB A100仍然OOM
# - 原因: RAG组件对每个batch的retrieved sequences
#   都要过完整的BERT (10层), 导致内存爆炸
# - 每个batch实际消耗: ~27GB (不是预期的9GB)
#
# 策略:
# 1. 极小batch size: 16 (从64降到1/4)
# 2. 梯度累积: 4 steps (等效batch=64)
# 3. 中等模型: dims=192, layers=10, heads=6
# 4. 牺牲速度换取稳定性
#
# 预期内存消耗:
# - Forward: ~6GB (vs 27GB with batch=64)
# - Backward: ~6GB
# - Total peak: ~15GB (安全范围)
# ==========================================

# 创建日志和数据目录
LOG_DIR="logs/v17_extreme_memfix"
METRICS_DIR="metrics/v17_extreme_memfix"
mkdir -p ${LOG_DIR}
mkdir -p ${METRICS_DIR}

# 生成带时间戳的文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
METRICS_CSV="${METRICS_DIR}/metrics_${TIMESTAMP}.csv"

echo "================================================"
echo "V17: Extreme Memory Fix (batch=16, accum=4)"
echo "================================================"
echo "Memory Issue Diagnosis:"
echo "  - OOM even with 81GB A100!"
echo "  - Root cause: RAG encodes retrieved seqs through full BERT"
echo "  - Each batch consumes ~27GB (not 9GB as expected)"
echo ""
echo "Fix Strategy:"
echo "  - Batch size: 64 → 16 (75% reduction)"
echo "  - Grad accum: 1 → 4 steps"
echo "  - Effective batch: still 64"
echo "  - Expected memory: ~6GB forward + 6GB backward = 12GB"
echo ""
echo "Model Architecture:"
echo "  - Dims: 192"
echo "  - Layers: 10"
echo "  - Heads: 6"
echo "  - Params: ~8M"
echo ""
echo "Training Config:"
echo "  - LR: 7.5e-5"
echo "  - Warmup: 15k steps"
echo "  - Focal gamma: 2.0"
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
    --output_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v17_memfix/rag_bert.model \
    \
    --dims 192 \
    --layers 10 \
    --attn_heads 6 \
    --train_batch_size 16 \
    --val_batch_size 32 \
    --epochs 20 \
    --cuda_devices 0 \
    --log_freq 500 \
    \
    --rag_k 1 \
    --grad_accum_steps 4 \
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
echo "Note: Training will be 4x slower due to grad accumulation"
echo "      But this is necessary to fit in memory with RAG"
