#!/bin/bash

# ==========================================
# V18 续训练脚本: 从 Epoch 0 Checkpoint 恢复
# Version: v18-embedding-rag-resume
#
# 背景:
# - 发现并修复了多进程 Mask 同步 Bug
# - Epoch 0 性能正常 (F1 0.92)
# - 需要从 Epoch 0 checkpoint 续训练验证修复效果
#
# 核心修复:
# - __getitem__ 实时生成 Mask (确定性)
# - Worker 进程与主进程 Mask 完全同步
# - 预期 Epoch 1 性能恢复正常
# ==========================================

# 创建日志和数据目录
LOG_DIR="logs/v18_embedding_rag"
METRICS_DIR="metrics/v18_embedding_rag"
mkdir -p ${LOG_DIR}
mkdir -p ${METRICS_DIR}

# 生成带时间戳的文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/resume_ep0_${TIMESTAMP}.log"
METRICS_CSV="${METRICS_DIR}/resume_ep0_${TIMESTAMP}.csv"

echo "================================================"
echo "V18: Resume from Epoch 0 (Mask Fix Verification)"
echo "================================================"
echo "Critical Fix Applied:"
echo "  - __getitem__ now generates Mask on-the-fly"
echo "  - Worker processes sync with main process"
echo "  - Expected: Epoch 1 F1 ~0.92 (vs buggy 0.50)"
echo ""
echo "Resume Config:"
echo "  - Checkpoint: Epoch 0 model"
echo "  - Start from: Epoch 1"
echo "  - Verify: Mask synchronization fix"
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

# === Checkpoint恢复配置 ===
# 请根据实际情况修改以下路径
RESUME_PATH="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v18_embrag_no_maf/rag_bert.model.ep0"
RESUME_EPOCH=1  # ← CRITICAL: 必须设置为 1！
                # 原因分析:
                # 1. rag_bert.model.ep0 是第1轮训练（epoch=0, "Epoch 1/20"）结束时保存的
                # 2. Bug 出现在第2轮训练（epoch=1, "Epoch 2/20"）
                # 3. 设置 RESUME_EPOCH=1 → range(1, 20) → 从 epoch=1 开始（即重新训练第2轮）
                # 4. 这样可以验证 Mask 同步修复后，第2轮训练的 F1 是否恢复到 0.92

echo "Resume Parameters:"
echo "  - Checkpoint: ${RESUME_PATH}"
echo "  - Resume Epoch Index: ${RESUME_EPOCH}"
echo "  - Will start training at: Epoch $((RESUME_EPOCH + 1)) (index=${RESUME_EPOCH})"
echo "  - Explanation: ep0 file = Epoch 1 trained → resume from index 1 = Epoch 2"
echo ""

# 运行训练
python -m src.train_embedding_rag \
    --train_dataset /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split_no_maf/train_split.h5 \
    --train_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split_no_maf/train_panel.txt \
    \
    --val_dataset /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split_no_maf/val_split.h5 \
    --val_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_RAG-SNVBERT-packup/data/train_val_split_no_maf/val_panel.txt \
    \
    --refpanel_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/VCF/Panel/KGP.chr21.Panel.vcf.gz \
    --freq_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/Freq/Freq.npy \
    --window_size 510 \
    --type_path data/type_to_idx.bin \
    --pop_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/pop_to_idx.bin \
    --pos_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/pos_to_idx.bin \
    \
    --output_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v18_embrag_no_maf/rag_bert.model \
    \
    --dims 384 \
    --layers 12 \
    --attn_heads 12 \
    --train_batch_size 72 \
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
echo "Resume training finished"
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
echo "Expected Results:"
echo "  - Epoch 1 F1 should be ~0.92 (vs buggy 0.50)"
echo "  - Mask synchronization verified"
echo "  - Training should continue normally"
echo ""
