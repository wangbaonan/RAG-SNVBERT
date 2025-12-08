#!/bin/bash

# ==========================================
# V18 Embedding RAG Inference Script
# ==========================================
#
# 功能:
# 1. 加载训练好的 V18 模型
# 2. 对 Target Sample 进行 Imputation
# 3. 生成完整的 VCF 文件
#
# 核心特性:
# - Imputation Masking: Mask 位置由数据缺失情况决定
# - 对称 Masking: Query 和 Reference 在相同位置 Mask
# - Lazy Encoding: 检索后按需编码 Complete Reference
# ==========================================

echo "================================================"
echo "V18 Embedding RAG Inference"
echo "================================================"

# ==========================================
# 配置区域 (需要根据实际情况修改)
# ==========================================

# === 模型 Checkpoint ===
# 使用训练好的最佳模型 checkpoint
CHECK_POINT="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_v18_embrag/rag_bert.model.ep11"

# === 模型架构参数 (必须与训练时一致!) ===
DIMS=384         # Hidden dimension (must match training)
LAYERS=6         # Number of layers (must match training)
HEADS=8          # Attention heads (must match training)

# === 数据路径 ===
# Reference Panel (用于构建 FAISS 索引)
REF_PANEL="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/data/train_val_split/train_split.h5"
REF_PANEL_INFO="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/data/train_val_split/train_panel.txt"

# Target Dataset (待填补的数据)
# 注意: 这里需要指定实际的待填补数据文件
TARGET_DATASET="/path/to/your/target/data.h5"  # TODO: 修改为实际路径
TARGET_PANEL="/path/to/your/target/panel.txt"  # TODO: 修改为实际路径

# Frequency and mapping files
FREQ_PATH="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/maf_data/Freq.npy"
TYPE_PATH="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/data/type_to_idx.txt"
POP_PATH="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/data/pop_to_idx.txt"
POS_PATH="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/data/pos_to_idx.txt"

# === 输出路径 ===
OUTPUT_DIR="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/infer_output_v18"
mkdir -p ${OUTPUT_DIR}

# === 推理参数 ===
BATCH_SIZE=16        # Inference batch size
NUM_WORKERS=4        # DataLoader workers
K_RETRIEVE=1         # Number of reference haplotypes to retrieve

# === GPU 设置 ===
CUDA_DEVICES="0"     # GPU device ID
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}

# ==========================================
# 打印配置信息
# ==========================================
echo ""
echo "Configuration:"
echo "  - Model Checkpoint: ${CHECK_POINT}"
echo "  - Architecture: dims=${DIMS}, layers=${LAYERS}, heads=${HEADS}"
echo "  - Reference Panel: ${REF_PANEL}"
echo "  - Target Dataset: ${TARGET_DATASET}"
echo "  - Output Directory: ${OUTPUT_DIR}"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - K Retrieve: ${K_RETRIEVE}"
echo "  - GPU Device: ${CUDA_DEVICES}"
echo ""

# ==========================================
# 启动推理
# ==========================================
echo "Starting V18 Inference..."
echo "================================================"

python -m src.infer_embedding_rag \
    --ref_panel ${REF_PANEL} \
    --infer_dataset ${TARGET_DATASET} \
    --infer_panel ${TARGET_PANEL} \
    --freq_path ${FREQ_PATH} \
    --type_path ${TYPE_PATH} \
    --pop_path ${POP_PATH} \
    --pos_path ${POS_PATH} \
    --check_point ${CHECK_POINT} \
    --output_path ${OUTPUT_DIR} \
    --dims ${DIMS} \
    --layers ${LAYERS} \
    --attn_heads ${HEADS} \
    --infer_batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --k_retrieve ${K_RETRIEVE} \
    --with_cuda True \
    --cuda_devices ${CUDA_DEVICES}

echo "================================================"
echo "V18 Inference Completed!"
echo "================================================"
echo "Output VCF: ${OUTPUT_DIR}/imputed.vcf"
echo ""

# ==========================================
# 后处理 (可选)
# ==========================================
# 如果需要对生成的 VCF 进行后处理，可以在这里添加命令
# 例如: 排序、压缩、索引等

# bcftools sort ${OUTPUT_DIR}/imputed.vcf -Oz -o ${OUTPUT_DIR}/imputed.sorted.vcf.gz
# bcftools index ${OUTPUT_DIR}/imputed.sorted.vcf.gz

echo "All done!"
