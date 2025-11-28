# 基础配置
MODEL_PATH="output_rag/bert.model.ep9.pth"
BASE_INPUT_DIR="data/New_VCF/Test/TestData/Test2"
BASE_OUTPUT_PREFIX="infer_output_rag_mask_Test2"
REF_PANEL="data/New_VCF/Panel/KGP.chr21.Panel.vcf.h5"
INFER_PANEL="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/34_SNV_KGP_BioData/VCF/KGP_Imputation_info_520_42.txt"

# 需要处理的mask值列表
for mask in $(seq 10 10 90); do
    # 构建输入文件路径
    input_vcf="${BASE_INPUT_DIR}/KGP.chr21.Test2.Mask${mask}.vcf.gz"
    
    # 构建输出目录路径
    output_dir="${BASE_OUTPUT_PREFIX}${mask}/"
    
    # 创建输出目录
    mkdir -p "$output_dir"
    
    echo "Processing Mask${mask}..."
    
    # 执行推理命令
    python infer.py \
        -c "$MODEL_PATH" \
        --infer_dataset "$input_vcf" \
        --ref_panel "$REF_PANEL" \
        --infer_panel "$INFER_PANEL" \
        --freq_path data/Freq.npy \
        --type_path data/type_to_idx.bin \
        --pop_path data/pop_to_idx.bin \
        --pos_path data/pos_to_idx.bin \
        --output_path "$output_dir" \
        --dims 16 \
        --layers 4 \
        --attn_heads 2 \
        --infer_batch_size 128 \
        --cuda_devices 0

    echo "Mask${mask} processing completed. Output saved to: $output_dir"
    echo "--------------------------------------------------"
done

echo "All mask levels processed."