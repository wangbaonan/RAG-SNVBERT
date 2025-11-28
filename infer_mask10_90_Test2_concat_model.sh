# 基础配置
MODEL_PATH="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/output_rag_20250320_shapeit5_newfreq_newseg/rag_bert.model.ep9.pth"
BASE_INPUT_DIR="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/VCF/Test/TestData/Test2"
BASE_OUTPUT_PREFIX="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/01_infer_all_concat_model/infer_output_rag_mask_Test2"
REF_PANEL="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/VCF/Panel/KGP.chr21.Panel.vcf.h5"
INFER_PANEL="/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/VCF/Test/TestData/Test2/infer.520.sample.panel"

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
        --freq_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/Freq/Freq.npy \
        --type_path data/type_to_idx.bin \
        --pop_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/pop_to_idx.bin \
        --pos_path /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/pos_to_idx.bin \
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