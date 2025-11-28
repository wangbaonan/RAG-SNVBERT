python infer.py \
    -c output_rag/bert.model.ep9.pth \
    --infer_dataset data/New_VCF/Test/TestData/Test1/KGP.chr21.Test1.Mask10.vcf.h5 \
    --ref_panel data/New_VCF/Panel/KGP.chr21.Panel.vcf.h5 \
    --infer_panel /cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/34_SNV_KGP_BioData/VCF/KGP_Imputation_info_520_42.txt \
    --freq_path data/Freq.npy \
    --type_path data/type_to_idx.bin \
    --pop_path data/pop_to_idx.bin \
    --pos_path data/pos_to_idx.bin \
    --output_path infer_output_rag_0310/ \
    --dims 16 \
    --layers 4 \
    --attn_heads 2 \
    --infer_batch_size 128 \
    --cuda_devices 0

# python transfer_vcf.py \
#     -c /home/user8/VCF-Bert/output/bert.model.ep9.pth
