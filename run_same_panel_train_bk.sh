# python run.py --output_path output/bert.model -c output/bert.model.mask30.ep1.pth --dims 64 --layers 4 --attn_heads 2 --train_batch_size 240 --epochs 1 --cuda_devices 5 6 7 --log_freq 1000
python run.py \
    --train_dataset data/KGP_train_520_42_chr21.h5 \
    --train_panel data/traindata_520_42_KGP_info_new.txt \
    --refpanel_path data/KGP_train_520_42_chr21.vcf.gz \
    --freq_path data/Freq.npy \
    --window_path data/segments_V1.2.csv \
    --type_path data/type_to_idx.bin \
    --pop_path data/pop_to_idx.bin \
    --pos_path data/pos_to_idx.bin \
    --output_path output_rag/bert.model \
    --dims 16 \
    --layers 4 \
    --attn_heads 2 \
    --train_batch_size 128 \
    --epochs 10 \
    --cuda_devices 0 \
    --log_freq 1000 \
