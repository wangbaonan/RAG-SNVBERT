python test_rag_traindataset.py \
  --train_dataset data/KGP_train_520_42_chr21.h5 \
  --train_panel data/traindata_520_42_KGP_info_new.txt \
  --freq_path data/Freq.npy \
  --window_path data/segments_V1.2_head20.csv \
  --type_path data/type_to_idx.bin \
  --pop_path data/pop_to_idx.bin \
  --pos_path data/pos_to_idx.bin \
  --refpanel_path data/KGP_train_520_42_chr21.vcf.gz
