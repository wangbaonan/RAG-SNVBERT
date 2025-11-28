#!/usr/bin/env python
# scripts/expand_and_search.py
"""
用法:
python expand_and_search.py \
  --ref_vcf /path/to/ref_chr21.vcf.gz \
  --target_vcf /path/to/target_chr21_missing.vcf.gz \
  --window_csv /path/to/segments_v1.4.csv \
  --sample_idx 0 \
  --top_k 5

脚本逻辑:
1) load reference => (var_ref, samp_ref,2), pos_ref
2) load target => (var_tgt, samp_tgt,2), pos_tgt
3) expand target => shape=(var_ref, samp_tgt,2) => where missing => (0,0)
   + build an array `target_mask` => shape=(var_ref, samp_tgt) => if missing => 1
4) for each window => slice reference => shape=(win_len, samp_ref,2), slice target => shape=(win_len, 2) + mask => dynamic partial index => topK
5) no saving winnpy => purely in memory
"""

import argparse
import pathlib
import os

import h5py
import numpy as np
import allel
import faiss
import pandas as pd
import time

def load_vcf_data(vcf_path:str):
    p= pathlib.Path(vcf_path)
    if p.suffix=='.vcf' or p.suffixes==['.vcf','.gz']:
        h5_path= p.with_suffix('.h5')
        if not h5_path.exists():
            print(f"[expand_and_search] converting {p} => {h5_path} via allel.vcf_to_hdf5")
            allel.vcf_to_hdf5(str(p), str(h5_path), fields='*', overwrite=True)
        p= h5_path
    with h5py.File(p,'r') as hf:
        vcf_data= hf['calldata/GT'][:]   # shape=(variants, samples,2)
        pos_data= hf['variants/POS'][:] # shape=(variants,)
    vcf_data[vcf_data>0]=1
    return vcf_data, pos_data

def expand_target_to_ref(ref_pos: np.ndarray,
                         tgt_data: np.ndarray,
                         tgt_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    将 target vcf_data 扩充到与 ref_pos 行数一致.
    Args:
      ref_pos => shape=(var_ref,)
      tgt_data => shape=(var_tgt, sample_tgt,2)
      tgt_pos  => shape=(var_tgt,)
    Return:
      expanded_data => shape=(var_ref, sample_tgt,2)
      missing_mask  => shape=(var_ref, sample_tgt), 1表示missing
    """
    var_ref= ref_pos.shape[0]
    var_tgt= tgt_data.shape[0]
    sample_tgt= tgt_data.shape[1]

    # 建立 pos => idx for target
    tgt_dict= {p:i for i,p in enumerate(tgt_pos)}

    expanded_data= np.zeros((var_ref, sample_tgt,2), dtype=np.uint8)
    missing_mask= np.zeros((var_ref, sample_tgt), dtype=np.uint8)

    for r_i, p in enumerate(ref_pos):
        if p in tgt_dict:
            t_idx= tgt_dict[p]
            # copy
            expanded_data[r_i]= tgt_data[t_idx]
            # if you'd like to see if this row is truly missing => 0,0? or check ??? 
            # but presumably it's not missing => missing_mask=0
        else:
            # not in target => fill (0,0) + missing_mask=1
            missing_mask[r_i]=1

    return expanded_data, missing_mask

def build_partial_index_l2(hap_1: np.ndarray,
                           hap_2: np.ndarray,
                           mask: np.ndarray,
                           ref_data_3d: np.ndarray,
                           top_k: int=5):
    """
    动态构建 L2 => partial => skip mask=1
    ref_data_3d => shape=(num_ref, w_len,2)
    """
    valid_idx= np.where(mask==0)[0]  # shape=(??)
    q1_sub= hap_1[valid_idx]
    q2_sub= hap_2[valid_idx]
    q_flat= np.concatenate([q1_sub,q2_sub], axis=0)
    q_float= q_flat.astype(np.float32)[None,:]

    num_ref= ref_data_3d.shape[0]
    dims= len(valid_idx)*2
    sub_ref= np.zeros((num_ref,dims), dtype=np.float32)
    for i in range(num_ref):
        sub_sl= ref_data_3d[i,valid_idx,:].reshape(-1)
        sub_ref[i]= sub_sl.astype(np.float32)
    index= faiss.IndexFlatL2(dims)
    t0=time.time()
    index.add(sub_ref)
    build_t= time.time()-t0

    t1=time.time()
    D,I= index.search(q_float, top_k)
    search_t= time.time()-t1
    return I[0], D[0], build_t, search_t

def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--ref_vcf", required=True)
    parser.add_argument("--target_vcf", required=True)
    parser.add_argument("--window_csv", required=True)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    args= parser.parse_args()

    # 1) load reference => (var_ref, samp_ref,2), pos_ref
    ref_data, pos_ref= load_vcf_data(args.ref_vcf)
    var_ref, samp_ref, _= ref_data.shape
    print(f"[INFO] ref_data => shape=({var_ref},{samp_ref},2)")

    # 2) load target => expand => shape=(var_ref, sample_tgt,2) + missing_mask
    tgt_data, pos_tgt= load_vcf_data(args.target_vcf)
    var_tgt, samp_tgt, _= tgt_data.shape
    print(f"[INFO] target => shape=({var_tgt},{samp_tgt},2) => expand to var_ref={var_ref}")
    if args.sample_idx<0 or args.sample_idx>= samp_tgt:
        print(f"[WARN] sample_idx={args.sample_idx} out-of-range => exit")
        return

    expanded_data, missing_mask= expand_target_to_ref(pos_ref, tgt_data, pos_tgt)
    # expanded_data => shape=(var_ref,sample_tgt,2)
    # missing_mask  => shape=(var_ref, sample_tgt)

    # 3) load window_csv => (start,end)
    df= pd.read_csv(args.window_csv, usecols=[0,1])
    start_arr= df.iloc[:,0].astype(int).to_numpy()
    end_arr  = df.iloc[:,1].astype(int).to_numpy()
    num_windows= len(start_arr)

    for w_idx in range(num_windows):
        start_i= start_arr[w_idx]
        end_i  = end_arr[w_idx]
        if start_i<0 or end_i>var_ref or start_i>=end_i:
            print(f"Skipping window {w_idx} => invalid range=({start_i},{end_i})")
            continue
        w_len= end_i - start_i
        # slice reference => shape=(w_len, samp_ref,2) => transpose =>(samp_ref,w_len,2)
        ref_sub= ref_data[start_i:end_i]  # (w_len,samp_ref,2)
        ref_sub= np.transpose(ref_sub,(1,0,2)) #(samp_ref,w_len,2)
        num_ref= ref_sub.shape[0]

        # slice target => shape=(w_len,2)
        sub_tgt= expanded_data[start_i:end_i, args.sample_idx,:]  # (w_len,2)
        hap_1= sub_tgt[:,0]
        hap_2= sub_tgt[:,1]

        # slice missing_mask => shape=(w_len,)
        sub_mask= missing_mask[start_i:end_i, args.sample_idx]
        # sub_mask=1 => 说明此pos不在 target => or truly missing => skip

        # 这里 你也可再加 "random" mask => sub_mask= sub_mask | random_mask ?

        # dynamic partial index
        I,D, build_t, search_t= build_partial_index_l2(hap_1, hap_2, sub_mask, ref_sub, top_k=args.top_k)
        print(f"[Window {w_idx}] range=({start_i},{end_i}), len={w_len}, #ref={num_ref}")
        print(f"  build_index_time={build_t:.4f}s, search_time={search_t:.4f}s")
        print(f"  topK idx={I}, dist={D}")


if __name__=="__main__":
    main()
