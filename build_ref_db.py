#!/usr/bin/env python
# scripts/build_ref_db.py

import argparse
import os
import pathlib

import h5py
import numpy as np
import allel
import faiss

from src.dataset import PanelData, Window  # 如果 Window, PanelData 在别的路径，就相应修改import


def build_ref_db(args):
    """
    离线处理参考数据：
      1) 若 ref_vcf 是 .vcf / .vcf.gz，则转成 .h5
      2) 加载 .h5 (读取 GT, POS)
      3) 读取 panel (population label) 并与 GT 样本对齐
      4) 读取 window.csv, 逐窗口切 slice，保存 (num_samples, window_len, 2) 到 window_*.npy
      5) 同时构建 FAISS 索引 (IndexFlatL2) 并写到 window_*.faiss
    """

    # ========== (1) 如果是 .vcf 或 .vcf.gz，就转成 .h5 ==========
    vcf_path = pathlib.Path(args.ref_vcf)
    
    # 如果输入是 .vcf.gz， vcf_path.suffix == '.gz'; vcf_path.suffixes == ['.vcf','.gz']
    # 如果输入是 .vcf，    vcf_path.suffix == '.vcf'
    # 如果输入是 .h5，     vcf_path.suffix == '.h5'
    
    if vcf_path.suffix == '.vcf' or vcf_path.suffixes == ['.vcf', '.gz']:
        # 生成一个 h5 文件路径
        #   如果原文件是 something.vcf.gz => something.vcf => something.h5
        #   如果原文件是 something.vcf    => something.h5
        #   如果原文件是 something_else.vcf.gz => something_else.h5
        h5_path = None
        
        if vcf_path.suffixes == ['.vcf', '.gz']:
            # 去掉最外层.gz
            tmp_no_gz = vcf_path.with_suffix('')
            # 现在 tmp_no_gz.suffix 可能是 '.vcf'
            h5_path = tmp_no_gz.with_suffix('.h5')
        else:
            # suffix == '.vcf'
            h5_path = vcf_path.with_suffix('.h5')

        if not h5_path.exists():
            print(f"[build_ref_db] Converting {vcf_path} to {h5_path} via allel.vcf_to_hdf5...")
            allel.vcf_to_hdf5(
                input=str(vcf_path),
                output=str(h5_path),
                fields='*',
                overwrite=True
            )
        print(f"[build_ref_db] Will read from {h5_path} now.")
        vcf_path = h5_path
    else:
        # 如果不是 .vcf / .vcf.gz，就认为是 .h5 (或者你自己确保输入是h5)
        pass

    # ========== (2) 打开 HDF5, 读取 GT & POS ==========
    print(f"[build_ref_db] Loading reference HDF5: {vcf_path}")
    h5f = h5py.File(vcf_path, 'r')
    
    # shape: (num_variants, num_samples, 2)
    vcf_data = h5f['calldata/GT'][:]
    # shape: (num_variants,)
    pos_data = h5f['variants/POS'][:] 

    # 把所有 >0 的值改为1, 保持0/1
    vcf_data[vcf_data > 0] = 1

    # ========== (3) 读取 panel（若需要 population info）==========
    print(f"[build_ref_db] Loading panel: {args.ref_panel}")
    panel = PanelData.from_file(args.ref_panel)

    # 保证 panel.pop_list 大小 与 vcf_data.shape[1] (samples)一致
    assert len(panel.pop_list) == vcf_data.shape[1], \
        f"Panel sample count ({len(panel.pop_list)}) != VCF sample count ({vcf_data.shape[1]})"

    # ========== (4) 读取 window.csv, 同目标切分一致 ==========
    print(f"[build_ref_db] Loading window csv: {args.window_csv}")
    window_obj = Window.from_file(args.window_csv)
    num_windows = window_obj.window_info.shape[0]
    print(f"[build_ref_db] Number of windows: {num_windows}")

    # ========== (5) 输出目录(存 .npy & .faiss) ==========
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== (6) 逐个 window 切分 & 构建 FAISS 索引 ==========
    for w_idx, (start_i, end_i) in enumerate(window_obj.window_info):
        # sub_data shape => (window_len, num_samples, 2)
        sub_data = vcf_data[start_i:end_i]
        window_len = end_i - start_i
        
        # 转置 => (num_samples, window_len, 2)
        sub_data = np.transpose(sub_data, (1, 0, 2))

        # (6.1) 保存原始 0/1 数据
        npy_path = os.path.join(args.output_dir, f"window_{w_idx}.npy")
        np.save(npy_path, sub_data)

        # 保存 population label
        pop_labels = panel.pop_list  # shape (num_samples,)
        label_path = os.path.join(args.output_dir, f"window_{w_idx}_pop.npy")
        np.save(label_path, pop_labels)

        # (6.2) 构建 FAISS 索引: 先 flatten (num_samples, window_len*2)
        flatten_data = sub_data.reshape(sub_data.shape[0], -1).astype(np.float32)

        # 建立简单的IndexFlatL2, 适合中小规模; 大规模可用 IVF,PQ 等
        index = faiss.IndexFlatL2(flatten_data.shape[1])
        index.add(flatten_data)

        # 写到磁盘
        faiss_path = os.path.join(args.output_dir, f"window_{w_idx}.faiss")
        faiss.write_index(index, faiss_path)

        # 每处理50个窗口打个log
        if (w_idx+1) % 50 == 0:
            print(f"[build_ref_db] Processed window {w_idx+1}/{num_windows}")

    h5f.close()
    print(f"[build_ref_db] Done! All reference windows & FAISS indices saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_vcf", type=str, required=True,
                        help="Reference VCF(.vcf/.vcf.gz) or HDF5(.h5) file")
    parser.add_argument("--ref_panel", type=str, required=True,
                        help="Reference panel txt file (2-col: sampleID,pop...)")
    parser.add_argument("--window_csv", type=str, required=True,
                        help="CSV file defining windows, with start and end in columns[0,1]")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for .npy & .faiss indices")
    args = parser.parse_args()

    build_ref_db(args)


if __name__ == "__main__":
    main()
