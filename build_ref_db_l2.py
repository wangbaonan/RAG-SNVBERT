#!/usr/bin/env python
# scripts/build_ref_db_l2.py

import argparse
import os
import pathlib

import h5py
import numpy as np
import allel
import faiss

from src.dataset import PanelData, Window  # 如果你的 PanelData, Window 定义在别的地方，请相应修改import

def build_ref_db_l2(args):
    """
    离线处理参考 VCF 并构建 L2 索引:
      1) 如果 ref_vcf 是 .vcf 或 .vcf.gz => 转成 .h5
      2) 读取 [calldata/GT] => (variants, samples, 2), 并把 >0 的设为1
      3) 按 window.csv 切分 => (num_samples, window_len, 2)
      4) flatten => (num_samples, window_len*2), 交给 IndexFlatL2
      5) 输出: window_{i}.faiss, window_{i}.npy, window_{i}_pop.npy
    """

    # 1) 如果是 .vcf / .vcf.gz，就转成 .h5
    vcf_path = pathlib.Path(args.ref_vcf)
    if vcf_path.suffix == '.vcf' or vcf_path.suffixes == ['.vcf', '.gz']:
        # 生成 .h5 路径
        h5_path = None
        if vcf_path.suffixes == ['.vcf', '.gz']:
            tmp_no_gz = vcf_path.with_suffix('')
            h5_path = tmp_no_gz.with_suffix('.h5')
        else:
            h5_path = vcf_path.with_suffix('.h5')

        if not h5_path.exists():
            print(f"[build_ref_db_l2] Converting {vcf_path} => {h5_path} via allel.vcf_to_hdf5...")
            allel.vcf_to_hdf5(str(vcf_path), str(h5_path), fields='*', overwrite=True)

        vcf_path = h5_path

    # 2) 读 .h5
    print(f"[build_ref_db_l2] Loading reference HDF5: {vcf_path}")
    h5f = h5py.File(vcf_path, 'r')
    vcf_data = h5f['calldata/GT'][:]   # (num_variants, num_samples, 2)
    pos_data = h5f['variants/POS'][:] # (num_variants,)
    h5f.close()

    # 将 >0 的都设为 1
    vcf_data[vcf_data > 0] = 1

    # 3) 读取 panel
    print(f"[build_ref_db_l2] Loading panel: {args.ref_panel}")
    panel = PanelData.from_file(args.ref_panel)
    assert len(panel.pop_list) == vcf_data.shape[1], \
        f"Panel sample count ({len(panel.pop_list)}) != VCF sample count ({vcf_data.shape[1]})"

    # 4) 读取 Window
    print(f"[build_ref_db_l2] Loading window csv: {args.window_csv}")
    window_obj = Window.from_file(args.window_csv)
    num_windows = window_obj.window_info.shape[0]
    print(f"[build_ref_db_l2] Number of windows = {num_windows}")

    # 5) 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 6) 逐 window 构建 L2 索引
    for w_idx, (start_i, end_i) in enumerate(window_obj.window_info):
        # sub_data => (window_len, num_samples, 2)
        sub_data = vcf_data[start_i:end_i]
        window_len = end_i - start_i

        # 转置 => (num_samples, window_len, 2)
        sub_data = np.transpose(sub_data, (1,0,2))

        # 保存到 .npy
        npy_path = os.path.join(args.output_dir, f"window_{w_idx}.npy")
        np.save(npy_path, sub_data)

        # population label
        pop_labels = panel.pop_list
        label_path = os.path.join(args.output_dir, f"window_{w_idx}_pop.npy")
        np.save(label_path, pop_labels)

        # flatten => shape (num_samples, window_len*2)
        flatten_data = sub_data.reshape(sub_data.shape[0], -1).astype(np.float32)

        # 建索引: IndexFlatL2
        index = faiss.IndexFlatL2(flatten_data.shape[1])
        index.add(flatten_data)

        faiss_path = os.path.join(args.output_dir, f"window_{w_idx}.faiss")
        faiss.write_index(index, faiss_path)

        if (w_idx+1) % 50 == 0:
            print(f"[build_ref_db_l2] Processed window {w_idx+1}/{num_windows}")

    print(f"[build_ref_db_l2] Done! All ref windows & L2 indices saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_vcf", type=str, required=True,
                        help="参考VCF(.vcf/.vcf.gz)或 HDF5文件")
    parser.add_argument("--ref_panel", type=str, required=True,
                        help="参考panel, 用来获取population信息")
    parser.add_argument("--window_csv", type=str, required=True,
                        help="与目标数据一致的window.csv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    args = parser.parse_args()

    build_ref_db_l2(args)

if __name__ == "__main__":
    main()
