#!/usr/bin/env python
# scripts/build_ref_db_intersect.py

import argparse
import os
import pathlib

import h5py
import numpy as np
import allel

from src.dataset import PanelData, Window  # 如果 PanelData, Window 在别处，请相应修改import

def build_ref_db_intersect(args):
    """
    离线处理参考VCF，仅存原始0/1数据 + positions + population，不建索引。

    1) 若 ref_vcf 为 .vcf/.vcf.gz, 转成 .h5
    2) 读 h5 => vcf_data(variants, samples, 2), pos_data(variants,)
    3) window.csv => 对 reference 切分 (start_i, end_i)
    4) 存:
       - window_{w}.npy => shape (num_ref, window_len, 2)
       - window_{w}_pos.npy => shape (window_len,) => 记录各SNP的坐标
       - window_{w}_pop.npy => population label
    """

    # 0) 若 ref_vcf 是 .vcf 或 .vcf.gz，则先转 .h5
    vcf_path = pathlib.Path(args.ref_vcf)
    if vcf_path.suffix == '.vcf' or vcf_path.suffixes == ['.vcf', '.gz']:
        h5_path = None
        if vcf_path.suffixes == ['.vcf', '.gz']:
            tmp_no_gz = vcf_path.with_suffix('')
            h5_path = tmp_no_gz.with_suffix('.h5')
        else:
            h5_path = vcf_path.with_suffix('.h5')

        if not h5_path.exists():
            print(f"[build_ref_db_intersect] Converting {vcf_path} => {h5_path} via allel.vcf_to_hdf5...")
            allel.vcf_to_hdf5(str(vcf_path), str(h5_path), fields='*', overwrite=True)

        vcf_path = h5_path

    print(f"[build_ref_db_intersect] Loading reference HDF5: {vcf_path}")
    with h5py.File(vcf_path, 'r') as h5f:
        vcf_data = h5f['calldata/GT'][:]   # (num_variants, num_samples, 2)
        pos_data = h5f['variants/POS'][:] # (num_variants,)

    # 将 >0 的全部设为1
    vcf_data[vcf_data > 0] = 1

    # 1) 读 panel
    panel = PanelData.from_file(args.ref_panel)
    assert len(panel.pop_list) == vcf_data.shape[1], \
        f"Panel sample count {len(panel.pop_list)} != vcf_data.shape[1]={vcf_data.shape[1]}"

    # 2) 读 window
    window_obj = Window.from_file(args.window_csv)
    num_windows = window_obj.window_info.shape[0]
    print(f"[build_ref_db_intersect] #windows = {num_windows}")

    # 3) 输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 4) 逐窗口存原始信息
    for w_idx, (start_i, end_i) in enumerate(window_obj.window_info):
        sub_data = vcf_data[start_i:end_i]   # (window_len, num_samples, 2)
        sub_pos  = pos_data[start_i:end_i]   # (window_len,)

        # 转置 => (num_samples, window_len, 2)
        sub_data = np.transpose(sub_data, (1,0,2))  # shape (num_ref, window_len, 2)

        # 保存
        np.save(os.path.join(args.output_dir, f"window_{w_idx}.npy"), sub_data)
        np.save(os.path.join(args.output_dir, f"window_{w_idx}_pos.npy"), sub_pos)
        np.save(os.path.join(args.output_dir, f"window_{w_idx}_pop.npy"), panel.pop_list)

        if (w_idx+1) % 50 == 0:
            print(f"[build_ref_db_intersect] processed window {w_idx+1}/{num_windows}")

    print(f"[build_ref_db_intersect] Done! Save raw data + positions + pop to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_vcf", type=str, required=True,
                        help="参考VCF(.vcf/.vcf.gz)或HDF5文件")
    parser.add_argument("--ref_panel", type=str, required=True,
                        help="panel文件, 读取pop_list")
    parser.add_argument("--window_csv", type=str, required=True,
                        help="window定义.csv")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    args = parser.parse_args()

    build_ref_db_intersect(args)

if __name__ == "__main__":
    main()
