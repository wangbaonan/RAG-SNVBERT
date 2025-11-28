#!/usr/bin/env python
# scripts/batch_test_faiss_l2.py

import argparse
import os
import pathlib
import time

import h5py
import numpy as np
import allel
import faiss

from src.dataset import Window  # 你的 Window 类

def load_target_vcf(vcf_file: str):
    """
    加载待测VCF => vcf_data(variants, samples, 2), pos_data(variants,)
    若是 .vcf / .vcf.gz => 转换成 .h5
    """
    p = pathlib.Path(vcf_file)

    if p.suffix == '.vcf' or p.suffixes == ['.vcf', '.gz']:
        # 转成 .h5
        h5_path = None
        if p.suffixes == ['.vcf', '.gz']:
            tmp = p.with_suffix('')
            h5_path = tmp.with_suffix('.h5')
        else:
            h5_path = p.with_suffix('.h5')

        if not h5_path.exists():
            print(f"[batch_test_faiss_l2] Converting {p} => {h5_path} via allel.vcf_to_hdf5()")
            allel.vcf_to_hdf5(str(p), str(h5_path), fields='*', overwrite=True)
        p = h5_path

    # 读取 h5
    print(f"[batch_test_faiss_l2] Loading target HDF5: {p}")
    with h5py.File(p, 'r') as h5f:
        vcf_data = h5f['calldata/GT'][:]   # (num_variants, num_samples, 2)
        pos_data = h5f['variants/POS'][:] # (num_variants,)

    # 将 >0 的都设为1
    vcf_data[vcf_data>0] = 1
    return vcf_data, pos_data


def batch_test_faiss_l2(args):
    """
    1) 加载 target_vcf => (num_variants, num_samples, 2)
    2) 读取 window.csv => 逐窗口
    3) 若 sample_idx=-1 => 全部样本一起 batch search; 否则只对指定样本
    4) 加载 window_{i}.faiss => IndexFlatL2
    5) flatten => (batch, dims) => index.search(...) => 输出
    6) 统计并输出耗时
    """

    # 1) 加载 待测VCF
    vcf_data, pos_data = load_target_vcf(args.target_vcf)
    n_variants, n_samples, _ = vcf_data.shape
    print(f"[batch_test_faiss_l2] target => n_variants={n_variants}, n_samples={n_samples}")

    # 2) window
    window_obj = Window.from_file(args.window_csv)
    num_windows = window_obj.window_info.shape[0]
    print(f"[batch_test_faiss_l2] total windows={num_windows}")

    # 样本范围
    if args.sample_idx < 0:
        sample_list = range(n_samples)
        print("[batch_test_faiss_l2] Will do BATCH retrieval for ALL samples.")
    else:
        if args.sample_idx >= n_samples:
            raise ValueError(f"sample_idx={args.sample_idx} >= {n_samples}")
        sample_list = [args.sample_idx]
        print(f"[batch_test_faiss_l2] Will do retrieval for SINGLE sample={args.sample_idx}")

    overall_start = time.time()

    for w_idx, (start_i, end_i) in enumerate(window_obj.window_info):
        window_len = end_i - start_i
        # 取 target 的 (n_samples, window_len, 2)
        sub_data = vcf_data[start_i:end_i].transpose((1,0,2))

        # 选取 sample_list => (batch, window_len, 2)
        sub_data_batch = sub_data[list(sample_list)]
        # flatten => (batch, window_len*2)
        batch_data_1d = sub_data_batch.reshape(sub_data_batch.shape[0], -1).astype(np.float32)

        # 加载相应faiss
        faiss_path = os.path.join(args.ref_db, f"window_{w_idx}.faiss")
        if not os.path.exists(faiss_path):
            continue
        index = faiss.read_index(faiss_path)

        # 加载参考数据
        ref_npy = os.path.join(args.ref_db, f"window_{w_idx}.npy")
        if not os.path.exists(ref_npy):
            continue
        ref_data = np.load(ref_npy)  # (num_ref, window_len, 2)

        # 可能有pop
        pop_npy = os.path.join(args.ref_db, f"window_{w_idx}_pop.npy")
        ref_pop = None
        if os.path.exists(pop_npy):
            ref_pop = np.load(pop_npy)  # (num_ref,)

        # do search
        t0 = time.time()
        D, I = index.search(batch_data_1d, args.top_k)  # D,I => (batch, top_k)
        t1 = time.time()
        dt = t1 - t0

        print(f"\n[Window {w_idx}] start={start_i}, end={end_i}, length={window_len}")
        print(f"  #query={len(sample_list)}, searchTime={dt:.4f}s")

        # 打印结果(仅top-1)
        for b_idx in range(D.shape[0]):
            real_sid = sample_list[b_idx]
            best_id = I[b_idx, 0]
            best_dist = D[b_idx, 0]
            print(f"    targetSample={real_sid}, bestID={best_id}, dist={best_dist:.4f}", end='')
            if ref_pop is not None and best_id < len(ref_pop):
                print(f", pop={ref_pop[best_id]}")
            else:
                print()

            if args.print_snippet:
                # 截断
                show_len = min(window_len, args.show_snp_len)
                target_snps = sub_data_batch[b_idx, :show_len, :].flatten()
                neighbor_snps = ref_data[best_id, :show_len, :].flatten()
                print(f"      target => {target_snps.tolist()}")
                print(f"      neighb => {neighbor_snps.tolist()}")

    overall_dt = time.time() - overall_start
    print(f"[batch_test_faiss_l2] Done all windows. totalTime={overall_dt:.2f}s.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_vcf", type=str, required=True,
                        help="待测vcf(.vcf/.vcf.gz/.h5)")
    parser.add_argument("--ref_db", type=str, required=True,
                        help="之前 build_ref_db_l2.py 生成的目录")
    parser.add_argument("--window_csv", type=str, required=True,
                        help="与ref_db一致的 window定义.csv")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--sample_idx", type=int, default=-1,
                        help="-1 => 全部样本一起batch; >=0 => 指定单样本")
    parser.add_argument("--show_snp_len", type=int, default=10,
                        help="打印SNP片段的截断长度")
    parser.add_argument("--print_snippet", action='store_true',
                        help="是否打印target/neighbor的0/1片段")
    args = parser.parse_args()

    batch_test_faiss_l2(args)


if __name__ == "__main__":
    main()
