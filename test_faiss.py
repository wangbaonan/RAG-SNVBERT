#!/usr/bin/env python
# scripts/batch_test_faiss.py

import argparse
import os
import pathlib
import time

import h5py
import numpy as np
import allel
import faiss

from src.dataset import Window  # 你已有的

def load_vcf_to_array(vcf_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    读取 .vcf / .vcf.gz / .h5，并返回:
      - vcf_data: (num_variants, num_samples, 2) 只保留 0/1
      - pos_data: (num_variants,)
    如果是 .vcf/.vcf.gz，自动调用 scikit-allel 转成 .h5。
    """
    p = pathlib.Path(vcf_path)

    # 判断后缀
    if p.suffix == '.vcf' or p.suffixes == ['.vcf', '.gz']:
        # 先转 .h5
        h5_path = None
        if p.suffixes == ['.vcf', '.gz']:
            # remove .gz
            tmp = p.with_suffix('')
            h5_path = tmp.with_suffix('.h5')
        else:
            # .vcf
            h5_path = p.with_suffix('.h5')

        if not h5_path.exists():
            print(f"[batch_test_faiss] Converting {p} => {h5_path} via allel.vcf_to_hdf5...")
            allel.vcf_to_hdf5(str(p), str(h5_path), fields='*', overwrite=True)

        p = h5_path

    # 现在 p 应该是 .h5
    print(f"[batch_test_faiss] Loading target HDF5: {p}")
    with h5py.File(p, 'r') as h5f:
        vcf_data = h5f['calldata/GT'][:]   # (num_variants, num_samples, 2)
        pos_data = h5f['variants/POS'][:] # (num_variants,)

    # 将 >0 的都设为 1
    vcf_data[vcf_data > 0] = 1

    return vcf_data, pos_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_vcf", type=str, required=True,
                        help="待测 VCF(.vcf/.vcf.gz) 或 HDF5(.h5)")
    parser.add_argument("--ref_db", type=str, required=True,
                        help="离线构建的参考数据库目录：含 window_{i}.faiss, window_{i}.npy 等")
    parser.add_argument("--window_csv", type=str, required=True,
                        help="和ref_db一致的window定义.csv")
    parser.add_argument("--top_k", type=int, default=5,
                        help="检索TopK大小")
    parser.add_argument("--sample_idx", type=int, default=-1,
                        help="默认-1表示对所有样本批量检索；>=0则只对指定样本。")
    parser.add_argument("--show_snp_len", type=int, default=10,
                        help="打印SNP片段时, 截断长度(避免输出爆炸)")
    parser.add_argument("--print_neighbor", action='store_true',
                        help="是否打印邻居的0/1片段(只显示show_snp_len个SNP)")

    args = parser.parse_args()

    # 1) 加载目标 VCF => vcf_data (num_variants, num_samples, 2)
    #                 pos_data (num_variants,)
    vcf_data, pos_data = load_vcf_to_array(args.target_vcf)
    n_variants, n_samples, _ = vcf_data.shape
    print(f"[batch_test_faiss] target VCF => #variants={n_variants}, #samples={n_samples}")

    # 2) 读取window
    window_obj = Window.from_file(args.window_csv)
    num_windows = window_obj.window_info.shape[0]
    print(f"[batch_test_faiss] #windows={num_windows}")

    # 确定要处理的“样本范围”
    if args.sample_idx < 0:
        sample_list = range(n_samples)  # 全部样本
        print("[batch_test_faiss] Will do batch retrieval for ALL samples.")
    else:
        if args.sample_idx >= n_samples:
            raise ValueError(f"sample_idx={args.sample_idx} >= #samples={n_samples}")
        sample_list = [args.sample_idx]
        print(f"[batch_test_faiss] Will do retrieval for SINGLE sample={args.sample_idx}.")

    # 3) 逐窗口进行批量检索
    #    => 先把 [sample_list] 对应的 (window_len, 2) flatten => (num_samples_in_list, window_len*2)
    #    => index.search(batch_data, top_k)
    overall_start_time = time.time()

    for w_idx, (start_i, end_i) in enumerate(window_obj.window_info):
        window_len = end_i - start_i
        # (window_len, 2)
        # 我们这里对 sample_list 做batch => shape: (num_samples_in_list, window_len, 2)
        sub_data = vcf_data[start_i:end_i, :, :]  # => (window_len, n_samples, 2)
        # 转置 => (n_samples, window_len, 2)
        sub_data = sub_data.transpose((1, 0, 2))  # (n_samples, window_len, 2)

        # 再取 sample_list
        sub_data_batch = sub_data[sample_list]  # shape: (#batch, window_len, 2)
        # flatten => (batch, window_len*2)
        batch_data_1d = sub_data_batch.reshape(sub_data_batch.shape[0], -1).astype(np.float32)

        # 加载FAISS索引 & ref_data
        faiss_path = os.path.join(args.ref_db, f"window_{w_idx}.faiss")
        if not os.path.exists(faiss_path):
            # 说明参考库没有这个窗口的索引
            continue
        index = faiss.read_index(faiss_path)

        # 参考 0/1 数据 => (num_ref, window_len, 2)
        ref_npy = os.path.join(args.ref_db, f"window_{w_idx}.npy")
        if not os.path.exists(ref_npy):
            continue
        ref_data = np.load(ref_npy)

        # 如果有pop文件
        pop_npy = os.path.join(args.ref_db, f"window_{w_idx}_pop.npy")
        if os.path.exists(pop_npy):
            ref_pop = np.load(pop_npy)  # shape (num_ref,)
        else:
            ref_pop = None

        # ============ 做检索 =============
        t0 = time.time()
        D, I = index.search(batch_data_1d, args.top_k)
        t1 = time.time()
        dt = t1 - t0

        print(f"\n[Window {w_idx}] start={start_i}, end={end_i}, len={window_len}")
        print(f"   #queries={len(sample_list)}, search_time={dt:.4f}s")

        # 4) 打印结果(只给出少量信息)
        #    D, I 分别是 (batch_size, top_k)
        for b_idx in range(D.shape[0]):
            # sampleID in target
            real_sample_idx = sample_list[b_idx]
            # 只演示top-1
            best_id = I[b_idx, 0]
            best_dist = D[b_idx, 0]
            print(f"   targetSample={real_sample_idx}, bestNeighborID={best_id}, dist={best_dist:.4f}", end='')
            if ref_pop is not None and best_id < len(ref_pop):
                print(f", pop={ref_pop[best_id]}")
            else:
                print()

            if args.print_neighbor:
                # 打印SNP截断
                show_len = min(args.show_snp_len, window_len)
                # 目标
                target_snps = sub_data_batch[b_idx, :show_len, :].flatten()
                # 邻居
                neighbor_snps = ref_data[best_id, :show_len, :].flatten()
                print(f"     target snippet => {target_snps.tolist()}")
                print(f"     neighb snippet => {neighbor_snps.tolist()}")

    overall_end_time = time.time()
    overall_dt = overall_end_time - overall_start_time
    print(f"\n[batch_test_faiss] Done all windows. Total time = {overall_dt:.2f}s.")


if __name__ == "__main__":
    main()
