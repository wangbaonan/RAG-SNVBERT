#!/usr/bin/env python
# scripts/test_faiss_intersect.py

import argparse
import os
import pathlib
import time

import h5py
import numpy as np
import allel
import faiss

from src.dataset import Window

def load_target_vcf(target_vcf):
    """
    读 target VCF(.vcf/.vcf.gz/.h5)，返回:
      vcf_data => (num_variants, num_samples, 2) 只保留0/1
      pos_data => (num_variants,)
    """
    p = pathlib.Path(target_vcf)
    if p.suffix == '.vcf' or p.suffixes == ['.vcf', '.gz']:
        h5_path = None
        if p.suffixes == ['.vcf', '.gz']:
            tmp = p.with_suffix('')
            h5_path = tmp.with_suffix('.h5')
        else:
            h5_path = p.with_suffix('.h5')

        if not h5_path.exists():
            print(f"[test_faiss_intersect] converting {p} => {h5_path} via allel.vcf_to_hdf5")
            allel.vcf_to_hdf5(str(p), str(h5_path), fields='*', overwrite=True)

        p = h5_path

    print(f"[test_faiss_intersect] Loading target HDF5: {p}")
    with h5py.File(p, 'r') as h5f:
        vcf_data = h5f['calldata/GT'][:]   # (num_variants, num_samples, 2)
        pos_data = h5f['variants/POS'][:] # (num_variants,)

    vcf_data[vcf_data > 0] = 1
    return vcf_data, pos_data


def bitpack_2d_array(arr_2d):
    """
    将 (N, d) 的0/1数组打包成二进制uint8 => shape (N, ceil(d/8))
    arr_2d[i,j] in {0,1}
    """
    arr_2d = arr_2d.astype(np.uint8)
    n, d = arr_2d.shape
    packed = np.packbits(arr_2d, axis=1)
    return packed


def test_faiss_intersect(args):
    """
    核心逻辑:
      1) 读 target_vcf => (num_variants, num_samples,2), pos =>(num_variants,)
      2) 读 window.csv => 逐窗口
      3) 对 ref_db 中 window_{w}.npy, window_{w}_pos.npy 做 '交集pos' => flatten => 构建临时索引
      4) distance_mode=l2 => IndexFlatL2; binary => IndexBinaryFlat
      5) search top-k, 输出
      6) 打印更多日志: 当窗口被跳过时也说明原因
    """

    # ========== 加载目标数据 ==========
    t0_all = time.time()
    target_data, target_pos = load_target_vcf(args.target_vcf)
    n_variants, n_samples, _ = target_data.shape
    print(f"[test_faiss_intersect] target => n_variants={n_variants}, n_samples={n_samples}")

    # ========== 加载 Window 定义 ==========
    window_obj = Window.from_file(args.window_csv)
    num_windows = window_obj.window_info.shape[0]
    print(f"[test_faiss_intersect] #windows={num_windows}")

    # 确定要检索的样本范围
    if args.sample_idx < 0:
        sample_list = range(n_samples)
        print("[test_faiss_intersect] Will do BATCH retrieval for ALL target samples.")
    else:
        if args.sample_idx >= n_samples:
            raise ValueError(f"sample_idx={args.sample_idx} out of range (n_samples={n_samples})")
        sample_list = [args.sample_idx]
        print(f"[test_faiss_intersect] Will do retrieval for single sample={args.sample_idx}")

    # ========== 逐窗口处理 ==========
    for w_idx, (start_i, end_i) in enumerate(window_obj.window_info):
        print(f"\n----- Window {w_idx} : range=({start_i}, {end_i}) -----")

        # 1) 加载参考数据
        ref_npy = os.path.join(args.ref_db, f"window_{w_idx}.npy")
        ref_pos_npy = os.path.join(args.ref_db, f"window_{w_idx}_pos.npy")
        ref_pop_npy = os.path.join(args.ref_db, f"window_{w_idx}_pop.npy")

        if not os.path.exists(ref_npy):
            print(f"Skipping window {w_idx}: ref_npy not found => {ref_npy}")
            continue
        if not os.path.exists(ref_pos_npy):
            print(f"Skipping window {w_idx}: ref_pos_npy not found => {ref_pos_npy}")
            continue

        ref_data = np.load(ref_npy)         # shape (num_ref, window_len, 2)
        ref_pos  = np.load(ref_pos_npy)     # shape (window_len,)
        ref_pop  = np.load(ref_pop_npy) if os.path.exists(ref_pop_npy) else None

        window_len = ref_pos.shape[0]
        print(f"    #ref={ref_data.shape[0]}, window_len={window_len}")

        # 2) 检查窗口是否超出 target 范围
        if end_i > target_data.shape[0]:
            print(f"Skipping window {w_idx}: end_i({end_i}) > target_data.shape[0]({target_data.shape[0]})")
            continue

        # slice target
        t_sub = target_data[start_i:end_i]  # (local_len, n_samples,2)
        t_pos = target_pos[start_i:end_i]   # (local_len,)

        # 转置 => (n_samples, local_len, 2)
        t_sub = t_sub.transpose((1, 0, 2))

        local_len = t_pos.size
        print(f"    target local_len={local_len}, #samples={n_samples}")

        # 3) 做交集
        shared_pos = np.intersect1d(ref_pos, t_pos)
        if shared_pos.size == 0:
            print(f"Skipping window {w_idx}: no shared SNP => shared_pos.size=0")
            continue
        intersection_len = shared_pos.size
        print(f"    intersection_len={intersection_len}")

        # 4) 根据 shared_pos 对 ref_data, target_data 做索引
        ref_idx = np.searchsorted(ref_pos, shared_pos)
        ref_data_sub = ref_data[:, ref_idx, :]  # (num_ref, intersection_len,2)

        t_idx = np.searchsorted(t_pos, shared_pos)
        t_data_sub = t_sub[:, t_idx, :]        # (n_samples, intersection_len,2)

        # 只取 sample_list
        t_data_sub = t_data_sub[list(sample_list)]
        print(f"    #queries={t_data_sub.shape[0]}")

        # 5) 构建索引并搜索
        if args.distance_mode == 'l2':
            print("    Using IndexFlatL2 (欧氏距离)")
            ref_flat = ref_data_sub.reshape(ref_data_sub.shape[0], -1).astype(np.float32)
            t_flat   = t_data_sub.reshape(t_data_sub.shape[0], -1).astype(np.float32)

            index = faiss.IndexFlatL2(ref_flat.shape[1])
            print(f"    Building L2 index... (#ref={ref_flat.shape[0]}, dims={ref_flat.shape[1]})")
            t0 = time.time()
            index.add(ref_flat)
            print(f"    Done building. cost={time.time()-t0:.4f}s")

            print("    Searching top-k...")
            t1 = time.time()
            D, I = index.search(t_flat, args.top_k)
            dt = time.time() - t1
            print(f"    Search done. cost={dt:.4f}s")

        elif args.distance_mode == 'binary':
            print("    Using IndexBinaryFlat (Hamming距离)")
            d_bits = 2 * intersection_len
            ref_flat_01 = ref_data_sub.reshape(ref_data_sub.shape[0], -1).astype(np.uint8)
            t_flat_01   = t_data_sub.reshape(t_data_sub.shape[0], -1).astype(np.uint8)

            ref_packed = bitpack_2d_array(ref_flat_01)
            t_packed   = bitpack_2d_array(t_flat_01)

            index = faiss.IndexBinaryFlat(d_bits)
            print(f"    Building Binary index... (#ref={ref_packed.shape[0]}, dims_bits={d_bits})")
            t0 = time.time()
            index.add(ref_packed)
            print(f"    Done building. cost={time.time()-t0:.4f}s")

            print("    Searching top-k...")
            t1 = time.time()
            D, I = index.search(t_packed, args.top_k)
            dt = time.time() - t1
            print(f"    Search done. cost={dt:.4f}s")

        else:
            print(f"Skipping window {w_idx}: unknown distance_mode={args.distance_mode}")
            continue

        # 6) 打印 top-1
        for b_idx in range(I.shape[0]):
            real_sid = sample_list[b_idx]
            best_id = I[b_idx, 0]
            best_dist = D[b_idx, 0]
            print(f"      sample={real_sid}, bestRefID={best_id}, dist={best_dist}", end='')
            if ref_pop is not None and best_id < ref_pop.size:
                print(f", pop={ref_pop[best_id]}")
            else:
                print()

            # 若需查看SNP片段
            if args.show_snps:
                show_len = min(intersection_len, args.show_snp_len)
                tar_snip = t_data_sub[b_idx, :show_len, :].flatten().tolist()
                ref_snip = ref_data_sub[best_id, :show_len, :].flatten().tolist()
                print(f"         target => {tar_snip}")
                print(f"         neighb => {ref_snip}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_vcf", type=str, required=True,
                        help="待测VCF(.vcf/.vcf.gz/.h5)")
    parser.add_argument("--ref_db", type=str, required=True,
                        help="由 build_ref_db_intersect.py 生成的目录, 包含 raw reference + pos + pop")
    parser.add_argument("--window_csv", type=str, required=True,
                        help="window定义.csv")
    parser.add_argument("--distance_mode", type=str, default='l2',
                        choices=['l2','binary'],
                        help="选择 'l2'(欧氏距离:IndexFlatL2) 或 'binary'(Hamming:IndexBinaryFlat)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="检索top_k")
    parser.add_argument("--sample_idx", type=int, default=-1,
                        help="-1 => 所有样本一起batch; >=0 => 只选定样本")
    parser.add_argument("--show_snps", action='store_true',
                        help="是否打印SNP片段")
    parser.add_argument("--show_snp_len", type=int, default=10,
                        help="最多打印多少个SNP(截断)")
    args = parser.parse_args()

    test_faiss_intersect(args)

if __name__ == "__main__":
    main()
