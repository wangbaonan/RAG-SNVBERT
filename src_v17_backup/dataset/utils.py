import numpy as np
import random
import vcfpy
import os
import gc

from time import time
from typing import Literal
from tqdm import tqdm


INFER_WINDOW_LEN = 1020
MAX_SEQ_LEN = 1030

GT_MAP = {
    0 : "0|0",
    1 : "0|1",
    2 : "1|0",
    3 : "1|1"
}


def timer(func):
    """Calculating Function's using time.
    """
    def wrapper(*args, **kwargs):

        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()

        print(f"Function: {func.__name__:<20}, Time using: {end_time - start_time:.2f}s.")

        return result
    
    return wrapper



class PanelProcessingModule():
    """
    All methods about processing '.panel' file data.

    """

    @staticmethod
    def split_dataset(population_based_index_dict : dict[str, list[int]],
                      split_ratio : float = 0.20
                      ) -> tuple[list[int], list[int]]:
        """Preprocess the dataset, split training set and testing set based on [POP].

        Args:

            population_based_index_dict: Map population to sample indices.

            split_ratio: Ratio of Test dataset.

        """
        
        train_list = []
        test_list  = []

        for value in population_based_index_dict.values():

            assert isinstance(value, list), "Data should be list type when splitting training data & testing data."

            test_sample_count = int(len(value) * split_ratio)

            test_samples = random.sample(value, test_sample_count)

            train_samples = list(set(value) - set(test_samples))

            train_list.extend(train_samples)

            test_list.extend(test_samples)

        return train_list, test_list




class VCFProcessingModule():
    """
    All methods about processing .vcf file data.
    """

    @staticmethod
    def genotype_mapping(single_sample_gt_sequence : np.ndarray) -> np.ndarray:
        """
        Map sample's genotype into a numpy.ndarray of int type.

        Args:

            single_sample_gt_sequence: An np.ndarray consists of 0 & 1 with shape like (n, 2).
            >>> single_sample_gt_sequence.shape
            >>> (n, 2)
        
        """
        assert single_sample_gt_sequence.shape[1] == 2, "Input array must have shape (n, 2)"

        return single_sample_gt_sequence[:, 0] + single_sample_gt_sequence[:, 1]
    

    @staticmethod
    def position_normalize(position_sequence : np.ndarray) -> np.ndarray:
        """
        0-1 Normalization for [variants/POS] sequence
        """

        min_pos = np.min(position_sequence)
        max_pos = np.max(position_sequence)

        range_pos = max_pos - min_pos

        norm_pos = (position_sequence - min_pos) / range_pos

        return norm_pos
    

    @staticmethod
    def sequence_padding(seq : np.ndarray, dtype : str = Literal['int', 'float']) -> np.ndarray:
        """Padding sequence based on specific data type.

        If dtype is 'int', then pad sequence with '0'. Or pad sequence with '0.' when dtype is 'float'
        """
        pad_num = 0 if dtype == 'int' else 0.
        
        # Deep copy
        pre = np.array([pad_num]) # [SOS] token
        post = np.array([pad_num for _ in range(MAX_SEQ_LEN - seq.shape[0] - 1)])       # padding  

        return np.concatenate((pre, seq, post))

    @staticmethod
    def process_gt_prob_mat(gt_prob_mat: np.ndarray) -> np.ndarray:
        """
        将四分类概率矩阵转换为 scikit-allel 兼容的 vcf_data 格式。
    
        参数:
            gt_prob_mat: 形状为 (variants, samples, 4) 的浮点数组，表示四分类概率。
    
        返回:
            vcf_data: 形状为 (variants, samples, 2) 的 int8 数组，值为 0 或 1。
        """
        n_variants, n_samples, _ = gt_prob_mat.shape
        vcf_data = np.zeros((n_variants, n_samples, 2), dtype=np.int8)
    
    
        for var_idx in range(n_variants):
            for sample_idx in range(n_samples):
                # Step 1: 取四分类概率并计算 argmax
                gt_array = gt_prob_mat[var_idx, sample_idx, :]
                gt_index = np.argmax(gt_array)
            
                # Step 2: 映射到基因型字符串
                gt_str = GT_MAP.get(gt_index, "0|0")  # 默认值为 0|0
            
                # Step 3: 解析基因型字符串为单倍型值
                if "|" in gt_str:
                    hap1, hap2 = gt_str.split("|")
                else:
                    hap1, hap2 = gt_str.split("/") if "/" in gt_str else ("0", "0")
            
                # Step 4: 转换为 0/1 并二值化
                hap1 = 0 if hap1 in ("0", ".") else 1
                hap2 = 0 if hap2 in ("0", ".") else 1
            
                vcf_data[var_idx, sample_idx, 0] = hap1
                vcf_data[var_idx, sample_idx, 1] = hap2
    
        # 强制二值化（所有 >0 的值设为 1）
        vcf_data[vcf_data > 0] = 1
        return vcf_data

    @staticmethod
    def process_gt_prob_mat_with_progress(gt_prob_mat: np.ndarray) -> np.ndarray:
        """
        带进度条的版本（修复 GT_MAP 类型错误）
        """
        n_variants = gt_prob_mat.shape[0]
        vcf_data = np.zeros((*gt_prob_mat.shape[:2], 2), dtype=np.int8)
    
        # 预定义映射表 (4, 2)，直接使用整数元组
        GT_MAP = {
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1)
        }
        hap_table = np.array([GT_MAP[i] for i in range(4)], dtype=np.int8)
    
        # 逐变异位点处理并显示进度
        for var_idx in tqdm(range(n_variants), desc="Processing Variants"):
            # 当前变异位点的所有样本的 argmax
            gt_indices = np.argmax(gt_prob_mat[var_idx], axis=1)  # 形状 (samples,)
        
            # 映射到单倍型值
            haps = hap_table[gt_indices]
        
            # 处理无效索引
            invalid_mask = (gt_indices < 0) | (gt_indices >= 4)
            haps[invalid_mask] = (0, 0)
        
            # 填充数据
            vcf_data[var_idx] = haps
    
        # 强制二值化（所有 >0 的值设为 1）
        vcf_data[vcf_data > 0] = 1
        return vcf_data

    @staticmethod
    def generate_vcf(chr_id : str,
                     file_path : str,
                     output_path : str,
                     arr_hap1 : np.ndarray,
                     arr_hap2 : np.ndarray,
                     arr_gt : np.ndarray,
                     arr_pos : np.ndarray,
                     arr_pos_flag : np.ndarray) -> None:
        """Impute VCF file.
        """
        # 打开现有的 VCF 文件进行读取
        reader = vcfpy.Reader.from_path(file_path)

        samples = reader.header.samples.names

        # 创建新的 VCF 文件，基于输入文件的头部
        writer = vcfpy.Writer.from_path(output_path, reader.header)

        # 将原始 VCF 文件的所有记录写入到新文件中
        # for record in tqdm(reader):
        #     writer.write_record(record)

        for idx in tqdm(range(arr_pos.shape[0])):
            if arr_pos_flag[idx] == True:
                # 创建一条新的 VCF 记录
                new_record = vcfpy.Record(
                    CHROM = chr_id,             # 染色体名称
                    POS = arr_pos[idx],               # 变异位置 (1-based)
                    ID = ["."],       # 变异的 ID，可以是一个字符串或 `None`
                    REF = ".",                  # 参考等位基因
                    ALT = [vcfpy.Substitution(type_="SNV", value=".")],  # 替换的 ALT 等位基因
                    QUAL = 0.0,                # 质量值
                    FILTER = ["PASS"],          # 过滤标记，例如 "PASS"
                    INFO = {},         # INFO 字段，字典形式
                    FORMAT = ["GT", "HDS", "GP", "DS"],      # FORMAT 字段
                    calls = [
                        vcfpy.Call(
                            sample = samples[s_id],
                            data = {
                                "GT" : GT_MAP[np.argmax(arr_gt[idx, s_id, :])],
                                "HDS" : [f"{arr_hap1[idx, s_id]:.3f}", f"{arr_hap2[idx, s_id]:.3f}"],
                                "GP" : [f"{arr_gt[idx, s_id, 0]:.3f}", f"{arr_gt[idx, s_id, 1] + arr_gt[idx, s_id, 2]:.3f}", f"{arr_gt[idx, s_id, 3]:.3f}"],
                                "DS" : [f"{arr_gt[idx, s_id, 1] + arr_gt[idx, s_id, 2] + 2 * arr_gt[idx, s_id, 3]:.3f}"]
                            }
                        )
                        for s_id in range(len(samples))
                    ]                       # 样本基因型
                )

                # 写入新记录
                writer.write_record(new_record)
            else:
                continue

        # 关闭文件
        reader.close()
        writer.close()

    @staticmethod
    def parse_vcf_header_only(file_path: str) -> vcfpy.Header:
        """仅解析 VCF 文件头部，返回 header 对象"""
        reader = vcfpy.Reader.from_path(file_path)
        header = reader.header
        reader.close()
        return header

    @staticmethod
    def generate_vcf_efficient(
        chr_id: str,
        file_path: str,
        output_path: str,
        arr_hap1: np.ndarray,
        arr_hap2: np.ndarray,
        arr_gt: np.ndarray,
        arr_pos: np.ndarray,
        arr_pos_flag: np.ndarray,
        chunk_size: int = 50000
    ) -> None:
        """
        在保持原有函数签名的前提下，进行“分块写 VCF + 主动释放临时对象 + 强制 GC”。
        
        依旧需要注意：
        - 如果 arr_* 都是一次性加载到内存中的很大数组，则它们本身占用的内存无法被部分释放。
        - 本函数仅能避免在“写 VCF”阶段生成过多额外临时对象，导致进一步的内存上涨。
        """

        print("generate_vcf_efficient (keep original signature, chunk + GC)")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")

        # 1) 只解析 VCF header
        header = VCFProcessingModule.parse_vcf_header_only(file_path)
        samples = header.samples.names

        # 2) 创建 VCF Writer
        writer = vcfpy.Writer.from_path(output_path, header)

        n_variants = arr_pos.shape[0]
        n_samples = len(samples)

        # 3) 分块写出
        for start in tqdm(range(0, n_variants, chunk_size), desc="Writing VCF by chunk"):
            end = min(start + chunk_size, n_variants)

            # -- 3.1 切片: 仅取本 chunk 的数据 --
            pos_chunk      = arr_pos[start:end]
            flag_chunk     = arr_pos_flag[start:end]
            hap1_chunk     = arr_hap1[start:end]       # shape: [chunk_len, n_samples]
            hap2_chunk     = arr_hap2[start:end]
            gt_chunk       = arr_gt[start:end]         # shape: [chunk_len, n_samples, 4]

            # -- 3.2 遍历本 chunk 写出 VCF --
            chunk_len = end - start
            for idx_in_chunk in range(chunk_len):
                if not flag_chunk[idx_in_chunk]:
                    continue

                pos_val = int(pos_chunk[idx_in_chunk])

                calls = []
                for s_id in range(n_samples):
                    gt_array = gt_chunk[idx_in_chunk, s_id, :]  # shape (4,)
                    gt_index = np.argmax(gt_array)
                    gt_str   = GT_MAP.get(gt_index, "0|0")

                    hap1_val = hap1_chunk[idx_in_chunk, s_id]
                    hap2_val = hap2_chunk[idx_in_chunk, s_id]

                    gp0 = gt_array[0]
                    gp1 = gt_array[1] + gt_array[2]
                    gp2 = gt_array[3]
                    ds_val = gp1 + 2.0 * gp2

                    call_data = {
                        "GT": gt_str,
                        "HDS": [f"{hap1_val:.3f}", f"{hap2_val:.3f}"],
                        "GP": [f"{gp0:.3f}", f"{gp1:.3f}", f"{gp2:.3f}"],
                        "DS": [f"{ds_val:.3f}"]
                    }
                    calls.append(vcfpy.Call(sample=samples[s_id], data=call_data))

                new_record = vcfpy.Record(
                    CHROM=chr_id,
                    POS=pos_val,
                    ID=["."],
                    REF=".",
                    ALT=[vcfpy.Substitution(type_="SNV", value=".")],
                    QUAL=0.0,
                    FILTER=["PASS"],
                    INFO={},
                    FORMAT=["GT", "HDS", "GP", "DS"],
                    calls=calls
                )

                writer.write_record(new_record)

            # -- 3.3 处理完本 chunk，显式删除切片引用，主动 GC --
            del pos_chunk, flag_chunk, hap1_chunk, hap2_chunk, gt_chunk
            gc.collect()

        # 4) 关闭 Writer
        writer.close()
        print("Done. (generate_vcf_efficient)")

    @staticmethod
    def generate_vcf_efficient_optimized(
        chr_id: str,
        file_path: str,
        output_path: str,
        arr_hap1: np.ndarray,
        arr_hap2: np.ndarray,
        arr_gt: np.ndarray,
        arr_pos: np.ndarray,
        arr_pos_flag: np.ndarray,
        chunk_size: int = 100000
    ) -> None:

        print("generate_vcf_efficient_optimized (maximized performance)")

        header = VCFProcessingModule.parse_vcf_header_only(file_path)
        samples = header.samples.names
        writer = vcfpy.Writer.from_path(output_path, header)

        reader = vcfpy.Reader.from_path(file_path)
        record_count = 0
        for record in tqdm(reader, desc="Writing original records"):
            writer.write_record(record)
            record_count += 1
            if record_count % chunk_size == 0:
                gc.collect()  # 每 chunk_size 条记录后清理
        reader.close()

        ALT_SUB = [vcfpy.Substitution(type_="SNV", value=".")]
        FORMAT_FIELDS = ["GT", "HDS", "GP", "DS"]

        n_variants = arr_pos.shape[0]
        n_samples = len(samples)
        
        gt_map_array = np.array([GT_MAP[i] for i in range(len(GT_MAP))], dtype=object)

        for start in tqdm(range(0, n_variants, chunk_size), desc="Writing new records"):
            end = min(start + chunk_size, n_variants)
            chunk_len = end - start

            pos_chunk = arr_pos[start:end].astype(int)
            flag_chunk = arr_pos_flag[start:end]
            hap1_chunk = arr_hap1[start:end]
            hap2_chunk = arr_hap2[start:end]
            gt_chunk = arr_gt[start:end]

            gt_indices = np.argmax(gt_chunk, axis=2)
            
            gp0 = gt_chunk[:, :, 0]
            gp1 = gt_chunk[:, :, 1] + gt_chunk[:, :, 2]
            gp2 = gt_chunk[:, :, 3]
            ds = gp1 + 2 * gp2

            hds_hap1_str = np.char.mod('%.3f', hap1_chunk)  # shape: [chunk_len, n_samples]
            hds_hap2_str = np.char.mod('%.3f', hap2_chunk)
            gp0_str = np.char.mod('%.3f', gp0)
            gp1_str = np.char.mod('%.3f', gp1)
            gp2_str = np.char.mod('%.3f', gp2)
            ds_str = np.char.mod('%.3f', ds)

            for idx_in_chunk in range(chunk_len):
                if not flag_chunk[idx_in_chunk]:
                    continue

                pos_val = int(pos_chunk[idx_in_chunk])
                current_gt = gt_indices[idx_in_chunk]       # shape: [n_samples]
                current_hap1 = hds_hap1_str[idx_in_chunk]   # shape: [n_samples]
                current_hap2 = hds_hap2_str[idx_in_chunk]
                current_gp0 = gp0_str[idx_in_chunk]
                current_gp1 = gp1_str[idx_in_chunk]
                current_gp2 = gp2_str[idx_in_chunk]
                current_ds = ds_str[idx_in_chunk]

                calls = [
                    vcfpy.Call(
                        sample=samples[s_id],
                        data={
                            "GT": gt_map_array[current_gt[s_id]],
                            "HDS": [current_hap1[s_id], current_hap2[s_id]],
                            "GP": [current_gp0[s_id], current_gp1[s_id], current_gp2[s_id]],
                            "DS": [current_ds[s_id]]
                        }
                    )
                    for s_id in range(n_samples)
                ]

                writer.write_record(vcfpy.Record(
                    CHROM=chr_id,
                    POS=pos_val,
                    ID=["."],
                    REF=".",
                    ALT=ALT_SUB,
                    QUAL=0.0,
                    FILTER=["PASS"],
                    INFO={},
                    FORMAT=FORMAT_FIELDS,
                    calls=calls
                ))

            del pos_chunk, flag_chunk, hap1_chunk, hap2_chunk, gt_chunk
            del gt_indices, gp0, gp1, gp2, ds  

        writer.close()

