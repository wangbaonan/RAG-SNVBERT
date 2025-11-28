import numpy as np
import random
import vcfpy
import os
import gc

from time import time
from typing import Literal
from tqdm import tqdm


MAX_SEQ_LEN = 980

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

        n_variants, n_samples, _ = gt_prob_mat.shape
        vcf_data = np.zeros((n_variants, n_samples, 2), dtype=np.int8)
    
    
        for var_idx in range(n_variants):
            for sample_idx in range(n_samples):
                
                gt_array = gt_prob_mat[var_idx, sample_idx, :]
                gt_index = np.argmax(gt_array)
            
                gt_str = GT_MAP.get(gt_index, "0|0")  # 默认值为 0|0
            
                if "|" in gt_str:
                    hap1, hap2 = gt_str.split("|")
                else:
                    hap1, hap2 = gt_str.split("/") if "/" in gt_str else ("0", "0")
            
                hap1 = 0 if hap1 in ("0", ".") else 1
                hap2 = 0 if hap2 in ("0", ".") else 1
            
                vcf_data[var_idx, sample_idx, 0] = hap1
                vcf_data[var_idx, sample_idx, 1] = hap2
    
        vcf_data[vcf_data > 0] = 1
        return vcf_data

    @staticmethod
    def process_gt_prob_mat_with_progress(gt_prob_mat: np.ndarray) -> np.ndarray:
        n_variants = gt_prob_mat.shape[0]
        vcf_data = np.zeros((*gt_prob_mat.shape[:2], 2), dtype=np.int8)
    
        GT_MAP = {
            0: (0, 0),
            1: (0, 1),
            2: (1, 0),
            3: (1, 1)
        }
        hap_table = np.array([GT_MAP[i] for i in range(4)], dtype=np.int8)
    
        for var_idx in tqdm(range(n_variants), desc="Processing Variants"):
            gt_indices = np.argmax(gt_prob_mat[var_idx], axis=1)  # 形状 (samples,)

            haps = hap_table[gt_indices]
        
            invalid_mask = (gt_indices < 0) | (gt_indices >= 4)
            haps[invalid_mask] = (0, 0)
        
            vcf_data[var_idx] = haps
    
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

        reader = vcfpy.Reader.from_path(file_path)

        samples = reader.header.samples.names


        writer = vcfpy.Writer.from_path(output_path, reader.header)


        for idx in tqdm(range(arr_pos.shape[0])):
            if arr_pos_flag[idx] == True:
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
                    ]                   
                )

                writer.write_record(new_record)
            else:
                continue


        reader.close()
        writer.close()

    @staticmethod
    def parse_vcf_header_only(file_path: str) -> vcfpy.Header:
        reader = vcfpy.Reader.from_path(file_path)
        header = reader.header
        reader.close()
        return header


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
                gc.collect()  
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

