import pickle
import numpy as np
import pandas as pd
import h5py
import pathlib
import gzip
import shutil
import allel
from typing import Dict, List
import logging
from src.dataset import PanelData
import os

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('freq_generation.log'),
        logging.StreamHandler()
    ]
)

# 全局路径配置
CONFIG = {
    "input": {
        "panel": "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/VCF/Train/train.980.sample.panel",
        "af": "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/Freq_chr21_0321/chr21_AF_V5.csv",
        "ref": "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/Freq_chr21_0321/chr21_GF_Ref_V5.csv",
        "het": "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/Freq_chr21_0321/chr21_GF_Het_V5.csv",
        "hom": "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/Freq_chr21_0321/chr21_GF_Hom_V5.csv",
        "vcf": "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/VCF/Train/KGP.chr21.Train.vcf.gz"
    },
    "output": {
        "freq_npy": "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/Freq/Freq.npy",
        "pop_index": "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/pop_to_idx.bin",
        "pos_index": "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/pos_to_idx.bin"
    }
}

class DataValidationError(Exception):
    """自定义数据验证异常"""
    pass

def validate_file(path: str) -> None:
    """验证文件是否存在且可读"""
    if not pathlib.Path(path).exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"文件不可读: {path}")
    logging.info(f"已验证文件可访问性: {path}")

def safe_read_csv(path: str, required_columns: List[str]) -> pd.DataFrame:
    """安全读取CSV文件并进行验证"""
    try:
        df = pd.read_csv(path)
        logging.info(f"成功读取CSV文件: {path} (行数: {len(df)})")
        
        # 验证必需列
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"CSV文件 {path} 缺少必需列: {missing_cols}")
        
        # 检查NaN值
        nan_counts = df.isna().sum()
        if nan_counts.any():
            logging.warning(f"CSV文件 {path} 包含NaN值:\n{nan_counts[nan_counts > 0]}")
            # 自动填充策略：用全局平均值填充NaN
            if 'Global' in df.columns:
                global_mean = df['Global'].mean()
                df.fillna(global_mean, inplace=True)
                logging.info("已用全局平均值填充NaN值")
        
        # 验证数值范围
        for col in required_columns:
            if col == 'POS':
                continue
            if (df[col] < 0).any() or (df[col] > 1).any():
                logging.warning(f"列 {col} 包含超出[0,1]范围的值")
                df[col] = df[col].clip(0, 1)  # 自动裁剪到合理范围
        
        return df
    except Exception as e:
        logging.error(f"读取CSV文件失败: {path} - {str(e)}")
        raise

def enhanced_vcf_to_h5(vcf_path: str) -> str:
    """增强型VCF转HDF5转换"""
    try:
        vcf_path = pathlib.Path(vcf_path)
        h5_path = vcf_path.with_suffix('.h5')
        
        if h5_path.exists():
            logging.info(f"发现已存在的HDF5文件: {h5_path}")
            return str(h5_path)
            
        if vcf_path.suffix == '.gz':
            logging.info("处理gzip压缩的VCF文件...")
            temp_vcf = vcf_path.with_suffix('')
            
            with gzip.open(vcf_path, 'rb') as f_in:
                with open(temp_vcf, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            allel.vcf_to_hdf5(str(temp_vcf), str(h5_path), fields='*', overwrite=True)
            temp_vcf.unlink()
        else:
            allel.vcf_to_hdf5(str(vcf_path), str(h5_path), fields='*', overwrite=True)
        
        logging.info(f"成功转换VCF到HDF5: {h5_path}")
        return str(h5_path)
    except Exception as e:
        logging.error(f"VCF转换失败: {str(e)}")
        raise

def build_frequency_matrix(
    ref_df: pd.DataFrame,
    het_df: pd.DataFrame,
    hom_df: pd.DataFrame,
    af_df: pd.DataFrame,
    pop_list: List[str],
    pos_count: int
) -> np.ndarray:
    """构建频率矩阵并进行验证"""
    try:
        # 初始化频率矩阵
        freq = np.zeros((4, len(pop_list)+1, pos_count), dtype=np.float32)
        logging.info(f"初始化频率矩阵完成 维度: {freq.shape}")
        
        # 填充各层数据
        layer_mapping = [
            (0, ref_df, "参考频率"),
            (1, het_df, "杂合率"), 
            (2, hom_df, "纯合率"),
            (3, af_df, "等位频率")
        ]
        
        for layer_idx, df, layer_name in layer_mapping:
            logging.info(f"正在填充 {layer_name} 层...")
            
            # 填充各群体数据
            for pop_idx, population in enumerate(pop_list):
                if population not in df.columns:
                    raise DataValidationError(f"缺失群体列: {population} 在 {layer_name} 层")
                
                values = df[population].values.astype(np.float32)
                freq[layer_idx, pop_idx] = values
            
            # 填充全局数据
            if 'Global' not in df.columns:
                raise DataValidationError(f"缺失Global列 在 {layer_name} 层")
            freq[layer_idx, -1] = df['Global'].values.astype(np.float32)
        
        # 最终验证
        if np.isnan(freq).any():
            nan_count = np.isnan(freq).sum()
            logging.error(f"频率矩阵包含NaN值 总数: {nan_count}")
            raise DataValidationError("检测到无效的NaN值")
            
        if (freq < 0).any() or (freq > 1).any():
            logging.warning("检测到超出[0,1]范围的值，自动裁剪...")
            freq = np.clip(freq, 0, 1)
        
        logging.info("频率矩阵构建和验证完成")
        return freq
    except Exception as e:
        logging.error("构建频率矩阵失败")
        raise

def main():
    """主处理流程"""
    try:
        logging.info("===== 开始频率矩阵生成流程 =====")
        
        # 验证所有输入文件
        for path in CONFIG['input'].values():
            validate_file(path)
        
        # 处理面板数据
        panel = PanelData.from_file(CONFIG['input']['panel'])
        pop_list = sorted(np.unique(panel.pop_list))
        logging.info(f"获取到 {len(pop_list)} 个唯一群体")
        
        # 生成群体索引映射
        pop_to_idx = {p: idx for idx, p in enumerate(pop_list)}
        pop_to_idx['Global'] = len(pop_list)
        with open(CONFIG['output']['pop_index'], 'wb') as f:
            pickle.dump(pop_to_idx, f)
        logging.info("群体索引映射已保存")
        
        # 转换VCF文件
        h5_path = enhanced_vcf_to_h5(CONFIG['input']['vcf'])
        
        # 获取位置信息
        with h5py.File(h5_path, 'r') as vcf_h5:
            positions = vcf_h5['variants/POS'][:]
            pos_count = len(positions)
            logging.info(f"获取到 {pos_count} 个变异位点")
        
        # 生成位置索引映射
        pos_to_idx = {int(pos): idx for idx, pos in enumerate(positions)}
        with open(CONFIG['output']['pos_index'], 'wb') as f:
            pickle.dump(pos_to_idx, f)
        logging.info("位置索引映射已保存")
        
        # 读取并验证所有CSV文件
        required_columns = pop_list + ['Global', 'POS']
        ref_df = safe_read_csv(CONFIG['input']['ref'], required_columns)
        het_df = safe_read_csv(CONFIG['input']['het'], required_columns)
        hom_df = safe_read_csv(CONFIG['input']['hom'], required_columns)
        af_df = safe_read_csv(CONFIG['input']['af'], required_columns)
        
        # 验证行数一致性
        for df, name in zip([ref_df, het_df, hom_df, af_df], ['REF', 'HET', 'HOM', 'AF']):
            if len(df) != pos_count:
                raise DataValidationError(
                    f"{name} CSV行数({len(df)})与VCF位点数({pos_count})不匹配"
                )
        
        # 构建频率矩阵
        freq_matrix = build_frequency_matrix(ref_df, het_df, hom_df, af_df, pop_list, pos_count)
        
        # 保存最终结果
        np.save(CONFIG['output']['freq_npy'], freq_matrix)
        logging.info(f"成功保存频率矩阵到 {CONFIG['output']['freq_npy']}")
        
        # 最终验证
        loaded = np.load(CONFIG['output']['freq_npy'])
        if not np.array_equal(freq_matrix, loaded):
            raise DataValidationError("保存的矩阵与内存矩阵不一致!")
        logging.info("最终验证通过")
        
    except Exception as e:
        logging.error(f"流程执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()
