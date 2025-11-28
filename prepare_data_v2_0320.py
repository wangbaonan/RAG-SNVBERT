import pickle
import numpy as np
import pandas as pd
import h5py
import pathlib
import gzip
import shutil
import allel  # 新增导入

from src.dataset import PanelData

# global path of input file.
PANEL_PATH = "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/VCF/Train/train.980.sample.panel"

AF_PATH = "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/Freq/chr21_AF_V5.csv"

REF_PATH = "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/Freq/chr21_GF_Ref_V5.csv"

HET_PATH = "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/Freq/chr21_GF_Het_V5.csv"

HOM_PATH = "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/Freq/chr21_GF_Hom_V5.csv"

VCF_PATH = "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/VCF/Train/KGP.chr21.Train.vcf.gz"

# global path of output file.
FREQ_OUTPUT = "/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/Freq/Freq.npy"

# Other global variants.
GENOTYPE_LIST = ['REF', 'HET', 'HOM']

def convert_vcf_to_h5(vcf_path):
    """将VCF/VCF.GZ转换为HDF5格式"""
    vcf_path = pathlib.Path(vcf_path)
    if vcf_path.name.endswith(('.vcf', '.vcf.gz')):
        output_path = vcf_path.with_suffix('.h5')
        
        try:
            # 尝试直接转换
            allel.vcf_to_hdf5(str(vcf_path), str(output_path), fields='*', overwrite=True)
            print(f"成功转换 {vcf_path.name} -> {output_path.name}")
        except Exception as e:
            # 处理压缩文件特殊情况
            if vcf_path.suffix == '.gz':
                print("检测到压缩文件读取问题，尝试解压转换...")
                temp_vcf = vcf_path.with_suffix('')  # 生成临时.vcf路径
                
                # 解压.gz文件
                with gzip.open(vcf_path, 'rb') as f_in:
                    with open(temp_vcf, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # 转换临时文件
                allel.vcf_to_hdf5(str(temp_vcf), str(output_path), fields='*', overwrite=True)
                temp_vcf.unlink()  # 删除临时文件
                print(f"通过解压完成转换 {vcf_path.name} -> {output_path.name}")
            else:
                raise RuntimeError(f"文件转换失败: {str(e)}")
        return output_path
    return vcf_path  # 已经是.h5文件时直接返回

def from_file(fpath: str):
    """Extract info from [.panel], [.txt] or [.csv] file."""
    pop_list : list[str] = []

    with open(fpath, "r") as f:
        for line in f:
            # index 1 : POP
            # index 2 : SUPER POP
            try:
                # Assuming .txt file format
                pop = line.strip().split(',')[2]
            except IndexError:
                # Fallback for .panel file format
                try:
                    pop = line.strip().split()[2]
                except IndexError:
                    raise ValueError(f"Invalid line format in panel file: {line.strip()}")
            
            pop_list.append(pop)

    # Omit the header
    return pop_list[1:]


def prepare_freq() -> None:
    panel = PanelData.from_file(PANEL_PATH)
    
    pop_labels = panel.pop_list  
    pop = np.unique(pop_labels)  
    pop = np.sort(pop)  

    pop_to_idx = {p: idx for idx, p in enumerate(pop)}
    pop_to_idx['Global'] = len(pop)
    
    with open('/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/pop_to_idx.bin', 'wb') as f:
        pickle.dump(pop_to_idx, f)

    h5_path = convert_vcf_to_h5(VCF_PATH)
    with h5py.File(h5_path, 'r') as vcf_h5:
        pos = vcf_h5['variants/POS'][:]
    
    pos_to_idx = {p: idx for idx, p in enumerate(pos)}
    with open('/cpfs01/projects-HDD/humPOG_HDD/wbn_24110700074/RAG_Version/VCF-Bert/00_Data_20250320/41_RAG-SNVBert_Data/pos_to_idx.bin', 'wb') as f:
        pickle.dump(pos_to_idx, f)

    freq = np.zeros((4, len(pop)+1, len(pos)), dtype=np.float32)
    
    for idx1, file in enumerate([REF_PATH, HET_PATH, HOM_PATH]):
        data = pd.read_csv(file)
        for idx2, p in enumerate(pop):
            freq[idx1, idx2] = data[p].values
        freq[idx1, -1] = data['Global'].values 
        
    af_data = pd.read_csv(AF_PATH)
    for idx, p in enumerate(pop):
        freq[3, idx] = af_data[p].values
    freq[3, -1] = af_data['Global'].values
    
    np.save(FREQ_OUTPUT, freq)

if __name__ == "__main__":
    prepare_freq()

    pass