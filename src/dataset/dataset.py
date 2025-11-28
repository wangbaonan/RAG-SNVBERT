import pickle
import allel
import h5py
import numpy as np
import tqdm
import torch
import random
import pathlib
import json
import os
import pandas as pd
import math

from typing import Optional, Literal, Callable
from torch.utils.data import Dataset

from .vocab import WordVocab
from .utils import timer, PanelProcessingModule, VCFProcessingModule



# DEFAULT MAXIMUM SEQUENCE LENGTH
#MAX_SEQ_LEN = 980
#INFER_WINDOW_LEN = 970

INFER_WINDOW_LEN = 1020
MAX_SEQ_LEN = 1030

REF = 0
HET = 1
HOM = 2
AF  = 3

GLOBAL = 5



class PanelData():
    """
    This Class contains samples' [POP] label from .panel or .txt file.
    

    Attributes:

        pop_list: Origin list of [POP].
        >>> pop_list
        >>> ['EUR', 'ASA', 'AFR', 'EUR']

        pop_class_dict: Mapping of unique [POP].
        >>> pop_class_dict
        >>> {'EUR': 0, 'ASA': 1}

        pop_class_list: Map pop_list from 'str' to 'int'
        >>> pop_class_list
        >>> [ 0, 1, 2, 0 ]

        pop_index_dict: A dict instance with [POP] as keys and corresponding samples' indices as values.
        >>> pop_index_dict
        >>> {'EUR' : [0, 3], 'ASA' : [1], 'AFR' : [2]}

    """
    def __init__(self, pop_list):
        self.pop_list : np.ndarray[int] = np.array(pop_list)

        self.pop_class_dict : dict[str, int] = self.class_dict_from_list()
        # self.pop_class_list : np.ndarray[int] = self.class_list_from_list()
        # self.pop_index_dict : dict[str, list[int]] = self.index_dict_from_list()


    @classmethod
    def from_file(cls, fpath : str):
        """Extract info from [.panel] or [.txt] file.
        """
        pop_list = []

        with open(fpath, "r") as f:
            for line in f:
                # index 1 : POP
                # index 2 : SUPER POP
                #pop = line.strip().split(',')[1]    # for .txt file
                pop = line.strip().split()[2]       # for .panel file
                pop_list.append(pop)

        # omit the header
        return cls(pop_list[1:])
    

    def class_dict_from_list(self) -> dict[str, int]:
        """Map [POP] to token.
        """
        class_dict = {}
        pop_cls = np.unique(self.pop_list)
        for idx, pop in enumerate(pop_cls):
            class_dict[pop] = idx

        # check the order of the population
        if not os.path.exists('POP.json'):
            with open('POP.json', 'w') as f:
                json.dump(class_dict, f, indent=4)
                
        return class_dict
    

    # def class_list_from_list(self) -> list[int]:
    #     """self.pop_list in token form.
    #     """
    #     return np.array([self.pop_class_dict[pop] for pop in self.pop_list])


    # def index_dict_from_list(self) -> dict[str, list[int]]:
    #     """Store samples' indices according to the [POP].
    #     """
    #     index_dict = { pop: [] for pop in self.pop_class_dict.keys() }
    #     for idx, pop in enumerate(self.pop_list):
    #         index_dict[pop].append(idx)
    #     return index_dict



class Window():
    """This Class contains information about how to split the sequence.

    Attributes:

        window_info: The following code shows the 1st window starts at index 0 and ends at index 3, 
            the 2nd window starts at index 3 and ends at index 5 so-and-so.
        >>> cls.window_info
        >>> [[0, 3],
        >>>  [3, 5]]
        
    """

    def __init__(self, 
                 start_index : np.ndarray[int], 
                 end_index : np.ndarray[int]):
        window_info = np.stack((start_index, end_index), axis=1)

        # window_info.shape = (window_counts, 2), index 0 in dim_1 means start index, index 1 in dim_1 means end index.
        self.window_info : np.ndarray = window_info

    
    @classmethod
    def from_file(cls, fpath : str, limit : int = None):
        """Extract info from [.csv] file.

        Parameters:

            limit: The maximum number of windows.
        """
        window_path = pathlib.Path(fpath)
        assert window_path.suffix == '.csv', "Window File format should be [.csv]"

        df = pd.read_csv(window_path, usecols=[0, 1])

        # the start index of every window.
        array1 = df.iloc[:, 0].astype(int).to_numpy()
        # the end index.
        array2 = df.iloc[:, 1].astype(int).to_numpy()

        if limit is None:
            return cls(array1, array2)
        else:
            assert limit >= 1 and limit < array1.shape[0], "Illegal integer for Window's limitation."

            return cls(array1[:limit], array2[:limit])



class TrainDataset(Dataset): 
    """Building a dataset only for training.

    Attributes:

        vcf : VCFData Class Instance.

        id_vocab : WordVocab Class Instance.

        long_fields : Keys with LongType values in output of __getitem__.
        >>> self.long_fields
        >>> ['hap_1', 'hap_2', 'gt', 'pop', 'hap_1_label', 'hap_2_label', 'mask']

        float_fields : Keys with FloatType values in output of __getitem__.
        >>> float_fields
        >>> ['pos', 'af', 'af_p', 'ref', 'het', 'hom']

        mask_rate : (Private.) To control the difficulty of task.
        >>> self.__mask_rate
        >>> [0.15, 0.30, 0.50, 0.60, 0.70, 0.80]

        level : (Private.) Set difficulty according to self.mask_rate.
        >>> self.__level
        >>> 0

        mask_strategy : Map to Span-mask or Random-mask function.
        >>> self.mask_strategy
        >>> {
        >>>     0 : self.span_mask,
        >>>     1 : self.random_mask,
        >>> }

        mask_strategy_count : total number of mask strategies.
        >>> mask_strategy_count
        >>> 2

    """
    def __init__(self, 
                 vocab : WordVocab,
                 vcf : np.ndarray,
                 pos : np.ndarray,
                 panel : PanelData,
                 freq : np.ndarray,  # GT/'AF' -> POP -> POS -> Freq
                 window : Window,
                 type_to_idx : dict[str, int],
                 pop_to_idx : dict[str, int],
                 pos_to_idx : dict[int, int]
                 ):
        
        self.vocab = vocab

        self.vcf = vcf

        self.pos = pos

        self.panel : PanelData = panel

        # frequency data.
        self.freq : dict = freq

        self.window : Window = window

        self.type_to_idx = type_to_idx

        self.pop_to_idx = pop_to_idx

        self.pos_to_idx = pos_to_idx

        self.window_count : int = self.window.window_info.shape[0]

        # sample个数为样本个数*窗口数*两个hap?
        self.sample_count = self.vcf.shape[1] * self.window_count # 移除2

        # Keys should be transferred into LongType when sampling.
        self.long_fields = ['hap_1', 'hap_2', 'hap_1_label', 'hap_2_label', 'gt_label', 'mask','hap1_nomask','hap2_nomask']

        # Keys should be transferred into FloatType when sampling.
        self.float_fields = ['pos', 'af', 'af_p', 'ref', 'het', 'hom']

        self.__mask_rate : list[float] = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

        self.__level : int = 0

        # self.GT_NAME = ['REF', 'HET', 'HOM']

        self.mask_strategy : dict[int, Callable] = {
            0 : self.span_mask,
            1 : self.random_mask,
        }

        self.mask_strategy_count = len(list(self.mask_strategy.keys()))

        # self.data = self.generate_data()

    
    @classmethod
    @timer
    def from_file(cls, vocab, vcfpath, panelpath, freqpath, windowpath, typepath, poppath, pospath):
        filepath = pathlib.Path(vcfpath)
        panelpath = pathlib.Path(panelpath)
        freqpath = pathlib.Path(freqpath)
        windowpath = pathlib.Path(windowpath)

        # assert filepath.suffix in ['.vcf', '.h5'], "File format should be either [.vcf] or [.h5]."
        assert filepath.name.endswith(('.vcf', '.vcf.gz', '.h5')), \
            "File format should be [.vcf/.vcf.gz] or [.h5]"
        assert panelpath.suffix in ['.panel', '.txt'], "Panel format should be [.panel] or [.txt]"
        assert freqpath.suffix == '.npy', "MAF format should be [.npy]"
        assert windowpath.suffix == '.csv', "Window File format should be [.csv]"

        #if filepath.suffix == '.vcf':
        #    allel.vcf_to_hdf5(filepath, filepath.with_suffix('.h5'), fields='*', overwrite=True)
        #    exit("Program ends with [.vcf.gz] file transferred into [.h5] file.")

        # 处理VCF和VCF.GZ的情况
        if filepath.name.endswith(('.vcf', '.vcf.gz')):
            # 生成正确的输出文件名（自动替换为.h5）
            output_path = filepath.with_suffix('.h5')  # 直接替换所有后缀为.h5
    
            # 如果输入是.vcf.gz，需确保scikit-allel支持读取压缩文件（若报错则需手动解压）
            try:
                allel.vcf_to_hdf5(str(filepath), str(output_path), fields='*', overwrite=True)
                print(f"成功转换 {filepath.name} 为 {output_path.name}")
            except Exception as e:
                # 处理可能的压缩文件读取错误：先解压再转换
                if filepath.suffix == '.gz':
                    # 解压为临时.vcf文件
                    temp_vcf = filepath.with_suffix('')  # 去除.gz得到.vcf路径
                    with gzip.open(filepath, 'rb') as f_in:
                        with open(temp_vcf, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    # 转换临时.vcf文件为.h5
                    allel.vcf_to_hdf5(str(temp_vcf), str(output_path), fields='*', overwrite=True)
                    temp_vcf.unlink()  # 删除临时文件
                    print(f"通过解压转换 {filepath.name} 为 {output_path.name}")
                else:
                    raise e
    
            exit(f"程序结束：已生成 {output_path}")


        print("Loading VCF...")
        vcf_h5 = h5py.File(filepath, mode='r')
        vcf_data = vcf_h5['calldata/GT'][:]
        vcf_data[vcf_data > 0] = 1
        # vcf_data = vcf_h5['calldata/GT']
        pos = vcf_h5['variants/POS']

        panel = PanelData.from_file(panelpath)

        print("Loading Frequency...")
        if freqpath.exists():
            freq = np.load(freqpath)
        else:
            exit(f"MAF File not found. Current file path is {freqpath}.")

        print("Loading Window...")
        if windowpath.exists():
            window = Window.from_file(windowpath)
        else:
            exit(f"Window File not found. Current file path is {windowpath}.")

        print("Loading Mapping file...")

        with open(typepath, 'rb') as f:
            type_to_idx = pickle.load(f)

        with open(poppath, 'rb') as f:
            pop_to_idx = pickle.load(f)

        with open(pospath, 'rb') as f:
            pos_to_idx = pickle.load(f)

        return cls(vocab, vcf_data, pos, panel, freq, window, type_to_idx, pop_to_idx, pos_to_idx)


    def __len__(self):
        # sample_count * window_count * haplotype_count(2)
        return self.sample_count
    
    
    def add_level(self) -> None:
        """Raise mask rate to next level.
        """
        self.__level += 1
        
        # Fix self.__level if out of range.
        if self.__level == len(self.__mask_rate):
            print("\n",
                  "=" * 10,
                  "Maximum Mask Rate",
                  "=" * 10,
                  "\n")
            self.__level -= 1

    
    def generate_mask(self,
                      length : int,
                      mask_ratio : float = None) -> np.ndarray[int]:
        """Generate span mask or random mask. 
        
        Return a np.ndarray instance with 1 as mask and 0 as the opposite.
        """
        # Randomly pick a method.
        # strategy_idx = np.random.randint(0, self.mask_strategy_count)

        # Cancel the following comment if SET MASK RATIO.
        # if mask_ratio is None:
        #     mask_ratio = self.__mask_rate[self.__level]

        # return self.mask_strategy[strategy_idx](length, mask_ratio)
    
        # return self.mask_strategy[strategy_idx](length, self.__mask_rate[self.__level])
        return self.mask_strategy[1](length, self.__mask_rate[self.__level])

    
    def span_mask(self,
                  length : int,
                  mask_ratio : float = None) -> np.ndarray[int]:
        """Generate a Span-mask sequence.

        Span-mask means continuously masking some postions.
        """
        # Initialize a zero-array.
        mask = np.zeros((length, ), dtype=int)

        mask_length = int(length * mask_ratio)

        start_index = random.randint(0, length - mask_length)

        # choose mask position.
        for i in range(start_index, start_index + mask_length):
            mask[i] = 1

        return mask


    def random_mask(self,
                    length : int,
                    mask_ratio : float = None) -> np.ndarray[int]:
        """Generate a Random-mask sequence.

        Random-mask uses random.random() to generate probability for every position.
        
        If the probability is less than mask_ratio, then the respective position will be masked.
        """
        # Initialize a zero-array.
        mask = np.zeros((length, ), dtype=int)

        # Every position could be masked based on mask_ratio.
        for i in range(length):

            prob = random.random()

            if prob < mask_ratio:
                mask[i] = 1

        return mask


    def __getitem__(self, item) -> dict:
        """Return a dict that contains several information.

            - A masked haplotype sequence with [SOS] token at first.
            >>> output['hap_1']
            >>> output['hap_2']

            - The GroundTruth of masked haplotype sequence with [SOS] token at first and padded with MAX_SEQ_LEN as length.
            >>> output['hap_1_label']
            >>> output['hap_2_label']

            - The GroundTruth of masked genotype sequence with [SOS] token at first and padded with MAX_SEQ_LEN as length.
            >>> output['gt_label']

            - Sample's population.
            >>> output['pop']

            - A 0-1 sequence where 1 means "masked" and 0 means opposite.
            >>> output['mask']

            - A sequence consists of normalized ['variants/POS'].
            >>> output['pos']

            - Global allel frequency.
            >>> output['af']

            - Population allel frequency.
            >>> output['af_p']

            - Population genotype frequency.
            >>> output['ref']
            >>> output['het']
            >>> output['hom']

        """
        output = {}

        # Respective indices to query data.
        # sample_idx = item // (self.window_count * 2)
        # window_idx = (item % (self.window_count * 2)) >> 1
        sample_idx = item // self.window_count
        window_idx = item % self.window_count
        output['window_idx'] = window_idx
        # hap_idx = item & 0b1
        
        # Start position & End position of haplotype sequence.
        # 当前的窗口
        current_slice = slice(self.window.window_info[window_idx, 0], self.window.window_info[window_idx, 1])
        # output['current_slice'] = current_slice
        """ =========================== Long-Type =========================== """

        # Sample's population.
        pop = self.panel.pop_list[sample_idx]

        # Haplotype label. np.ndarray[int]
        hap_1 = self.vcf[current_slice, sample_idx, 0]
        # hap_1[hap_1 > 0] = 1
        hap_2 = self.vcf[current_slice, sample_idx, 1]
        # hap_2[hap_2 > 0] = 1
        # 未经过tokenize的就是原始的label 经过填充的自然就是0 后续也不会变化0
        output['hap_1_label'] = VCFProcessingModule.sequence_padding(hap_1, dtype='int')
        output['hap_2_label'] = VCFProcessingModule.sequence_padding(hap_2, dtype='int')

        # Genotype label.
        gt_label = (hap_1 << 1) + hap_2
        output['gt_label'] = VCFProcessingModule.sequence_padding(gt_label, dtype='int')

        # Generate a mask sequence.
        mask = self.generate_mask(gt_label.shape[0])
        mask = VCFProcessingModule.sequence_padding(mask, dtype='int')
        output['mask'] = mask

        # 新添加没有mask的原始hap_1,hap_2
        output['hap1_nomask'] = hap_1
        output['hap2_nomask'] = hap_2
        
        # Generate training sample.
        hap_1 = self.tokenize(hap_1, mask)
        hap_2 = self.tokenize(hap_2, mask)

        
        

        output['hap_1'] = hap_1
        output['hap_2'] = hap_2


        """ ===========================  Float-Type =========================== """
        # Position info.
        pos = self.pos[current_slice]
        _pos = VCFProcessingModule.position_normalize(pos)
        _pos = VCFProcessingModule.sequence_padding(_pos, dtype='float')
        output['pos'] = _pos

        # Global Allel frequency.
        pop_key = self.pop_to_idx[pop]
        f = np.array([self.freq[AF][GLOBAL][self.pos_to_idx[p]] for p in pos])
        output['af'] = VCFProcessingModule.sequence_padding(f, dtype='float')

        # Population Allel frequency.
        f = np.array([self.freq[AF][pop_key][self.pos_to_idx[p]] for p in pos])
        output['af_p'] = VCFProcessingModule.sequence_padding(f, dtype='float')

        # REF genotype frequency.
        f = np.array([self.freq[REF][pop_key][self.pos_to_idx[p]] for p in pos])
        output['ref'] = VCFProcessingModule.sequence_padding(f, dtype='float')

        # HET genotype frequency.
        f = np.array([self.freq[HET][pop_key][self.pos_to_idx[p]] for p in pos])
        output['het'] = VCFProcessingModule.sequence_padding(f, dtype='float')

        # HOM genotype frequency.
        f = np.array([self.freq[HOM][pop_key][self.pos_to_idx[p]] for p in pos])
        output['hom'] = VCFProcessingModule.sequence_padding(f, dtype='float')

        # return a dict instance.
        for key in self.long_fields:
            output[key] = torch.LongTensor(output[key])
        for key in self.float_fields:
            output[key] = torch.FloatTensor(output[key])
        return output


    def tokenize(self,
                 seq : np.ndarray[int],
                 mask : np.ndarray[int] = None) -> np.ndarray[int]:
        """Process sequence to get 'masked sequence' with a fixed 'mask'.
        """
        ret, _ = self.vocab.to_seq(seq, with_sos=True, with_len=True, seq_len=MAX_SEQ_LEN)
        ret = np.array(ret)

        ret[mask.astype(bool)] = self.vocab.mask_index

        return ret
    
    def tokenize(self, seq: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Vectorized version supporting both single and batch processing"""
        # 保存原始形状用于恢复维度
        original_shape = seq.shape
        seq = seq.reshape(-1, original_shape[-1])  # 展平为 (B, original_seq_len)
        
        # 批量处理
        processed = []
        for s in seq:
            # 将numpy数组转为list后处理
            seq_item, _ = self.vocab.to_seq(
                s.tolist(),
                seq_len=MAX_SEQ_LEN,
                with_sos=True,
                with_len=True
            )
            processed.append(seq_item)
            
        processed = np.array(processed)  # (B, MAX_SEQ_LEN)
        
        # 应用mask（支持广播机制）
        if mask is not None:
            # 统一mask维度：如果mask是1D则广播到所有样本
            if processed.ndim > mask.ndim:
                mask = np.expand_dims(mask, axis=0)
            processed = np.where(mask.astype(bool), self.vocab.mask_index, processed)
        
        # 恢复原始维度（最后一个维度变为MAX_SEQ_LEN）
        return processed.reshape(*original_shape[:-1], MAX_SEQ_LEN)

    

class InferDataset(Dataset): 
    """Building a dataset only for testing.

    Attributes:

        vcf : VCFData Class Instance.

        id_vocab : WordVocab Class Instance.

        long_fields : Keys with LongType values in output of __getitem__.
        >>> self.long_fields
        >>> ['hap_1', 'hap_2', 'gt', 'pop', 'hap_1_label', 'hap_2_label', 'mask']

        float_fields : Keys with FloatType values in output of __getitem__.
        >>> float_fields
        >>> ['pos', 'af', 'af_p', 'ref', 'het', 'hom']

        mask_rate : (Private.) To control the difficulty of task.
        >>> self.__mask_rate
        >>> [0.15, 0.30, 0.50, 0.60, 0.70, 0.80]

        level : (Private.) Set difficulty according to self.mask_rate.
        >>> self.__level
        >>> 0

        mask_strategy : Map to Span-mask or Random-mask function.
        >>> self.mask_strategy
        >>> {
        >>>     0 : self.span_mask,
        >>>     1 : self.random_mask,
        >>> }

        mask_strategy_count : total number of mask strategies.
        >>> mask_strategy_count
        >>> 2

    """
    def __init__(self, 
                 vocab : WordVocab,
                 vcf : np.ndarray,
                 pos : np.ndarray,
                 panel : PanelData,
                 freq : np.ndarray,  # GT/'AF' -> POP -> POS -> Freq
                 type_to_idx : dict[str, int],
                 pop_to_idx : dict[str, int],
                 pos_to_idx : dict[int, int]
                 ):
        
        self.vocab = vocab
        self.vcf = vcf
        self.pos = pos
        self.panel : PanelData = panel
        # frequency data.
        self.freq : dict = freq
        self.type_to_idx = type_to_idx
        self.pop_to_idx = pop_to_idx
        self.pos_to_idx = pos_to_idx
    

        # else data
        self.ori_pos = np.array(list(pos_to_idx.keys()))
        # 
        self.position_needed = ~np.isin(self.ori_pos, self.pos, assume_unique=True)
        self.test_pos_to_idx = {p : idx for idx, p in enumerate(self.pos)}

        # Keys should be transferred into LongType when sampling.
        self.long_fields = ['hap_1', 'hap_2', 'mask', 'sample_idx', 'start_idx', 'end_idx','hap1_nomask','hap2_nomask']
        # Keys should be transferred into FloatType when sampling.
        self.float_fields = ['pos', 'af', 'af_p', 'ref', 'het', 'hom']

        #self.window_count = self.vcf.shape[0] // INFER_WINDOW_LEN + 1
        self.window_count = math.ceil(self.ori_pos.shape[0] / INFER_WINDOW_LEN)

        self.sample_count = self.vcf.shape[1] * self.window_count

    
    @classmethod
    @timer
    def from_file(cls, vocab, vcfpath, panelpath, freqpath, typepath, poppath, pospath):
        filepath = pathlib.Path(vcfpath)
        panelpath = pathlib.Path(panelpath)
        freqpath = pathlib.Path(freqpath)

        #assert filepath.suffix in ['.vcf', '.h5'], "File format should be either [.vcf] or [.h5]."
        assert panelpath.suffix in ['.panel', '.txt'], "Panel format should be [.panel] or [.txt]"
        assert freqpath.suffix == '.npy', "MAF format should be [.npy]"

        # assert filepath.suffix in ['.vcf', '.h5'], "File format should be either [.vcf] or [.h5]."
        assert filepath.name.endswith(('.vcf', '.vcf.gz', '.h5')), \
            "File format should be [.vcf/.vcf.gz] or [.h5]"

        # 处理VCF和VCF.GZ的情况
        if filepath.name.endswith(('.vcf', '.vcf.gz')):
            # 生成正确的输出文件名（自动替换为.h5）
            output_path = filepath.with_suffix('.h5')  # 直接替换所有后缀为.h5
    
            # 如果输入是.vcf.gz，需确保scikit-allel支持读取压缩文件（若报错则需手动解压）
            try:
                allel.vcf_to_hdf5(str(filepath), str(output_path), fields='*', overwrite=True)
                print(f"成功转换 {filepath.name} 为 {output_path.name}")
            except Exception as e:
                # 处理可能的压缩文件读取错误：先解压再转换
                if filepath.suffix == '.gz':
                    # 解压为临时.vcf文件
                    temp_vcf = filepath.with_suffix('')  # 去除.gz得到.vcf路径
                    with gzip.open(filepath, 'rb') as f_in:
                        with open(temp_vcf, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    # 转换临时.vcf文件为.h5
                    allel.vcf_to_hdf5(str(temp_vcf), str(output_path), fields='*', overwrite=True)
                    temp_vcf.unlink()  # 删除临时文件
                    print(f"通过解压转换 {filepath.name} 为 {output_path.name}")
                else:
                    raise e
    
            #exit(f"程序结束：已生成 {output_path}")
            filepath = output_path

        print("Loading VCF...")
        vcf_h5 = h5py.File(filepath, mode='r')
        vcf_data = vcf_h5['calldata/GT'][:]
        vcf_data[vcf_data > 0] = 1
        # vcf_data = vcf_h5['calldata/GT']
        pos = vcf_h5['variants/POS'][:]

        panel = PanelData.from_file(panelpath)

        print("Loading Frequency...")
        if freqpath.exists():
            freq = np.load(freqpath)
        else:
            exit(f"MAF File not found. Current file path is {freqpath}.")

        print("Loading Mapping file...")

        with open(typepath, 'rb') as f:
            type_to_idx = pickle.load(f)

        with open(poppath, 'rb') as f:
            pop_to_idx = pickle.load(f)

        with open(pospath, 'rb') as f:
            pos_to_idx = pickle.load(f)

        return cls(vocab, vcf_data, pos, panel, freq, type_to_idx, pop_to_idx, pos_to_idx)


    def __len__(self) -> int:
        return self.sample_count
    

    def __getitem__(self, item) -> dict:
        """Return a dict that contains several information.

            - A masked haplotype sequence with [SOS] token at first.
            >>> output['hap_1']
            >>> output['hap_2']

            - The GroundTruth of masked haplotype sequence with [SOS] token at first and padded with MAX_SEQ_LEN as length.
            >>> output['hap_1_label']
            >>> output['hap_2_label']

            - The GroundTruth of masked genotype sequence with [SOS] token at first and padded with MAX_SEQ_LEN as length.
            >>> output['gt_label']

            - Sample's population.
            >>> output['pop']

            - A 0-1 sequence where 1 means "masked" and 0 means opposite.
            >>> output['mask']

            - A sequence consists of normalized ['variants/POS'].
            >>> output['pos']

            - Global allel frequency.
            >>> output['af']

            - Population allel frequency.
            >>> output['af_p']

            - Population genotype frequency.
            >>> output['ref']
            >>> output['het']
            >>> output['hom']

        """
        output = {}

        # Respective indices to query data.
        sample_idx = item // self.window_count
        window_idx = item % self.window_count
        # hap_idx = item & 0b1

        # Start position & End position of haplotype sequence.
        start_idx = INFER_WINDOW_LEN * window_idx
        end_idx = min(start_idx + INFER_WINDOW_LEN, self.ori_pos.shape[0])
        current_slice = slice(start_idx, end_idx)

        output['sample_idx'] = [sample_idx]
        output['start_idx'] = [start_idx]
        output['end_idx'] = [end_idx]

        """ =========================== Long-Type =========================== """

        # Sample's population.
        pop = self.panel.pop_list[sample_idx]

        # Haplotype label.
        hap_1 = []
        hap_2 = []
        mask = []
        for i in range(0, end_idx - start_idx):
            idx = start_idx + i
            if self.position_needed[idx]:
                hap_1.append(0)
                hap_2.append(0)
                mask.append(1)
            else:
                hap_1.append(self.vcf[self.test_pos_to_idx[self.ori_pos[idx]], sample_idx, 0])
                hap_2.append(self.vcf[self.test_pos_to_idx[self.ori_pos[idx]], sample_idx, 1])
                mask.append(0)

        hap_1, hap_2, mask = map(lambda x: np.array(x), [hap_1, hap_2, mask])

        mask = VCFProcessingModule.sequence_padding(mask, dtype='int')
        output['mask'] = mask

        # 新添加没有mask的原始hap_1,hap_2
        output['hap1_nomask'] = hap_1
        output['hap2_nomask'] = hap_2

        # Generate training sample.
        hap_1 = self.tokenize(hap_1, mask)
        hap_2 = self.tokenize(hap_2, mask)
        output['hap_1'] = hap_1
        output['hap_2'] = hap_2


        """ ===========================  Float-Type =========================== """
        # Position info.
        pos = self.ori_pos[current_slice]
        _pos = VCFProcessingModule.position_normalize(pos)
        _pos = VCFProcessingModule.sequence_padding(_pos, dtype='float')
        output['pos'] = _pos

        # Global Allel frequency.
        pop_key = self.pop_to_idx[pop]
        f = np.array([self.freq[AF][GLOBAL][self.pos_to_idx[p]] for p in pos])
        output['af'] = VCFProcessingModule.sequence_padding(f, dtype='float')

        # Population Allel frequency.
        f = np.array([self.freq[AF][pop_key][self.pos_to_idx[p]] for p in pos])
        output['af_p'] = VCFProcessingModule.sequence_padding(f, dtype='float')

        # REF genotype frequency.
        f = np.array([self.freq[REF][pop_key][self.pos_to_idx[p]] for p in pos])
        output['ref'] = VCFProcessingModule.sequence_padding(f, dtype='float')

        # HET genotype frequency.
        f = np.array([self.freq[HET][pop_key][self.pos_to_idx[p]] for p in pos])
        output['het'] = VCFProcessingModule.sequence_padding(f, dtype='float')

        # HOM genotype frequency.
        f = np.array([self.freq[HOM][pop_key][self.pos_to_idx[p]] for p in pos])
        output['hom'] = VCFProcessingModule.sequence_padding(f, dtype='float')

        # return a dict instance.
        for key in self.long_fields:
            output[key] = torch.LongTensor(output[key])
        for key in self.float_fields:
            output[key] = torch.FloatTensor(output[key])
        return output


    def tokenize(self, seq: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Vectorized version supporting both single and batch processing"""
        # 保存原始形状用于恢复维度
        original_shape = seq.shape
        seq = seq.reshape(-1, original_shape[-1])  # 展平为 (B, original_seq_len)
        
        # 批量处理
        processed = []
        for s in seq:
            # 将numpy数组转为list后处理
            seq_item, _ = self.vocab.to_seq(
                s.tolist(),
                seq_len=MAX_SEQ_LEN,
                with_sos=True,
                with_len=True
            )
            processed.append(seq_item)
            
        processed = np.array(processed)  # (B, MAX_SEQ_LEN)
        
        # 应用mask（支持广播机制）
        if mask is not None:
            # 统一mask维度：如果mask是1D则广播到所有样本
            if processed.ndim > mask.ndim:
                mask = np.expand_dims(mask, axis=0)
            processed = np.where(mask.astype(bool), self.vocab.mask_index, processed)
        
        # 恢复原始维度（最后一个维度变为MAX_SEQ_LEN）
        return processed.reshape(*original_shape[:-1], MAX_SEQ_LEN)
