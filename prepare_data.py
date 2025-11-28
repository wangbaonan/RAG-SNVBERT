import pickle
import numpy as np
import pandas as pd
import torch

import h5py
import tqdm

from src.dataset import PanelData


# global path of input file.
PANEL_PATH = "data/traindata_520_42_KGP_info_new.txt"

AF_PATH = "00_new_Freq_orig_20250318/Freq/chr21_AF_V5B.csv"

REF_PATH = "00_new_Freq_orig_20250318/Freq/chr21_GF_Ref_V5.csv"

HET_PATH = "00_new_Freq_orig_20250318/Freq/chr21_GF_Het_V5.csv"

HOM_PATH = "00_new_Freq_orig_20250318/Freq/chr21_GF_Hom_V5.csv"

VCF_PATH = "data/New_VCF/Train/KGP.chr21.Train.vcf.h5"


# global path of output file.
FREQ_OUTPUT = "00_new_Freq_orig_20250318/Freq/Freq.bin"


# Other global variants.
GENOTYPE_LIST = ['REF', 'HET', 'HOM']


def prepare_freq() -> None:
    """Generate a .bin file storing all the frequency data.
    """

    panel = PanelData.from_file(PANEL_PATH)
    file_list = [REF_PATH, HET_PATH, HOM_PATH]

    result = {}

    pos = h5py.File(VCF_PATH)['variants/POS'][:]

    # genotype frequency.
    for gt, file in zip(GENOTYPE_LIST, file_list):
        gt_dict = {}
        data = pd.read_csv(file)
        for pop in tqdm.tqdm(list(panel.pop_class_dict.keys())):
            pop_dict = {}
            for p, f in zip(pos, data[pop]):
                pop_dict[p] = f
            gt_dict[pop] = pop_dict
        result[gt] = gt_dict
        
    # allel frequency.
    af_dict = {}
    data = pd.read_csv(AF_PATH)

    population_list = list(panel.pop_class_dict.keys()) + ['Global']
    for pop in tqdm.tqdm(population_list):
        pop_dict = {}
        for p, f in zip(pos, data[pop]):
            pop_dict[p] = f
        af_dict[pop] = pop_dict
    result['AF'] = af_dict

    # write
    with open('data/Freq.bin', 'wb') as f:
        pickle.dump(result, f)


def prepare_train_data() -> None:
    """Generate a .bin file storing instance for Dataloader.
    """
    pass
    



if __name__ == "__main__":
    prepare_freq()

    pass