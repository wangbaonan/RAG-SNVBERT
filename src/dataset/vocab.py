import pickle
import json
import numpy as np

from collections import Counter

from .utils import timer


class TorchVocab(object):
    """
    Defines a vocabulary object that will be used to numericalize a field.

    Attributes:

        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
            
        itos: A list of token strings indexed by their numerical identifiers.

    """

    def __init__(self, 
                 counter : Counter,
                 max_size : int = None, 
                 min_freq : int = 1, 
                 specials : list[str] = ['<pad>', '<oov>']
                 ):
        """Create a Vocab object from a collections.Counter.

        Arguments:

            counter: collections.Counter object holding the frequencies of
                each value found in the data.

            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.

            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.

            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
        """
        # The parameter min_freq & max_size are included for potential future use and is not used in the current implementation.
        min_freq = max(min_freq, 1)

        # special tokens are initialized by default together with the vocabulary.
        self.itos = specials
        for tok in specials:
            del counter[tok]

        # max_size = None if max_size is None else max_size + len(self.itos)

        for word, freq in counter.items():
            # if freq < min_freq or len(self.itos) == max_size:
            #     break
            self.itos.append(word)

        self.stoi = {tok: i for i, tok in enumerate(self.itos)}


    def __eq__(self, other : 'TorchVocab') -> bool:
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True


    def __len__(self):
        return len(self.itos)


    def extend(self, others : 'TorchVocab') -> None:
        for w in others.itos:
            if w not in self.stoi:
                self.stoi[w] = len(self.itos)
                self.itos.append(w)



class Vocab(TorchVocab):
    def __init__(self, 
                 counter : Counter, 
                 max_size : int = None, 
                 min_freq : int = 1
                 ):
        
        self.pad_index = 0
        self.unk_index = 1
        self.sos_index = 2
        self.eos_index = 3
        self.mask_index = 4
        
        super().__init__(counter, specials=["<pad>", "<unk>", "<sos>", "<eos>", "<mask>"],
                         max_size=max_size, min_freq=min_freq)


    # def to_seq(self, sentece, seq_len, with_sos=False) -> list:
    #     pass


    # def from_seq(self, seq, join=False, with_pad=False):
    #     pass


    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


    def save_vocab(self, vocab_path) -> None:
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)



# Building Vocab with .vcf files or .h5 file
class WordVocab(Vocab):

    @timer
    def __init__(self,
                 pop_vocab : list[str],
                 max_size : int = None, 
                 min_freq : int = 1
                 ):
        """
        Build Vocab with population-info.

        The Vocabulary will contain [Special Token], [Haplotype], [POP]. 
        """

        # unphased Genotype
        # if using phased GT, please fix Function .utils.VCFProcessingModule.genotype_mapping()
        # counter_gt = Counter([0, 1, 2])

        # phased data
        counter_gt = Counter([0, 1])

        counter_pop = Counter(pop_vocab)

        # Merge all data
        merged_counter = Counter()

        merged_counter.update(counter_gt)
        merged_counter.update(counter_pop)

        super().__init__(merged_counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_sos=False, with_len=False):

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_sos:
            seq = [self.sos_index] + seq
            seq.append(self.eos_index)

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq.extend([self.pad_index for _ in range(seq_len - len(seq))])
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq
    

    # def from_seq(self, seq, join=False, with_pad=False):
    #     words = [self.itos[idx]
    #              if idx < len(self.itos)
    #              else "<%d>" % idx
    #              for idx in seq
    #              if not with_pad or idx != self.pad_index]

    #     return " ".join(words) if join else words


    def save_json(self,
                  save_path : str = 'data/example.json'
                  ) -> None:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.stoi, f, indent=4)


