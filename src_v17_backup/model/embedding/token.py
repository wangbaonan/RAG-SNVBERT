import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    """
    A Class equivalent to torch.nn.Embedding.
    Embedding for ID in .vcf file.

    Arguments:

        vocab_size: int.
        embed_size: int, 512 as default.
    """

    def __init__(self,
                 vocab_size: int,
                 embed_size: int = 512
                 ):
        
        super().__init__(vocab_size, embed_size, padding_idx=0)
