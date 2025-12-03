import torch
import torch.nn as nn

from .utils import FeedForward
from .bert import BERT, BERTWithRAG


MAX_SCORE = 6


class BERTFoundationModel(nn.Module):
    """
    BERT Foundation Model
    """

    def __init__(self, bert: BERTWithRAG):

        super().__init__()
        self.bert = bert
        self.hap_classifier = HaplotypeClassifier(self.bert.dims, 2)
        self.gt_classifier = GenotypeClassifier(2, 4)

    def forward(self, x):
        # Haplotype.
        hap_1_after, hap_2_after, hap_1_before, hap_2_before = self.bert.forward(x)
        hap_1 = self.hap_classifier.forward(hap_1_after, x['af_p'])  # hap_1.shape == (batch, seq_len, emb_dim)
        hap_2 = self.hap_classifier.forward(hap_2_after, x['af_p'])

        gt = self.gt_classifier.forward(hap_1, hap_2, x['ref'], x['het'], x['hom'])

        return [hap_1, hap_2, gt, hap_1_before, hap_2_before, hap_1_after, hap_2_after]

    
class EnhancedHaplotypeClassifier(nn.Module):
    """增强版单体型分类器"""
    def __init__(self, dims, vocab_size=2):
        super().__init__()
        # AF融合层
        self.af_fusion = nn.Sequential(
            nn.Linear(dims + 2, 4*dims),
            nn.GELU(),
            nn.Linear(4*dims, dims),
            nn.LayerNorm(dims)
        )
        
        # 主网络
        self.net = nn.Sequential(
            nn.Linear(dims, 4*dims),
            nn.GELU(),
            nn.Linear(4*dims, vocab_size)
        )
        
        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.1)

    def forward(self, x, global_af, pop_af):
        """
        输入：
            x:        [B, L, D] 特征
            global_af: [B, L] 全局频率
            pop_af:    [B, L] 群体频率
        """
        # 拼接AF特征
        af_feat = torch.stack([global_af, pop_af], dim=-1)  # [B, L, 2]
        fused = torch.cat([x, af_feat], dim=-1)  # [B, L, D+2]
        
        # 特征融合
        fused = self.af_fusion(fused)
        
        # 分类输出
        logits = self.net(fused)
        return F.softmax(logits, dim=-1)


class HaplotypeClassifier(nn.Module):
    """Impute Haplotype.
    """

    def __init__(self, dims : int, vocab_size : int = 2):
        """
        Args:
            dims : output size of BERT model.
            vocab_size : for haplotype-imputation, it's 2.
        """
        super().__init__()
        # ======================= HaploType =========================== #

        self.af_fusion = nn.Linear(dims + 1, dims)
        self.af_act = nn.LeakyReLU(negative_slope=0.01)
        self.af_norm = nn.LayerNorm(dims)

        self.layer = FeedForward(dims, dims, dropout=0.1)

        self.classifier = nn.Linear(dims, vocab_size)
        
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x : torch.Tensor, af_p : torch.Tensor):
        """
        x.shape == (batch, seq_len, emb_dim)

        af_p.shape == (batch, seq_len)
        
        """
        # ======================= HaploType =========================== #

        af_p_feat = af_p.unsqueeze(-1)
        af_p_feat = torch.cat((x, af_p_feat), dim=-1)
        af_p_feat = self.af_norm(x + self.af_act(self.af_fusion(af_p_feat)))

        out = self.layer(af_p_feat)
        
        out = self.classifier(out)

        return self.softmax(out)



class GenotypeClassifier(nn.Module):
    """Impute Genotype.
    """

    def __init__(self, augment_factor : int = 2, vocab_size : int = 3):
        """
        Args:
            augment_factor : related to hidden dims.
            vocab_size : for genotype-imputation, it's 3.
        """
        super().__init__()

        # ======================= GenoType =========================== #
        self.hidden_dims = 4 ** augment_factor

        # input_dim = 2(hap_1) + 2(hap_2) + 3(genotype_frequency)
        self.gf_fusion = nn.Linear(7, self.hidden_dims)
        self.gf_act = nn.LeakyReLU(negative_slope=0.01)
        self.gf_norm = nn.LayerNorm(self.hidden_dims)

        self.layer = FeedForward(self.hidden_dims, self.hidden_dims, dropout=0.1)

        self.classifier = nn.Linear(self.hidden_dims, vocab_size)
        
        self.softmax = nn.Softmax(dim=-1)

    

    def forward(self,
                hap_1 : torch.Tensor,
                hap_2 : torch.Tensor,
                ref : torch.Tensor,
                het : torch.Tensor,
                hom : torch.Tensor):

        # ======================= GenoType =========================== #

        ref_feat = ref.unsqueeze(-1)
        het_feat = het.unsqueeze(-1)
        hom_feat = hom.unsqueeze(-1)

        gf_feat = torch.cat((hap_1, hap_2, ref_feat, het_feat, hom_feat), dim=-1)
        gf_feat = self.gf_norm(self.gf_act(self.gf_fusion(gf_feat)))

        out = self.layer(gf_feat)
        
        out = self.classifier(out)

        return self.softmax(out)

