import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CrossAFInteraction(nn.Module):
    """跨层次AF交互模块（维度修正版）"""
    def __init__(self, dims):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, dims),
            nn.Sigmoid()
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(2, dims),
            nn.LayerNorm(dims),
            nn.GELU()
        )
        # 新增维度适配器
        self.af_adapter = nn.Linear(1, dims)
        self.res_scale = nn.Parameter(torch.tensor(0.1))
        self.dims = dims
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, global_af, pop_af):
        # 维度校验
        B, L = global_af.size()
        
        # 数值稳定性处理
        global_af = global_af.clamp(min=1e-4, max=0.9999)
        pop_af = pop_af.clamp(min=1e-4, max=0.9999)
        
        # 特征融合核心逻辑
        combined = torch.stack([global_af, pop_af], dim=-1)  # [B, L, 2]
        gate = self.gate_net(combined)  # [B, L, D]
        encoded = self.joint_encoder(combined)  # [B, L, D]
        
        # 维度修正：将global_af扩展为[B, L, D]
        global_af_expanded = self.af_adapter(global_af.unsqueeze(-1))  # [B, L, 1] -> [B, L, D]
        
        return global_af_expanded + self.res_scale * (gate * encoded)


class EnhancedRareVariantFusion(nn.Module):
    """维度修正版罕见变异融合模块"""
    def __init__(self, dims, ld_window=128):
        super().__init__()
        self.dims = dims
        self.af_interaction = CrossAFInteraction(dims)
        
        # AF特征适配器
        self.af_adapter = nn.Sequential(
            nn.Linear(dims, 4*dims),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4*dims, dims),
            nn.Sigmoid()
        )
        
        # 位置感知投影
        self.pos_proj = nn.Sequential(
            nn.Linear(dims, dims),
            nn.LayerNorm(dims),
            nn.GELU()
        )
        
        # LD增强的聚合层（维度对齐修正）
        self.ld_aggregator = nn.Sequential(
            nn.Linear(2*dims, dims),
            nn.GELU(),
            nn.LayerNorm(dims)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(2*dims, 4*dims),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4*dims, dims),
            nn.LayerNorm(dims, eps=1e-5)
        )
        
        # 稳定性参数
        self.maf_scale = nn.Parameter(torch.ones(1)*0.5)
        self.res_scale = nn.Parameter(torch.tensor(0.1))
        self.ld_window = ld_window
        
        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def _get_ld_mask(self, positions):
        """生成设备感知的LD衰减掩码（显存优化版）"""
        B, L = positions.size()
        device = positions.device
        
        # 优化后的位置矩阵计算
        pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)
        return torch.exp(-torch.abs(pos_diff) / self.ld_window).to(device)

    def forward(self, orig_feat, rag_feat, global_af, pop_af, snp_positions):
        # 输入维度校验
        B, K, L, D = rag_feat.size()
        assert D == self.dims, f"输入维度{D}与模型定义维度{self.dims}不匹配"
        
        # ====== AF特征融合 ======
        fused_af = self.af_interaction(global_af, pop_af)  # [B, L, D]
        af_weight = self.af_adapter(fused_af).unsqueeze(1)  # [B, 1, L, D]
        
        # ====== 位置增强参考特征 ======
        pos_emb = self.pos_proj(orig_feat).unsqueeze(1)  # [B, 1, L, D]
        weighted_ref = rag_feat * af_weight * pos_emb  # [B, K, L, D]
        
        # ====== LD引导的序列内聚合 ======
        ld_mask = self._get_ld_mask(snp_positions)  # [B, L, L]
        
        # 优化后的矩阵运算
        chunk_size = max(1, 512 // L)
        aggregated_ref = []
        for i in range(0, K, chunk_size):
            chunk = weighted_ref[:, i:i+chunk_size]  # [B, C, L, D]
            
            # 优化后的爱因斯坦求和表示
            ld_energy = torch.einsum('bclk,bclm->bckm', 
                                   chunk.permute(0,1,3,2), 
                                   chunk.permute(0,1,3,2))  # [B, C, L, L]
            ld_energy = ld_energy * ld_mask.unsqueeze(1)  # [B, C, L, L]
            
            ld_weights = F.softmax(ld_energy / math.sqrt(D), dim=-1)
            agg_chunk = torch.einsum('bckm,bcld->bckd', 
                                   ld_weights, 
                                   chunk.permute(0,1,3,2)).permute(0,1,3,2)
            aggregated_ref.append(agg_chunk)
        
        aggregated_ref = torch.cat(aggregated_ref, dim=1)  # [B, K, L, D]
        
        # ====== 维度对齐的跨参考序列聚合 ======
        orig_expanded = orig_feat.unsqueeze(1).expand(-1, K, -1, -1)  # [B, K, L, D]
        context_input = torch.cat([aggregated_ref, orig_expanded], dim=-1)  # [B, K, L, 2D]
        context_emb = self.ld_aggregator(context_input)  # [B, K, L, D]
        
        # 注意力池化（增加维度校验）
        attn_scores = torch.einsum('bkld,bld->bkl', context_emb, orig_feat) / math.sqrt(D)
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, K, L]
        pooled_ref = torch.einsum('bkl,bkld->bld', attn_weights, aggregated_ref)  # [B, L, D]
        
        # ====== 残差融合 ======
        fused = self.fusion(torch.cat([orig_feat, pooled_ref], dim=-1))  # [B, L, D]
        
        # ====== MAF稳定加权 ======
        maf = torch.min(global_af, 1 - global_af).unsqueeze(-1).clamp(min=1e-4)
        maf_weight = (self.maf_scale / maf).clamp(max=5.0)
        
        return orig_feat + self.res_scale * (fused * maf_weight)



class PositionFeatModule(nn.Module):
    """Process the feature of 'pos'.
    """
    def __init__(self, 
                 hidden_channels : int = 4,
                 kernel_size : int = 9,
                 stride : int = 1,
                 padding : int = 4,
                 *args, **kwargs) -> None:
        
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=hidden_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.act1 = nn.LeakyReLU(negative_slope=0.05)

        self.conv2 = nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels,
                               kernel_size=kernel_size, stride=stride, padding=padding)

        self.act2 = nn.LeakyReLU(negative_slope=0.05)

        self.conv3 = nn.Conv1d(in_channels=hidden_channels, out_channels=1,
                               kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.act3 = nn.LeakyReLU(negative_slope=0.05)
        
        self.norm1 = nn.BatchNorm1d(num_features=hidden_channels)

        self.norm2 = nn.BatchNorm1d(num_features=hidden_channels)

    
    def forward(self, pos : torch.Tensor):
        # pos.shape == (batch, seq_len)
        out = pos.unsqueeze(1)
        out = self.norm1(self.act1(self.conv1(out)))
        out = self.norm2(self.act2(self.conv2(out)))
        out = self.act3(self.conv3(out))
        return out.squeeze()



class EmbeddingFusionModule(nn.Module):
    """Process the features of 'hap', 'pos' & 'af'
    """

    def __init__(self, emb_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # pos-related.
        self.pos_feat = PositionFeatModule()

        self.fusion = nn.Linear(emb_size + 2, emb_size) # 将维度映射回 emb_size
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.norm = nn.LayerNorm(emb_size)


    def forward(self, emb : torch.Tensor, pos : torch.Tensor, af : torch.Tensor):
        """
        emb.shape == (batch, seq_len, emb_dim).
        
        pos.shape == (batch, seq_len). unsqueeze -> (batch, seq_len, 1)
        
        af.shape == (batch, seq_len), unsqueeze -> (batch, seq_len, 1)
        """

        # pos.
        pos_feat = self.pos_feat(pos)
        pos_feat = pos_feat.unsqueeze(-1)

        af_feat = af.unsqueeze(-1)

        all_feat = torch.cat((emb, pos_feat, af_feat), dim=-1)
        all_feat = self.act(self.fusion(all_feat))

        return self.norm(emb + all_feat)

