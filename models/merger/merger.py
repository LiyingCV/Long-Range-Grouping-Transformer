# -*- coding: utf-8 -*-
#
# Refer UMIFormer <https://github.com/GaryZhu1996/UMIFormer>


import torch
from einops import rearrange

from models.transformer_base.encoder.transformer_block import STMBlock


class Merger(torch.nn.Module):
    def __init__(self, cfg):
        super(Merger, self).__init__()
        
        if torch.distributed.get_rank() == 0:
            print('Merger: Similar Token Merger (STM)')
        
        self.blocks = torch.nn.ModuleList(
            [STMBlock(dim=cfg.NETWORK.MERGER.STM.DIM, out_token_len=token_len,
                     k=cfg.NETWORK.MERGER.STM.K, num_heads=cfg.NETWORK.MERGER.STM.NUM_HEAD)
             for token_len in cfg.NETWORK.MERGER.STM.OUT_TOKEN_LENS])

    def forward(self, feature):
        # feature [B, V, P, D]
        feature = rearrange(feature, 'b v l d -> b (v l) d')  # [B, V*P, D]
        
        for blk in self.blocks:
            feature = blk(feature)

        return feature
