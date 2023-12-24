# -*- coding: utf-8 -*-
#
# Developed by Liying Yang <lyyang69@gmail.com>

import torch
from timm.models.vision_transformer import Mlp
from models.transformer_base.attention import Attention


# original decoder block
class Block(torch.nn.Module):
    def __init__(
            self, dim, num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            act_layer=torch.nn.GELU,
            norm_layer=torch.nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn1 = Attention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop)
        self.norm2 = norm_layer(dim)
        self.attn2 = Attention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm3 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)
        
    def forward(self, x, context):
        attn1 = self.attn1(x=self.norm1(x))
        x = x + attn1
        attn2 = self.attn2(x=self.norm2(x), context=context)
        x = x + attn2
        x = x + self.mlp(self.norm3(x))
        return x