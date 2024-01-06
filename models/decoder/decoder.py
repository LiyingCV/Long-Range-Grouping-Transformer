# -*- coding: utf-8 -*-
#
# Developed by Liying Yang <lyyang69@gmail.com>

import torch
from einops import rearrange
from timm.models.vision_transformer import Mlp, partial
from models.transformer_base.decoder.standard_layers import Block as Blocks
from inspect import isfunction
import numpy as np


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.to_q = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x, context=None, mask=None):
        B, N, C = x.shape
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is None:
            attn = attn.softmax(dim=-1) 
        else:
            attn = self.softmax_with_policy(attn, mask) 

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


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
        self.attn = Attention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

    def forward(self, context, mask=None):
        context = context + self.attn(x=self.norm1(context), mask=mask)
        context = context + self.mlp(self.norm2(context))
        return context


class TransformerDecoder(torch.nn.Module):
    def __init__(
            self,
            patch_num=4 ** 3,
            embed_dim=768,
            num_heads=12,
            depth=8,
            mlp_ratio=4,
            qkv_bias=False, 
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=None):
        super().__init__()
        self.patch_num = patch_num
        self.input_side = round(np.power(patch_num, float(1) / float(3)))
        norm_layer = norm_layer or partial(torch.nn.LayerNorm)  # eps=1e-6  
        self.blocks1 = torch.nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                  proj_drop=proj_drop, norm_layer=norm_layer) for _ in range(depth // 4)])
        self.blocks2 = torch.nn.ModuleList([
            Block(dim=embed_dim // 8, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                  proj_drop=proj_drop, norm_layer=norm_layer) for _ in range(depth // 4)]) # 64

        self.blocks3 = torch.nn.ModuleList([
            Block(dim=embed_dim // 8, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                  proj_drop=proj_drop, norm_layer=norm_layer) for _ in range(depth // 4)]) # 96

        self.blocks4 = torch.nn.ModuleList([
            Block(dim=embed_dim // 4, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                  proj_drop=proj_drop, norm_layer=norm_layer) for _ in range(depth // 4)]) # 192

        self.upconv1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(768, 96, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(96),
            torch.nn.ReLU()
        )

        self.upconv2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(96, 12, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(12),
            torch.nn.ReLU()
        ) 

        self.upconv3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(12, 3, kernel_size=4, stride=2, bias=False, padding=1),
            torch.nn.BatchNorm3d(3),
            torch.nn.ReLU()
        ) 

        self.upconv4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(3, 1, kernel_size=1, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, context): 
        B = context.shape[0]
        side = self.input_side

        for blk in self.blocks1: # LR
            context = blk(context)

        context = rearrange(context, 'b (h w l) d -> b d h w l', h=side, w=side, l=side)  # [B, 768, 4, 4, 4]
        context = self.upconv1(context)  # [B, 96, 8, 8, 8]
        side *= 2
        context = rearrange(context, 'b d h w l -> b (h w l) d')  # [B, 512, 96]

        for blk in self.blocks2: # LR
            context = blk(context)

        context = rearrange(context, 'b (h w l) d -> b d h w l', h=side, w=side, l=side)  # [B, 96, 8, 8, 8]
        context = self.upconv2(context)  # [B, 12, 16, 16, 16]
        side *= 2
        context_temp = \
            rearrange(context, 'b d (h hs) (w ws) (l ls) -> b (h w l) (d hs ws ls)', hs=2, ws=2, ls=2)  # [B, 512, 96]

        for blk in self.blocks3: # HR
            context_temp = blk(context_temp)

        context_temp = rearrange(
            context_temp, 'b (h w l) (d hs ws ls) -> b d (h hs) (w ws) (l ls)', h=8, w=8, l=8, hs=2, ws=2, ls=2)
        context = context + context_temp  # [B, 12, 16, 16, 16]
        context = self.upconv3(context)  # [B, 3, 32, 32, 32]
        context_temp = rearrange(
            context, 'b d (h hs) (w ws) (l ls) -> b (h w l) (d hs ws ls)', hs=4, ws=4, ls=4)

        for blk in self.blocks4: # HR
            context_temp = blk(context_temp)

        context_temp = rearrange(
            context_temp, 'b (h w l) (d hs ws ls) -> b d (h hs) (w ws) (l ls)', h=8, w=8, l=8, hs=4, ws=4, ls=4)
        context = context + context_temp  # [B, 3, 32, 32, 32]
        out = self.upconv4(context)  # [B, 1, 32, 32, 32]
        return out


class Transformer(torch.nn.Module):
    def __init__(
            self,
            patch_num=4 ** 3,
            embed_dim=768,
            num_heads=12,
            depth=1,
            mlp_ratio=4,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_num = patch_num
        norm_layer = norm_layer or partial(torch.nn.LayerNorm)  # eps=1e-6  
        self.emb = torch.nn.Embedding(patch_num, embed_dim)
        self.blocks = torch.nn.ModuleList([
            Blocks(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                  proj_drop=proj_drop, norm_layer=norm_layer) for _ in range(depth)])

    def forward(self, context):
        x = self.emb(torch.arange(self.patch_num, device=context.device))
        x = x.unsqueeze(0).repeat(context.shape[0], 1, 1)
        for blk in self.blocks:
            x = blk(x=x, context=context)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()

        if cfg.NETWORK.DECODER.VOXEL_SIZE % 4 != 0:
            raise ValueError('voxel_size must be dividable by patch_num')
        if torch.distributed.get_rank() == 0:
            print('Decoder: Progressive Upsampling Transformer-Based')

        self.patch_num = 4 ** 3
        self.trans_patch_size = 4
        self.voxel_size = cfg.NETWORK.DECODER.VOXEL_SIZE
        self.patch_size = cfg.NETWORK.DECODER.VOXEL_SIZE // self.patch_num

        self.transformer_decoder = TransformerDecoder(
            embed_dim=cfg.NETWORK.DECODER.GROUP.DIM,
            num_heads=cfg.NETWORK.DECODER.GROUP.HEADS,
            depth=cfg.NETWORK.DECODER.GROUP.DEPTH,
            attn_drop=cfg.NETWORK.DECODER.GROUP.SOFTMAX_DROPOUT,
            proj_drop=cfg.NETWORK.DECODER.GROUP.ATTENTION_MLP_DROPOUT)

        self.prepare = Transformer()
        self.layer_norm = torch.nn.LayerNorm(cfg.NETWORK.DECODER.GROUP.DIM)

    def forward(self, context):
        # [B, P, D]
        context = self.prepare(context=context)
        context = self.layer_norm(context)
        out = self.transformer_decoder(context=context)
        return out


