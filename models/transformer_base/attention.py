# -*- coding: utf-8 -*-
#
# Developed by Liying Yang <lyyang69@gmail.com>

import torch
from einops import rearrange
from inspect import isfunction
import numpy as np


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

class LearnableSigmoid(torch.nn.Module):
    def __init__(self):
        super(LearnableSigmoid, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, 1, 1), requires_grad=True)

    def forward(self, input):
        return (1 + torch.exp(self.weight)) / (1 + torch.exp(self.weight - input))

# Default Attenton
class Attention(torch.nn.Module):
    def __init__(
            self, dim, num_heads,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            patch_group=7):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.group_num = patch_group
        self.qkv_bias = qkv_bias
        self.head_dim = head_dim

        self.to_q = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = torch.nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        
    def qkv_cal(self, q, k, v, mask=None):
        # [B, P, D]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        # [B, H, P, d]

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if mask is not None:
            dots = dots + mask
        attn = dots.softmax(dim=-1)  # [B, H, P_q, P_kv]
        attn = self.attn_drop(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # [B, H, P_q, d]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, P_q, D]
        return out

    def forward(self, x, context=None, mask=None):
        b, n, _ = x.shape
        kv_input = default(context, x)
        q_input = x

        q = self.to_q(q_input)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        # [B, P, D]

        out = self.qkv_cal(q, k, v, mask)
        # [B, P, D]

        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# Long-Range Grouping Layer
class LGAtention(Attention):
    def __init__(
            self, *args, **kwargs):
        super(LGAtention, self).__init__(*args, **kwargs)
        self.IFS = torch.nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim)

        if torch.distributed.get_rank() == 0:
            print('Attention type: Long-Range Grouping Attention')

    def InterViewSignature(self, x):
        B, H, N, C = x.shape
        hight = weight = round(np.power(N, float(1) / float(2)))
        x = rearrange(x, 'b h (h1 w1) c -> b (h c) h1 w1', h1=hight, w1=weight)

        ifs = self.IFS(x)
        ifs = rearrange(ifs, 'b (h c) h1 w1 -> b h (h1 w1) c', h=H, c=C)

        return ifs

    def forward(self, x, view_num=None, context=None, mask=None, lga=False):
        b, n, _ = x.shape  
        kv_input = default(context, x)
        q_input = x

        q = self.to_q(q_input)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        # [B, P, D]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        # [B, H, P, d]

        if lga is True:
            b, _, _, _ = q.shape
            b_s = int(b // view_num)
            token_num = 14 // self.group_num
            q_cls, k_cls, v_cls = map(lambda t: rearrange(t, '(b v) h (t1 s1 t2 s2) d -> b (s1 s2) h (t1 t2 v) d',
                                                          b=b_s, v=view_num, t1=token_num, s1=self.group_num, t2=token_num,
                                                          s2=self.group_num), (q, k, v))

            dots = torch.einsum('b n h i d, b n h j d -> b n h i j', q_cls, k_cls) * self.scale  
            if mask is not None:
                dots = dots + mask

            attn = dots.softmax(dim=-1)  # [B, H, P_q, P_kv]
            attn = self.attn_drop(attn)
            out = torch.einsum('b n h i j, b n h j d -> b n h i d', attn, v_cls)  # [B, H, P_q, d]

            out = rearrange(out, 'b (s1 s2) h (t1 t2 v) d -> (b v) h (t1 s1 t2 s2) d', v=view_num, t1=token_num,
                            s1=self.group_num, t2=token_num, s2=self.group_num)
            ifs = self.InterViewSignature(v)
            out = out + ifs
        else:
            dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            if mask is not None:
                dots = dots + mask
            attn = dots.softmax(dim=-1)  # [B, H, P_q, P_kv]
            attn = self.attn_drop(attn)
            out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # [B, H, P_q, d]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, P_q, D]

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# STM block
class STMAttention(Attention):
    def forward(self, q_input, kv_input, token_score):
        # q [B, P_q, D]; kv [B, P_kv, D]; token_score [B, P_kv, 1]
        b, n, _ = q_input.shape
        
        q = self.to_q(q_input)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        # [B, P, D]

        # qkv_cal
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        # [B, H, P, d]

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale # b,12(h),196,196*v
        token_score = token_score.squeeze(-1)[:, None, None, :]  # [B, 1, 1, P_kv] # b,1,1,196*v
        attn = (dots + token_score).softmax(dim=-1)  # [B, H, P_q, P_kv]
        attn = self.attn_drop(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # [B, H, P_q, d]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, P_q, D]

        out = self.proj(out)
        out = self.proj_drop(out)
        return out
