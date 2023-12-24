# -*- coding: utf-8 -*-
#
# Developed by Liying Yang <lyyang69@gmail.com>
# Refer UMIFormer <https://github.com/GaryZhu1996/UMIFormer>
# Refer TCFormer <https://github.com/zengwang430521/TCFormer>

import torch
from einops import rearrange
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath, to_2tuple
from models.transformer_base.attention import LGAtention, STMAttention


class PatchEmbed(torch.nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else torch.nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# Long-Range Grouping block
class Block(torch.nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.,
                 act_layer=torch.nn.GELU,
                 norm_layer=torch.nn.LayerNorm,
                 patch_group=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = LGAtention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=proj_drop, patch_group=patch_group)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else torch.nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

    def forward(self, x, view_num=None, overlap=False):
        attn = self.attn(self.norm1(x), view_num=view_num, overlap=overlap)
        x = x + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# STM block
class STMBlock(torch.nn.Module):
    def __init__(self,
                 dim,
                 out_token_len,
                 k,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 act_layer=torch.nn.GELU,
                 norm_layer=torch.nn.LayerNorm):
        super(STMBlock, self).__init__()
        self.dim = dim
        self.norm1 = torch.nn.LayerNorm(dim)

        # cluster
        self.out_token_len = out_token_len
        self.k = k

        # merger
        self.score_mlp = torch.nn.Linear(dim, 1)

        # transformer block
        self.attention_layer = \
            STMAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=proj_drop)

    def _index_points(self, points, idx):
        """Sample features following the index.
            Returns:
                new_points:, indexed points data, [B, S, C]

            Args:
                points: input points data, [B, N, C]
                idx: sample index data, [B, S]
            """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def cluster(self, feature, batch_size):
        with torch.no_grad():
            distance_matrix = torch.cdist(feature, feature) / (self.dim ** 0.5)

            # get local density
            distance_nearest, index_nearest = \
                torch.topk(distance_matrix, k=self.k, dim=-1, largest=False)
            density = (-(distance_nearest ** 2).mean(dim=-1)).exp()

            # add a little noise to ensure no tokens have the same density.
            density = density + torch.rand(density.shape, device=density.device, dtype=density.dtype) * 1e-6

            # get distance indicator
            mask = density[:, None, :] > density[:, :, None]
            mask = mask.type(feature.dtype)
            dist_max = distance_matrix.flatten(1).max(dim=-1)[0][:, None, None]
            dist, index_parent = (distance_matrix * mask + dist_max * (1 - mask)).min(dim=-1)

            # select clustering center according to score
            score = dist * density
            _, index_down = torch.topk(score, k=self.out_token_len, dim=-1)  

            # assign tokens to the nearest center
            distance_matrix = self._index_points(distance_matrix, index_down)

            idx_cluster = distance_matrix.argmin(dim=1)

            # make sure cluster center merge to itself
            idx_batch = \
                torch.arange(batch_size, device=feature.device)[:, None].expand(batch_size, self.out_token_len)
            idx_tmp = \
                torch.arange(self.out_token_len, device=feature.device)[None, :].expand(batch_size, self.out_token_len)
            idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)

        return idx_cluster

    def merger(self, feature, idx_cluster, token_score, batch_size, patch_num):
        idx_batch = torch.arange(batch_size, device=feature.device)[:, None]  
        idx = idx_cluster + idx_batch * self.out_token_len  

        token_weight = token_score.exp()
        all_weight = token_weight.new_zeros(batch_size * self.out_token_len, 1)
        all_weight.index_add_(dim=0, index=idx.reshape(batch_size * patch_num),
                              source=token_weight.reshape(batch_size * patch_num, 1))
        all_weight = all_weight + 1e-6
        norm_weight = token_weight / all_weight[idx]

        # average token features
        merged_feature = feature.new_zeros(batch_size * self.out_token_len, self.dim)
        source = feature * norm_weight
        merged_feature.index_add_(dim=0, index=idx.reshape(batch_size * patch_num),
                                  source=source.reshape(batch_size * patch_num, self.dim).type(feature.dtype))
        merged_feature = merged_feature.reshape(batch_size, self.out_token_len, self.dim)

        return merged_feature

    def transformer_block(self, q_input, kv_input, token_score):
        attn = self.attention_layer(q_input, kv_input, token_score)
        feature = q_input + attn
        feature = feature + self.mlp(self.norm2(feature))
        return feature

    def forward(self, x):
        # feature [B, V*P, D]
        batch_size, patch_num, _ = x.shape

        x = self.norm1(x)
        token_score = self.score_mlp(x)  

        idx_cluster = self.cluster(x, batch_size) 
        q_input = self.merger(x, idx_cluster, token_score, batch_size, patch_num)
        kv_input = x

        feature = self.transformer_block(q_input, kv_input, token_score)  # [B, OUT_TOKEN_LEN, D]

        return feature




