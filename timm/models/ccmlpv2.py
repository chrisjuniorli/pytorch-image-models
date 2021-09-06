import os
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .layers import DropPath, trunc_normal_, to_2tuple
from .registry import register_model

import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv
import pdb

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'cycle_S': _cfg(crop_pct=0.9),
    'cycle_M': _cfg(crop_pct=0.9),
    'cycle_L': _cfg(crop_pct=0.875),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class BaseBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., seq_len=1, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(seq_len*seq_len)
        self.tokenMLP = Mlp(in_features=seq_len*seq_len, hidden_features=int(seq_len*seq_len*mlp_ratio), act_layer=act_layer)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.channelMLP = Mlp(in_features=dim, hidden_features=int(dim*4), act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.tokenMLP(self.norm1(x.transpose(1,2))).transpose(1,2))
        x = x + self.drop_path(self.channelMLP(self.norm2(x)))
        return x

class PatchEmbedOverlapping(nn.Module):
    """ 2D Image to Patch Embedding with overlapping
    """
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=None, groups=1):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size
        # remove image_size in model init to support dynamic image size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        return x

class ConvStem(nn.Module):
    def __init__(self, kernel_size=7, stride=2, padding=1, in_chans=3, embed_dim=768, norm_layer=None, groups=1):
        super().__init__()
        # remove image_size in model init to support dynamic image size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        return x

class Downsample(nn.Module):
    """ Downsample transition stage
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        B, N, C = x.shape
        H = int(math.sqrt(N))
        x = x.transpose(1,2).reshape(B,C,H,H)
        x = self.proj(x)  # B, C, H, W
        Bd, Cd, Hd, Wd = x.shape
        x = x.reshape(Bd,Cd,Hd*Wd).transpose(1,2)
        return x

def basic_blocks(dim, index, layers, mlp_ratio=3., seq_len=1, drop_path_rate=0., **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(BaseBlock(dim, mlp_ratio=mlp_ratio, seq_len=seq_len, drop_path=block_dpr))
    blocks = nn.Sequential(*blocks)

    return blocks


class CCMLPNet(nn.Module):
    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, seq_len=None, drop_rate=0., drop_path_rate=0.,
        norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbedOverlapping(patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=64)

        self.conv_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dims[0], kernel_size=1,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=1,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims[0], embed_dims[0], kernel_size=3,stride=2, padding=1, bias=False)
            )

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i], seq_len=seq_len[i], drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i+1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i+1], patch_size))

        self.network = nn.ModuleList(network)

        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x):
        #pdb.set_trace()
        x = self.forward_embeddings(x)
        x = self.conv_block(x)
        B, C, H, W = x.shape
        x = x.reshape(B,C,H*W).transpose(1,2)
        ## B,C,H,W -> B,N,C
        x = self.forward_tokens(x)
        x = self.norm(x)
        cls_out = self.head(x.mean(1))
        return cls_out

@register_model
def CCMLP_B1(pretrained=False, **kwargs):
    transitions = [True, True, True]
    layers = [2, 4, 2]
    mlp_ratios = [1, 2, 4]
    seq_len = [28, 14, 7]
    embed_dims = [128, 256, 512]
    model = CCMLPNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, seq_len=seq_len, **kwargs)
    model.default_cfg = default_cfgs['cycle_S']
    return model


@register_model
def CCMLP_B2(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 3, 10, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = CCMLPNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, **kwargs)
    model.default_cfg = default_cfgs['cycle_S']
    return model


@register_model
def CCMLP_B3(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [3, 4, 18, 3]
    mlp_ratios = [8, 8, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = CCMLPNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, **kwargs)
    model.default_cfg = default_cfgs['cycle_M']
    return model


@register_model
def CCMLP_B4(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [3, 8, 27, 3]
    mlp_ratios = [8, 8, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = CCMLPNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, **kwargs)
    model.default_cfg = default_cfgs['cycle_L']
    return model


@register_model
def CCMLP_B5(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [3, 4, 24, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [96, 192, 384, 768]
    model = CCMLPNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, **kwargs)
    model.default_cfg = default_cfgs['cycle_L']
    return model