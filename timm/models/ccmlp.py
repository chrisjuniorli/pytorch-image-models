import logging
import math
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg, overlay_external_default_cfg
from .layers import DropPath, trunc_normal_
from .registry import register_model
from .vision_transformer import checkpoint_filter_fn
import pdb 

_logger = logging.getLogger(__name__)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'ccmlp_s': _cfg(
        url='',
        input_size=(3, 224, 224), crop_pct=0.9)
}

class Tokenizer(nn.Module):
    def __init__(self, img_size=224, use_conv_blocks=False, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, embed_dim, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
            )
        self.use_conv_blocks = use_conv_blocks
        if self.use_conv_blocks:
            self.conv_blocks = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 128, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
            )
        
    def forward(self, x):
        x = self.conv_layers(x)
        if self.use_conv_blocks:
            x = self.conv_blocks(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shuffle=False, group=4):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.shuffle = shuffle
        self.group = group
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        #pdb.set_trace()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shuffle and (x.shape[1] % self.group == 0):
            ## x shape [b, hidden_features, hidden_features]
            batchsize, channel, N = x.shape
            channel_per_group = channel // self.group
            x = x.view(batchsize, self.group, channel_per_group, N)
            x = torch.transpose(x, 1, 2).contiguous()
            x = x.view(batchsize, -1, N)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MLPBlock(nn.Module):
    def __init__(self, seq_len, d_model, dim_feedforward=2048, dropout=0.1,
                 drop_path_rate=0.1, mlp_shuffle=False, depthconv=False):
        super(MLPBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mlp = Mlp(seq_len, d_model, shuffle=mlp_shuffle)
        self.norm2 = nn.LayerNorm(d_model)
        self.depthconv = depthconv
        if self.depthconv:
            self.connect = nn.Conv1d(d_model, d_model, kernel_size=3, stride=1, padding=1, groups=d_model, bias=False)
            self.connect_norm = nn.LayerNorm(d_model)
        self.channel_mlp = Mlp(d_model, dim_feedforward, shuffle=mlp_shuffle)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, src):
        ### src [B, N, C]
        src = src + self.drop_path(self.token_mlp(self.norm1(src).transpose(1, 2)).transpose(1, 2))
        if self.depthconv:
            src = self.connect(self.connect_norm(src).transpose(1, 2)).transpose(1, 2)
        src = src + self.drop_path(self.channel_mlp(self.norm2(src)))
        return src

class PatchMerging(nn.Module):
    """ 
    Adapted from Swin
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction = nn.Linear(in_dim*4, out_dim, bias=False)
        self.norm = nn.LayerNorm(in_dim*4)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        H, W = int(np.sqrt(L)), int(np.sqrt(L))
        assert C == self.in_dim
        x = x.reshape(B,H,W,C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicStage(nn.Module):
    def __init__(self, num_blocks, seq_len, embedding_dim=[192,384], mlp_ratio=1, dropout_rate=0.1,
                stochastic_depth=0.1, downsample=True, mlp_shuffle=False, depthconv=False, positional_embedding='learnable'):
        super().__init__()
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_blocks)]
        for i in range(num_blocks):
            block = MLPBlock(seq_len=seq_len,
                            d_model=embedding_dim[0],
                            dim_feedforward=int(embedding_dim[0] * mlp_ratio),
                            dropout=dropout_rate, drop_path_rate=dpr[i],
                            mlp_shuffle=mlp_shuffle, depthconv=depthconv
                            )
            self.blocks.append(block)

        self.positional_emb = positional_embedding
        if self.positional_emb != 'none':
            if self.positional_emb == 'learnable':
                self.positional_emb = nn.Parameter(torch.zeros(1, seq_len, embedding_dim[0]),
                                                requires_grad=True)
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = nn.Parameter(self.sinusoidal_embedding(seq_len, embedding_dim[0]),
                                                requires_grad=False)
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.positional_emb = None

        self.downsample = downsample
        if self.downsample:
            self.downsample_mlp = PatchMerging(embedding_dim[0], embedding_dim[1])
    
    def sinusoidal_embedding(self, n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)

    def forward(self, x):
        if self.positional_emb is not None:
            x += self.positional_emb
            x = self.dropout(x)
        
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample_mlp(x)
        return x

class CCMLP_STAGE(nn.Module):
    def __init__(self,
                 img_size=224,
                 downsample_rate=4,
                 mlp_ratio=[1,2],
                 embedding_dim=[192,384],
                 num_stage=2,
                 num_blocks=[2,4],
                 num_classes=1000,
                 positional_embedding='none',
                 use_conv_blocks=False,
                 use_mlp_shuffle=False,
                 use_depthconv=False,
                 *args, **kwargs):
        super(CCMLP_STAGE, self).__init__()
        self.num_classes = num_classes
        self.tokenizer = Tokenizer(img_size=img_size,
                                   use_conv_blocks=use_conv_blocks,
                                   embed_dim=64)
        self.stages = nn.ModuleList()
        for i in range(0,num_stage):
            if i == (num_stage-1):
                downsample = False
            else:
                downsample = True
            stage = BasicStage(num_blocks=num_blocks[i],
                               seq_len=((img_size//downsample_rate)//(2**i))**2,
                               embedding_dim=embedding_dim[i:i+2],
                               mlp_ratio=mlp_ratio[i],
                               dropout_rate=0.1,
                               stochastic_depth=0.1,
                               downsample=downsample,
                               mlp_shuffle=use_mlp_shuffle,
                               depthconv=use_depthconv,
                               positional_embedding=positional_embedding)
            self.stages.append(stage)
        self.norm = nn.LayerNorm(embedding_dim[-1])
        self.head = nn.Linear(embedding_dim[-1], num_classes)
        self.apply(self._init_weight)

    def forward(self, x):
        #pdb.set_trace()
        x = self.tokenizer(x)
        B,C,H,W = x.shape
        x = x.reshape(B,C,H*W).transpose(1,2)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x
    
    def _init_weight(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear, nn.Conv1d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

def _create_ccmlp(variant, pretrained=False, default_cfg=None, **kwargs):
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-1]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        CCMLP_STAGE, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model

@register_model
def ccmlp_s(pretrained=False, **kwargs):
    model_kwargs = dict(
        num_stage=3, num_blocks=[2,4,2], downsample_rate=8, mlp_ratio=[2,2,2], embedding_dim=[128,256,512], use_conv_blocks=True, **kwargs)
    return _create_ccmlp('ccmlp_s', pretrained=pretrained, **model_kwargs)

@register_model
def ccmlp_s_pe(pretrained=False, **kwargs):
    model_kwargs = dict(
        num_stage=3, num_blocks=[2,4,2], downsample_rate=8, mlp_ratio=[2,2,2], embedding_dim=[128,256,512], use_conv_blocks=True, positional_embedding='learnable', **kwargs)
    return _create_ccmlp('ccmlp_s', pretrained=pretrained, **model_kwargs)

@register_model
def ccmlp_s_pe_shuffle(pretrained=False, **kwargs):
    model_kwargs = dict(
        num_stage=3, num_blocks=[2,4,2], downsample_rate=8, mlp_ratio=[2,2,2], embedding_dim=[128,256,512], 
        use_conv_blocks=True, use_mlp_shuffle=True, positional_embedding='learnable', **kwargs)
    return _create_ccmlp('ccmlp_s', pretrained=pretrained, **model_kwargs)

@register_model
def ccmlp_s_pe_shuffle_depthconv(pretrained=False, **kwargs):
    model_kwargs = dict(
        num_stage=3, num_blocks=[2,4,2], downsample_rate=8, mlp_ratio=[2,2,2], embedding_dim=[128,256,512], 
        use_conv_blocks=True, use_mlp_shuffle=True, use_depthconv=True, positional_embedding='learnable', **kwargs)
    return _create_ccmlp('ccmlp_s', pretrained=pretrained, **model_kwargs)

@register_model
def ccmlp_s_baseline(pretrained=False, **kwargs):
    model_kwargs = dict(
        num_stage=4, num_blocks=[2,2,4,2], downsample_rate=4, mlp_ratio=[4,4,4,4], embedding_dim=[64,128,256,512], use_conv_blocks=False, **kwargs)
    return _create_ccmlp('ccmlp_s', pretrained=pretrained, **model_kwargs)

@register_model
def ccmlp_s_192(pretrained=False, **kwargs):
    model_kwargs = dict(
        num_stage=4, num_blocks=[2,2,4,2], downsample_rate=4, mlp_ratio=[2,2,4,4], embedding_dim=[192,192,384,384], **kwargs)
    return _create_ccmlp('ccmlp_s_192', pretrained=pretrained, **model_kwargs)