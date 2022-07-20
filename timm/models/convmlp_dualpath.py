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

from .efficientnet_blocks import *

_logger = logging.getLogger(__name__)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'convmlp_dp_b0': _cfg(
        url='',
        input_size=(3, 224, 224), crop_pct=0.875),
}

class Tokenizer(nn.Module):
    def __init__(self, img_size=224, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, embed_dim//2, kernel_size=3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
            )
        
    def forward(self, x):
        x = self.conv_layers(x)
        return x

class ConvHead(nn.Module):
    def __init__(self, num_blocks=1, embed_dim_in=64, embed_dim_out=128):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.Sequential(
                    nn.Conv2d(embed_dim_in, embed_dim_out, kernel_size=1,stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(embed_dim_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim_out, embed_dim_out, kernel_size=3,stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(embed_dim_out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(embed_dim_out, embed_dim_in, kernel_size=1,stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(embed_dim_in),
                    nn.ReLU(inplace=True)
                    )
            self.conv_blocks.append(block)
        self.downsample = nn.Conv2d(embed_dim_in, embed_dim_out, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x) + x
        x = self.downsample(x)
        return x

class EfficientConvHead(nn.Module):
    def __init__(self, num_blocks=1, embed_dim_in=64, embed_dim_out=128, headtype='ir'):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        for i in range(num_blocks):
            if headtype == 'ir':
                block = InvertedResidual(in_chs=embed_dim_in, out_chs=embed_dim_in, exp_ratio=2.0)
            elif headtype == 'cc':
                block = CondConvResidual(in_chs=embed_dim_in, out_chs=embed_dim_in, num_experts=4)
            self.conv_blocks.append(block)
        self.downsample = nn.Conv2d(embed_dim_in, embed_dim_out, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = self.downsample(x)
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
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        if self.shuffle and (x.shape[-1] % self.group == 0):
            batchsize, height, width, channel = x.shape
            channel_per_group = channel // self.group
            x = x.view(batchsize, height, width, self.group, channel_per_group)
            x = torch.transpose(x, 3, 4).contiguous()
            x = x.view(batchsize, height, width, -1)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MLPBlock(nn.Module):
    def __init__(self, seq_len, d_model, dim_feedforward=2048, dropout=0.1,
                 drop_path_rate=0.1, mlp_shuffle=False, depthconv=False):
        super(MLPBlock, self).__init__()
        self.mlp_shuffle = mlp_shuffle
        self.group = 4
        self.norm1 = nn.LayerNorm(d_model)
        self.channel_mlp1 = Mlp(d_model, dim_feedforward, shuffle=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.depthconv = depthconv
        if self.depthconv:
            self.connect = nn.Conv2d(d_model, d_model, kernel_size=3, stride=1, padding=1, groups=d_model, bias=False)
            self.connect_norm = nn.LayerNorm(d_model)
        self.channel_mlp2 = Mlp(d_model, dim_feedforward, shuffle=False)
        #pdb.set_trace()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, src):
        src = src + self.drop_path(self.channel_mlp1(self.norm1(src)))
        if self.mlp_shuffle:
            batchsize, height, width, channel = src.shape
            channel_per_group = channel // self.group
            src = src.view(batchsize, height, width, self.group, channel_per_group)
            src = torch.transpose(src, 3, 4).contiguous()
            src = src.view(batchsize, height, width, -1)
        if self.depthconv:
            src = self.connect(self.connect_norm(src).permute(0,3,1,2)).permute(0,2,3,1)
        src = src + self.drop_path(self.channel_mlp2(self.norm2(src)))
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
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        #x = x.reshape(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x

class ConvDownsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        x = x.permute(0, 2, 3, 1)
        return x

class BasicStage(nn.Module):
    def __init__(self, num_blocks, seq_len, embedding_dim=[192,384], mlp_ratio=1, dropout_rate=0.1,
                stochastic_depth=0.1, downsample=True, downsample_type='merge', mlp_shuffle=False, 
                depthconv=False, positional_embedding='learnable'):
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
                self.positional_emb = nn.Parameter(torch.zeros(1, seq_len, seq_len, embedding_dim[0]),
                                                requires_grad=True)
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = nn.Parameter(self.sinusoidal_embedding_2d(seq_len, seq_len, embedding_dim[0]),
                                                requires_grad=False)
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.positional_emb = None

        self.downsample = downsample
        
        if self.downsample:
            if downsample_type == 'merge':
                self.downsample_mlp = PatchMerging(embedding_dim[0], embedding_dim[1])
            else:
                self.downsample_mlp = ConvDownsample(embedding_dim[0], embedding_dim[1])

    def sinusoidal_embedding_2d(self, height, width, dim):
        assert dim % 4 == 0
        pe = torch.zeros(height,width,dim)
        d_model = dim // 2
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        ### a^b = e^ (log(a^b)) =  e^(b*log(a)) = e^(-2i/d * log(10000)) = e^(2i * -log(10000)/d)
        div_term = torch.exp(torch.arange(0., d_model, 2)* -(math.log(10000.0) / d_model))

        pe[:, :, 0:d_model:2] = torch.sin(pos_w * div_term).unsqueeze(0).repeat(height,1,1)
        pe[:, :, 1:d_model:2] = torch.cos(pos_w * div_term).unsqueeze(0).repeat(height,1,1)
        pe[:, :, d_model::2] = torch.sin(pos_h * div_term).unsqueeze(1).repeat(1,width,1)
        pe[:, :, d_model+1::2] = torch.cos(pos_h * div_term).unsqueeze(1).repeat(1,width,1)

        return pe.unsqueeze(0)

    def sinusoidal_embedding_1d(self, seq_len, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(seq_len)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)

    def forward(self, x):
        #pdb.set_trace()
        if self.positional_emb is not None:
            x += self.positional_emb
            x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample_mlp(x)
        return x

class CONVMLP(nn.Module):
    def __init__(self,
                 img_size=224,
                 downsample_rate=4,
                 mlp_ratio=[1,2],
                 conv_dim=64,
                 embedding_dim=[192,384],
                 num_stage=3,
                 num_blocks=[2,4,2],
                 num_classes=1000,
                 num_conv_block=2,
                 positional_embedding='none',
                 downsample_type='conv',
                 use_conv_blocks=False,
                 use_mlp_shuffle=False,
                 use_depthconv=False,
                 *args, **kwargs):
        super(CONVMLP, self).__init__()
        self.num_classes = num_classes
        self.use_conv_blocks = use_conv_blocks

        if self.use_conv_blocks:
            self.tokenizer = Tokenizer(img_size=img_size, embed_dim=conv_dim)
            self.conv_head = ConvHead(num_conv_block, embed_dim_in=conv_dim, embed_dim_out=embedding_dim[0])
        
        else:
            self.tokenizer = Tokenizer(img_size=img_size, embed_dim=embedding_dim[0])

        self.stages = nn.ModuleList()
        for i in range(0,num_stage):
            if i == (num_stage-1):
                downsample = False
            else:
                downsample = True

            stage = BasicStage(num_blocks=num_blocks[i],
                               seq_len=((img_size//downsample_rate)//(2**i)),
                               embedding_dim=embedding_dim[i:i+2],
                               mlp_ratio=mlp_ratio[i],
                               dropout_rate=0.1,
                               stochastic_depth=0.1,
                               downsample=downsample,
                               downsample_type=downsample_type,
                               mlp_shuffle=use_mlp_shuffle,
                               depthconv=use_depthconv,
                               positional_embedding=positional_embedding)
            self.stages.append(stage)
        
        self.norm = nn.LayerNorm(embedding_dim[-1])
        self.head = nn.Linear(embedding_dim[-1], num_classes)
        self.apply(self._init_weight)

    def forward(self, x):
        x = self.tokenizer(x)
        if self.use_conv_blocks:
            x = self.conv_head(x)
        x = x.permute(0, 2, 3, 1)
        for stage in self.stages:
            x = stage(x)
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
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

def _create_convmlp_dp(variant, pretrained=False, default_cfg=None, **kwargs):
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
        CONVMLP, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model

@register_model
def convmlp_dp_b0(pretrained=False, **kwargs):
    model_kwargs = dict(
        num_stage=3, num_blocks=[2,4,2], downsample_rate=8, mlp_ratio=[2,2,2], embedding_dim=[128,256,512], 
        use_conv_blocks=True, num_conv_block=2, downsample_type='conv', use_depthconv=True, **kwargs)
    return _create_convmlp_dp('convmlp_dp_b0', pretrained=pretrained, **model_kwargs)