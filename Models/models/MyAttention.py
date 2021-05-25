from vit_pytorch.vit import Attention
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_

class ResAttention(Attention):
    '''
    实现自定义attention，并在其中选择是否加入残差连接
    Args:
        use_res(Bool) : 制定是否使用参差连接
    '''
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., use_res=True, pos_embed=False):
        super().__init__(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.use_res = use_res
        self.pos_embed = pos_embed
        if pos_embed:
            # 默认输入特征图尺度为Resnet提取过后的5*5
            self.window_size = (5, 5)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)


            trunc_normal_(self.relative_position_bias_table, std=.02)


    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if self.pos_embed:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots = dots + relative_position_bias.unsqueeze(0)
            
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # 添加残差连接
        if self.use_res:
            out = x + out
        return out

if __name__ == "__main__":
    attn = ResAttention(640, pos_embed=True)
    x = torch.randn((16, 25, 640))

    out = attn(x)
    print(out.shape)