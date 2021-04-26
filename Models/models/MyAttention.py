from vit_pytorch.vit import Attention
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ResAttention(Attention):
    '''
    实现自定义attention，并在其中选择是否加入残差连接
    Args:
        use_res(Bool) : 制定是否使用参差连接
    '''
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., use_res=True):
        super().__init__(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.use_res = use_res

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        # 添加残差连接
        if self.use_res:
            out = x + out
        return out

if __name__ == "__main__":
    attn = ResAttention(640)
    x = torch.randn((16, 25, 640))

    out = attn(x)
    print(out.shape)