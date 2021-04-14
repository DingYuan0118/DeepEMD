import torch
from torch import nn, einsum
from torch.nn import parameter
import torch.nn.functional as F
from vit_pytorch import ViT

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class ViT_feature(ViT):
    def __init__(self, *, args, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__(image_size=image_size, patch_size=patch_size, num_classes=num_classes, dim=dim, depth=depth,
                         heads=heads, mlp_dim=mlp_dim, pool=pool, channels=channels, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)
        # 为计算EMD距离做准备
        self.args = args
        self.patch_size = patch_size
        self.image_size = image_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert image_size % patch_size == 0
        # 每行的patch数
        self.n_patch = self.image_size // self.patch_size

        # 使用命令行参数覆盖pool方法 ["mean", "cls"]
        self.pool = self.args.vit_mode
        if  self.args.not_use_clstoken:
            print("不使用cls token")
            # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        print("预测分类时使用{}模式".format(self.args.vit_mode))

            
    # 在forward pass中除去最后一层,不修改网络参数也许可以方便存储与读取
    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        if not self.args.not_use_clstoken:
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
        else:
            # 不用添加额外的cls token维度
            x += self.pos_embedding[:, :n]

        x = self.dropout(x)

        x = self.transformer(x)  # x输出形状为[batch_size, patch_size+1, dim] or [batch_size, patch_size, dim]

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # x = self.to_latent(x)
        return x  # 作为特征提取器最终输出为[batch_size, dim] eg [64, 1024]


if __name__ == "__main__":
    model = ViT(
        image_size=256,
        patch_size=32,
        num_classes=1000,
        dim=512,
        depth=4,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    img = torch.randn((1, 3, 256, 256))
    out = model(img)
    print(out.shape)
    print()
