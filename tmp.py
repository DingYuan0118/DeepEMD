import torch
from vit_pytorch import ViT

class ViT_feature(ViT):
    '''
    只取最终分类前的特征作为输出,讲ViT变为特征提取器
    '''
    def forward(self, img):
        return super().forward(img)



if __name__ == "__main__":
    model = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    img = torch.randn(1, 3, 256, 256)

    preds = model(img) # (1, 1000)