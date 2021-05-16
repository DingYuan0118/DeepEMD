from math import sqrt
import torch.nn as nn
import torch
import torch.nn.functional as F
from vit_pytorch.vit import Transformer
from einops import rearrange
from Models.models.MyAttention import ResAttention
# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate,
                            training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, args, block=BasicBlock, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(
            block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(
            block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 320, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 640, stride=2, drop_rate=drop_rate, drop_block=True,
                                       block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.args = args

        if args.with_SA and not args.no_mlp:
            # 使用self attention机制
            self.transformer = Transformer(dim=640, depth=args.SA_depth, heads=args.SA_heads,
                                           dim_head=args.SA_dim_head, dropout=args.SA_dropout, mlp_dim=args.SA_mlp_dim)
        elif args.with_SA and args.no_mlp and args.SA_res:
            self.attention = ResAttention(dim=640, dim_head=args.SA_dim_head, dropout=args.SA_dropout, heads=args.SA_heads, use_res=True, pos_embed=args.pos_embed)
            self.SA_bn = nn.BatchNorm2d(640)
        
        elif args.with_SA and args.no_mlp and not args.SA_res:
            self.attention = ResAttention(dim=640, dim_head=args.SA_dim_head, dropout=args.SA_dropout, heads=args.SA_heads, use_res=False, pos_embed=args.pos_embed)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x) # [bs, 640, 5, 5]


        if self.args.with_SA and not self.args.no_mlp:
            # 需要reshape,使用einops库的rearrange
            x = rearrange(x, 'b dim rows cols -> b (rows cols) dim')
            x = self.transformer(x)
            # 需要reshape,使用einops库的rearrange
            x = rearrange(x, 'b (rows cols) dim -> b dim rows cols', cols=int(sqrt(x.shape[1])))
        
        elif self.args.with_SA and self.args.no_mlp:
            x = rearrange(x, 'b dim rows cols -> b (rows cols) dim')
            x = self.attention(x)
            # 需要reshape,使用einops库的rearrange
            x = rearrange(x, 'b (rows cols) dim -> b dim rows cols', cols=int(sqrt(x.shape[1])))
            x = self.SA_bn(x)

        return x


if __name__ == '__main__':
    args = None
    v = ResNet(args)
    input = torch.FloatTensor(5, 3, 80, 80)
    out = v(input)
    print(out.shape)