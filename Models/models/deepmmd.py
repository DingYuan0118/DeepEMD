from os import truncate
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .emd_utils import *
from .mmd_utils import *
from .resnet import ResNet
from einops import rearrange


class DeepMMD(nn.Module):
    """
    使用MMD作为度量准则
    """
    def __init__(self, args, mode='meta'):
        super().__init__()

        self.mode = mode
        self.args = args

        # 为换backbone做准备,不改变原程序实现
        if self.args.model == 'resnet':
            self.encoder = ResNet(args=args)
        else:
            raise ValueError("没有{}模型".format(self.args.model))

        if self.mode == 'pre_train':
            self.fc = nn.Linear(640, self.args.num_class)

    def forward(self, input):
        # three modes. "meta" for meta-train, "pre_train" for pretrain
        if self.mode == 'meta':
            # support shape: [1, 5, 640, 5, 5], query shape: [75, 640, 5, 5]
            support, query = input
            return self.emd_forward_1shot(support, query)

        elif self.mode == 'pre_train':
            return self.pre_train_forward(input)

        elif self.mode == 'encoder':
            if self.args.deepemd == 'fcn':
                dense = True
            else:
                dense = False
            return self.encode(input, dense)
        else:
            raise ValueError('Unknown mode')

    def pre_train_forward(self, input):
        out = self.fc(self.encode(input, dense=False).squeeze(-1).squeeze(-1))
        return out

    # 实现MMD版本的emd_forward_1shot
    def emd_forward_1shot(self, proto, query):
        proto = proto.squeeze(0)
        # center normalize
        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)
        logits = self.get_mmd_distance(proto, query)
        return logits

    # TODO:考虑shot > 1时的微调, 在其SFC基础上更新, 因为更改的是emd_forward_1shot接口, 好像不用改动
    def get_sfc(self, support):
        support = support.squeeze(0)
        # init the proto
        SFC = support.view(self.args.shot, -1, 640,
                           support.shape[-2], support.shape[-1]).mean(dim=0).clone().detach()
        SFC = nn.Parameter(SFC.detach(), requires_grad=True)

        optimizer = torch.optim.SGD(
            [SFC], lr=self.args.sfc_lr, momentum=0.9, dampening=0.9, weight_decay=0)

        # crate label for finetune
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        label_shot = label_shot.type(torch.cuda.LongTensor)

        with torch.enable_grad():
            for k in range(0, self.args.sfc_update_step):
                rand_id = torch.randperm(self.args.way * self.args.shot).cuda()
                for j in range(0, self.args.way * self.args.shot, self.args.sfc_bs):
                    selected_id = rand_id[j: min(
                        j + self.args.sfc_bs, self.args.way * self.args.shot)]
                    batch_shot = support[selected_id, :]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.emd_forward_1shot(SFC, batch_shot.detach())
                    loss = F.cross_entropy(logits, batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC
    
    # 实现get_mmd_distance
    def get_mmd_distance(self, proto, query):
        """
        计算query与proto之间的MMD距离
            Args:
                proto(tensor):shape [n_way, dim, patch, patch] eg [5,640,5,5]
                query(tensor):shape [n_query, dim, patch, path] eg [75,640,5,5]
            
            Return:
                logits(tensor):shape [n_query, n_way]
        """
        num_query = query.shape[0]
        num_proto = proto.shape[0]
        proto = rearrange(proto, 'b dim rows cols -> b (rows cols) dim')
        query = rearrange(query, 'b dim rows cols -> b (rows cols) dim')
        # init score函数
        score = torch.randn(num_query, num_proto).cuda(proto.device) # shape eg.[75,5]
        for i in range(num_query):
            for j in range(num_proto):
                score[i][j] = mmd(query[i], proto[j])

        # 测试距离相似度转换使用的函数 1.使用y=-x(protonet) 2.使用y=1/x. 第二个函数引入非线性,可能会导致loss爆炸吗?
        # TODO:是否加入temperature
        logits = - score
        return logits


    def normalize_feature(self, x):
        if self.args.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
            return x

    def encode(self, x, dense=True):

        if x.shape.__len__() == 5:  # batch of image patches
            num_data, num_patch = x.shape[:2]
            x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
            x = self.encoder(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.reshape(num_data, num_patch,
                          x.shape[1], x.shape[2], x.shape[3])
            x = x.permute(0, 2, 1, 3, 4)
            x = x.squeeze(-1)
            return x

        else:
            if self.args.model == "resnet":
                x = self.encoder(x)  # [batchsize, 640, 5, 5]
                if dense == False:
                    x = F.adaptive_avg_pool2d(x, 1) # [batchsize, 640, 1, 1]
                    return x
                if self.args.feature_pyramid is not None:
                    x = self.build_feature_pyramid(x)

            # 使用ViT网络作为特征提取器,将矩阵变形后计算EMD距离
            elif self.args.model == "ViT":
                x = self.encoder(x) # [batchsize, patchsize+1, 512] or [batchsize, patchsize, 512]
                if dense == False:
                    # 在pretrain阶段使用
                    x = x.mean(dim = 1) if self.encoder.pool == 'mean' else x[:, 0]
                    x = self.encoder.to_latent(x) # [batchsize, 512]
                else:
                    # 在验证是计算EMD距离时使用
                    if self.args.vit_mode == "cls":
                        x = x[:,1:] #[batchsize, patchsize, 512]
                    bs, ps, dim = x.shape[0], x.shape[1], x.shape[2]
                    x = x.reshape((bs, dim, self.encoder.n_patch, self.encoder.n_patch)) #[batchsize, dim, patch, patch]
        return x

    def build_feature_pyramid(self, feature):
        feature_list = []
        for size in self.args.feature_pyramid:
            feature_list.append(F.adaptive_avg_pool2d(feature, size).view(
                feature.shape[0], feature.shape[1], 1, -1))
        feature_list.append(feature.view(
            feature.shape[0], feature.shape[1], 1, -1))
        out = torch.cat(feature_list, dim=-1)
        return out
