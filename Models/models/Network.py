from os import truncate
import torch
import torch.nn as nn
import torch.nn.functional as F
from .emd_utils import *
from .resnet import ResNet
from .vit_feature import ViT_feature
import timm


class DeepEMD(nn.Module):

    def __init__(self, args, mode='meta'):
        super().__init__()

        self.mode = mode
        self.args = args

        # 为换backbone做准备,不改变原程序实现
        if self.args.model == 'resnet':
            self.encoder = ResNet(args=args)
        elif self.args.model == 'ViT':
            self.encoder = ViT_feature(args=args, image_size=256,
                                       patch_size=32,
                                       num_classes=5,
                                       dim=512,
                                       depth=self.args.vit_depth,
                                       heads=16,
                                       mlp_dim=1024,
                                       dropout=0.1,
                                       emb_dropout=0.1)
        elif self.args.model == "vit_small_patch16_224":
            if self.args.not_imagenet_pretrain:
                self.encoder = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=self.args.num_class)
            else:
                self.encoder = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=self.args.num_class)
        
        else:
            raise ValueError("没有{}模型".format(self.args.model))

        if self.mode == 'pre_train':
            if self.args.model == 'resnet':
                self.fc = nn.Linear(640, self.args.num_class)
            elif self.args.model == 'ViT':
                self.mlp_head  = self.encoder.mlp_head # 引用传递,同时改变
                self.mlp_head[1] = nn.Linear(512, self.args.num_class)
            elif self.args.model == 'vit_small_patch16_224':
                # TODO
                raise Exception("代码未完善")

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
        if self.args.model == "resnet":
            out = self.fc(self.encode(input, dense=False).squeeze(-1).squeeze(-1))

        elif self.args.model == "ViT":
            out = self.mlp_head(self.encode(input, dense=False))
        
        elif self.args.model == "vit_small_patch16_224":
            out = self.encoder(input)

        return out

    def get_weight_vector(self, A, B):

        M = A.shape[0]  # 75
        N = B.shape[0]  # 5

        B = F.adaptive_avg_pool2d(B, [1, 1])
        B = B.repeat(1, 1, A.shape[2], A.shape[3])

        A = A.unsqueeze(1)  # [75, 1, 640, 5, 5]
        B = B.unsqueeze(0)  # [1, 5, 640, 5, 5]

        A = A.repeat(1, N, 1, 1, 1)  # [75, 5, 640, 5, 5]
        B = B.repeat(M, 1, 1, 1, 1)  # [75, 5, 640, 5, 5]

        combination = (A * B).sum(2)  # [75, 5, 5, 5]
        combination = combination.view(M, N, -1)
        combination = F.relu(combination) + 1e-3
        return combination

    def emd_forward_1shot(self, proto, query):
        proto = proto.squeeze(0)  # [5, 640, 5, 5]

        weight_1 = self.get_weight_vector(query, proto)  # [75, 5, 25]
        weight_2 = self.get_weight_vector(proto, query)  # [5, 75, 25]
        # center normalize
        proto = self.normalize_feature(proto)
        query = self.normalize_feature(query)

        similarity_map = self.get_similiarity_map(proto, query)
        if self.args.solver == 'opencv' or (not self.training):
            logits = self.get_emd_distance(
                similarity_map, weight_1, weight_2, solver='opencv')
        else:
            logits = self.get_emd_distance(
                similarity_map, weight_1, weight_2, solver='qpth')
        return logits

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

    def get_emd_distance(self, similarity_map, weight_1, weight_2, solver='opencv'):
        # params: similarity_map(tensor) : shape[75, 5, 25, 25]
        #         weight_1(tensor) : shape[75, 5, 25]
        #         weight_2(tensor) : shape[5, 75, 25]
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node = weight_1.shape[-1]

        if solver == 'opencv':  # use openCV solver

            for i in range(num_query):
                for j in range(num_proto):
                    _, flow = emd_inference_opencv(
                        1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])
                    similarity_map[i, j, :, :] = (
                        similarity_map[i, j, :, :])*torch.from_numpy(flow).cuda()

            temperature = (self.args.temperature/num_node)
            logitis = similarity_map.sum(-1).sum(-1) * temperature
            return logitis

        elif solver == 'qpth':
            weight_2 = weight_2.permute(1, 0, 2)  # [75, 5, 25]
            similarity_map = similarity_map.view(num_query * num_proto, similarity_map.shape[-2],
                                                 similarity_map.shape[-1])  # [375, 25, 25]
            weight_1 = weight_1.view(
                num_query * num_proto, weight_1.shape[-1])  # [375, 25]
            weight_2 = weight_2.reshape(
                num_query * num_proto, weight_2.shape[-1])  # [375, 25]

            _, flows = emd_inference_qpth(
                1 - similarity_map, weight_1, weight_2, form=self.args.form, l2_strength=self.args.l2_strength)

            logitis = (flows*similarity_map).view(num_query,
                                                  num_proto, flows.shape[-2], flows.shape[-1])
            temperature = (self.args.temperature / num_node)
            logitis = logitis.sum(-1).sum(-1) * temperature
        else:
            raise ValueError('Unknown Solver')

        return logitis

    def normalize_feature(self, x):
        if self.args.norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
            return x

    def get_similiarity_map(self, proto, query):
        way = proto.shape[0]  # 5
        num_query = query.shape[0]  # 75
        query = query.view(query.shape[0], query.shape[1], -1)  # [75, 640, 25]
        proto = proto.view(proto.shape[0], proto.shape[1], -1)  # [5, 640, 25]

        proto = proto.unsqueeze(0).repeat(
            [num_query, 1, 1, 1])  # [75, 5, 640, 25]
        query = query.unsqueeze(1).repeat([1, way, 1, 1])  # [75, 5, 640, 25]
        proto = proto.permute(0, 1, 3, 2)  # [75,5,25,640]
        query = query.permute(0, 1, 3, 2)  # [75,5,25,640]
        feature_size = proto.shape[-2]  # [25]

        if self.args.metric == 'cosine':
            proto = proto.unsqueeze(-3)  # [75, 5, 1, 25, 640]
            query = query.unsqueeze(-2)  # [75, 5, 25, 1, 640]
            # [75, 5, 25, 25, 640]
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = F.cosine_similarity(
                proto, query, dim=-1)  # [75, 5, 25, 25]
        if self.args.metric == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map

        return similarity_map

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
