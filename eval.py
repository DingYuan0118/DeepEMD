import argparse

import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from Models.dataloader.samplers import CategoriesSampler
from Models.models.Network import DeepEMD
from Models.utils import *
from Models.dataloader.data_utils import *

DATA_DIR = './datasets'
# DATA_DIR='/home/zhangchi/dataset'
MODEL_DIR = 'deepemd_trained_model/miniimagenet/fcn/max_acc.pth'


parser = argparse.ArgumentParser()
# about task
parser.add_argument('-way', type=int, default=5)
parser.add_argument('-shot', type=int, default=1)
# number of query image per class
parser.add_argument('-query', type=int, default=15)
parser.add_argument('-dataset', type=str, default='miniimagenet', choices=['miniimagenet', 'cub', 'tieredimagenet', 'fc100',
                                                                           'tieredimagenet_yao', 'cifar_fs', 'recognition36', 'recognition36_crop'])  # 增加自定义数据集
parser.add_argument('-set', type=str, default='test',
                    choices=['train', 'val', 'test'])
# about model
parser.add_argument('-temperature', type=float, default=12.5)
parser.add_argument('-metric', type=str, default='cosine', choices=['cosine'])
parser.add_argument('-norm', type=str, default='center', choices=['center'])
parser.add_argument('-deepemd', type=str, default='fcn',
                    choices=['fcn', 'grid', 'sampling'])
# deepemd fcn only
parser.add_argument('-feature_pyramid', type=str, default=None)
# deepemd sampling only
parser.add_argument('-num_patch', type=int, default=9)
# deepemd grid only patch_list
parser.add_argument('-patch_list', type=str, default='2,3')
parser.add_argument('-patch_ratio', type=float, default=2)
# solver
parser.add_argument('-solver', type=str, default='opencv', choices=['opencv'])
# SFC
parser.add_argument('-sfc_lr', type=float, default=100)
parser.add_argument('-sfc_wd', type=float, default=0,
                    help='weight decay for SFC weight')
parser.add_argument('-sfc_update_step', type=float, default=100)
parser.add_argument('-sfc_bs', type=int, default=4)
# others
parser.add_argument('-test_episode', type=int, default=1000)
parser.add_argument('-gpu', default='0,1')
parser.add_argument('-data_dir', type=str, default=DATA_DIR)
parser.add_argument('-model_dir', type=str, default=MODEL_DIR)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-extra_dir', type=str, default=None,
                    help='extra information that is added to checkpoint dir, e.g. hyperparameters')
parser.add_argument('--image_size', type=int, default=84,
                    help='extra information that is added to checkpoint dir, e.g. hyperparameters')
# ====================================自定义模型参数====================================
# 额外参数
parser.add_argument('--model', type=str, default='resnet',
                    help='选择要使用的backbone(为vit transformer做准备), 使用ViT作为backbone时请使用FCN模式')
parser.add_argument("--pre_lr", type=float, default=0.01, help="预训练时学习率")
parser.add_argument('--pre_gamma', type=float, default=0.2, help="预训练时学习率衰减效率")
parser.add_argument('--pre_optim', type=str, default='SGD', help='预训练时选择优化器')
parser.add_argument("--pre_epoch", type=int, default=120, help='预训练使用的epoch数')
parser.add_argument('--pre_step_size', type=int,
                    default=30, help='预训练使用的step_size')
parser.add_argument('--not_use_clstoken', action="store_true",
                    help='viT模型可选项是否添加cls token, 默认使用')
parser.add_argument('--vit_mode', type=str, default='cls',
                    choices=['cls', 'mean'], help='选择使用cls token或者mean(平均所有patch)的方式')
parser.add_argument('--vit_depth', type=int, default=4, help="使用ViT时的深度")
parser.add_argument('--not_imagenet_pretrain',
                    action="store_true", help="是否使用imagenet的pretrain参数")
# resnet下使用注意力机制的相关参数
parser.add_argument('--with_SA', action='store_true',
                    help="在resnet基础上使用self-attention模式")
parser.add_argument('--SA_heads', type=int, default=8, help="resnet使用heads的数目")
parser.add_argument('--SA_mlp_dim', type=int, default=1024,
                    help="resnet中SA模块使用的mlp中隐藏层的数目")
parser.add_argument('--SA_depth', type=int, default=1, help='resnet下SA模块的层数')
parser.add_argument('--SA_dim_head', type=int, default=64,
                    help="resnet下SA模块每个head的维度")
parser.add_argument('--SA_dropout', type=float,
                    default=0.1, help="resnet下SA模块的dropout率")
# ====================================自定义模型参数====================================
args = parser.parse_args()
if args.feature_pyramid is not None:
    args.feature_pyramid = [int(x) for x in args.feature_pyramid.split(',')]
args.patch_list = [int(x) for x in args.patch_list.split(',')]

if args.model == "resnet":
    if args.with_SA:
        args.model_name = '{model}_SA({depth}_{heads}_{dim_head}_{mlp_dim})'.format(
            model=args.model, depth=args.SA_depth, heads=args.SA_heads, dim_head=args.SA_dim_head, mlp_dim=args.SA_mlp_dim)
    else:
        args.model_name = '{model}'.format(model=args.model)

# 不管测试时是5shot 还是1shot，均使用5shot训练后的模型
args.model_dir = 'checkpoint/meta_train/miniimagenet/{model_name}/{shot}shot-{way}way/max_acc.pth'.format(
    model_name=args.model_name, shot=5, way=5)

args.res_save_path = "result/{dataset}/{model_name}/{shot}shot-{way}way/".format(
    dataset=args.dataset, model_name=args.model_name, shot=args.shot, way=args.way)
if os.path.exists(args.res_save_path):
    pass
else:
    os.makedirs(args.res_save_path)

pprint(vars(args))
if os.path.exists(args.model_dir):
    print("使用模型路径:{}".format(args.model_dir))
else:
    raise ValueError("找不到模型参数文件:", args.model_dir)

set_seed(args.seed)
num_gpu = set_gpu(args)
Dataset = set_up_datasets(args)


# model
model = DeepEMD(args)
model = load_model(model, args.model_dir)
model = nn.DataParallel(model, list(range(num_gpu)))
model = model.cuda()
model.eval()

# test dataset
test_set = Dataset(args.set, args)
sampler = CategoriesSampler(
    test_set.label, args.test_episode, args.way, args.shot + args.query)
loader = DataLoader(test_set, batch_sampler=sampler,
                    num_workers=8, pin_memory=True)
tqdm_gen = tqdm.tqdm(loader)

# label of query images
ave_acc = Averager()
test_acc_record = np.zeros((args.test_episode,))
label = torch.arange(args.way).repeat(args.query)
label = label.type(torch.cuda.LongTensor)

with torch.no_grad():
    for i, batch in enumerate(tqdm_gen, 1):
        # "_"取出label，但后续均未使用，每一个batch都重置了label
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot
        model.module.mode = 'encoder'
        data = model(data)
        # shot: 5,3,84,84  query:75,3,84,84
        data_shot, data_query = data[:k], data[k:]
        model.module.mode = 'meta'
        if args.shot > 1:
            data_shot = model.module.get_sfc(data_shot)
        logits = model((data_shot.unsqueeze(0).repeat(
            num_gpu, 1, 1, 1, 1), data_query))
        acc = count_acc(logits, label) * 100
        ave_acc.add(acc)
        test_acc_record[i - 1] = acc
        m, pm = compute_confidence_interval(test_acc_record[:i])
        tqdm_gen.set_description(
            'batch {}: This episode:{:.2f}  average: {:.4f}+{:.4f}'.format(i, acc, m, pm))

    m, pm = compute_confidence_interval(test_acc_record)
    result_list = ['test Acc {:.4f}'.format(ave_acc.item())]
    result_list.append('Test Acc {:.4f} + {:.4f}'.format(m, pm))
    print(result_list[0])
    print(result_list[1])
    # TODO
    save_list_to_txt(os.path.join(args.res_save_path, 'results.txt'), result_list)
