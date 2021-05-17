import os
import shutil
import time
import pprint
import torch
import numpy as np
import os.path as osp
import random

from torch.utils.data import dataset


def save_list_to_txt(name, input_list):
    f = open(name, mode='a')
    for item in input_list:
        f.write(item+'\n')
    f.close()


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)
    print("save path: ", path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def load_model(model, dir):
    model_dict = model.state_dict()
    print('loading model from :', dir)
    pretrained_dict = torch.load(dir)['params']
    if 'encoder' in list(pretrained_dict.keys())[0]:
        # load from a parallel meta-trained model
        if 'module' in list(pretrained_dict.keys())[0]:
            pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items()}
    else:
        # load from a pretrained model
        pretrained_dict = {'encoder.' + k: v for k,
                           v in pretrained_dict.items()}
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    # update the param in encoder, remain others still
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)  # strict 默认为True

    return model


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def detect_grad_nan(model):
    for param in model.parameters():
        if (param.grad != param.grad).float().sum() != 0:  # nan detected
            param.grad.zero_()


def print_model_params(model, params):
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(q.numel() for q in model.buffers())
    print("\033[1;32;m{}\033[0m model \033[1;32;m{}\033[0m backbone have \033[1;32;m{}\033[0m parameters.".format(
        model.__class__.__name__, params.model, total_params + total_buffers))
    total_trainable_params = sum(p.numel()
                                 for p in model.parameters() if p.requires_grad)
    print("\033[1;32;m{}\033[0m model \033[1;32;m{}\033[0m backbone have \033[1;32;m{}\033[0m training parameters.".format(
        model.__class__.__name__, params.model, total_trainable_params))


def print_save_path(args):
    # pretrain阶段使用
    # 打印储存空间
    # 使用可变变量引用传参，不用显示赋值
    args.train_type = "epoch{epoch}_optim{optim}_lr{lr:.4f}_stepsize{stepsize}_gamma{gamma:.2f}_imagesize{imagesize}".format(
        lr=args.lr, stepsize=args.step_size, gamma=args.gamma, imagesize=args.image_size, optim=args.optim, epoch=args.max_epoch)
    if args.model == "resnet":
        if args.with_SA:
            if args.no_mlp and not args.SA_res:
                args.model_type = "{model}_MySA({heads}_{dim_head})".format(
                    model=args.model, heads=args.SA_heads, dim_head=args.SA_dim_head)
            elif args.no_mlp and args.SA_res:
                args.model_type = "{model}_MyResSA({heads}_{dim_head})".format(
                    model=args.model, heads=args.SA_heads, dim_head=args.SA_dim_head)
            else:          
                args.model_type = "{model}_SA({depth}_{heads}_{dim_head}_{mlp_dim})".format(
                    model=args.model, heads=args.SA_heads, dim_head=args.SA_dim_head, mlp_dim=args.SA_mlp_dim, depth=args.SA_depth)
            if args.pos_embed:
                args.model_type += "_pos-embed"
            args.save_path = "pre_train/{dataset}/".format(dataset=args.dataset) + args.model_type + "_" + args.train_type

        else:
            args.save_path = 'pre_train/{dataset}/{model}_epoch{epoch}_optim{optim}_lr{lr:.4f}_stepsize{stepsize}_gamma{gamma:.2f}_imagesize{imagesize}'.format(
                dataset=args.dataset, model=args.model, lr=args.lr, stepsize=args.step_size, gamma=args.gamma, imagesize=args.image_size, optim=args.optim, epoch=args.max_epoch)

    elif args.model == "ViT":
        args.save_path = 'pre_train/{dataset}/{model}_depth{depth}_epoch{epoch}_optim{optim}_lr{lr:.4f}_stepsize{stepsize}_gamma{gamma:.2f}_imagesize{imagesize}_use-clstoken({class_token})_vit-mode({vit_mode})'.format(
            dataset=args.dataset, model=args.model, lr=args.lr, stepsize=args.step_size, gamma=args.gamma, imagesize=args.image_size, class_token=str(not args.not_use_clstoken), vit_mode=args.vit_mode, optim=args.optim,
            epoch=args.max_epoch, depth=args.vit_depth)

    elif args.model == "vit_small_patch16_224":
        args.save_path = 'pre_train/{dataset}/{model}_depth{depth}_epoch{epoch}_optim{optim}_lr{lr:.4f}_stepsize{stepsize}_gamma{gamma:.2f}_imagesize{imagesize}_use-imagenet-params({imagenet_pretrain}))'.format(
            dataset=args.dataset, model=args.model, lr=args.lr, stepsize=args.step_size, gamma=args.gamma, imagesize=args.image_size, optim=args.optim, epoch=args.max_epoch, depth=args.vit_depth, imagenet_pretrain=str(not args.not_imagenet_pretrain))

    args.save_path = osp.join('checkpoint', args.save_path)
    if args.extra_dir is not None:
        args.save_path = osp.join(args.save_path, args.extra_dir)
    ensure_path(args.save_path)
    return args.save_path


def pretrain_save_path(args):
    # train_meta阶段使用，找寻pretrain model的存储地址
    args.train_type = "epoch{epoch}_optim{optim}_lr{lr:.4f}_stepsize{stepsize}_gamma{gamma:.2f}_imagesize{imagesize}".format(
        lr=args.pre_lr, stepsize=args.pre_step_size, gamma=args.pre_gamma, imagesize=args.image_size, optim=args.pre_optim, epoch=args.pre_epoch)
    if args.model == "resnet":
        if args.with_SA:
            if args.no_mlp and not args.SA_res:
                args.model_type = "{model}_MySA({heads}_{dim_head})".format(
                    model=args.model, heads=args.SA_heads, dim_head=args.SA_dim_head)
            elif args.no_mlp and args.SA_res:
                args.model_type = "{model}_MyResSA({heads}_{dim_head})".format(
                    model=args.model, heads=args.SA_heads, dim_head=args.SA_dim_head)
            else:          
                args.model_type = "{model}_SA({depth}_{heads}_{dim_head}_{mlp_dim})".format(
                    model=args.model, heads=args.SA_heads, dim_head=args.SA_dim_head, mlp_dim=args.SA_mlp_dim, depth=args.SA_depth)
            if args.pos_embed:
                args.model_type += "_pos-embed"
            args.pre_save_path = "pre_train/{dataset}/".format(dataset=args.dataset) + args.model_type + "_" + args.train_type

        else:
            args.pre_save_path = 'pre_train/{dataset}/{model}_epoch{epoch}_optim{optim}_lr{lr:.4f}_stepsize{stepsize}_gamma{gamma:.2f}_imagesize{imagesize}'.format(
                dataset=args.dataset, model=args.model, lr=args.pre_lr, stepsize=args.pre_step_size, gamma=args.pre_gamma, imagesize=args.image_size, optim=args.pre_optim, epoch=args.pre_epoch)
    elif args.model == "ViT":
        args.pre_save_path = 'pre_train/{dataset}/{model}_depth{depth}_epoch{epoch}_optim{optim}_lr{lr:.4f}_stepsize{stepsize}_gamma{gamma:.2f}_imagesize{imagesize}_use-clstoken({class_token})_vit-mode({vit_mode})'.format(
            dataset=args.dataset, model=args.model, lr=args.pre_lr, stepsize=args.pre_step_size, gamma=args.pre_gamma, imagesize=args.image_size, class_token=str(not args.not_use_clstoken), vit_mode=args.vit_mode, optim=args.pre_optim,
            epoch=args.pre_epoch, depth=args.vit_depth)

    elif args.model == "vit_small_patch16_224":
        args.pre_save_path = 'pre_train/{dataset}/{model}_depth{depth}_epoch{epoch}_optim{optim}_lr{lr:.4f}_stepsize{stepsize}_gamma{gamma:.2f}_imagesize{imagesize}_use-imagenet-params({imagenet_pretrain}))'.format(
            dataset=args.dataset, model=args.model, lr=args.pre_lr, stepsize=args.pre_step_size, gamma=args.pre_gamma, imagesize=args.image_size, optim=args.pre_optim, epoch=args.pre_epoch, depth=args.vit_depth, imagenet_pretrain=str(not args.not_imagenet_pretrain))

    args.pre_save_path = osp.join('checkpoint', args.pre_save_path)
    if args.extra_dir is not None:
        args.pre_save_path = osp.join(args.pre_save_path, args.extra_dir)
    if os.path.exists(args.pre_save_path):
        print("预训练模型路径:{}".format(args.pre_save_path))
    else:
        raise ValueError("没有该路径:{}".format(args.pre_save_path))
    return args.pre_save_path


def parse_tune_pretrain(args):
    if args.model == "ViT" and args.deepemd != 'fcn':
        print("选用ViT时未将deepemd参数置为fcn模式,当前为{}模式,将转换为fcn模式".format(args.deepemd))
        args.deepemd = 'fcn'

    if args.model == "ViT" and args.image_size != '256':
        print("选用ViT时未将image_size调整为256, 当前image size = {},将转换为256".format(
            args.image_size))
        args.image_size = 256

    if args.model == "vit_small_patch16_224" and args.image_size != '224':
        print("选用vit_small_patch16_224时未将image_size调整为224, 当前image size = {},将转换为224".format(
            args.image_size))
        args.image_size = 224

    if args.model == 'resnet' and args.image_size != '84':
        print("选用resnet时未将image_size调整为84, 当前image size = {},将转换为84".format(
            args.image_size))
        args.image_size = 84

    if args.model == 'resnet' and args.with_SA:
        print("使用带self attention的resnet")


def meta_save_path(args):
    # meta train阶段使用
    epoch_index = args.pre_save_path.find("_epoch")
    args.model_name = args.pre_save_path[:epoch_index].split("/")[-1]
    if args.sfc_update_step == 100:
        args.save_path = "{dataset}/{model_name}/{shot}shot-{way}way".format(dataset=args.dataset,
                        model_name=args.model_name, shot=args.shot, way=args.way, sfc_update_step=int(args.sfc_update_step))
    else:
        args.save_path = "{dataset}/{model_name}/{shot}shot-{way}way_SFC{sfc_update_step}".format(dataset=args.dataset,
                        model_name=args.model_name, shot=args.shot, way=args.way, sfc_update_step=int(args.sfc_update_step))
    args.save_path = osp.join('checkpoint/meta_train',
                              args.save_path + "_{}".format(args.solver))
    if args.extra_dir is not None:
        args.save_path = osp.join(args.save_path, args.extra_dir)
    ensure_path(args.save_path)
    return args.save_path, args.model_name


def format_model_name(args):
    if args.model_name == "resnet_MyResSA":
        args.model_name = args.model_name + \
            "({}_{})".format(args.SA_heads, args.SA_dim_head)
        if args.pos_embed:
            args.model_name += "_pos-embed"

    elif args.model_name == "resnet":
        pass

    else:
        # TODO: 实现其他方法
        print("")
        raise ValueError("没有该model_name")
