import argparse
from numpy.core.fromnumeric import argsort
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from Models.dataloader.samplers import CategoriesSampler
from Models.utils import *
from Models.dataloader.data_utils import *
from Models.models.deepmmd import DeepMMD
from torch.utils.tensorboard import SummaryWriter
import tqdm
import time

PRETRAIN_DIR = 'deepemd_pretrain_model/'
# DATA_DIR='/home/zhangchi/dataset'
DATA_DIR = './datasets'

parser = argparse.ArgumentParser()
# about dataset and training
parser.add_argument('-dataset', type=str, default='miniimagenet', choices=[
                    'miniimagenet', 'cub', 'tieredimagenet', 'fc100', 'tieredimagenet_yao', 'cifar_fs'])
parser.add_argument('-data_dir', type=str,
                    default=DATA_DIR, help='dir of datasets')
parser.add_argument('-set', type=str, default='val', choices=[
                    'test', 'val'], help='the set used for validation')  # set used for validation
# about training
parser.add_argument('-bs', type=int, default=1, help='batch size of tasks')
parser.add_argument('-max_epoch', type=int, default=100)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-temperature', type=float,                         # 暂时未使用
                    default=12.5, help="control the loss value")
parser.add_argument('-step_size', type=int, default=10)
parser.add_argument('-gamma', type=float, default=0.5)
parser.add_argument('-val_frequency', type=int, default=50)
parser.add_argument('-random_val_task', action='store_true',
                    help='random samples tasks for validation at each epoch')
parser.add_argument('-save_all', action='store_true',
                    help='save models on each epoch')
parser.add_argument('--image_size', type=int, default=84,
                    help='extra information that is added to checkpoint dir, e.g. hyperparameters')
# about task
parser.add_argument('-way', type=int, default=5)
parser.add_argument('-shot', type=int, default=1)
parser.add_argument('-query', type=int, default=15,
                    help='number of query image per class')
parser.add_argument('-val_episode', type=int, default=1000,
                    help='number of validation episode')
parser.add_argument('-test_episode', type=int, default=5000,
                    help='number of testing episodes after training')
# about model
parser.add_argument('-pretrain_dir', type=str, default=PRETRAIN_DIR)
parser.add_argument('-metric', type=str, default='cosine', choices=['cosine'])
parser.add_argument('-norm', type=str, default='center',
                    choices=['center'], help='feature normalization')
parser.add_argument('-deepemd', type=str, default='fcn',
                    choices=['fcn', 'grid', 'sampling'])
# deepemd fcn only
parser.add_argument('-feature_pyramid', type=str,
                    default=None, help='you can set it like: 2,3')
# deepemd sampling only
parser.add_argument('-num_patch', type=int, default=9)
# deepemd grid only patch_list
parser.add_argument('-patch_list', type=str, default='2,3',
                    help='the size of grids at every image-pyramid level')
parser.add_argument('-patch_ratio', type=float, default=2,
                    help='scale the patch to incorporate context around the patch')
# slvoer about
parser.add_argument('-solver', type=str, default='opencv',
                    choices=['opencv', 'qpth'])
parser.add_argument('-form', type=str, default='L2', choices=['QP', 'L2'])
parser.add_argument('-l2_strength', type=float, default=0.000001)
# SFC
parser.add_argument('-sfc_lr', type=float, default=0.1,
                    help='learning rate of SFC')
parser.add_argument('-sfc_wd', type=float, default=0,
                    help='weight decay for SFC weight')
parser.add_argument('-sfc_update_step', type=int,
                    default=100, help='number of updating step of SFC')
parser.add_argument('-sfc_bs', type=int, default=4,
                    help='batch size for finetune sfc')
# OTHERS
parser.add_argument('-gpu', default='0,1')
parser.add_argument('-extra_dir', type=str, default=None,
                    help='extra information that is added to checkpoint dir, e.g. hyperparameters')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=0.0005, help="meta train 使用的weight decay")

# =========================== 自定义额外参数 ===============================
parser.add_argument('--model', type=str, default='resnet',
                    help='选择要使用的backbone(为vit transformer做准备), 使用ViT作为backbone时请使用FCN模式')
parser.add_argument("--pre_lr", type=float, default=0.01, help="预训练时学习率")
parser.add_argument('--pre_gamma', type=float, default=0.2, help="预训练时学习率衰减效率")
parser.add_argument('--pre_optim', type=str, default='SGD', help='预训练时选择优化器')
parser.add_argument("--pre_epoch", type=int, default=120, help='预训练使用的epoch数')
parser.add_argument('--pre_step_size', type=int,
                    default=30, help='预训练使用的step_size')
parser.add_argument('--pre_weight_decay', type=float,
                    default=0.0005, help='预训练使用的weight_decay')
parser.add_argument('--not_use_clstoken', action="store_true",
                    help='viT模型可选项是否添加cls token, 默认使用')
parser.add_argument('--vit_mode', type=str, default='cls',
                    choices=['cls', 'mean'], help='选择使用cls token或者mean(平均所有patch)的方式')
parser.add_argument('--vit_depth', type=int, default=4, help="使用ViT时的深度")
parser.add_argument('--not_imagenet_pretrain',
                    action="store_true", help="是否使用imagenet的pretrain参数")
# ========================= resnet下使用注意力机制的相关参数 =========================
parser.add_argument('--with_SA', action='store_true', help="在resnet基础上使用self-attention模式")
# parser.add_argument('--SA_heads', type=int, default=8, help="resnet使用heads的数目")
# parser.add_argument('--SA_mlp_dim', type=int, default=1024, help="resnet中SA模块使用的mlp中隐藏层的数目")
# parser.add_argument('--SA_depth', type=int, default=1, help='resnet下SA模块的层数')
# parser.add_argument('--SA_dim_head', type=int, default=64, help="resnet下SA模块每个head的维度")
# parser.add_argument('--SA_dropout', type=float, default=0.1, help="resnet下SA模块的dropout率")
# parser.add_argument('--SA_res', action="store_true", help="使用残差连接")
# parser.add_argument('--no_mlp', action="store_true", help="去除mlp层")
# parser.add_argument('--pos_embed', action="store_true", help="加入相对位置编码(relative position embedding)")


args = parser.parse_args()
pprint(vars(args))

# transform str parameter into list
if args.feature_pyramid is not None:
    args.feature_pyramid = [int(x) for x in args.feature_pyramid.split(',')]
args.patch_list = [int(x) for x in args.patch_list.split(',')]

set_seed(args.seed)
num_gpu = set_gpu(args)
Dataset = set_up_datasets(args)
parse_tune_pretrain(args)
pretrain_save_path(args)
# model
# args.pretrain_dir=osp.join(args.pretrain_dir,'%s/resnet12/max_acc.pth'%(args.dataset))
args.pretrain_dir = osp.join(args.pre_save_path, 'max_acc.pth').replace("checkpoint", "checkpoint_mmd")
if os.path.exists(args.pretrain_dir):
    print("使用此处的模型:{}".format(args.pretrain_dir))
else:
    raise ValueError("未找到预训练模型")

model = DeepMMD(args)
model = load_model(model, args.pretrain_dir)
model = nn.DataParallel(model, list(range(num_gpu)))
model = model.cuda()
model.eval()



meta_save_path(args)
with open(os.path.join(args.save_path, "config.txt"), "w") as fileob:
    print("pre weight decay:{}, SA_dropout:{}".format(args.pre_weight_decay, args.SA_dropout), file=fileob)
    print("meta weight decay:{}".format(args.weight_decay), file=fileob)

with open(os.path.join(args.save_path, "args.txt"), "w") as fileob:
    print(vars(args), file=fileob)
# args.save_path = '%s/%s/%dshot-%dway/'%(args.dataset,args.deepemd,args.shot,args.way)
# args.save_path = '%s/%s/%s/%dshot-%dway/' % (
#     args.dataset, args.deepemd, args.solver, args.shot, args.way)

# args.save_path = osp.join('checkpoint', args.save_path)
# if args.extra_dir is not None:
#     args.save_path = osp.join(args.save_path, args.extra_dir)
# ensure_path(args.save_path)


trainset = Dataset('train', args)
train_sampler = CategoriesSampler(
    trainset.label, args.val_frequency*args.bs, args.way, args.shot + args.query)
train_loader = DataLoader(
    dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

valset = Dataset(args.set, args)
val_sampler = CategoriesSampler(
    valset.label, args.val_episode, args.way, args.shot + args.query)
val_loader = DataLoader(
    dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)

if not args.random_val_task:
    print('fix val set for all epochs')
    val_loader = [x for x in val_loader]
print('save all checkpoint models:', (args.save_all is True))

# label for query set, always in the same pattern 重新按批次定义label
# 012340123401234...
label = torch.arange(args.way, dtype=torch.int8).repeat(args.query)
label = label.type(torch.LongTensor)
label = label.cuda()


optimizer = torch.optim.SGD([{'params': model.parameters(), 'lr': args.lr}], momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=args.step_size, gamma=args.gamma)


def save_model(name):
    torch.save(dict(params=model.state_dict()),
               osp.join(args.save_path, name + '.pth'))


trlog = {}
trlog['args'] = vars(args)
trlog['train_loss'] = []
trlog['val_loss'] = []
trlog['train_acc'] = []
trlog['val_acc'] = []
trlog['max_acc'] = 0.0
trlog['max_acc_epoch'] = 0

global_count = 0
writer = SummaryWriter(osp.join(args.save_path, 'tf'))

result_list = [args.save_path]
for epoch in range(1, args.max_epoch + 1):
    print(args.save_path)
    start_time = time.time()

    tl = Averager()
    ta = Averager()

    tqdm_gen = tqdm.tqdm(train_loader)
    model.train()
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm_gen, 1):

        global_count = global_count + 1
        data, _ = [_.cuda() for _ in batch]

        k = args.way * args.shot
        model.module.mode = 'encoder'
        data = model(data)
        data_shot, data_query = data[:k], data[k:]
        model.module.mode = 'meta'
        if args.shot > 1:
            data_shot = model.module.get_sfc(data_shot)
        # prepare data for each gpu [num_gpu, 5, 640, 5, 5]
        logits = model((data_shot.unsqueeze(0).repeat(
            num_gpu, 1, 1, 1, 1), data_query))
        loss = F.cross_entropy(logits, label)

        acc = count_acc(logits, label)
        writer.add_scalar('data/loss', float(loss), global_count)
        writer.add_scalar('data/acc', float(acc), global_count)

        total_loss = loss/args.bs  # batch of tasks, done by accumulate gradients
        writer.add_scalar('data/total_loss', float(total_loss), global_count)
        tqdm_gen.set_description('epo {}, total loss={:.4f} acc={:.4f}'
                                 .format(epoch, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)
        total_loss.backward()

        detect_grad_nan(model)
        if i % args.bs == 0:  # batch of tasks, done by accumulate gradients
            optimizer.step()
            optimizer.zero_grad()

    tl = tl.item()
    ta = ta.item()
    vl = Averager()
    va = Averager()

    # validation
    model.eval()
    with torch.no_grad():
        tqdm_gen = tqdm.tqdm(val_loader)
        for i, batch in enumerate(tqdm_gen, 1):
            data, _ = [_.cuda() for _ in batch]
            k = args.way * args.shot
            model.module.mode = 'encoder'
            data = model(data)
            data_shot, data_query = data[:k], data[k:]
            model.module.mode = 'meta'
            if args.shot > 1:
                data_shot = model.module.get_sfc(data_shot)
            logits = model((data_shot.unsqueeze(0).repeat(
                num_gpu, 1, 1, 1, 1), data_query))

            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            vl.add(loss.item())
            va.add(acc)

    vl = vl.item()
    va = va.item()
    writer.add_scalar('data/val_loss', float(vl), epoch)
    writer.add_scalar('data/val_acc', float(va), epoch)
    tqdm_gen.set_description(
        'epo {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    print('val acc:%.4f' % va)
    if va >= trlog['max_acc']:
        print('*********A better model is found*********')
        trlog['max_acc'] = va
        trlog['max_acc_epoch'] = epoch
        save_model('max_acc')

    trlog['train_loss'].append(tl)
    trlog['train_acc'].append(ta)
    trlog['val_loss'].append(vl)
    trlog['val_acc'].append(va)

    result_list.append('epoch:%03d,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
        epoch, tl, ta, vl, va))

    torch.save(trlog, osp.join(args.save_path, 'trlog'))
    if args.save_all and epoch % 30 == 0:
        save_model('epoch-%d' % epoch)
        torch.save(optimizer.state_dict(), osp.join(
            args.save_path, 'optimizer_latest.pth'))
    print('best epoch {}, best val acc={:.4f}'.format(
        trlog['max_acc_epoch'], trlog['max_acc']))
    print('This epoch takes %d seconds' % (time.time()-start_time),
          '\nstill need %.2f hour to finish' % ((time.time()-start_time)*(args.max_epoch-epoch)/3600))
    lr_scheduler.step()

writer.close()


# Test Phase
trlog = torch.load(osp.join(args.save_path, 'trlog'))
test_set = Dataset('test', args)
sampler = CategoriesSampler(
    test_set.label, args.test_episode, args.way, args.shot + args.query)
loader = DataLoader(test_set, batch_sampler=sampler,
                    num_workers=8, pin_memory=True)
test_acc_record = np.zeros((args.test_episode,))
model.load_state_dict(torch.load(
    osp.join(args.save_path, 'max_acc' + '.pth'))['params'])
model.eval()

ave_acc = Averager()
label = torch.arange(args.way).repeat(args.query)
if torch.cuda.is_available():
    label = label.type(torch.cuda.LongTensor)
else:
    label = label.type(torch.LongTensor)

tqdm_gen = tqdm.tqdm(loader)
with torch.no_grad():
    for i, batch in enumerate(tqdm_gen, 1):
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot
        model.module.mode = 'encoder'
        data = model(data)
        data_shot, data_query = data[:k], data[k:]
        model.module.mode = 'meta'
        if args.shot > 1:
            data_shot = model.module.get_sfc(data_shot)
        logits = model((data_shot.unsqueeze(0).repeat(
            num_gpu, 1, 1, 1, 1), data_query))
        acc = count_acc(logits, label) * 100
        ave_acc.add(acc)
        test_acc_record[i-1] = acc
        tqdm_gen.set_description(
            'batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item(), acc))


m, pm = compute_confidence_interval(test_acc_record)

result_list.append('Val Best Epoch {},\nbest val Acc {:.4f}, \nbest est Acc {:.4f}'.format(
    trlog['max_acc_epoch'], trlog['max_acc'], ave_acc.item()))
result_list.append('Test Acc {:.4f} + {:.4f}'.format(m, pm))
print(result_list[-2])
print(result_list[-1])
save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)
