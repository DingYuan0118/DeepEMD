import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch
import os
import json

class cars(Dataset):

    def __init__(self, setname, args):
        IMAGE_PATH = os.path.join(args.data_dir, 'cars/images')
        SPLIT_PATH = os.path.join(args.data_dir, 'cars')
        if setname in ["test", "novel_all"]:
            json_path = osp.join(SPLIT_PATH, 'novel.json') # 当测试数据集与源数据集不一致时使用
        else:
            json_path = osp.join(SPLIT_PATH, setname + '.json')
        with open(json_path, "r") as f:
            self.meta = json.load(f) # json file(dict) ： {"label_names:[...], "image_names":[...], "image_labels":[...]}

        data = self.meta["image_names"]
        label = self.meta["image_labels"] # dataset 返回的label并无实际作用,只是当做一个类别标识
        data = [_.replace("filelists", "datasets") for _ in data]
 
        self.data = data  # data path of all data
        # self.label = label  # label of all data
        # 将label重新排列变为连续的0-N
        new_label = [0]
        init_class = 0
        for i in range(1, len(label)):
            if label[i] != label[i-1]:
                init_class += 1
            new_label.append(init_class)
        self.label = new_label
        self.num_class = len(set(label))

        if 'num_patch' not in vars(args).keys():
            print ('do not assign num_patch parameter, set as default: 9')
            self.num_patch=9
        else:
            self.num_patch=args.num_patch
            
        image_size = 84
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                 np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        patch_list=[]
        for _ in range(self.num_patch):
            patch_list.append(self.transform(Image.open(path).convert('RGB')))
        patch_list=torch.stack(patch_list,dim=0)
        return patch_list, label

if __name__ == '__main__':
    pass