import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np
import json

class recognition36Crop(Dataset):

    def __init__(self, setname, args):
        IMAGE_PATH = os.path.join(args.data_dir, 'recognition36_crop/images')
        SPLIT_PATH = os.path.join(args.data_dir, 'recognition36_crop')
        if setname in ["test", "novel_all"]:
            json_path = osp.join(SPLIT_PATH, 'novel_all.json') # 当测试数据集与源数据集不一致时使用
        else:
            json_path = osp.join(SPLIT_PATH, setname + '.json')
        with open(json_path, "r") as f:
            self.meta = json.load(f) # json file(dict) ： {"label_names:[...], "image_names":[...], "image_labels":[...]}

        data = self.meta["image_names"]
        label = self.meta["image_labels"] # dataset 返回的label并无实际作用,只是当做一个类别标识
        data = [_.replace("filelists", "datasets") for _ in data]
 
        self.data = data  # data path of all data
        self.label = label  # label of all data
        self.num_class = len(set(label))

        if setname == 'val' or setname == 'test':

            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif setname == 'train':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


if __name__ == '__main__':
    pass