import glob
import sys
import PIL

from torchvision import transforms, datasets
from PIL import Image
from torch.utils.data import Dataset
from mytransforms import mytransforms
import numpy as np
import random
import torch
import rasterio

# class dataload_valid(Dataset):
#     def __init__(self, path='', aug=False, mode='img'):
#         self.data_info = datasets.ImageFolder(root=path)
#         self.data_num = len(self.data_info)
#         self.path_mtx = np.array(self.data_info.samples).reshape(self.data_num, 2)
#         self.mask_num = int(len(self.path_mtx[0]))
#         self.aug = aug
#         self.mode = mode
#
#         if mode == 'img':
#             self.path = self.data_info
#             self.data_num = 1
#         elif mode == 'dir':
#             self.path = glob.glob(path + '/*.png')
#             self.data_num = len(self.path)
#
#         self.mask_trans = transforms.Compose([transforms.Resize((224,224)),
#                                               mytransforms.Affine(0, translate=[0, 0], scale=1, fillcolor=0),
#                                               transforms.ToTensor()])
#         self.norm = mytransforms.Compose([transforms.Normalize((0.5,), (0.5,))])
#
#     def __len__(self):
#         return self.data_num
#
#     def __getitem__(self, idx):
#         if self.mode == 'img':
#             input = Image.open(self.path)
#         if self.mode == 'dir':
#             input = Image.open(self.path[idx])
#
#         input = self.mask_trans(input)
#
#         return input
#

class dataload_train(Dataset):
    def __init__(self,  path='dataset/Dog Segmentation', aug=True, phase='train'):
        self.data_info = datasets.ImageFolder(root=path)
        # print(self.data_info.samples)
        # sys.exit()
        self.data_num = len(self.data_info)
        # print(self.data_num)
        self.path_mtx = np.array(self.data_info.samples)[:, :1].reshape(2, int(self.data_num/2))
        self.mask_num = int(len(self.path_mtx[0]))
        # print(self.path_mtx)
        self.phase = phase
        # print(self.mask_num)
        # sys.exit()

        self.aug=aug

        self.mask_trans = transforms.Compose([transforms.Resize((224, 224)),
                                              mytransforms.Affine(0, translate=[0, 0], scale=1, fillcolor=0),
                                              transforms.ToTensor()])
        self.col_trans = transforms.Compose([transforms.ColorJitter(brightness=random.random())])
        self.norm = mytransforms.Compose([transforms.Normalize((0.5,), (0.5,))])


    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.aug:
            self.mask_trans.transforms[0].degrees = random.randrange(-25, 25)
            self.mask_trans.transforms[0].translate = [random.uniform(0, 0.05), random.uniform(0, 0.05)]
            self.mask_trans.transforms[0].scale = random.uniform(0.9, 1.1)

        if self.phase == 'train':
            _input = self.path_mtx[0, idx+1]
            mask = self.path_mtx[1, idx+1]

        else:
            _input = self.path_mtx[0, idx+1]
            _input = self.mask_trans(_input)
            return _input

        _input, mask = self.mask_trans(_input), self.mask_trans(mask)
        _input, mask = self.col_trans(_input), mask
        _input, mask = self.norm(_input), self.norm(mask)

        return [_input, mask]
