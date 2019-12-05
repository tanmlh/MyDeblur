import os.path
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch
from PIL import Image
import pdb
from tqdm import tqdm

class AlignedDataset():
    def __init__(self, loader_conf, phase='train'):
        self.loader_conf = loader_conf
        self.dataset_dir = loader_conf['dataset_dir']

        if phase == 'train':
            self.imgs_desc_file = loader_conf['train_imgs_desc_file']
        elif phase == 'val':
            self.imgs_desc_file = loader_conf['val_imgs_desc_file']
        else:
            raise ValueError

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

        with open(self.imgs_desc_file) as f:
            lines = f.read().splitlines()
            path = list(map(lambda x: x.split(' '), lines))
            random.shuffle(path)

        imgs_path = []
        for i, line in enumerate(path):
            if int(line[-1]) >= 2304 and int(line[-2]) >= 3456:
                imgs_path.append(line)

        # if phase == 'val':
        #     imgs_path = imgs_path[:100]

        if loader_conf['pre_load'] is True:
            imgs_path = imgs_path[:100]
            imgs_A = []; imgs_B = []
            tq = tqdm(imgs_path)
            tq.set_description('loading dataset... ')
            for img_A_path, img_B_path in tq:
                img_A = Image.open(os.path.join(self.dataset_dir, img_A_path)).convert('RGB')
                img_B = Image.open(os.path.join(self.dataset_dir, img_B_path)).convert('RGB')
                imgs_A.append(img_A)
                imgs_B.append(img_B)

            self.imgs_A = imgs_A
            self.imgs_B = imgs_B

        self.imgs_path = imgs_path

    def __getitem__(self, idx):
        if self.loader_conf['pre_load'] is True:
            A = self.imgs_A[idx]
            B = self.imgs_B[idx]

        else:
            A = Image.open(os.path.join(self.dataset_dir, self.imgs_path[idx][0])).convert('RGB')
            B = Image.open(os.path.join(self.dataset_dir, self.imgs_path[idx][1])).convert('RGB')

        crop_params = transforms.RandomCrop.get_params(A, output_size=self.loader_conf['crop_size'])
        A = TF.crop(A, *crop_params)
        B = TF.crop(B, *crop_params)

        A = self.transform(A)
        B = self.transform(B)

        A_path = self.imgs_path[idx][0]
        B_path = self.imgs_path[idx][1]

        res = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        return res

    def __len__(self):
        return len(self.imgs_path)
