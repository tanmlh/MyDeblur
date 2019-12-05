import torch
from models import networks
from PIL import Image
import time
import numpy as np

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

net = networks.define_G(3, 3, 64, 'resnet_9blocks', 'instance', False, True)
state = torch.load('./checkpoints_dir_v1/conditional_gan/network_merged.pkl')['nets_state']['generator']
net.load_state_dict(state)
net.cuda()

A = Image.open('/home/opt603/Code/src/DeblurGAN-master/sample_imgs/0001.jpg')

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5),
                                       (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

crop_params = transforms.RandomCrop.get_params(A, output_size=(720, 720))
A = TF.crop(A, *crop_params)
A = transform(A)
A = A.unsqueeze(0).cuda()

time1 = time.time()
fake_B = net(A)[0]
time2 = time.time()

B = fake_B.detach().cpu().numpy().transpose(1, 2, 0)
B = (B+1) / 2.0 * 255.0
B = B.astype(np.uint8)

Image.fromarray(B).save('./temp/fake_B.jpg')

