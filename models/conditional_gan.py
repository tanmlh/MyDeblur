import random
import numpy as np
import pdb

import torch
import torchvision as tv
import torch.nn as nn
import torch.autograd as autograd

from .base_module import BaseModule
from .base_solver import BaseSolver
from .architectures import networks
from .architectures import fpn_mobilenet
from . import metrics

def get_solver(conf):
    return ConditionalGANSolver(conf)

def get_model(conf):
    return ConditionalGANModule(conf)

def get_visual(real_A, fake_B, real_B):
    real_As, fake_Bs, real_Bs = [], [], []

    for i in range(real_A.shape[0]):
        real_As.append(real_A[i])
    for i in range(fake_B.shape[0]):
        fake_Bs.append(fake_B[i])
    for i in range(real_B.shape[0]):
        real_Bs.append(real_B[i])

    real_As = np.concatenate(real_As, axis=1)
    fake_Bs = np.concatenate(fake_Bs, axis=1)
    real_Bs = np.concatenate(real_Bs, axis=1)

    diff_fake_B = np.abs(real_As - fake_Bs)
    diff_real_B = np.abs(real_Bs - real_As)

    visual = np.concatenate([real_As, fake_Bs, real_Bs, diff_fake_B, diff_real_B], axis=2)
    visual  = (visual + 1) / 2.0 * 255.0
    visual = visual.astype(np.uint8)

    return visual

def get_generator(net_conf):
    G_conf = net_conf['generator']

    if net_conf['generator_type'] == 'ResNet':
        generator = networks.define_G(
            G_conf['num_in_chns'],
            G_conf['num_out_chns'],
            G_conf['ngf'],
            G_conf['backbone_name'],
            G_conf['norm_type'],
            G_conf['use_dropout'],
            G_conf['learn_residual'])
        """
        features = generator.features
        res = nn.Sequential()
        res.add_module('ResNet_pre_downsample_0', nn.Conv2d(3, G_conf['num_in_chns'], 11, stride=2, padding=5))
        for i, layer in enumerate(list(generator)):
            res.add_module('ResNet_'+str(i), layer)
        res.add_module('ResNet_post_upsample_0', nn.ConvTranspose2d(G_conf['num_out_chns'], 3, 3, stride=2, padding=1))
        generator = res
        """

    elif net_conf['generator_type'] == 'MobileNet':
        generator = fpn_mobilenet.FPNMobileNet(G_conf['norm_type'], kernel_size=G_conf['kernel_size'])

    return generator

def get_discriminator(net_conf):
    D_conf = net_conf['discriminator']
    use_sigmoid = net_conf['gan_type'] == 'gan'

    discriminator = networks.define_D(
        D_conf['num_out_chns'],
        D_conf['ndf'],
        D_conf['backbone_name'],
        D_conf['num_layers'],
        D_conf['norm_type'],
        use_sigmoid)

    return discriminator

def get_perceptual_net(net_conf):
    conv_3_3_layer = 14
    vgg19 = tv.models.vgg19(pretrained=True).features
    perceptual_net = nn.Sequential()
    perceptual_net.add_module('vgg_avg_pool', nn.AvgPool2d(kernel_size=4, stride=4))
    for i,layer in enumerate(list(vgg19)):
        perceptual_net.add_module('vgg_'+str(i),layer)
        if i == conv_3_3_layer:
            break
    for param in perceptual_net.parameters():
        param.requires_grad = False

    # pdb.set_trace()
    """
    perceptual_net = networks.Generator_Encoder()
    state = torch.load('./checkpoints/45_NO_Skip.pth')
    perceptual_net.load_state_dict(state['state_dict'], strict=False)
    for param in perceptual_net.parameters():
        param.requires_grad = False
    """

    return perceptual_net

class ConditionalGANSolver(BaseSolver):
    def init_tensors(self):
        self.tensors = {}
        self.tensors['real_A'] = torch.FloatTensor()
        self.tensors['real_B'] = torch.FloatTensor()
        self.tensors['fake_B'] = torch.FloatTensor()


        self.tensors['gan_real_label'] = torch.FloatTensor()
        self.tensors['gan_fake_label'] = torch.FloatTensor()


    def set_tensors(self, batch):
        real_A, real_B = batch['A'], batch['B']
        real_A = nn.functional.avg_pool2d(real_A, kernel_size=2, stride=2)
        real_B = nn.functional.avg_pool2d(real_B, kernel_size=2, stride=2)

        self.tensors['real_A'].resize_(real_A.size()).copy_(real_A)
        self.tensors['real_B'].resize_(real_B.size()).copy_(real_B)


    def process_batch(self, batch, phase='train'):
        torch.cuda.empty_cache()
        optimizers = self.optimizers
        self.set_tensors(batch)

        if phase == 'train':
            self.nets.train()
            self.nets.zero_grad()
        else:
            self.nets.train()

        if phase == 'train':
            optimizers['generator'].zero_grad()
            loss_G, state = self.nets.forward_G(self.tensors)
            loss_G.backward()
            optimizers['generator'].step()
        else:
            with torch.no_grad():
                loss_G, state = self.nets.forward_G(self.tensors)

        for i in range(self.net_conf['num_D_backward']):
            if phase == 'train':
                optimizers['discriminator'].zero_grad()
                loss_D, state2 = self.nets.forward_D(self.tensors, 'train')
                loss_D.backward(retain_graph=True)
                optimizers['discriminator'].step()
            else:
                state2 = {}

        # for key, value in state2.items():
        #     state[key] = value

        real_A = self.tensors['real_A'].detach().cpu().numpy()
        real_B = self.tensors['real_B'].detach().cpu().numpy()
        fake_B = self.tensors['fake_B'].detach().cpu().numpy()

        visual = get_visual(real_A[:, 0:3, :, :], fake_B, real_B)

        psnr = []
        for i in range(real_B.shape[0]):
            sample_real_B, sample_fake_B = map(lambda x: (x+1)/2.0*255.0, [real_B[i], fake_B[i]])
            psnr.append(metrics.PSNR(sample_real_B, sample_fake_B))

        state['image|image'] = visual
        state['scalar|psnr'] = sum(psnr) / len(psnr)

        state['else|fake_B'] = nn.functional.interpolate(self.tensors['fake_B'].detach().cpu(),
                                                         scale_factor=(2, 2), mode='bilinear',
                                                         align_corners=False)

        # state['else|fake_B'] = self.tensors['fake_B'].detach().cpu()
        return state

class ConditionalGANModule(BaseModule):
    def __init__(self, net_conf):
        super(ConditionalGANModule, self).__init__()

        self.net_conf = net_conf
        self.net = {}

        self.net['generator'] = get_generator(net_conf)
        if net_conf['loss_weights']['perceptual'] != 0:
            self.net['perceptual'] = get_perceptual_net(net_conf)
        if net_conf['loss_weights']['GAN'] != 0:
            self.net['discriminator'] = get_discriminator(net_conf)

        self.mse_loss_fun = nn.MSELoss()
        self.l1_loss_fun = nn.L1Loss()

    def forward_G(self, tensors):

        real_A = tensors['real_A']
        real_B = tensors['real_B']

        res = self.net['generator'](real_A)
        fake_B = res['output']
        tensors['fake_B'] = fake_B

        loss_weights = self.net_conf['loss_weights'] # perceptual, pix2pix, ssim, GAN
        loss_perceptual, loss_pix2pix, loss_ssim, loss_GAN = 0, 0, 0, 0

        """ perceptual loss """
        if loss_weights['perceptual'] != 0:
            f_fake_B = self.net['perceptual'](fake_B)
            with torch.no_grad():
                f_real_B = self.net['perceptual'](real_B)

            loss_perceptual = self.mse_loss_fun(f_fake_B, f_real_B.detach())

        """ mse loss"""
        if loss_weights['pix2pix'] != 0:
            loss_pix2pix = self.mse_loss_fun(fake_B, real_B)
            # loss_pix2pix_1 = self.mse_loss_fun(nn.functional.avg_pool2d(fake_B, 3, 3),
            #                                  nn.functional.avg_pool2d(real_B, 3, 3))
            # loss_pix2pix_2 = self.mse_loss_fun(nn.functional.avg_pool2d(fake_B, 9, 9),
            #                                  nn.functional.avg_pool2d(real_B, 9, 9))
            # loss_pix2pix_3 = self.mse_loss_fun(nn.functional.avg_pool2d(fake_B, 27, 27),
            #                                  nn.functional.avg_pool2d(real_B, 27, 27))
            # loss_pix2pix = (loss_pix2pix_1 + loss_pix2pix_2 + loss_pix2pix_3 + loss_pix2pix_4) / 4.0


        """ ssim loss """
        if loss_weights['ssim'] != 0:
            loss_ssim, ssim_map = metrics.SSIM(fake_B, real_B)
            loss_ssim = 1 - loss_ssim

        """ GAN loss """
        if loss_weights['GAN'] != 0:
            if self.net_conf['gan_type'] == 'wgan-gp':
                loss_GAN = self.get_wgan_gp_loss(tensors, loss_type='generator')

            elif self.net_conf['gan_type'] == 'gan':
                loss_GAN = self.get_gan_loss(tensors, loss_type='generator')

            elif self.net_conf['gan_type'] is None:
                loss_GAN = 0

        """ Final loss """
        loss_total = loss_weights['perceptual'] * loss_perceptual + loss_weights['pix2pix'] * loss_pix2pix + loss_weights['ssim'] * loss_ssim + loss_weights['GAN'] * loss_GAN

        # feature_img_real_A = (torch.clamp(real_A_feature[0, 0], -1, 1).unsqueeze(0).detach().cpu().numpy()) / 2.0 * 255.0
        # feature_img_real_B = (torch.clamp(real_B_feature[0, 0], -1, 1).unsqueeze(0).detach().cpu().numpy()) / 2.0 * 255.0
        # feature_img = np.concatenate([feature_img_real_A, feature_img_real_B], axis=2)
        ssim_map = (torch.clamp(ssim_map[0], -1, 1).detach().cpu()) / 2.0 if loss_weights['ssim'] != 0 else None

        # pdb.set_trace()

        state = {'scalar|loss_perceptual': loss_perceptual.item() if loss_weights['perceptual'] != 0 else 0,
                 'scalar|loss_pix2pix': loss_pix2pix.item() if loss_weights['pix2pix'] != 0 else 0,
                 'scalar|loss_ssim': loss_ssim.item() if loss_weights['ssim'] != 0 else 0,
                 'scalar|loss_G_GAN': loss_GAN.item() if loss_weights['GAN'] != 0 else 0,
                 'scalar|loss_G_total': loss_total.item() if type(loss_total) != int else 0,
                 'image|ssim_map': ssim_map if loss_weights['ssim'] != 0 else None}

        return loss_total, state

    def forward_D(self, tensors, phase='train'):
        real_A = tensors['real_A']
        real_B = tensors['real_B']
        fake_B = tensors['fake_B']

        D_fake = self.net['discriminator'].forward(fake_B.detach())
        D_real = self.net['discriminator'].forward(real_B)

        if self.net_conf['gan_type'] == 'wgan-gp':
            loss_D = self.get_wgan_gp_loss(tensors, loss_type='discriminator', phase=phase)

        elif self.net_conf['gan_type'] == 'gan':
            loss_D = self.get_gan_loss(tensors, loss_type='discriminator')

        else:
            raise ValueError

        state = {'scalar|loss_D': loss_D.item()}
        return loss_D, state

    def get_gan_loss(self, tensors, loss_type='generator'):

        if loss_type == 'generator':
            D_fake = self.net['discriminator'](tensors['fake_B'])
            tensors['gan_real_label'].resize_(D_fake.size()).fill_(1)
            loss_GAN = self.l1_loss_fun(D_fake, tensors['gan_real_label'])

        else:
            D_fake = self.net['discriminator'](tensors['fake_B'].detach())
            D_real = self.net['discriminator'](tensors['real_B'])

            tensors['gan_fake_label'].resize_(D_fake.size()).fill_(0)
            tensors['gan_real_label'].resize_(D_fake.size()).fill_(1)

            loss_D1 = self.l1_loss_fun(D_fake, tensors['gan_fake_label'])
            loss_D2 = self.l1_loss_fun(D_real, tensors['gan_real_label'])
            loss_GAN = loss_D1 + loss_D2

        return loss_GAN

    def get_wgan_gp_loss(self, tensors, loss_type='generator', phase='train'):

        def _calc_gradient_penalty(tensors):
            real_B = tensors['real_B'].data
            fake_B = tensors['fake_B'].data

            alpha = random.random()

            mix_B = alpha * real_B + ((1 - alpha) * fake_B)
            mix_B.requires_grad = True

            D_mix_B = self.net['discriminator'](mix_B)

            tensors['gan_real_label'].resize_(D_mix_B.size()).fill_(1)
            gradients = autograd.grad(
                outputs=D_mix_B, inputs=mix_B, grad_outputs=tensors['gan_real_label'],
                create_graph=True, retain_graph=True, only_inputs=True)[0]

            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
            return gradient_penalty


        if loss_type == 'generator':
            D_fake = self.net['discriminator'](tensors['fake_B'])
            loss_GAN = -D_fake.mean()

        else:
            D_fake = self.net['discriminator'](tensors['fake_B'].detach())
            D_real = self.net['discriminator'](tensors['real_B'])

            D_fake = D_fake.mean()
            D_real = D_real.mean()
            loss_GAN = D_fake - D_real

            if phase == 'train': # if it's not train, gradient is required
                gradient_penalty = _calc_gradient_penalty(tensors)
                loss_GAN = loss_GAN + gradient_penalty

        return loss_GAN


