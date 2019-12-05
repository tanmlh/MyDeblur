import torch
import torch.nn as nn
# from torch.nn import init
import functools
# from torch.autograd import Variable
import numpy as np
import pdb


###############################################################################
# Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def define_G(num_in_chns, num_out_chns, ngf, backbone_name, norm='batch',
             use_dropout=False, learn_residual=False):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if backbone_name == 'resnet_9blocks':
        netG = ResnetGenerator(num_in_chns, num_out_chns, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                               learn_residual=learn_residual)
    elif backbone_name == 'resnet_6blocks':
        netG = ResnetGenerator(num_in_chns, num_out_chns, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                               learn_residual=learn_residual)
    elif backbone_name == 'unet_128':
        netG = UnetGenerator(num_in_chns, num_out_chns, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             learn_residual=learn_residual)
    elif backbone_name == 'unet_256':
        netG = UnetGenerator(num_in_chns, num_out_chns, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             learn_residual=learn_residual)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % backbone_name)
    netG.apply(weights_init)
    return netG


def define_D(num_in_chns, ndf, backbone_name, num_layers=3, norm='batch', use_sigmoid=False):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if backbone_name == 'basic':
        netD = NLayerDiscriminator(num_in_chns, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif backbone_name == 'n_layers':
        netD = NLayerDiscriminator(num_in_chns, ndf, num_layers, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % backbone_name)

    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(
            self, num_in_chns, num_out_chns, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.num_in_chns = num_in_chns
        self.num_out_chns = num_in_chns
        self.ngf = ngf
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.first = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(num_in_chns, 64, kernel_size=7, bias=use_bias, dilation=1),
            norm_layer(64),
            nn.ReLU(True),
        ]
        self.first = nn.Sequential(*self.first)

        # 下采样
        # for i in range(n_downsampling): # [0,1]
        # 	mult = 2**i
        #
        # 	model += [
        # 		nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
        # 		norm_layer(ngf * mult * 2),
        # 		nn.ReLU(True)
        # 	]

        kernel_size = 5
        dilation = 2
        stride = 2
        padding = 4
        output_padding = 1


        self.down_sample_0 = [nn.Conv2d(64, 64, kernel_size=kernel_size, stride=stride,
                                        padding=padding, bias=use_bias, dilation=dilation),
                                        norm_layer(64), nn.ReLU(True)]
        self.down_sample_0 = nn.Sequential(*self.down_sample_0)


        self.down_sample_1 = [nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride,
                                        padding=padding, bias=use_bias, dilation=dilation),
                                        norm_layer(128), nn.ReLU(True)]
        self.down_sample_1 = nn.Sequential(*self.down_sample_1)

        self.down_sample_2 = [nn.Conv2d(128, 256, kernel_size=kernel_size, stride=stride,
                                        padding=padding, bias=use_bias, dilation=dilation),
                                        norm_layer(256), nn.ReLU(True)]
        self.down_sample_2 = nn.Sequential(*self.down_sample_2)



        """
        model += [
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias, dilation=dilation),
            norm_layer(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias, dilation=dilation),
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias, dilation=dilation),
            norm_layer(256),
            nn.ReLU(True)
        ]
        """

        # 中间的残差网络
        # mult = 2**n_downsampling
        model = []
        for i in range(n_blocks):
            # model += [
            # 	ResnetBlock(
            # 		ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
            # 		use_dropout=use_dropout, use_bias=use_bias)
            # ]
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]

        # 上采样
        # for i in range(n_downsampling):
        # 	mult = 2**(n_downsampling - i)
        #
        # 	model += [
        # 		nn.ConvTranspose2d(
        # 			ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
        # 			padding=1, output_padding=1, bias=use_bias),
        # 		norm_layer(int(ngf * mult / 2)),
        # 		nn.ReLU(True)
        # 	]
        self.up_sample_0 = [nn.ConvTranspose2d(256, 128, kernel_size=kernel_size, stride=stride,
                                               padding=padding, output_padding=output_padding,
                                               bias=use_bias, dilation=dilation),
                            norm_layer(128),
                            nn.ReLU(True)]
        self.up_sample_0 = nn.Sequential(*self.up_sample_0)

        self.up_sample_1 = [nn.ConvTranspose2d(128*2, 64, kernel_size=kernel_size, stride=stride,
                                               padding=padding, output_padding=output_padding, bias=use_bias, dilation=dilation),
                            norm_layer(64),
                            nn.ReLU(True)]
        self.up_sample_1 = nn.Sequential(*self.up_sample_1)

        self.up_sample_2 = [nn.ConvTranspose2d(64*2, 64, kernel_size=kernel_size, stride=stride,
                                               padding=padding, output_padding=output_padding, bias=use_bias, dilation=dilation),
                            norm_layer(64),
                            nn.ReLU(True)]
        self.up_sample_2 = nn.Sequential(*self.up_sample_2)

        """
        model += [
            nn.ConvTranspose2d(256, 128, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=0, bias=use_bias, dilation=dilation),
            norm_layer(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=0, bias=use_bias, dilation=dilation),
            norm_layer(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, stride=stride, padding=padding,
                               output_padding=0, bias=use_bias, dilation=dilation),
            norm_layer(32),
            nn.ReLU(True),
        ]
        """

        self.final = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(128, num_out_chns, kernel_size=7, dilation=1),
            nn.Tanh()
        ]
        self.final = nn.Sequential(*self.final)

        self.model = nn.Sequential(*model)

    def forward(self, input, return_feature=False):

        # pdb.set_trace()
        x0 = self.first(input)

        x1 = self.down_sample_0(x0)
        x2 = self.down_sample_1(x1)
        x3 = self.down_sample_2(x2)

        x4 = self.model(x3)
        if return_feature:
            return {'feature1': x0, 'feature2': x4}

        x5 = self.up_sample_0(x4)
        x6 = self.up_sample_1(torch.cat([x5, x2], dim=1))
        x7 = self.up_sample_2(torch.cat([x6, x1], dim=1))

        output = self.final(torch.cat([x7, x0], dim=1))

        if self.learn_residual:
            output = input + output

        output = torch.clamp(output, min=-1, max=1)

        res = {'output': output, 'feature1': x0, 'feature2': x4}

        return res


# Define a resnet block
class ResnetBlock(nn.Module):

	def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		super(ResnetBlock, self).__init__()

		padAndConv = {
			'reflect': [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
			'replicate': [
                nn.ReplicationPad2d(1),
                nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
			'zero': [
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
		}

		try:
			blocks = padAndConv[padding_type] + [
				norm_layer(dim),
				nn.ReLU(True)
            ] + [
				nn.Dropout(0.5)
			] if use_dropout else [] + padAndConv[padding_type] + [
				norm_layer(dim)
			]
		except:
			raise NotImplementedError('padding [%s] is not implemented' % padding_type)

		self.conv_block = nn.Sequential(*blocks)

		# self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
		# def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
		#     padAndConv = {
		#         'reflect': [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
		#         'replicate': [nn.ReplicationPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
		#         'zero': [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
		#     }
		#     try:
		#         blocks = [
		#             padAndConv[padding_type],
		#
		#             norm_layer(dim),
		#             nn.ReLU(True),
		#             nn.Dropout(0.5) if use_dropout else None,
		#
		#             padAndConv[padding_type],
		#
		#             norm_layer(dim)
		#         ]
		#     except:
		#         raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		#
		#     return nn.Sequential(*blocks)

		# blocks = []
		# if padding_type == 'reflect':
		# 	blocks += [nn.ReflectionPad2d(1),  nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
		# elif padding_type == 'replicate':
		# 	blocks += [nn.ReplicationPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
		# elif padding_type == 'zero':
		# 	blocks += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
		# else:
		# 	raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		#
		# blocks += [
		# 	norm_layer(dim),
		# 	nn.ReLU(True),
		# 	nn.Dropout(0.5) if use_dropout else None
		# ]
		#
		# if padding_type == 'reflect':
		# 	blocks += [nn.ReflectionPad2d(1),  nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
		# elif padding_type == 'replicate':
		# 	blocks += [nn.ReplicationPad2d(1), nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)]
		# elif padding_type == 'zero':
		# 	blocks += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
		# else:
		# 	raise NotImplementedError('padding [%s] is not implemented' % padding_type)
		#
		# blocks += [
		# 	norm_layer(dim)
		# ]
		#
		# return nn.Sequential(*blocks)

	def forward(self, x):
		out = x + self.conv_block(x)
		return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(
            self, num_in_chns, num_out_chns, num_downs, ngf=64, norm_layer=nn.BatchNorm2d,
            use_dropout=False, learn_residual=False):
        super(UnetGenerator, self).__init__()
        self.learn_residual = learn_residual
        # currently support only num_in_chns == num_out_chns
        assert (num_in_chns == num_out_chns)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(num_out_chns, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        output = self.model(input)
        if self.learn_residual:
            output = input + output
            output = torch.clamp(output, min=-1, max=1)
        return output


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(
            self, outer_nc, inner_nc, submodule=None,
            outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        dConv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        dRelu = nn.LeakyReLU(0.2, True)
        dNorm = norm_layer(inner_nc)
        uRelu = nn.ReLU(True)
        uNorm = norm_layer(outer_nc)

        if outermost:
            uConv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            dModel = [dConv]
            uModel = [uRelu, uConv, nn.Tanh()]
            model = [
                dModel,
                submodule,
                uModel
            ]
        # model = [
        # 	# Down
        # 	nn.Conv2d( outer_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
        #
        # 	submodule,
        # 	# Up
        # 	nn.ReLU(True),
        # 	nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1),
        # 	nn.Tanh()
        # ]
        elif innermost:
            uConv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel = [dRelu, dConv]
            uModel = [uRelu, uConv, uNorm]
            model = [
                dModel,
                uModel
            ]
        # model = [
        # 	# down
        # 	nn.LeakyReLU(0.2, True),
        # 	# up
        # 	nn.ReLU(True),
        # 	nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
        # 	norm_layer(outer_nc)
        # ]
        else:
            uConv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            dModel = [dRelu, dConv, dNorm]
            uModel = [uRelu, uConv, uNorm]

            model = [
                dModel,
                submodule,
                uModel
            ]
            model += [nn.Dropout(0.5)] if use_dropout else []

        # if use_dropout:
        # 	model = down + [submodule] + up + [nn.Dropout(0.5)]
        # else:
        # 	model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, num_in_chns, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(num_in_chns, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
