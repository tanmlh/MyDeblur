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
        if hasattr(m, 'weight'):
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

        kernel_size = [5, 5, 5, 5]
        dilation = [2, 2, 2, 2]
        stride = [3, 3, 3, 3]
        padding = [4, 4, 4, 4]
        output_padding = [2, 2, 2, 2]

        channels = [64, 128, 256, 512, 1024]
        """
        kernel_size = [9, 7, 5, 3]
        dilation = [1, 1, 1, 1]
        stride = [4, 4, 4, 2]
        padding = [4, 3, 2, 1]
        output_padding = [3, 3, 3, 1]
        channels = [48, 64, 128, 256, 512]
        """

        self.first = [
            nn.ReflectionPad2d(5),
            nn.Conv2d(num_in_chns, channels[0], kernel_size=11, bias=use_bias, dilation=1),
            norm_layer(channels[0]),
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




        self.down_sample_0 = [nn.Conv2d(channels[0], channels[1], kernel_size=kernel_size[0], stride=stride[0],
                                        padding=padding[0], bias=use_bias, dilation=dilation[0]),
                                        norm_layer(channels[1]), nn.ReLU(True)]
        self.down_sample_0 = nn.Sequential(*self.down_sample_0)


        self.down_sample_1 = [nn.Conv2d(channels[1], channels[2], kernel_size=kernel_size[1], stride=stride[1],
                                        padding=padding[1], bias=use_bias, dilation=dilation[1]),
                                        norm_layer(channels[2]), nn.ReLU(True)]
        self.down_sample_1 = nn.Sequential(*self.down_sample_1)

        self.down_sample_2 = [nn.Conv2d(channels[2], channels[3], kernel_size=kernel_size[2], stride=stride[2],
                                        padding=padding[2], bias=use_bias, dilation=dilation[2]),
                                        norm_layer(channels[3]), nn.ReLU(True)]
        self.down_sample_2 = nn.Sequential(*self.down_sample_2)

        self.down_sample_3 = [nn.Conv2d(channels[3], channels[4], kernel_size=kernel_size[3], stride=stride[3],
                                        padding=padding[3], bias=use_bias, dilation=dilation[3]),
                                        norm_layer(channels[4]), nn.ReLU(True)]
        self.down_sample_3 = nn.Sequential(*self.down_sample_3)

        model = []
        for i in range(n_blocks):
            model += [
                ResnetBlock(channels[4], padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]


        self.up_sample_0 = [nn.ConvTranspose2d(channels[4], channels[3], kernel_size=kernel_size[3], stride=stride[3],
                                               padding=padding[3], output_padding=output_padding[3],
                                               bias=use_bias, dilation=dilation[3]),
                            norm_layer(channels[3]),
                            nn.ReLU(True)]
        self.up_sample_0 = nn.Sequential(*self.up_sample_0)

        self.up_sample_1 = [nn.ConvTranspose2d(channels[3], channels[2], kernel_size=kernel_size[2], stride=stride[2],
                                               padding=padding[1], output_padding=output_padding[2],
                                               bias=use_bias, dilation=dilation[2]),
                            norm_layer(channels[2]),
                            nn.ReLU(True)]
        self.up_sample_1 = nn.Sequential(*self.up_sample_1)

        self.up_sample_2 = [nn.ConvTranspose2d(channels[2], channels[1], kernel_size=kernel_size[1], stride=stride[1],
                                               padding=padding[1], output_padding=output_padding[1],
                                               bias=use_bias, dilation=dilation[1]),
                            norm_layer(channels[1]),
                            nn.ReLU(True)]
        self.up_sample_2 = nn.Sequential(*self.up_sample_2)

        self.up_sample_3 = [nn.ConvTranspose2d(channels[1], channels[0], kernel_size=kernel_size[0], stride=stride[0],
                                               padding=padding[0], output_padding=output_padding[0],
                                               bias=use_bias, dilation=dilation[0]),
                            norm_layer(channels[0]),
                            nn.ReLU(True)]
        self.up_sample_3 = nn.Sequential(*self.up_sample_3)


        self.final = [
            nn.ReflectionPad2d(5),
            nn.Conv2d(channels[0]*2, num_out_chns, kernel_size=11, dilation=1),
            nn.Tanh()
        ]
        self.final = nn.Sequential(*self.final)

        self.model = nn.Sequential(*model)

        """
        self.feature_net = Generator_Encoder()
        state = torch.load('./checkpoints/45_NO_Skip.pth')
        self.feature_net.load_state_dict(state['state_dict'], strict=False)
        for param in self.feature_net.parameters():
            param.requires_grad = False
        """

        torch.cuda.empty_cache()

    def forward(self, input):

        # y = self.feature_net(input)
        # y = nn.functional.interpolate(y, size=(input.shape[2]//27, input.shape[3]//27)).detach()

        feature = self.first(input)

        x = self.down_sample_0(feature)
        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)

        # x = torch.cat([x, y], dim=1)
        x = self.model(x)

        x = self.up_sample_0(x)
        x = self.up_sample_1(x)
        x = self.up_sample_2(x)
        x = self.up_sample_3(x)

        output = self.final(torch.cat([x, feature], dim=1))
        # output = nn.functional.interpolate(output, scale_factor=(2, 2))

        if self.learn_residual:
            output = input[:, 0:3, :, :] + output

        output = torch.clamp(output, min=-1, max=1)

        # pdb.set_trace()

        res = {'output': output, 'feature1': None, 'feature2': None}

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

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

    def cuda(self):
        self.conv_block.cuda()



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

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation = 1):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size//2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.instance = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.instance(out)
        return (out)

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels = 64, out_channels = 64, scale = 1, dilation = 1, stride = 1, attention = False, nonliner = 'LeakyReLU'):
        super(ResidualBlock, self).__init__()

        self.Attention = attention

        self.conv1 = ConvLayer(in_channels, in_channels, kernel_size=3, stride=stride, dilation = dilation)
        self.conv2 = ConvLayer(in_channels, out_channels, kernel_size=3, stride=1, dilation = 1)
        if nonliner == 'LeakyReLU':
          self.activation = nn.LeakyReLU(0.2)
        else:
          self.activation = nn.ReLU()

        self.downsample = None
        if in_channels != out_channels or stride !=1:
          self.downsample = nn.Sequential(
                              nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0),
                              nn.InstanceNorm2d(out_channels),
          )
        
        if attention:
          self.linear1 = nn.Linear(out_channels, out_channels//32)
          self.linear2 = nn.Linear(out_channels//32, out_channels)
          self.global_pooling = nn.AdaptiveAvgPool2d((1,1))
        
        #self.dropout = nn.Dropout(0.2)
        
    def attention(self, x):
        N, C, H, W = x.size()
        out = torch.flatten(self.global_pooling(x), 1)
        out = nn.functional.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out)).view(N, C, 1, 1)
    
        return out*x
        
    def forward(self, x):
        residual = x
        if self.downsample is not None:
          residual = self.downsample(residual)
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        if self.Attention:
          out = self.attention(out)
        out = torch.add(out, residual)
        out = self.activation(out)
        #out = self.dropout(out)
        return out


class Generator_Encoder(nn.Module):
    def __init__(self, res_blocks=18):
        super(Generator_Encoder, self).__init__()
        self.scale_factor = 1
        rgb_mean = (0.5204, 0.5167, 0.5129)
        '''
        ResNet = models.resnet34(pretrained = True)
        self.conv_input = nn.Sequential(
                            ResNet.conv1,
                            ResNet.bn1,
                            ResNet.relu,
                            ResNet.maxpool
                            )
        
        self.conv2x = ResNet.layer1
        
        self.conv4x = ResNet.layer2
        
        self.conv8x = ResNet.layer3
       
        self.conv16x = ResNet.layer4
        del ResNet
        torch.cuda.empty_cache()
        '''
        self.conv_input = nn.Sequential(
                          ConvLayer(3, 64*self.scale_factor, kernel_size = 7, stride = 1),
                          nn.ReLU(),
                          #nn.ReflectionPad2d(3),
                          #nn.Conv2d(3, 64*self.scale_factor, kernel_size = 7, stride = 1)
                          )
                          
        self.conv2x = nn.Sequential(
                          ConvLayer(64*self.scale_factor, 64*self.scale_factor, kernel_size=3, stride=1),
                          nn.ReLU(),
                          ConvLayer(64*self.scale_factor, 128*self.scale_factor, kernel_size=3, stride=2),
                          nn.ReLU(),
                          ConvLayer(128*self.scale_factor, 128*self.scale_factor, kernel_size=3, stride=1),
                          nn.ReLU(),
                          ConvLayer(128*self.scale_factor, 256*self.scale_factor, kernel_size=3, stride=2),
                          nn.ReLU(),
                          )
                          
        self.conv4x = nn.Sequential(
                          ResidualBlock(256*self.scale_factor, 512*self.scale_factor, stride = 2, attention = True, nonliner = 'ReLU'),
                          )
                          
        self.conv8x = nn.Sequential(
                          ResidualBlock(512*self.scale_factor, 1024*self.scale_factor, stride = 2, attention = True, nonliner = 'ReLU'),
                          )
                          
        # self.conv16x = nn.Sequential(
        #                   ResidualBlock(1024*self.scale_factor, 2048*self.scale_factor, stride = 2, attention = True, nonliner = 'ReLU'),
        #                   )

    def forward(self, x):
        x = nn.functional.avg_pool2d(x, 3, 3)
        res1x = self.conv_input(x)
        res2x = self.conv2x(res1x)
        res4x  = self.conv4x(res2x)
        res8x = self.conv8x(res4x)
        # res16x = self.conv16x(res8x)
        return res8x
