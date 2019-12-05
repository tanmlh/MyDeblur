import pdb
import torch
import torch.nn as nn
from .mobilenet_v2 import MobileNetV2
import functools

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def conv_nxn(num_in, num_out, bias=True, kernel_size=3, stride=1):
    # return nn.Conv2d(num_in, num_out, kernel_size=kernel_size, padding=kernel_size//2, bias=bias, stride=stride)
    return nn.Conv2d(num_in, num_out, kernel_size=kernel_size, padding=kernel_size//2, bias=bias,
                     stride=stride, dilation=5)

class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out, kernel_size=3):
        super().__init__()

        self.block0 = conv_nxn(num_in, num_mid, bias=False, kernel_size=kernel_size)
        self.block1 = conv_nxn(num_mid, num_out, bias=False, kernel_size=kernel_size)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x

class FPNMobileNet(nn.Module):

    def __init__(self, norm_type='instance', output_ch=3, num_filters=64, num_filters_fpn=128,
                 pretrained=False, kernel_size=3):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.
        norm_layer = get_norm_layer(norm_type)
        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer, pretrained=pretrained,
                       kernel_size=kernel_size)

        # The segmentation heads on top of the FPN

        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters, kernel_size=kernel_size)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters, kernel_size=kernel_size)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters, kernel_size=kernel_size)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters, kernel_size=kernel_size)

        self.smooth = nn.Sequential(
            conv_nxn(4 * num_filters, num_filters, kernel_size=kernel_size),
            norm_layer(num_filters),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            conv_nxn(num_filters, num_filters // 2, kernel_size=kernel_size),
            norm_layer(num_filters // 2),
            nn.ReLU(),
        )

        self.final = conv_nxn(num_filters // 2, output_ch, kernel_size=kernel_size)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):

        map0, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.interpolate(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.interpolate(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.interpolate(self.head2(map2), scale_factor=2, mode="nearest")
        map1 = nn.functional.interpolate(self.head1(map1), scale_factor=1, mode="nearest")

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.interpolate(smoothed, scale_factor=2, mode="nearest")
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.interpolate(smoothed, scale_factor=8, mode="nearest")

        final = self.final(smoothed)
        res = torch.tanh(final) + x

        return torch.clamp(res, min=-1, max=1)


class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters=128, pretrained=False, kernel_size=3):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()
        net = MobileNetV2(n_class=1000, kernel_size=kernel_size)

        if pretrained:
            #Load weights into the project directory
            state_dict = torch.load('mobilenetv2.pth.tar') # add map_location='cpu' if no gpu
            net.load_state_dict(state_dict)
        self.features = net.features

        # self.pre_down_sample = conv_nxn(3, 32, kernel_size=kernel_size, stride=4)
        self.pre_down_sample = nn.AvgPool2d(kernel_size=4, stride=4)

        self.enc0 = nn.Sequential(*self.features[0:2])
        self.enc1 = nn.Sequential(*self.features[2:4])
        self.enc2 = nn.Sequential(*self.features[4:7])
        self.enc3 = nn.Sequential(*self.features[7:11])
        self.enc4 = nn.Sequential(*self.features[11:16])

        self.td1 = nn.Sequential(conv_nxn(num_filters, num_filters, kernel_size=kernel_size),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(conv_nxn(num_filters, num_filters, kernel_size=kernel_size),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(conv_nxn(num_filters, num_filters, kernel_size=kernel_size),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))

        self.lateral4 = nn.Conv2d(160, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(32, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(24, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(16, num_filters // 2, kernel_size=1, bias=False)

        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True


    def forward(self, x):

        # Bottom-up pathway, from ResNet
        # pdb.set_trace()
        x = self.pre_down_sample(x)

        enc0 = self.enc0(x)

        enc1 = self.enc1(enc0) # 256

        enc2 = self.enc2(enc1) # 512

        enc3 = self.enc3(enc2) # 1024

        enc4 = self.enc4(enc3) # 2048

        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)

        # Top-down pathway
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.interpolate(map4, scale_factor=2, mode="nearest"))
        map2 = self.td2(lateral2 + nn.functional.interpolate(map3, scale_factor=2, mode="nearest"))
        map1 = self.td3(lateral1 + nn.functional.interpolate(map2, scale_factor=2, mode="nearest"))
        return lateral0, map1, map2, map3, map4

