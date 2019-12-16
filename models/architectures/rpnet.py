import torch
import torch.nn as nn
import torchvision.models as models
from model.net import ConvLayer, UpsampleConvLayer, ResidualBlock, WhiteRebalance, Self_Attention
import torch.nn.functional as F

def normalize(x, mean, std):
    #mean = torch.Tensor((0.485, 0.456, 0.406)).cuda().view(1, 3, 1, 1)
    #std = torch.Tensor((0.229, 0.224, 0.255)).cuda().view(1, 3, 1, 1)
    assert x.size(1) == 3, 'Only support 3-D input'
    mean = torch.Tensor(mean).cuda().view(1, 3, 1, 1)
    std = torch.Tensor(std).cuda().view(1, 3, 1, 1)
    return x.add(-mean.expand_as(x)).div(std.expand_as(x))

def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        m.weight.data.normal_(0.0, 0.05)

        if hasattr(m.bias, 'data'):

            m.bias.data.fill_(0)

    elif classname.find('BatchNorm2d') != -1:

        m.weight.data.normal_(1.0, 0.02)

        m.bias.data.fill_(0)
        
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
                          
        # self.conv8x = nn.Sequential(
        #                   ResidualBlock(512*self.scale_factor, 1024*self.scale_factor, stride = 2, attention = True, nonliner = 'ReLU'),
        #                   )
                          
        # self.conv16x = nn.Sequential(
        #                   ResidualBlock(1024*self.scale_factor, 2048*self.scale_factor, stride = 2, attention = True, nonliner = 'ReLU'),
        #                   )
        
        weights_init(self)
        
        
        
    def forward(self, x):
        res1x = self.conv_input(x)
        res2x = self.conv2x(res1x)
        res4x  = self.conv4x(res2x)
        return res4x
        # res8x = self.conv8x(res4x)
        # res16x = self.conv16x(res8x)
        
        # return res1x, res2x, res4x, res8x, res16x
          
class Generator_Decoder(nn.Module):
  def __init__(self):
      super(Generator_Decoder, self).__init__()
      self.scale_factor = 1
      self.convd16x = nn.Sequential(
                          ResidualBlock(2048*self.scale_factor, 1024*self.scale_factor, stride = 1, attention = False, nonliner = 'ReLU'),
                          )
                          
                          
      self.convd8x = nn.Sequential(
                          ResidualBlock(1024*self.scale_factor, 512*self.scale_factor, stride = 1, attention = False, nonliner = 'ReLU'),
                          )
                          
      self.convd4x = nn.Sequential(
                          ResidualBlock(512*self.scale_factor, 256*self.scale_factor, stride = 1, attention = False, nonliner = 'ReLU'),
                          )
                          
      self.convd2x = nn.Sequential(
                          ConvLayer(256*self.scale_factor, 128*self.scale_factor, kernel_size=3, stride=1),
                          nn.ReLU(),
                          ConvLayer(128*self.scale_factor, 128*self.scale_factor, kernel_size=3, stride=1),
                          nn.ReLU(),
                          nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                          ConvLayer(128*self.scale_factor, 64*self.scale_factor, kernel_size=3, stride=1),
                          nn.ReLU(),
                          ConvLayer(64*self.scale_factor, 64*self.scale_factor, kernel_size=3, stride=1),
                          nn.ReLU(),
                          nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                          )
                         
       
      self.convd1x = nn.Sequential(
                          nn.ReflectionPad2d(3),
                          nn.Conv2d(64*self.scale_factor, 3, kernel_size=7, stride=1),
                          )

      self.Refinenet = self.refinenet(branches = 4)
      
      self.conv_out = nn.Sequential(
                          nn.ReflectionPad2d(0),
                          nn.Conv2d(3*2, 3, kernel_size = 1, stride = 1),
                          )
      
      
      weights_init(self)
  
  def refinenet(self, branches):
      refinenet= nn.ModuleList()
      for i in range(branches):
        if i == 0:
          branche = nn.Sequential(
              nn.ReflectionPad2d(1),
              nn.Conv2d(3, 3, kernel_size = 3, stride = 1, groups = 3),
              )
        else:
          branche = nn.Sequential(
              nn.Upsample(scale_factor = 1.0/(2**(i)), mode = 'bilinear'),
              nn.ReflectionPad2d(1),
              nn.Conv2d(3, 3, kernel_size = 3, stride = 1, groups = 3),
              nn.Upsample(scale_factor = (2**(i)), mode = 'bilinear')
            )
        refinenet.append(branche)
      return refinenet
                          
      
  def forward(self, x, res1x, res2x, res4x, res8x, res16x):
      output = []
      resd8x = (F.interpolate(self.convd16x(res16x), res8x.size()[2:], mode = 'bilinear').add(res8x))
      resd4x = (F.interpolate(self.convd8x(resd8x), res4x.size()[2:], mode = 'bilinear').add(res4x))
      resd2x = (F.interpolate(self.convd4x(resd4x), res2x.size()[2:], mode = 'bilinear').add(res2x))
      resd1x = (self.convd2x(resd2x)).add(res1x)
      out = (self.convd1x(resd1x))
      output.append(x)
      output.append(out)
      '''
      for i in range(len(self.Refinenet)):
        output.append(self.Refinenet[i](output[1]))
      '''
      output = torch.cat((output), dim = 1)
      return self.conv_out(output)
      
class Generator(nn.Module):
  def __init__(self):
      super(Generator, self).__init__()
      
      self.encoder = Generator_Encoder()
      self.decoder = Generator_Decoder()
      
  def forward(self, x):
      res4x_1, res4x_2, res8x, res16x, res32x = self.encoder(x)
      return self.decoder(x, res4x_1, res4x_2, res8x, res16x, res32x)
      
              
class Discrimitor(nn.Module):
  def __init__(self):
      super(Discrimitor, self).__init__()
      
      dim = 64
      
      self.image_to_features = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(3, dim, 7, stride = 2),
        nn.LeakyReLU(0.2),
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, 2*dim, 3, stride = 2),
        nn.InstanceNorm2d(2*dim),
        nn.LeakyReLU(0.2),
        nn.ReflectionPad2d(1),
        nn.Conv2d(2 * dim, 4*dim, 3, stride = 2),
        nn.InstanceNorm2d(4*dim),
        nn.LeakyReLU(0.2),
        nn.ReflectionPad2d(1),
        nn.Conv2d(4 * dim, 8*dim, 3, stride = 2),
        nn.InstanceNorm2d(8*dim),
        nn.LeakyReLU(0.2),
        nn.ReflectionPad2d(1),
        nn.Conv2d(8*dim, 1, kernel_size = 3, stride = 1),
        )
      
      
  def forward(self, x):
      return ((self.image_to_features(x)))
      
      
if __name__ == '__main__':
  input = torch.randn(1, 3, 720, 720)
  netG = Generator()
  output = netG(input)
  print(output.size())
