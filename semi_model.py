
import torch
import torch.nn as nn
import functools
from Encoder import Encoder_Semi
from Decoder import Decoder_Semi


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class Model_Semi_base(nn.Module):
    def __init__(self, input=3,decoder_num=2,channel=32):
        super(Model_Semi_base, self).__init__()
        self.channel = channel
        self.encoder1 = Encoder_Semi(channel=self.channel,input=input)
        self.decoder1 = Decoder_Semi(channel=self.channel)
        weights_init(self.encoder1)
        weights_init(self.decoder1)
    def forward(self, x,y=None):
        if y is not None:
            x = torch.cat([x,y],dim=1)
        encoder_layers = self.encoder1(x)
        out1 = self.decoder1(encoder_layers)
        return out1


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


