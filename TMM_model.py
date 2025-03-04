import torch
import torch.nn as nn
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import GuidedFilter
import numpy as np
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []

        m.append(conv(n_feats, n_feats//4, 1, bias=bias))
        m.append(act)
        m.append(conv(n_feats//4, n_feats//4, 3, bias=bias))
        m.append(act)
        m.append(conv(n_feats // 4, n_feats , 1, bias=bias))
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class DCPDehazeGenerator(nn.Module):
    """Create a DCP Dehaze generator"""
    def __init__(self, win_size=5, r=15, eps=1e-3):
        super(DCPDehazeGenerator, self).__init__()

        self.guided_filter = GuidedFilter(r=r, eps=eps)
        self.neighborhood_size = win_size
        self.omega = 0.95

    def get_dark_channel(self, img, neighborhood_size):
        shape = img.shape
        if len(shape) == 4:
            img_min,_ = torch.min(img, dim=1)

            padSize = np.int(np.floor(neighborhood_size/2))
            if neighborhood_size % 2 == 0:
                pads = [padSize, padSize-1 ,padSize ,padSize-1]
            else:
                pads = [padSize, padSize ,padSize ,padSize]

            img_min = F.pad(img_min, pads, mode='constant', value=1)
            dark_img = -F.max_pool2d(-img_min, kernel_size=neighborhood_size, stride=1)
        else:
            raise NotImplementedError('get_tensor_dark_channel is only for 4-d tensor [N*C*H*W]')

        dark_img = torch.unsqueeze(dark_img, dim=1)
        return dark_img

    def atmospheric_light(self, img, dark_img):
        num,chl,height,width = img.shape
        topNum = np.int(0.01*height*width)

        A = torch.Tensor(num,chl,1,1)
        if img.is_cuda:
            A = A.cuda()

        for num_id in range(num):
            curImg = img[num_id,...]
            curDarkImg = dark_img[num_id,0,...]

            _, indices = curDarkImg.reshape([height*width]).sort(descending=True)
            #curMask = indices < topNum

            for chl_id in range(chl):
                imgSlice = curImg[chl_id,...].reshape([height*width])
                A[num_id,chl_id,0,0] = torch.mean(imgSlice[indices[0:topNum]])

        return A


    def forward(self, x):
        if x.shape[1] > 1:
            guidance = 0.2989 * x[:,0,:,:] + 0.5870 * x[:,1,:,:] + 0.1140 * x[:,2,:,:]           # rgb2gray

        else:
            guidance = x
        # rescale to [0,1]
        guidance = (guidance + 1)/2
        guidance = torch.unsqueeze(guidance, dim=1)
        imgPatch = (x + 1)/2

        num,chl,height,width = imgPatch.shape

        # dark_img and A with the range of [0,1]
        dark_img = self.get_dark_channel(imgPatch, self.neighborhood_size)
        A = self.atmospheric_light(imgPatch, dark_img)

        map_A = A.repeat(1,1,height,width)
        # make sure channel of trans_raw == 1
        trans_raw = 1 - self.omega*self.get_dark_channel(imgPatch/map_A, self.neighborhood_size)

        # get initial results
        T_DCP = self.guided_filter(guidance, trans_raw)
        J_DCP = (imgPatch - map_A)/T_DCP.repeat(1,3,1,1) + map_A

        # import cv2
        # temp = cv2.cvtColor(J_DCP[0].numpy().transpose([1,2,0]), cv2.COLOR_BGR2RGB)
        # cv2.imshow('J_DCP',temp)
        # cv2.imshow('T_DCP',T_DCP[0].numpy().transpose([1,2,0]))
        # cv2.waitKey(0)
        # exit()
        return J_DCP, T_DCP, map_A
        # return J_DCP, T_DCP, torch.squeeze(A)

class Encoder_Semi(nn.Module):
    def __init__(self, channel,input=3,conv=default_conv, res_blokcs=16):
        super(Encoder_Semi, self).__init__()
        self.channels = channel
        self.conv_d1 = nn.Conv2d(input, self.channels, 7, 1, 3)
        self.conv_d2 = nn.Conv2d(self.channels, self.channels*2, 4, 2, 1)
        self.conv_d3 = nn.Conv2d(self.channels*2, self.channels * 4, 4, 2, 1)
        self.conv_d4 = nn.Conv2d(self.channels*4, self.channels * 8, 4, 2, 1)
        self.act = nn.ReLU()
        m_body = [ResBlock(conv, self.channels*8, 3, act=self.act, res_scale=1
                           ) for _ in range(res_blokcs)
                  ]
        self.body = nn.Sequential(*m_body)
        self.relu = nn.ReLU()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.weight0 = nn.Parameter(torch.Tensor([0]))
        # self.weight1 = nn.Parameter(torch.Tensor([0]))
        # self.weight2 = nn.Parameter(torch.Tensor([0]))
        # self.weight3 = nn.Parameter(torch.Tensor([0]))
    def forward(self, x):
        encoder_layer = []
        x = self.conv_d1(x)
        encoder_layer.append(x)
        x = self.conv_d2(self.relu(x))
        encoder_layer.append(x)
        x = self.conv_d3(self.relu(x))
        encoder_layer.append(x)
        x = self.conv_d4(self.relu(x))
        # encoder_layer.append(x)
        out = self.body(self.relu(x))
        out = out+x
        encoder_layer.append(out)
        return encoder_layer
class Upsample_blk(nn.Module):
    def __init__(self,inplanes,outplanes,scale=2,size=None):
        super(Upsample_blk, self).__init__()
        self.up_op = nn.Upsample(size,scale_factor=2,mode='bicubic',align_corners=True)
        self.conv1 = nn.Conv2d(inplanes,outplanes,3,1,1)
        self.conv1X1 = nn.Conv2d(outplanes, outplanes, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        out = self.up_op(x)
        out0 = self.conv1(out)
        out = self.conv1X1(self.relu(out0))
        return out+out0

class Decoder_Semi(nn.Module):
    def  __init__(self,channel):
        super(Decoder_Semi, self).__init__()
        self.channels = channel
        self.conv_t1 = Upsample_blk(self.channels*8,self.channels*4)
        self.conv_t2 = Upsample_blk(self.channels*8,self.channels*2)
        self.conv_t3 = Upsample_blk(self.channels*4,self.channels)
        self.conv_t4 = nn.Conv2d(self.channels*2,3,3,1,1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self,x):
        # y = self.conv_t1(self.relu(in_en+ x[2]))
        y = self.conv_t1(self.relu(x[-1]))
        y = torch.cat([y , x[-2]],dim=1)
        y = self.conv_t2(self.relu(y))
        y =torch.cat( [y , x[-3]],dim=1)
        y = self.conv_t3(self.relu(y))
        y = torch.cat([y, x[0]], dim=1)
        y = self.conv_t4(self.relu(y))
        return self.tanh(y)

class Discriminator(nn.Module):
    def __init__(self,in_channel,ndf=64,scope='D',reuse=None):
        super(Discriminator, self).__init__()
        self.in_channel = in_channel
        self.ndf=ndf
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(self.in_channel, self.ndf, kernel_size=4, stride=2,padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(self.ndf, self.ndf*2, kernel_size=4, stride=2, padding=1))
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(self.ndf*2, self.ndf*4, kernel_size=4, stride=2, padding=1))
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(self.ndf*4, self.ndf*8, kernel_size=4, stride=2, padding=1))
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(self.ndf * 8, 3, kernel_size=4, stride=1, padding=1))
    def forward(self,input):
        layers = []
        # convolved = discrim_conv(self.input, self.ndf, stride=2,use_sn=True)
        convolved = self.conv1(input)
        # convolved = conv2d(self.input,self.ndf,4,4,1,1,use_sn=True)
        rectified = F.leaky_relu(convolved, 0.2)
        layers.append(rectified)
        convolved = self.conv2(rectified)
        rectified = F.leaky_relu(convolved, 0.2)
        layers.append(rectified)
        convolved = self.conv3(rectified)
        rectified = F.leaky_relu(convolved, 0.2)
        layers.append(rectified)
        convolved = self.conv4(rectified)
        rectified = F.leaky_relu(convolved, 0.2)
        layers.append(rectified)
        convolved = self.conv5(rectified)
        # output = F.sigmoid(convolved)
        layers.append(convolved)
        return  convolved

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