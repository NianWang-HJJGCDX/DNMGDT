
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from skimage.measure.simple_metrics import compare_psnr
import torch.nn.functional as F
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean(),ssim_map
    else:
        return ssim_map.mean(1).mean(1).mean(1),ssim_map

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        elif gan_mode in ['hinge']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def loss_hinge_gen(self,dis_fake, weight_fake=None):
        loss = -torch.mean(dis_fake)
        return loss

    def loss_hinge_dis(self,dis_fake, dis_real, weight_real=None, weight_fake=None):
        loss_real = torch.mean(F.relu(1. - dis_real))
        loss_fake = torch.mean(F.relu(1. + dis_fake))
        return loss_real, loss_fake

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'higne':
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = 0
        return loss
def create_window(window_size, channel,sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
def contrast_energy(img,window_size=21):
    energys = []
    (_, channel, _, _) = img.size()
    window = create_window(window_size,1,sigma=0.12)
    if img.is_cuda:
        window = window.cuda(img.get_device())
    for i in range(img.shape[0]):
        (R, G, B) = img[i][0], img[i][1], img[i][2]
        rg = torch.abs(R - G)
        yb = torch.abs(0.5 * (R + G) - B)
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        # window = create_window(window_size, 1,sigma=0.12)
        rg = torch.unsqueeze(rg,dim=0)
        rg = torch.unsqueeze(rg, dim=0)
        ce_rg = F.conv2d(rg, window, padding=window_size // 2, groups=1)
        yb = torch.unsqueeze(yb, dim=0)
        yb = torch.unsqueeze(yb, dim=0)
        ce_yb = F.conv2d(yb, window, padding=window_size // 2, groups=1)
        gray = torch.unsqueeze(gray, dim=0)
        gray = torch.unsqueeze(gray, dim=0)
        ce_gray = F.conv2d(gray, window, padding=window_size // 2, groups=1)

        alpha_rg = torch.max(ce_rg)
        ce_rg = alpha_rg*ce_rg/(ce_rg+alpha_rg*0.1)-0.0528
        alpha_yb = torch.max(ce_yb)
        ce_yb = alpha_yb * ce_yb / (ce_yb + alpha_yb * 0.1) - 0.2287
        alpha_gray = torch.max(ce_gray)
        ce_gray = alpha_gray * ce_gray / (ce_gray + alpha_gray * 0.1) - 0.2353
        ce_img = torch.cat([ce_gray,ce_yb,ce_rg],dim=1)
        energys.append(ce_img)
    return energys
def get_colorful(img):
    colorfulness =[]
    # colorfulness = 0
    for i in range(img.shape[0]):
        (R, G, B) = img[i][0], img[i][1], img[i][2]
        rg = torch.abs(R - G)
        yb = torch.abs(0.5 * (R + G) - B)
        (rbMean, rbStd) = (torch.mean(rg), torch.std(rg))
        (ybMean, ybStd) = (torch.mean(yb), torch.std(yb))
        stdRoot = torch.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = torch.sqrt((rbMean ** 2) + (ybMean ** 2))
        colorfulness.append(stdRoot + (0.3 * meanRoot))
    return colorfulness

def haze_evaluate(img1):
    f_s  = contrast_energy(img1)[0].mean()
    f_c = get_colorful(img1)[0]
    if torch.isnan(f_s):
        f_s=0
    if torch.isnan(f_c):
        f_c =0
    weight_h = 10 * f_s + 10 * f_c
    return weight_h
def content_weight(img,window_size=11):
    (_, channel, _, _) = img.size()
    window = create_window(window_size, channel,sigma=2)
    if img.is_cuda:
       window = window.cuda(img.get_device())
    mu1 = F.conv2d(img, window, padding=window_size // 2, groups=channel)
    # mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    # img1 is the reference

    mu1_sq = mu1.pow(2)
    # mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img * img, window, padding=window_size // 2, groups=channel) - mu1_sq
    # sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma1 = torch.sqrt(torch.abs(sigma1_sq))/100
    # weight_map = 1 / (torch.sqrt(torch.abs(sigma1_sq)) + 0.5)
    max_sigma = torch.max(sigma1)
    min_sigma = torch.min(sigma1)
    return (sigma1-min_sigma)/(max_sigma-min_sigma)

def batch_PSNR(img, imclean, data_range=None):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:])
    return (PSNR/Img.shape[0])