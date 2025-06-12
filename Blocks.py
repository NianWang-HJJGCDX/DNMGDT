
import math
import torch.nn.functional as F
import  numpy as np

import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

# def _make_scratch(in_shape, out_shape, groups=1, expand=False):
#     scratch = nn.Module()
#
#     out_shape1 = out_shape
#     out_shape2 = out_shape
#     out_shape3 = out_shape
#     out_shape4 = out_shape
#
#     if expand==True:
#         out_shape1 = out_shape
#         out_shape2 = out_shape*2
#         out_shape3 = out_shape*4
#         out_shape4 = out_shape*8
#
#     scratch.layer1_rn = nn.Sequential(
#         nn.ReflectionPad2d(1),
#         nn.Conv2d(
#         in_shape[0], out_shape1, kernel_size=3, stride=1, padding=0, bias=False, groups=groups
#     ))
#     scratch.layer2_rn = nn.Sequential(
#         nn.ReflectionPad2d(1),
#         nn.Conv2d(
#         in_shape[1], out_shape2, kernel_size=3, stride=1, padding=0, bias=False, groups=groups
#     ))
#     scratch.layer3_rn = nn.Sequential(
#         nn.ReflectionPad2d(1),
#         nn.Conv2d(
#         in_shape[2], out_shape3, kernel_size=3, stride=1, padding=0, bias=False, groups=groups
#     ))
#     scratch.layer4_rn = nn.Sequential(
#         nn.ReflectionPad2d(1),
#         nn.Conv2d(
#         in_shape[3], out_shape4, kernel_size=3, stride=1, padding=0, bias=False, groups=groups
#     ))
#
#     return scratch
#
#
# class ResidualConvUnit_custom(nn.Module):
#     """Residual convolution module.
#     """
#
#     def __init__(self, features, activation, bn):
#         """Init.
#
#         Args:
#             features (int): number of features
#         """
#         super().__init__()
#
#         self.bn = bn
#
#         self.groups = 1
#
#         self.conv1 = nn.Conv2d(
#             features, features, kernel_size=3, stride=1, padding=0, bias=True, groups=self.groups
#         )
#
#         self.conv2 = nn.Conv2d(
#             features, features, kernel_size=3, stride=1, padding=0, bias=True, groups=self.groups
#         )
#
#         self.pad1 = nn.ReflectionPad2d(1)
#
#         if self.bn == True:
#             self.bn1 = nn.BatchNorm2d(features)
#             self.bn2 = nn.BatchNorm2d(features)
#
#         self.activation = activation
#
#         self.skip_add = nn.quantized.FloatFunctional()
#
#     def forward(self, x):
#         """Forward pass.
#
#         Args:
#             x (tensor): input
#
#         Returns:
#             tensor: output
#         """
#
#         out = self.activation(x)
#         out = self.pad1(out)
#         out = self.conv1(out)
#         if self.bn == True:
#             out = self.bn1(out)
#
#         out = self.activation(out)
#         out = self.pad1(out)
#         out = self.conv2(out)
#         if self.bn == True:
#             out = self.bn2(out)
#
#         if self.groups > 1:
#             out = self.conv_merge(out)
#
#         return self.skip_add.add(out, x)
# class FeatureFusionBlock_custom(nn.Module):
#     """Feature fusion block.
#     """
#
#     def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
#         """Init.
#
#         Args:
#             features (int): number of features
#         """
#         super(FeatureFusionBlock_custom, self).__init__()
#
#         self.deconv = deconv
#         self.align_corners = align_corners
#
#         self.groups = 1
#
#         self.expand = expand
#         out_features = features
#         if self.expand == True:
#             out_features = features // 2
#
#         self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
#
#         self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
#         self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
#
#         self.skip_add = nn.quantized.FloatFunctional()
#
#     def forward(self, *xs):
#         """Forward pass.
#
#         Returns:
#             tensor: output
#         """
#         output = xs[0]
#
#         if len(xs) == 2:
#             res = self.resConfUnit1(xs[1])
#             output = self.skip_add.add(output, res)
#             # output += res
#
#         output = self.resConfUnit2(output)
#
#         output = nn.functional.interpolate(
#             output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
#         )
#
#         output = self.out_conv(output)
#
#         return output
# class Interpolate(nn.Module):
#     """Interpolation module.
#     """
#
#     def __init__(self, scale_factor, mode):
#         """Init.
#
#         Args:
#             scale_factor (float): scaling
#             mode (str): interpolation mode
#         """
#         super(Interpolate, self).__init__()
#
#         self.interp = nn.functional.interpolate
#         self.scale_factor = scale_factor
#         self.mode = mode
#
#     def forward(self, x):
#         """Forward pass.
#
#         Args:
#             x (tensor): input
#
#         Returns:
#             tensor: interpolated data
#         """
#
#         x = self.interp(
#             x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False
#         )
#
#         return x
#
#
# class LayerNorm(nn.Module):
#     r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
#     """
#
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError
#         self.normalized_shape = (normalized_shape,)
#
#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x
#
#
# class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
#     def __init__(self, dim_in, dim_out, x=8, y=8):
#         super().__init__()
#
#         c_dim_in = dim_in // 4
#         k_size = 3
#         pad = (k_size - 1) // 2
#
#         self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
#         nn.init.ones_(self.params_xy)
#         self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
#                                      nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))
#
#         self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
#         nn.init.ones_(self.params_zx)
#         self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
#                                      nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))
#
#         self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
#         nn.init.ones_(self.params_zy)
#         self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
#                                      nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))
#
#         self.dw = nn.Sequential(
#             nn.Conv2d(c_dim_in, c_dim_in, 1),
#             nn.GELU(),
#             nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
#         )
#
#         self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
#         self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
#
#         self.ldw = nn.Sequential(
#             nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
#             nn.GELU(),
#             nn.Conv2d(dim_in, dim_out, 1),
#         )
#
#     def forward(self, x):
#         x = self.norm1(x)
#         x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
#         B, C, H, W = x1.size()
#         # ----------xy----------#
#         params_xy = self.params_xy
#         x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4], mode='bilinear', align_corners=True))
#         # ----------zx----------#
#         x2 = x2.permute(0, 3, 1, 2)
#         params_zx = self.params_zx
#         x2 = x2 * self.conv_zx(
#             F.interpolate(params_zx, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
#         x2 = x2.permute(0, 2, 3, 1)
#         # ----------zy----------#
#         x3 = x3.permute(0, 2, 1, 3)
#         params_zy = self.params_zy
#         x3 = x3 * self.conv_zy(
#             F.interpolate(params_zy, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
#         x3 = x3.permute(0, 2, 1, 3)
#         # ----------dw----------#
#         x4 = self.dw(x4)
#         # ----------concat----------#
#         x = torch.cat([x1, x2, x3, x4], dim=1)
#         # ----------ldw----------#
#         x = self.norm2(x)
#         x = self.ldw(x)
#         return x
# #From the egeunet
# class group_aggregation_bridge(nn.Module):
#     def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1,2,5,7]):
#         super().__init__()
#         self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
#         group_size = dim_xl // 2
#         self.g0 = nn.Sequential(
#             LayerNorm(normalized_shape=group_size, data_format='channels_first'),
#             nn.Conv2d(group_size , group_size , kernel_size=3, stride=1,
#                       padding=(k_size+(k_size-1)*(d_list[0]-1))//2,
#                       dilation=d_list[0], groups=group_size )
#         )
#         self.g1 = nn.Sequential(
#             LayerNorm(normalized_shape=group_size, data_format='channels_first'),
#             nn.Conv2d(group_size , group_size , kernel_size=3, stride=1,
#                       padding=(k_size+(k_size-1)*(d_list[1]-1))//2,
#                       dilation=d_list[1], groups=group_size )
#         )
#         self.g2 = nn.Sequential(
#             LayerNorm(normalized_shape=group_size, data_format='channels_first'),
#             nn.Conv2d(group_size , group_size , kernel_size=3, stride=1,
#                       padding=(k_size+(k_size-1)*(d_list[2]-1))//2,
#                       dilation=d_list[2], groups=group_size )
#         )
#         self.g3 = nn.Sequential(
#             LayerNorm(normalized_shape=group_size, data_format='channels_first'),
#             nn.Conv2d(group_size , group_size , kernel_size=3, stride=1,
#                       padding=(k_size+(k_size-1)*(d_list[3]-1))//2,
#                       dilation=d_list[3], groups=group_size )
#         )
#         self.tail_conv = nn.Sequential(
#             LayerNorm(normalized_shape=dim_xl * 2 , data_format='channels_first'),
#             nn.Conv2d(dim_xl * 2 , dim_xl, 1)
#         )
#     def forward(self, xh, xl):
#         xh = self.pre_project(xh)
#         xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
#         xh = torch.chunk(xh, 4, dim=1)
#         xl = torch.chunk(xl, 4, dim=1)
#         x0 = self.g0(torch.cat((xh[0], xl[0]), dim=1))
#         x1 = self.g1(torch.cat((xh[1], xl[1]), dim=1))
#         x2 = self.g2(torch.cat((xh[2], xl[2]), dim=1))
#         x3 = self.g3(torch.cat((xh[3], xl[3]), dim=1))
#         x = torch.cat((x0,x1,x2,x3), dim=1)
#         x = self.tail_conv(x)
#         return x
#
# class Grouped_multi_scale_Hadamard_Product_Attention(nn.Module):
#     def __init__(self, dim_in, dim_out, s_list=[3,5,7,9]):
#         super().__init__()
#
#         c_dim_in = dim_in // 4
#         k_size = 3
#         pad = (k_size - 1) // 2
#
#         self.params_xy_1 = nn.Parameter(torch.Tensor(1, c_dim_in, s_list[0], s_list[0]), requires_grad=True)
#         nn.init.ones_(self.params_xy_1)
#         self.conv_1 = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
#                                      nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))
#
#         self.params_xy_2 = nn.Parameter(torch.Tensor(1, c_dim_in, s_list[1], s_list[1]), requires_grad=True)
#         nn.init.ones_(self.params_xy_2)
#         self.conv_2 = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
#                                      nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))
#
#         self.params_xy_3 = nn.Parameter(torch.Tensor(1, c_dim_in, s_list[2], s_list[2]), requires_grad=True)
#         nn.init.ones_(self.params_xy_3)
#         self.conv_3 = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
#                                      nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))
#
#         self.params_xy_4 = nn.Parameter(torch.Tensor(1, c_dim_in, s_list[3], s_list[3]), requires_grad=True)
#         nn.init.ones_(self.params_xy_4)
#         self.conv_4 = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
#                                     nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))
#
#
#
#         self.ldw = nn.Sequential(
#             nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
#             nn.Sigmoid(),
#         )
#
#         w = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
#         w = torch.nn.Parameter(w, requires_grad=True)
#         self.w = w
#     def forward(self, xin1,xin2):
#         x = xin1+xin2
#         x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
#         x11, x12, x13, x14 = torch.chunk(xin1, 4, dim=1)
#         x21, x22, x23, x24 = torch.chunk(xin2, 4, dim=1)
#         B, C, H, W = x1.size()
#         # ----------3----------#
#         params_1 = self.params_xy_1
#         x1 = x1 * self.conv_1(F.interpolate(params_1, size=x1.shape[2:4], mode='bilinear', align_corners=True))
#         x1_a = F.sigmoid(x1)
#         # ----------5----------#
#
#         params_2 = self.params_xy_2
#         x2 = x2 * self.conv_2(F.interpolate(params_2, size=x2.shape[2:4], mode='bilinear', align_corners=True))
#         x2_a = F.sigmoid(x2)
#         # ----------7----------#
#
#         params_3 = self.params_xy_3
#         x3 = x3 * self.conv_3(
#             F.interpolate(params_3, size=x3.shape[2:4], mode='bilinear', align_corners=True))
#         x3_a = F.sigmoid(x3)
#         # ----------9----------#
#         params_4 = self.params_xy_4
#         x4 = x4 * self.conv_4(
#             F.interpolate(params_4, size=x4.shape[2:4], mode='bilinear', align_corners=True))
#         x4_a = F.sigmoid(x4)
#         # ----------dw----------#
#         x5_a = self.ldw(x)
#         # ----------concat----------#
#         x = torch.cat([x1_a*x11+(1-x1_a)*x21, x2_a*x12+(1-x2_a)*x22, x3_a*x13+(1-x3_a)*x23, x4_a*x14+(1-x4_a)*x24], dim=1)
#         x_w = x5_a*xin1+(1-x5_a)*xin2
#         # ----------ldw----------#
#         self.w = F.sigmoid(self.w)
#         x =  self.w*x_w+(1- self.w)*x
#         return x
#
# class GlobalLocalFilter(nn.Module):
#     def __init__(self, dim, h=14, w=8):
#         super().__init__()
#         self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
#         self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
#         trunc_normal_(self.complex_weight, std=.02)
#         self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
#         self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
#
#     def forward(self, x):
#         x = self.pre_norm(x)
#         x1, x2 = torch.chunk(x, 2, dim=1)
#         x1 = self.dw(x1)
#
#         x2 = x2.to(torch.float32)
#         B, C, a, b = x2.shape
#         x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')
#
#         weight = self.complex_weight
#         if not weight.shape[1:3] == x2.shape[2:4]:
#             weight = F.interpolate(weight.permute(3,0,1,2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1,2,3,0)
#
#         weight = torch.view_as_complex(weight.contiguous())
#
#         x2 = x2 * weight
#         x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')
#
#         x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
#         x = self.post_norm(x)
#         return x
#
# def _make_pretrained_efficientnet_lite3(use_pretrained, exportable=False):
#     efficientnet = torch.hub.load(
#         "rwightman/gen-efficientnet-pytorch",
#         "tf_efficientnet_lite3",
#         pretrained=use_pretrained,
#         exportable=exportable
#     )
#     return _make_efficientnet_backbone(efficientnet)
# def _make_efficientnet_backbone(effnet):
#     pretrained = nn.Module()
#
#     pretrained.layer1 = nn.Sequential(
#         effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
#     )
#     pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
#     pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
#     pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])
#
#     return pretrained
def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Sequential(
        nn.ReflectionPad2d((kernel_size//2)),
        nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=0,stride=stride, bias=bias),)
# class depthwise_separable_conv(nn.Module):
#     def __init__(self, nin, nout):
#         super().__init__()
#         self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
#         self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)
#     def forward(self, x):
#         out = self.depthwise(x)
#         out = self.pointwise(out)
#         return out
#
#
# class DepthWiseConv2d(nn.Module):
#     def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
#         super().__init__()
#
#         self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=(kernel_size-1)//2,
#                                stride=stride, dilation=dilation, groups=dim_in)
#         self.norm_layer = nn.GroupNorm(4, dim_in)
#         self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)
#
#     def forward(self, x):
#         return self.conv2(self.norm_layer(self.conv1(x)))

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
# class ResBlock_new(nn.Module):8952-good
#     def __init__(
#         self, conv, n_feats, kernel_size,
#         bias=True, bn=False, act=nn.PReLU(), res_scale=1,height=None,width=None):
#
#         super(ResBlock_new, self).__init__()
#         self.conv1 = conv(n_feats, n_feats//4, 1, bias=bias)
#         self.act = act
#         self.conv2 = conv(n_feats//4, n_feats//4, 3, bias=bias)
#         self.conv3 = conv(n_feats // 4, n_feats , 1, bias=bias)
#         self.res_scale = res_scale
#         self.conv_fuse = conv(n_feats // 2, n_feats//4 , 1, bias=bias)
# #self.coefficient = nn.Parameter(torch.Tensor(np.ones(1, n_feats//4, 1, 1)), requires_grad=attention)
#     def forward(self, x,pre_out=None):
#         out1 = self.act(self.conv1(x))
#         out2 = self.conv2(out1)
#         if pre_out is not None:
#             out2 = self.conv_fuse(torch.cat((out2,pre_out),dim=1))
#         out3 = self.conv3(self.act(out2))
#         res = x*self.res_scale+out3
#
#         return res,out2

# class ResBlock_new(nn.Module):
#     def __init__(
#         self, conv, n_feats, kernel_size,
#         bias=True, bn=False, act=nn.PReLU(), res_scale=1,height=None,width=None):
#
#         super(ResBlock_new, self).__init__()
#         self.conv1 = conv(n_feats, n_feats//4, 1, bias=bias)
#         self.act = act
#         self.conv2 = conv(n_feats//4, n_feats//4, 3, bias=bias)
#         self.conv3 = conv(n_feats // 4, n_feats , 1, bias=bias)
#         self.res_scale = res_scale
#         self.conv_fuse = conv(n_feats // 2, n_feats//4 , 3, bias=bias)
# #self.coefficient = nn.Parameter(torch.Tensor(np.ones(1, n_feats//4, 1, 1)), requires_grad=attention)
#     def forward(self, x,pre_out=None):
#         out1 = self.act(self.conv1(x))
#         out2 = self.conv2(out1)
#         if pre_out is not None:
#             out2 = self.conv_fuse(torch.cat((out2,pre_out),dim=1))
#         out3 = self.conv3(self.act(out2))
#         res = x*self.res_scale+out3
#
#         return res,out2
#
# class ResBlock_new2(nn.Module):
#     def __init__(
#         self, conv, n_feats, kernel_size,
#         bias=True, bn=False, act=nn.PReLU(), res_scale=1,height=None,width=None):
#
#         super(ResBlock_new2, self).__init__()
#         self.conv1 = conv(n_feats, n_feats//4, 1, bias=bias)
#         self.act = act
#         self.conv2 = conv(n_feats//4, n_feats//4, 3, bias=bias)
#         self.conv3 = conv(n_feats // 4, n_feats , 1, bias=bias)
#         self.res_scale = res_scale
#         self.fuse = C_CBAM(n_feats//4,n_feats//4)
#         self.conv_fuse1 = conv(n_feats // 2, n_feats//4 , 1, bias=bias)
#         # self.conv_fuse2 = conv(n_feats*3 // 2, n_feats // 4, 1, bias=bias)
# #self.coefficient = nn.Parameter(torch.Tensor(np.ones(1, n_feats//4, 1, 1)), requires_grad=attention)
#     def forward(self, x,pre_out=None,high_out=None):
#         out1 = self.act(self.conv1(x))
#         out2 = self.conv2(out1)
#         if pre_out is not None:
#             if high_out is not None:
#                 temp = self.conv_fuse1(self.act(torch.cat((high_out,pre_out),dim=1)))
#                 out2 = self.fuse(temp, out2)
#                 # temp = self.conv_fuse2(self.act(torch.cat((out2, pre_out,high_out), dim=1)))
#             else:
#                 out2 = self.fuse(pre_out,out2)
#                 # temp = self.conv_fuse1(self.act(torch.cat((out2,pre_out),dim=1)))
#         # out2 = self.fuse(out2)
#         out3 = self.conv3(out2)
#         res = x*self.res_scale+out3
#
#         return res,out2
# class ResBlock_new3(nn.Module):
#     def __init__(self, conv, n_feats, kernel_size,
#         bias=True, bn=False, act=nn.PReLU(), res_scale=1,height=None,width=None):
#
#         super(ResBlock_new3, self).__init__()
#         self.conv1 = conv(n_feats, n_feats//4, 1, bias=bias)
#         self.act = act
#         self.conv2 = conv(n_feats//4, n_feats//4, 3, bias=bias)
#         self.conv3 = conv(n_feats // 4, n_feats , 1, bias=bias)
#         self.res_scale = res_scale
#         # self.fuse =C_CBAM(n_feats//4,n_feats//4)
#         self.conv_fuse1 = conv(n_feats // 2, n_feats//4 , 1, bias=bias)
#         self.conv_fuse2 = conv(n_feats*3 // 4, n_feats // 4, 1, bias=bias)
#         self.fuse = conv(n_feats // 4, n_feats // 4, 3, bias=bias)
# #self.coefficient = nn.Parameter(torch.Tensor(np.ones(1, n_feats//4, 1, 1)), requires_grad=attention)
#     def forward(self, x,pre_out=None,high_out=None):
#         out1 = self.act(self.conv1(x))
#         out2 = self.conv2(out1)
#         if pre_out is not None:
#             if high_out is not None:
#                 temp = self.conv_fuse2(self.act(torch.cat((high_out,pre_out,out2),dim=1)))
#                 out2 = self.fuse(temp)
#                 # temp = self.conv_fuse2(self.act(torch.cat((out2, pre_out,high_out), dim=1)))
#             else:
#                 temp = self.conv_fuse1(self.act(torch.cat((pre_out, out2), dim=1)))
#                 out2 = self.fuse(temp)
#                 # temp = self.conv_fuse1(self.act(torch.cat((out2,pre_out),dim=1)))
#         # out2 = self.fuse(out2)
#         out3 = self.conv3(self.act(out2))
#         res = x*self.res_scale+out3
#
#         return res,out2
#
# class ResBlock_new3A(nn.Module):
#     def __init__(
#         self, conv, n_feats, kernel_size,
#         bias=True, bn=False, act=nn.PReLU(), res_scale=1,height=None,width=None):
#
#         super(ResBlock_new3A, self).__init__()
#         self.conv1 = conv(n_feats, n_feats//4, 1, bias=bias)
#         self.act = act
#         self.conv2 = depthwise_separable_conv(n_feats//4, n_feats//4)
#         self.conv3 = conv(n_feats // 4, n_feats , 1, bias=bias)
#         self.res_scale = res_scale
#         # self.fuse =C_CBAM(n_feats//4,n_feats//4)
#         self.conv_fuse1 = conv(n_feats // 2, n_feats//4 , 1, bias=bias)
#         self.conv_fuse2 = conv(n_feats*3 // 4, n_feats // 4, 1, bias=bias)
#         self.fuse = depthwise_separable_conv(n_feats // 4, n_feats // 4)
# #self.coefficient = nn.Parameter(torch.Tensor(np.ones(1, n_feats//4, 1, 1)), requires_grad=attention)
#     def forward(self, x,pre_out=None,high_out=None):
#         out1 = self.act(self.conv1(x))
#         out2 = self.conv2(out1)
#         if pre_out is not None:
#             if high_out is not None:
#                 temp = self.conv_fuse2(self.act(torch.cat((high_out,pre_out,out2),dim=1)))
#                 out2 = self.fuse(temp)
#                 # temp = self.conv_fuse2(self.act(torch.cat((out2, pre_out,high_out), dim=1)))
#             else:
#                 temp = self.conv_fuse1(self.act(torch.cat((pre_out, out2), dim=1)))
#                 out2 = self.fuse(temp)
#                 # temp = self.conv_fuse1(self.act(torch.cat((out2,pre_out),dim=1)))
#         # out2 = self.fuse(out2)
#         out3 = self.conv3(self.act(out2))
#         res = x*self.res_scale+out3
#
#         return res,out2
# class ResBlock_new3AB(nn.Module):
#     def __init__(
#         self, conv, n_feats, kernel_size,
#         bias=True, bn=False, act=nn.PReLU(), res_scale=1,height=None,width=None):
#
#         super(ResBlock_new3AB, self).__init__()
#         # self.conv1 = conv(n_feats, n_feats//4, 1, bias=bias)
#         self.act = act
#         self.conv2 = depthwise_separable_conv(n_feats, n_feats)
#         # self.conv3 = conv(n_feats // 4, n_feats , 1, bias=bias)
#         self.res_scale = res_scale
#         # self.fuse =C_CBAM(n_feats//4,n_feats//4)
#         self.conv_fuse1 = ECAAttention_Ex()
#         # self.conv_fuse1 =conv(n_feats *2, n_feats , 1, bias=bias)
#         # self.conv_fuse2 = conv(n_feats*3 , n_feats , 1, bias=bias)
#         self.conv_fuse2 = ECAAttention_Ex()
#         self.fuse = depthwise_separable_conv(n_feats , n_feats)
# #self.coefficient = nn.Parameter(torch.Tensor(np.ones(1, n_feats//4, 1, 1)), requires_grad=attention)
#     def forward(self, x,pre_out=None,high_out=None):
#         out2 = self.act(self.conv2(x))
#         # out2 = self.conv2(out1)
#         if pre_out is not None:
#             if high_out is not None:
#                 out2 = self.conv_fuse2(torch.cat((high_out,pre_out,out2),dim=1))
#                 # out2 = self.fuse(temp)
#                 # temp = self.conv_fuse2(self.act(torch.cat((out2, pre_out,high_out), dim=1)))
#             else:
#                 out2 = self.conv_fuse1(torch.cat((pre_out, out2), dim=1))
#                 # out2 = self.fuse(temp)
#                 # temp = self.conv_fuse1(self.act(torch.cat((out2,pre_out),dim=1)))
#         out3 = self.fuse(out2)
#         # out3 = self.conv3(out2)
#         res = x*self.res_scale+out3
#
#         return res,out2
# #test on 2021-06-25
# # class ResBlock_new3(nn.Module):
# #     def __init__(
# #             self, conv, n_feats, kernel_size,
# #             bias=True, bn=False, act=nn.PReLU(), res_scale=1, height=None, width=None):
# #
# #         super(ResBlock_new3, self).__init__()
# #         self.conv1 = conv(n_feats, n_feats // 4, 1, bias=bias)
# #         self.act = act
# #         self.conv2 = conv(n_feats // 4, n_feats // 4, 3, bias=bias)
# #         self.conv3 = conv(n_feats // 4, n_feats, 1, bias=bias)
# #         self.res_scale = res_scale
# #         # self.fuse =C_CBAM(n_feats//4,n_feats//4)
# #         self.conv_fuse1 = conv(n_feats // 2, n_feats // 4, 1, bias=bias)
# #         self.conv_fuse2 = conv(n_feats * 3 // 4, n_feats // 4, 1, bias=bias)
# #         self.fuse = conv(n_feats // 4, n_feats // 4, 3, bias=bias)
# #
# #     # self.coefficient = nn.Parameter(torch.Tensor(np.ones(1, n_feats//4, 1, 1)), requires_grad=attention)
# #     def forward(self, x, pre_out=None, high_out=None):
# #         out1 = self.act(self.conv1(x))
# #
# #         if pre_out is not None:
# #             if high_out is not None:
# #                 temp = self.conv_fuse2(self.act(torch.cat((high_out, pre_out, out1), dim=1)))
# #                 out2 = self.fuse(temp)
# #                 out2 = self.conv2(out2)
# #                 # temp = self.conv_fuse2(self.act(torch.cat((out2, pre_out,high_out), dim=1)))
# #             else:
# #                 temp = self.conv_fuse1(self.act(torch.cat((pre_out, out1), dim=1)))
# #                 out2 = self.fuse(temp)
# #                 out2 = self.conv2(out2)
# #                 # temp = self.conv_fuse1(self.act(torch.cat((out2,pre_out),dim=1)))
# #         # out2 = self.fuse(out2)
# #         else:
# #             out2 = self.conv2(out1)
# #         out3 = self.conv3(self.act(out2))
# #         res = x * self.res_scale + out3
# #
# #         return res, out2
#
# class GroupBlock(nn.Module):
#     def __init__(self, conv,  channel=64,height=None, width=None):
#         super(GroupBlock, self).__init__()
#         self.channels = channel
#         self.res_blk0 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
#         self.res_blk1 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
#         self.res_blk2 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
#         self.res_blk3 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
#
#         self.res_blk4 = ResBlock_new3(default_conv, self.channels * 4, 3, act=nn.ReLU())
#         self.res_blk5 = ResBlock_new3(default_conv, self.channels * 4, 3, act=nn.ReLU())
#         self.res_blk6 = ResBlock_new3(default_conv, self.channels * 4, 3, act=nn.ReLU())
#
#         self.res_blk7 = ResBlock_new3(default_conv, self.channels * 2, 3, act=nn.ReLU())
#         self.res_blk8 = ResBlock_new3(default_conv, self.channels * 2, 3, act=nn.ReLU())
#
#         self.res_blk9 = ResBlock_new3(default_conv, self.channels, 3, act=nn.ReLU())
#
#         self.conv1X11 = nn.Conv2d(self.channels * 8, self.channels , 1, 1, 0)
#
#         self.conv1X12 = nn.Conv2d(self.channels * 3, self.channels//2, 1, 1, 0)
#
#         self.conv1X13 = nn.Conv2d(self.channels , self.channels // 4, 1, 1, 0)
#         self.relu = nn.ReLU()
#     def forward(self, x, pre_out=None):
#         if pre_out is not None:
#             x0, m0 = self.res_blk0(x[-1],pre_out[-1])
#         else:
#             x0, m0 = self.res_blk0(x[-1])
#         x1, m1 = self.res_blk1(x0, m0)
#         x2, m2 = self.res_blk2(x1, m1)
#         x3, m3 = self.res_blk3(x2, m2)
#         resxm = self.conv1X11(self.relu(torch.cat([m0, m1, m2, m3], dim=1)))
#
#         resxm = F.interpolate(resxm, x[-2].size()[2:], mode='bilinear')
#         if pre_out is not None:
#             x4, m4 = self.res_blk4(x[-2],resxm,pre_out[-2])
#         else:
#             x4, m4 = self.res_blk4(x[-2],resxm)
#         x5, m5 = self.res_blk5(x4, m4, resxm)
#         # 此处有Bug，x4应为x5
#         # x6, m6 = self.res_blk6(x4, m5, resxm)
#         x6, m6 = self.res_blk6(x5, m5, resxm)
#
#         res2xm = self.conv1X12(self.relu(torch.cat([m4, m5, m6], dim=1)))
#         res2xm = F.interpolate(res2xm, x[-3].size()[2:], mode='bilinear')
#         if pre_out is not None:
#             x7, m7 = self.res_blk7(x[-3],res2xm,pre_out[-3])
#         else:
#             x7, m7 = self.res_blk7(x[-3], res2xm)
#         x8, m8 = self.res_blk8(x7, m7, res2xm)
#
#         res3xm = self.conv1X13(self.relu(torch.cat([m7, m8], dim=1)))
#         res3xm = F.interpolate(res3xm, x[0].size()[2:], mode='bilinear')
#         if pre_out is not None:
#             x9, m9 = self.res_blk9(x[0], res3xm,pre_out[0])
#         else:
#             x9, m9 = self.res_blk9(x[0], res3xm)
#         return [x9,x8,x6,x3],[m9,m8,m6,m3]
#
# class GroupBlock1(nn.Module):
#     def __init__(self, conv,  channel=64,height=None, width=None):
#         super(GroupBlock1, self).__init__()
#         self.channels = channel
#         self.res_blk0 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
#         self.res_blk1 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
#         self.res_blk2 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
#         self.res_blk3 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
#
#         self.res_blk4 = ResBlock_new3(default_conv, self.channels * 4, 3, act=nn.ReLU())
#         self.res_blk5 = ResBlock_new3(default_conv, self.channels * 4, 3, act=nn.ReLU())
#         self.res_blk6 = ResBlock_new3(default_conv, self.channels * 4, 3, act=nn.ReLU())
#
#         self.res_blk7 = ResBlock_new3(default_conv, self.channels * 2, 3, act=nn.ReLU())
#         self.res_blk8 = ResBlock_new3(default_conv, self.channels * 2, 3, act=nn.ReLU())
#
#         self.res_blk9 = ResBlock_new3(default_conv, self.channels, 3, act=nn.ReLU())
#
#         self.conv1X11_h = nn.Conv2d(self.channels * 8, self.channels , 1, 1, 0)
#         self.conv1X11_w = nn.Conv2d(self.channels * 8, self.channels*2, 1, 1, 0)
#         self.conv1X12_h = nn.Conv2d(self.channels * 3, self.channels//2, 1, 1, 0)
#         self.conv1X12_w = nn.Conv2d(self.channels * 3, self.channels , 1, 1, 0)
#
#         self.conv1X13_h = nn.Conv2d(self.channels , self.channels // 4, 1, 1, 0)
#         self.conv1X13_w = nn.Conv2d(self.channels, self.channels // 2, 1, 1, 0)
#         self.relu = nn.ReLU()
#     def forward(self, x, pre_out=None):
#         if pre_out is not None:
#             x0, m0 = self.res_blk0(x[-1],pre_out[-1])
#         else:
#             x0, m0 = self.res_blk0(x[-1])
#         x1, m1 = self.res_blk1(x0, m0)
#         x2, m2 = self.res_blk2(x1, m1)
#         x3, m3 = self.res_blk3(x2, m2)
#         resxm_h = self.conv1X11_h(self.relu(torch.cat([m0, m1, m2, m3], dim=1)))
#         resxm_w = self.conv1X11_w(self.relu(torch.cat([m0, m1, m2, m3], dim=1)))
#         resxm = F.interpolate(resxm_h, x[-2].size()[2:], mode='bilinear')
#         if pre_out is not None:
#             x4, m4 = self.res_blk4(x[-2],resxm,pre_out[-2])
#         else:
#             x4, m4 = self.res_blk4(x[-2],resxm)
#         x5, m5 = self.res_blk5(x4, m4, resxm)
#         # 此处有Bug，x4应为x5
#         # x6, m6 = self.res_blk6(x4, m5, resxm)
#         x6, m6 = self.res_blk6(x5, m5, resxm)
#
#         res2xm_h = self.conv1X12_h(self.relu(torch.cat([m4, m5, m6], dim=1)))
#         res2xm_w = self.conv1X12_w(self.relu(torch.cat([m4, m5, m6], dim=1)))
#         res2xm = F.interpolate(res2xm_h, x[-3].size()[2:], mode='bilinear')
#         if pre_out is not None:
#             x7, m7 = self.res_blk7(x[-3],res2xm,pre_out[-3])
#         else:
#             x7, m7 = self.res_blk7(x[-3], res2xm)
#         x8, m8 = self.res_blk8(x7, m7, res2xm)
#
#         res3xm_h = self.conv1X13_h(self.relu(torch.cat([m7, m8], dim=1)))
#         res3xm_w = self.conv1X13_w(self.relu(torch.cat([m7, m8], dim=1)))
#         res3xm = F.interpolate(res3xm_h, x[0].size()[2:], mode='bilinear')
#         if pre_out is not None:
#             x9, m9 = self.res_blk9(x[0], res3xm,pre_out[0])
#         else:
#             x9, m9 = self.res_blk9(x[0], res3xm)
#         return [x9,x8,x6,x3],[m9,res3xm_w,res2xm_w,resxm_w]
#
# class GroupBlockA(nn.Module):
#     def __init__(self, conv,  channel=64,height=None, width=None):
#         super(GroupBlockA, self).__init__()
#         self.channels = channel
#         self.res_blk0 = ResBlock_new3A(default_conv, self.channels * 8, 3, act=nn.GELU())
#         self.res_blk1 = ResBlock_new3A(default_conv, self.channels * 8, 3, act=nn.GELU())
#         self.res_blk2 = ResBlock_new3A(default_conv, self.channels * 8, 3, act=nn.GELU())
#         self.res_blk3 = ResBlock_new3A(default_conv, self.channels * 8, 3, act=nn.GELU())
#
#         self.res_blk4 = ResBlock_new3A(default_conv, self.channels * 4, 3, act=nn.GELU())
#         self.res_blk5 = ResBlock_new3A(default_conv, self.channels * 4, 3, act=nn.GELU())
#         self.res_blk6 = ResBlock_new3A(default_conv, self.channels * 4, 3, act=nn.GELU())
#
#         self.res_blk7 = ResBlock_new3A(default_conv, self.channels * 2, 3, act=nn.GELU())
#         self.res_blk8 = ResBlock_new3A(default_conv, self.channels * 2, 3, act=nn.GELU())
#
#         self.res_blk9 = ResBlock_new3A(default_conv, self.channels, 3, act=nn.GELU())
#
#         self.conv1X11 = nn.Conv2d(self.channels * 8, self.channels , 1, 1, 0)
#
#         self.conv1X12 = nn.Conv2d(self.channels * 3, self.channels//2, 1, 1, 0)
#
#         self.conv1X13 = nn.Conv2d(self.channels , self.channels // 4, 1, 1, 0)
#         # self.relu = nn.GeLU()
#     def forward(self, x, pre_out=None):
#         if pre_out is not None:
#             x0, m0 = self.res_blk0(x[-1],pre_out[-1])
#         else:
#             x0, m0 = self.res_blk0(x[-1])
#         x1, m1 = self.res_blk1(x0, m0)
#         x2, m2 = self.res_blk2(x1, m1)
#         x3, m3 = self.res_blk3(x2, m2)
#         resxm = self.conv1X11(F.gelu(torch.cat([m0, m1, m2, m3], dim=1)))
#
#         resxm = F.interpolate(resxm, x[-2].size()[2:], mode='bilinear')
#         if pre_out is not None:
#             x4, m4 = self.res_blk4(x[-2],resxm,pre_out[-2])
#         else:
#             x4, m4 = self.res_blk4(x[-2],resxm)
#         x5, m5 = self.res_blk5(x4, m4, resxm)
#         # 此处有Bug，x4应为x5
#         # x6, m6 = self.res_blk6(x4, m5, resxm)
#         x6, m6 = self.res_blk6(x5, m5, resxm)
#
#         res2xm = self.conv1X12(F.gelu(torch.cat([m4, m5, m6], dim=1)))
#         res2xm = F.interpolate(res2xm, x[-3].size()[2:], mode='bilinear')
#         if pre_out is not None:
#             x7, m7 = self.res_blk7(x[-3],res2xm,pre_out[-3])
#         else:
#             x7, m7 = self.res_blk7(x[-3], res2xm)
#         x8, m8 = self.res_blk8(x7, m7, res2xm)
#
#         res3xm = self.conv1X13(F.gelu(torch.cat([m7, m8], dim=1)))
#         res3xm = F.interpolate(res3xm, x[0].size()[2:], mode='bilinear')
#         if pre_out is not None:
#             x9, m9 = self.res_blk9(x[0], res3xm,pre_out[0])
#         else:
#             x9, m9 = self.res_blk9(x[0], res3xm)
#         return [x9,x8,x6,x3],[m9,m8,m6,m3]
# class GroupBlockAB(nn.Module):
#     def __init__(self, conv,  channel=64,height=None, width=None):
#         super(GroupBlockAB, self).__init__()
#         self.channels = channel
#         self.res_blk0 = ResBlock_new3AB(default_conv, self.channels * 8, 3, act=nn.GELU())
#         self.res_blk1 = ResBlock_new3AB(default_conv, self.channels * 8, 3, act=nn.GELU())
#         self.res_blk2 = ResBlock_new3AB(default_conv, self.channels * 8, 3, act=nn.GELU())
#         self.res_blk3 = ResBlock_new3AB(default_conv, self.channels * 8, 3, act=nn.GELU())
#
#         self.res_blk4 = ResBlock_new3AB(default_conv, self.channels * 4, 3, act=nn.GELU())
#         self.res_blk5 = ResBlock_new3AB(default_conv, self.channels * 4, 3, act=nn.GELU())
#         self.res_blk6 = ResBlock_new3AB(default_conv, self.channels * 4, 3, act=nn.GELU())
#
#         self.res_blk7 = ResBlock_new3AB(default_conv, self.channels * 2, 3, act=nn.GELU())
#         self.res_blk8 = ResBlock_new3AB(default_conv, self.channels * 2, 3, act=nn.GELU())
#
#         self.res_blk9 = ResBlock_new3AB(default_conv, self.channels, 3, act=nn.GELU())
#
#         self.conv1X11 = nn.Conv2d(self.channels * 8, self.channels , 1, 1, 0)
#
#         self.conv1X12 = nn.Conv2d(self.channels * 3, self.channels//2, 1, 1, 0)
#
#         self.conv1X13 = nn.Conv2d(self.channels , self.channels // 4, 1, 1, 0)
#         # self.relu = nn.GeLU()
#     def forward(self, x, pre_out=None):
#         if pre_out is not None:
#             x0, m0 = self.res_blk0(x[-1],pre_out[-1])
#         else:
#             x0, m0 = self.res_blk0(x[-1])
#         x1, m1 = self.res_blk1(x0, m0)
#         x2, m2 = self.res_blk2(x1, m1)
#         x3, m3 = self.res_blk3(x2, m2)
#         resxm = self.conv1X11(F.gelu(torch.cat([m0, m1, m2, m3], dim=1)))
#
#         resxm = F.interpolate(resxm, x[-2].size()[2:], mode='bilinear')
#         if pre_out is not None:
#             x4, m4 = self.res_blk4(x[-2],resxm,pre_out[-2])
#         else:
#             x4, m4 = self.res_blk4(x[-2],resxm)
#         x5, m5 = self.res_blk5(x4, m4, resxm)
#         # 此处有Bug，x4应为x5
#         # x6, m6 = self.res_blk6(x4, m5, resxm)
#         x6, m6 = self.res_blk6(x5, m5, resxm)
#
#         res2xm = self.conv1X12(F.gelu(torch.cat([m4, m5, m6], dim=1)))
#         res2xm = F.interpolate(res2xm, x[-3].size()[2:], mode='bilinear')
#         if pre_out is not None:
#             x7, m7 = self.res_blk7(x[-3],res2xm,pre_out[-3])
#         else:
#             x7, m7 = self.res_blk7(x[-3], res2xm)
#         x8, m8 = self.res_blk8(x7, m7, res2xm)
#
#         res3xm = self.conv1X13(F.gelu(torch.cat([m7, m8], dim=1)))
#         res3xm = F.interpolate(res3xm, x[0].size()[2:], mode='bilinear')
#         if pre_out is not None:
#             x9, m9 = self.res_blk9(x[0], res3xm,pre_out[0])
#         else:
#             x9, m9 = self.res_blk9(x[0], res3xm)
#         return [x9,x8,x6,x3],[m9,m8,m6,m3]
#
# # class GroupBlock(nn.Module):
# #     def __init__(self, conv,  channel=64,height=None, width=None):
# #         super(GroupBlock, self).__init__()
# #         self.channels = channel
# #         self.res_blk0 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
# #         self.res_blk1 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
# #         self.res_blk2 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
# #         self.res_blk3 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
# #
# #         self.res_blk4 = ResBlock_new3(default_conv, self.channels * 4, 3, act=nn.ReLU())
# #         self.res_blk5 = ResBlock_new3(default_conv, self.channels * 4, 3, act=nn.ReLU())
# #         self.res_blk6 = ResBlock_new3(default_conv, self.channels * 4, 3, act=nn.ReLU())
# #
# #         self.res_blk7 = ResBlock_new3(default_conv, self.channels * 2, 3, act=nn.ReLU())
# #         self.res_blk8 = ResBlock_new3(default_conv, self.channels * 2, 3, act=nn.ReLU())
# #
# #         self.res_blk9 = ResBlock_new3(default_conv, self.channels, 3, act=nn.ReLU())
# #
# #         self.conv1X11 = nn.Conv2d(self.channels * 8, self.channels , 1, 1, 0)
# #
# #         self.conv1X12 = nn.Conv2d(self.channels * 3, self.channels//2, 1, 1, 0)
# #
# #         self.conv1X13 = nn.Conv2d(self.channels , self.channels // 4, 1, 1, 0)
# #         self.relu = nn.ReLU()
# #     def forward(self, x, pre_out=None):
# #         if pre_out is not None:
# #             x0, m0 = self.res_blk0(x[-1],pre_out[-1])
# #         else:
# #             x0, m0 = self.res_blk0(x[-1])
# #         x1, m1 = self.res_blk1(x0, m0)
# #         x2, m2 = self.res_blk2(x1, m1)
# #         x3, m3 = self.res_blk3(x2, m2)
# #         resxm = self.conv1X11(self.relu(torch.cat([m0, m1, m2, m3], dim=1)))
# #         m3 = m0 + m1 + m2 + m3
# #         resxm = F.interpolate(resxm, x[-2].size()[2:], mode='bilinear')
# #         if pre_out is not None:
# #             x4, m4 = self.res_blk4(x[-2],resxm,pre_out[-2])
# #         else:
# #             x4, m4 = self.res_blk4(x[-2],resxm)
# #         x5, m5 = self.res_blk5(x4, m4, resxm)
# #         # 此处有Bug，x4应为x5
# #         x6, m6 = self.res_blk6(x5, m5, resxm)
# #         m6 = m4 + m5 + m6
# #         res2xm = self.conv1X12(self.relu(torch.cat([m4, m5, m6], dim=1)))
# #         res2xm = F.interpolate(res2xm, x[-3].size()[2:], mode='bilinear')
# #         if pre_out is not None:
# #             x7, m7 = self.res_blk7(x[-3],res2xm,pre_out[-3])
# #         else:
# #             x7, m7 = self.res_blk7(x[-3], res2xm)
# #         x8, m8 = self.res_blk8(x7, m7, res2xm)
# #         m8 = m7 + m8
# #         res3xm = self.conv1X13(self.relu(torch.cat([m7, m8], dim=1)))
# #         res3xm = F.interpolate(res3xm, x[0].size()[2:], mode='bilinear')
# #         if pre_out is not None:
# #             x9, m9 = self.res_blk9(x[0], res3xm,pre_out[0])
# #         else:
# #             x9, m9 = self.res_blk9(x[0], res3xm)
# #         return [x9,x8,x6,x3],[m9,m8,m6,m3]
#
# # revise 2021-04-28
# # class GroupBlock(nn.Module):
# #     def __init__(self, conv,  channel=64,height=None, width=None):
# #         super(GroupBlock, self).__init__()
# #         self.channels = channel
# #         self.res_blk0 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
# #         self.res_blk1 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
# #         self.res_blk2 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
# #         self.res_blk3 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
# #
# #         self.res_blk4 = ResBlock_new3(default_conv, self.channels * 4, 3, act=nn.ReLU())
# #         self.res_blk5 = ResBlock_new3(default_conv, self.channels * 4, 3, act=nn.ReLU())
# #         self.res_blk6 = ResBlock_new3(default_conv, self.channels * 4, 3, act=nn.ReLU())
# #
# #         self.res_blk7 = ResBlock_new3(default_conv, self.channels * 2, 3, act=nn.ReLU())
# #         self.res_blk8 = ResBlock_new3(default_conv, self.channels * 2, 3, act=nn.ReLU())
# #
# #         self.res_blk9 = ResBlock_new3(default_conv, self.channels, 3, act=nn.ReLU())
# #
# #         self.conv1X11 = nn.Conv2d(self.channels * 8, self.channels , 1, 1, 0)
# #         # self.conv1X11_1 = nn.Conv2d(self.channels*8 , self.channels*2, 1, 1,0)
# #
# #         self.conv1X12 = nn.Conv2d(self.channels * 3, self.channels//2, 1, 1, 0)
# #         # self.conv1X11_2= nn.Conv2d(self.channels * 3, self.channels , 1, 1, 0)
# #
# #         self.conv1X13 = nn.Conv2d(self.channels , self.channels // 4, 1, 1, 0)
# #         # self.conv1X11_3 = nn.Conv2d(self.channels , self.channels//2, 1, 1, 0)
# #         self.relu = nn.ReLU()
# #     def forward(self, x, pre_out=None):
# #         if pre_out is not None:
# #             x0, m0 = self.res_blk0(x[-1],pre_out[-1])
# #         else:
# #             x0, m0 = self.res_blk0(x[-1])
# #         x1, m1 = self.res_blk1(x0, m0)
# #         x2, m2 = self.res_blk2(x1, m1)
# #         x3, m3 = self.res_blk3(x2, m2)
# #         resxm = self.conv1X11(self.relu(torch.cat([m0, m1, m2, m3], dim=1)))
# #         x3 = x0+x1+x2+x3
# #         m3 = m0+m1+m2+m3
# #         resxm_up = F.interpolate(resxm, x[-2].size()[2:], mode='bilinear')
# #         if pre_out is not None:
# #             x4, m4 = self.res_blk4(x[-2],resxm_up,pre_out[-2])
# #         else:
# #             x4, m4 = self.res_blk4(x[-2],resxm_up)
# #         x5, m5 = self.res_blk5(x4, m4, resxm_up)
# #         # 此处有Bug，x4应为x5
# #         x6, m6 = self.res_blk6(x5, m5, resxm_up)
# #         x6 = x4+x5+x6
# #         m6 = m4+m5+m6
# #         res2xm = self.conv1X12(self.relu(torch.cat([m4, m5, m6], dim=1)))
# #         res2xm_up = F.interpolate(res2xm, x[-3].size()[2:], mode='bilinear')
# #         if pre_out is not None:
# #             x7, m7 = self.res_blk7(x[-3],res2xm_up,pre_out[-3])
# #         else:
# #             x7, m7 = self.res_blk7(x[-3], res2xm_up)
# #         x8, m8 = self.res_blk8(x7, m7, res2xm_up)
# #         x8 = x7+x8
# #         m8 = m8+m7
# #         res3xm = self.conv1X13(self.relu(torch.cat([m7, m8], dim=1)))
# #         res3xm_up = F.interpolate(res3xm, x[0].size()[2:], mode='bilinear')
# #         if pre_out is not None:
# #             x9, m9 = self.res_blk9(x[0], res3xm_up,pre_out[0])
# #         else:
# #             x9, m9 = self.res_blk9(x[0], res3xm_up)
# #         return [x9,x8,x6,x3],[m9,m8,m6,m3]
#
# class GroupBlock_base(nn.Module):
#     def __init__(self, conv, channel=64, height=None, width=None):
#         super(GroupBlock_base, self).__init__()
#         self.channels = channel
#         self.res_blk0 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
#         self.res_blk1 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
#         self.res_blk2 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
#         self.res_blk3 = ResBlock(default_conv, self.channels * 8, 3, act=nn.ReLU())
#
#         self.res_blk4 = ResBlock(default_conv, self.channels * 4, 3, act=nn.ReLU())
#         self.res_blk5 = ResBlock(default_conv, self.channels * 4, 3, act=nn.ReLU())
#         self.res_blk6 = ResBlock(default_conv, self.channels * 4, 3, act=nn.ReLU())
#
#         self.res_blk7 = ResBlock(default_conv, self.channels * 2, 3, act=nn.ReLU())
#         self.res_blk8 = ResBlock(default_conv, self.channels * 2, 3, act=nn.ReLU())
#
#         self.res_blk9 = ResBlock(default_conv, self.channels, 3, act=nn.ReLU())
#
#     def forward(self, x):
#         x0 = self.res_blk0(x[-1])
#         x1 = self.res_blk1(x0)
#         x2 = self.res_blk2(x1)
#         x3 = self.res_blk3(x2)
#
#         x4 = self.res_blk4(x[-2])
#         x5 = self.res_blk5(x4)
#         x6 = self.res_blk6(x5)
#
#         x7 = self.res_blk7(x[-3])
#         x8 = self.res_blk8(x7)
#
#         x9 = self.res_blk9(x[0])
#         return [x9, x8, x6, x3]
#
#
# class GroupBlock_base2(nn.Module):
#     def __init__(self, conv, channel=64, height=None, width=None):
#         super(GroupBlock_base2, self).__init__()
#         self.channels = channel
#         self.res_blk0 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
#         self.res_blk1 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
#         self.res_blk2 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
#         self.res_blk3 = ResBlock_new3(default_conv, self.channels * 8, 3, act=nn.ReLU())
#
#         self.res_blk4 = ResBlock_new3(default_conv, self.channels * 4, 3, act=nn.ReLU())
#         self.res_blk5 = ResBlock_new3(default_conv, self.channels * 4, 3, act=nn.ReLU())
#         self.res_blk6 = ResBlock_new3(default_conv, self.channels * 4, 3, act=nn.ReLU())
#
#         self.res_blk7 = ResBlock_new3(default_conv, self.channels * 2, 3, act=nn.ReLU())
#         self.res_blk8 = ResBlock_new3(default_conv, self.channels * 2, 3, act=nn.ReLU())
#
#         self.res_blk9 = ResBlock_new3(default_conv, self.channels, 3, act=nn.ReLU())
#
#         self.relu = nn.ReLU()
#
#     def forward(self, x, pre_out=None):
#         if pre_out is not None:
#             x0, m0 = self.res_blk0(x[-1], pre_out[-1])
#         else:
#             x0, m0 = self.res_blk0(x[-1])
#         x1, m1 = self.res_blk1(x0, m0)
#         x2, m2 = self.res_blk2(x1, m1)
#         x3, m3 = self.res_blk3(x2, m2)
#
#         if pre_out is not None:
#             x4, m4 = self.res_blk4(x[-2], pre_out[-2])
#         else:
#             x4, m4 = self.res_blk4(x[-2])
#         x5, m5 = self.res_blk5(x4, m4)
#         x6, m6 = self.res_blk6(x5, m5)
#
#         if pre_out is not None:
#             x7, m7 = self.res_blk7(x[-3], pre_out[-3])
#         else:
#             x7, m7 = self.res_blk7(x[-3])
#         x8, m8 = self.res_blk8(x7, m7)
#
#         if pre_out is not None:
#             x9, m9 = self.res_blk9(x[0], pre_out[0])
#         else:
#             x9, m9 = self.res_blk9(x[0])
#         return [x9, x8, x6, x3], [m9, m8, m6, m3]
#
# class MResBlock(nn.Module):
#     def __init__(
#         self, conv, n_feats, kernel_size,
#         bias=True, bn=False, act=nn.PReLU(), res_scale=1):
#
#         super(MResBlock, self).__init__()
#         m1 = []
#
#         m1.append(conv(n_feats, n_feats//4, 1, bias=bias))
#         m1.append(act)
#         m1.append(conv(n_feats//4, n_feats//4, 3, bias=bias))
#         m1.append(act)
#         m1.append(conv(n_feats // 4, n_feats , 1, bias=bias))
#         self.A = nn.Sequential(*m1)
#         m2 = []
#
#         m2.append(conv(n_feats, n_feats // 4, 1, bias=bias))
#         m2.append(act)
#         m2.append(conv(n_feats // 4, n_feats // 4, 3, bias=bias))
#         m2.append(act)
#         m2.append(conv(n_feats // 4, n_feats, 1, bias=bias))
#         self.T = nn.Sequential(*m2)
#         self.relu = nn.ReLU()
#         self.delta = nn.Sigmoid()
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         A = self.relu(self.A(x).mul(self.res_scale))
#         T = self.delta(self.T(x).mul(self.res_scale))
#         res = (1+T)*x+(1-T)*A
#
#         return res
#
# class SOS_blk(nn.Module):
#     def __init__(self, in_Channel, out_Channel,dialation=1):
#         super(SOS_blk, self).__init__()
#         self.path1_0= nn.Conv2d(in_Channel,in_Channel//4,1,1,0)
#         self.path2_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1,0)
#         self.path3_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1, 0)
#
#         self.path1_1 = nn.Conv2d(in_Channel // 4, in_Channel // 4, 3, 1, padding=dialation, dilation=dialation)
#         self.path2_1 = nn.Conv2d(in_Channel // 4, in_Channel // 4, 3, 1,padding=dialation,dilation=dialation)
#
#         self.path_up = nn.Conv2d(in_Channel //4, out_Channel, 1, 1,0)
#         self.relu = nn.ReLU(inplace=True)
#         self.weight0 = nn.Parameter(torch.Tensor([1]))
#         self.weight1 = nn.Parameter(torch.Tensor([1]))
#     def forward(self,x,x_pre):
#         # y = self.relu(self.path1(x)+x)
#         # y1 = self.path2(y)
#         # # x = self.se_ptah(x)
#         # out = self.relu(torch.cat([x,y1],dim=1))
#         # out = self.conv1X1(out)
#         # out = self.conv1(self.relu(out))
#         y0 = self.path1_0(x)
#         y1 = self.path2_0(x_pre)
#         y2 = self.path3_0(x_pre)
#
#         y0 = self.path1_1(self.relu(y0))
#         y1 = self.path2_1(self.relu(y1))
#         y = y0+y1-y2
#         out = self.path_up(self.relu(y))
#         return out
# class Nlocal_dehaze_blk_Connect(nn.Module):
#     def __init__(self, in_Channel, out_Channel, dialation=1):
#         super(Nlocal_dehaze_blk_Connect, self).__init__()
#         self.path1_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1, 0)
#         self.path2_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1, 0)
#         self.path3_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1, 0)
#
#         self.path1_1 = nn.Conv2d(in_Channel // 4, in_Channel // 4, 3, 1, padding=dialation, dilation=dialation)
#         self.path2_1 = nn.Conv2d(in_Channel // 4, in_Channel // 4, 3, 1, padding=dialation, dilation=dialation)
#
#         self.conv1 = nn.Conv2d(in_Channel // 2, in_Channel//4, 1, 1, 0)
#         self.conv2 = nn.Conv2d(in_Channel // 2, in_Channel // 4, 1, 1, 0)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.path_Up = nn.Conv2d(in_Channel // 4, out_Channel, 1, 1, 0)
#
#         self.relu = nn.ReLU(inplace=True)
#
#         self.weight0 = nn.Sigmoid()
#         # self.weight1 = nn.Parameter(torch.Tensor([1]))
#
#     def forward(self, x, x_pre_A=None, x_pre_T=None):
#         # y = self.relu(self.path1(x)+x)
#         # y1 = self.path2(y)
#         # # x = self.se_ptah(x)
#         # out = self.relu(torch.cat([x,y1],dim=1))
#         # out = self.conv1X1(out)
#         # out = self.conv1(self.relu(out))
#         A = self.path1_0(x)
#
#         T = self.path2_0(x)
#         I = self.path3_0(x)
#         # y = self.relu(y0+y1)
#         A = self.path1_1(self.relu(A))
#         if x_pre_A is not None:
#             A = self.conv1(torch.cat([A, x_pre_A], dim=1))
#         b, c, _, _ = A.size()
#         gap = self.avg_pool(A).view(b, c)
#         T = self.path2_1(self.relu(T))
#         if x_pre_T is not None:
#             T = self.conv2(torch.cat([T, x_pre_T], dim=1))
#         W = self.weight0(T)
#         # y = self.relu(y0+y1)
#         J = I * (1 + W) + gap * (1 - W)
#         out = self.path_Up(J)
#         # out = self.relu(x+y)
#         return out, A, T
#
# class Nlocal_dehaze_blk(nn.Module):
#     def __init__(self, in_Channel, out_Channel,dialation=1):
#         super(Nlocal_dehaze_blk, self).__init__()
#         self.path1_0= nn.Conv2d(in_Channel,in_Channel//4,1,1,0)
#         self.path2_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1, 0)
#         self.path3_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1, 0)
#
#         self.path1_1 = nn.Conv2d(in_Channel//4, in_Channel // 4, 3,1,padding=dialation,dilation=dialation)
#         self.path2_1 = nn.Conv2d(in_Channel // 4, in_Channel // 4, 3, 1,padding=dialation,dilation=dialation)
#
#         self.path_Up = nn.Conv2d(in_Channel // 4, out_Channel, 1, 1, 0)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.weight0 = nn.Sigmoid()
#         # self.weight1 = nn.Parameter(torch.Tensor([1]))
#     def forward(self,x):
#
#         A = self.path1_0(x)
#         T = self.path2_0(x)
#         I = self.path3_0(x)
#
#         A = self.relu(self.path1_1(self.relu(A)))
#         T = self.path2_1(self.relu(T))
#         T = self.weight0(T)
#
#         J = I*(1+T)+A*(1-T)
#         out = self.path_Up((J))
#
#         return out
#
#
# class Nlocal_dehaze_blk_GAP(nn.Module):
#     def __init__(self, in_Channel, out_Channel, dialation=1):
#         super(Nlocal_dehaze_blk_GAP, self).__init__()
#         self.path1_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1, 0)
#         self.path2_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1, 0)
#         self.path3_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1, 0)
#
#         self.path1_1 = nn.Conv2d(in_Channel // 4, in_Channel // 4, 3, 1, padding=dialation, dilation=dialation)
#         self.path2_1 = nn.Conv2d(in_Channel // 4, in_Channel // 4, 3, 1, padding=dialation, dilation=dialation)
#
#
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.path_Up = nn.Conv2d(in_Channel // 4, out_Channel, 1, 1, 0)
#
#         self.relu = nn.ReLU(inplace=True)
#
#         self.weight0 = nn.Sigmoid()
#         # self.weight1 = nn.Parameter(torch.Tensor([1]))
#
#     def forward(self, x):
#         A = self.path1_0(x)
#
#         T = self.path2_0(x)
#         I = self.path3_0(x)
#
#         A = self.path1_1(self.relu(A))
#
#         b, c, _, _ = A.size()
#         gap = self.avg_pool(A).view(b, c)
#         T = self.path2_1(self.relu(T))
#
#         W = self.weight0(T)
#
#         J = I * (1 + W) + gap * (1 - W)
#         out = self.path_Up(J)
#
#         return out
#
#
# class Nlocal_dehaze_blk_attention(nn.Module):
#     def __init__(self, in_Channel, out_Channel, dialation=1,reduction=8):
#         super(Nlocal_dehaze_blk_attention, self).__init__()
#         self.path1_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1, 0)
#         self.path2_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1, 0)
#         self.path3_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1, 0)
#
#         self.path1_1 = nn.Conv2d(in_Channel // 4, in_Channel // 4, 3, 1, padding=dialation, dilation=dialation)
#         self.path2_1 = nn.Conv2d(in_Channel // 4, in_Channel // 4, 3, 1, padding=dialation, dilation=dialation)
#         self.path3_1 = nn.Conv2d(in_Channel // 4, in_Channel // 4, 3, 1, padding=dialation, dilation=dialation)
#
#         self.conv1 = nn.Conv2d(in_Channel // 2, in_Channel // 4, 1, 1, 0)
#         self.conv2 = nn.Conv2d(in_Channel // 2, in_Channel // 4, 1, 1, 0)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.path_Up = nn.Conv2d(in_Channel // 4, out_Channel, 1, 1, 0)
#         self.channel = in_Channel//4
#         self.relu = nn.ReLU()
#         self.fc_0 = nn.Linear(self.channel, self.channel // reduction, bias=False)
#         self.fc_1 = nn.Linear(self.channel // reduction, self.channel, bias=False)
#         self.fc_2 = nn.Linear(self.channel // reduction, self.channel, bias=False)
#         self.fc_3 = nn.Linear(self.channel // reduction, self.channel, bias=False)
#         self.softmax = nn.Softmax(dim=1)
#
#
#         self.weight0 = nn.Sigmoid()
#
#
# # self.weight1 = nn.Parameter(torch.Tensor([1]))
#
#     def forward(self, x, x_pre_A=None, x_pre_T=None):
#         # y = self.relu(self.path1(x)+x)
#         # y1 = self.path2(y)
#         # # x = self.se_ptah(x)
#         # out = self.relu(torch.cat([x,y1],dim=1))
#         # out = self.conv1X1(out)
#         # out = self.conv1(self.relu(out))
#         A = self.path1_0(x)
#         T = self.path2_0(x)
#         I = self.path3_0(x)
#         KK = self.relu(A) + self.relu(T) + self.relu(I)
#
#         b, c, _, _ = KK.size()
#
#
#         at_g = self.avg_pool(KK).view(b, c)
#         at_g = self.relu(self.fc_0(at_g))
#         at_g1 = self.fc_1(at_g)
#         at_g2 = self.fc_2(at_g)
#         at_g3 = self.fc_3(at_g)
#         attention_v = self.softmax(torch.cat([at_g1, at_g2, at_g3], dim=1))
#         attention_v = attention_v.unsqueeze(-1).unsqueeze(-1)
#         A = attention_v[:, :self.channel, :, :] * A
#         T = attention_v[:, self.channel:2 * self.channel, :, :] * T
#         I = attention_v[:, 2 * self.channel:, :, :] * I
#         # y = self.relu(y0+y1)
#         A = self.relu(self.path1_1(A))
#         T = self.path2_1(T)
#         I = self.relu(self.path3_1(I))
#         W = self.weight0(T)
#
#         J = I * (1 + W) + A * (1 - W)
#         out = self.path_Up(J)
#         # out = self.relu(x+y)
#         return out
# # class Nlocal_dehaze_blk_Connect(nn.Module):
# #     def __init__(self, in_Channel, out_Channel, dialation=1):
# #         super(Nlocal_dehaze_blk_Connect, self).__init__()
# #         self.path1_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1, 0)
# #         self.path2_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1, 0)
# #         self.path3_0 = nn.Conv2d(in_Channel, in_Channel // 4, 1, 1, 0)
# #
# #         self.path1_1 = nn.Conv2d(in_Channel // 4, in_Channel // 4, 3, 1, padding=dialation, dilation=dialation)
# #         self.path2_1 = nn.Conv2d(in_Channel // 4, in_Channel // 4, 3, 1, padding=dialation, dilation=dialation)
# #
# #         self.conv1 = nn.Conv2d(in_Channel // 2, in_Channel//4, 1, 1, 0)
# #         self.conv2 = nn.Conv2d(in_Channel // 2, in_Channel // 4, 1, 1, 0)
# #         self.avg_pool = nn.AdaptiveAvgPool2d(1)
# #         self.path_Up = nn.Conv2d(in_Channel // 4, out_Channel, 1, 1, 0)
# #
# #         self.relu = nn.ReLU(inplace=True)
# #
# #         self.weight0 = nn.Sigmoid()
# #         # self.weight1 = nn.Parameter(torch.Tensor([1]))
# #
# #     def forward(self, x, x_pre_A=None, x_pre_T=None):
# #         # y = self.relu(self.path1(x)+x)
# #         # y1 = self.path2(y)
# #         # # x = self.se_ptah(x)
# #         # out = self.relu(torch.cat([x,y1],dim=1))
# #         # out = self.conv1X1(out)
# #         # out = self.conv1(self.relu(out))
# #         A = self.path1_0(x)
# #
# #         T = self.path2_0(x)
# #         I = self.path3_0(x)
# #         # y = self.relu(y0+y1)
# #         A = self.path1_1(self.relu(A))
# #         if x_pre_A is not None:
# #             A = self.conv1(torch.cat([A, x_pre_A], dim=1))
# #         b, c, _, _ = A.size()
# #         gap = self.avg_pool(A).view(b, c)
# #         T = self.path2_1(self.relu(T))
# #         if x_pre_T is not None:
# #             T = self.conv2(torch.cat([T, x_pre_T], dim=1))
# #         W = self.weight0(T)
# #         # y = self.relu(y0+y1)
# #         J = I * (1 + W) + gap * (1 - W)
# #         out = self.path_Up(J)
# #         # out = self.relu(x+y)
# #         return out, A, T
#
# class Nonlocal_Block(nn.Module):
#     def __init__(self, in_feats,out_feats,inter_channels):
#         super(Nonlocal_Block, self).__init__()
#         self.in_channel = in_feats
#         self.out_channel = out_feats
#         self.inter_channels = inter_channels
#         if self.inter_channels is None:
#             self.inter_channels = in_feats // 4
#         self.q = default_conv(self.in_channel,self.inter_channels,1,1)
#         self.k = default_conv(self.in_channel, self.inter_channels, 1, 1)
#         self.v = default_conv(self.in_channel, self.inter_channels, 1, 1)
#         self.up = default_conv(self.inter_channels, self.out, 1, 1)
#     def forward(self, x):
#         batch_size = x.size(0)
#         g_x = self.v(x).view(batch_size, self.inter_channels, -1)
#         g_x = g_x.permute(0, 2, 1)
#
#         theta_x = self.k(x).view(batch_size, self.inter_channels, -1)
#         theta_x = theta_x.permute(0, 2, 1)
#         phi_x = self.q(x).view(batch_size, self.inter_channels, -1)
#         f = torch.matmul(theta_x, phi_x)
#         f_div_C = F.softmax(f, dim=-1)
#         y = torch.matmul(f_div_C, g_x)
#         y = y.permute(0, 2, 1).contiguous()
#         y = y.view(batch_size, self.inter_channels, *x.size()[2:])
#         y = self.up(y)+x
#         return y
# class Nonlocal_Block2(nn.Module):
#     def __init__(self, in_feats,out_feats,inter_channels):
#         super(Nonlocal_Block2, self).__init__()
#         self.in_channel = in_feats
#         self.out_channel = out_feats
#         self.inter_channels = inter_channels
#         if self.inter_channels is None:
#             self.inter_channels = in_feats // 4
#         self.q = default_conv(self.in_channel,self.inter_channels,1,1)
#         self.k = default_conv(self.in_channel, self.inter_channels, 1, 1)
#         self.v = default_conv(self.in_channel, self.inter_channels, 1, 1)
#         self.up = default_conv(self.inter_channels, self.out, 1, 1)
#     def forward(self, x,x_pre):
#         batch_size = x.size(0)
#         # value
#         g_x = self.v(x_pre).view(batch_size, self.inter_channels, -1)
#         g_x = g_x.permute(0, 2, 1)
#         #key
#         theta_x = self.k(x_pre).view(batch_size, self.inter_channels, -1)
#         theta_x = theta_x.permute(0, 2, 1)
#         #query
#         phi_x = self.q(x).view(batch_size, self.inter_channels, -1)
#         f = torch.matmul(theta_x, phi_x)
#         f_div_C = F.softmax(f, dim=-1)
#         #embedding gaussian
#         y = torch.matmul(f_div_C, g_x)
#         y = y.permute(0, 2, 1).contiguous()
#         y = y.view(batch_size, self.inter_channels, *x.size()[2:])
#         # skip
#         y = self.up(y)+x
#         return y
# class Nonlocal_Block3(nn.Module):
#     def __init__(self, in_feats,out_feats,inter_channels):
#         super(Nonlocal_Block3, self).__init__()
#         self.in_channel = in_feats
#         self.out_channel = out_feats
#         self.inter_channels = inter_channels
#         if self.inter_channels is None:
#             self.inter_channels = in_feats // 4
#         self.q = default_conv(self.in_channel,self.inter_channels,1,1)
#         self.k = default_conv(self.in_channel, self.inter_channels, 1, 1)
#         self.v = default_conv(self.in_channel, self.inter_channels, 1, 1)
#         self.up = default_conv(self.inter_channels, self.out, 1, 1)
#     def forward(self, x,x_pre):
#         batch_size = x.size(0)
#         # value
#         g_x = self.v(x).view(batch_size, self.inter_channels, -1)
#         g_x = g_x.permute(0, 2, 1)
#         #key
#         theta_x = self.k(x).view(batch_size, self.inter_channels, -1)
#         theta_x = theta_x.permute(0, 2, 1)
#         #query
#         phi_x = self.q(x_pre).view(batch_size, self.inter_channels, -1)
#         f = torch.matmul(theta_x, phi_x)
#         f_div_C = F.softmax(f, dim=-1)
#         #embedding gaussian
#         y = torch.matmul(f_div_C, g_x)
#         y = y.permute(0, 2, 1).contiguous()
#         y = y.view(batch_size, self.inter_channels, *x.size()[2:])
#         # skip
#         y = self.up(y)+x
#         return y
# # class ResBlock(nn.Module):
# #     def __init__(
# #         self, conv, n_feats, kernel_size,
# #         bias=True, bn=False, act=nn.PReLU(), res_scale=1):
# #
# #         super(ResBlock, self).__init__()
# #         m = []
# #         for i in range(2):
# #             m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
# #             if bn:
# #                 m.append(nn.BatchNorm2d(n_feats))
# #             if i == 0:
# #                 m.append(act)
# #
# #         self.body = nn.Sequential(*m)
# #         self.res_scale = res_scale
# #
# #     def forward(self, x):
# #         res = self.body(x).mul(self.res_scale)
# #         res += x
# #
# #         return res
# class Upsampler(nn.Sequential):
#     def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
#
#         m = []
#         if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
#             for _ in range(int(math.log(scale, 2))):
#                 m.append(conv(n_feats, 4 * n_feats, 3, bias))
#                 m.append(nn.PixelShuffle(2))
#                 if bn:
#                     m.append(nn.BatchNorm2d(n_feats))
#                 if act == 'relu':
#                     m.append(nn.ReLU(True))
#                 elif act == 'prelu':
#                     m.append(nn.PReLU(n_feats))
#
#         elif scale == 3:
#             m.append(conv(n_feats, 9 * n_feats, 3, bias))
#             m.append(nn.PixelShuffle(3))
#             if bn:
#                 m.append(nn.BatchNorm2d(n_feats))
#             if act == 'relu':
#                 m.append(nn.ReLU(True))
#             elif act == 'prelu':
#                 m.append(nn.PReLU(n_feats))
#         else:
#             raise NotImplementedError
#
#         super(Upsampler, self).__init__(*m)
# class Wave_unet_blk(nn.Module):
#     def __init__(self, in_Channel, out_Channel):
#         super(Wave_unet_blk, self).__init__()
#         self.conv1 = nn.Conv2d(in_Channel,in_Channel*2,4,2,1)
#         self.conv2 = nn.Conv2d(in_Channel*2,in_Channel*4,4,2,1)
#         self.conv3 = nn.Conv2d(in_Channel*4,in_Channel*8,4,2,1)
#
#         self.convt1 = nn.ConvTranspose2d(in_Channel*8, in_Channel * 4, 4, 2, 1)
#         self.convt2 = nn.ConvTranspose2d(in_Channel*4, in_Channel * 2, 4, 2, 1)
#         self.convt3 = nn.ConvTranspose2d(in_Channel*2, in_Channel, 4, 2, 1)
#         self.relu = nn.ReLU(inplace=True)
#
#         self.conv1X1 = nn.Conv2d(in_Channel*2,out_Channel,1,1,0)
#     def forward(self,x):
#         x_d1 = self.conv1(x)
#         x_d2 = self.conv2(self.relu(x_d1))
#         x_d3 = self.conv3(self.relu(x_d2))
#         y_up1 = self.convt1(x_d3)
#         y_up1 = self.relu(y_up1+x_d2)
#         y_up2 = self.convt2(y_up1)
#         y_up3 = self.relu(y_up2 + x_d1)
#         y_up3 = self.convt3(y_up3)
#
#         # x = self.se_ptah(x)
#         out = self.relu(torch.cat([x,y_up3],dim=1))
#         out = self.conv1X1(out)
#         return out
# class Res_in_res_blk(nn.Module):
#     def __init__(self, in_Channel, out_Channel):
#         super(Res_in_res_blk, self).__init__()
#         self.path1 = nn.Sequential(
#                     nn.ReflectionPad2d(1),
#                     nn.Conv2d(in_Channel,in_Channel*3,3,1,0),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(in_Channel*3, out_Channel, 1, 1, 0),
#                     nn.ReLU(inplace=True),
#                     )
#         self.path2= nn.Sequential(
#             nn.ReflectionPad2d(1),
#             nn.Conv2d(out_Channel, out_Channel*3, 3, 1, 0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_Channel*3, out_Channel, 1, 1, 0),
#             nn.ReLU(inplace=True),
#         )
#         self.relu = nn.ReLU(inplace=True)
#         # self.se_ptah = SELayer(out_Channel)
#         self.conv1X1 = nn.Conv2d(in_Channel+out_Channel,out_Channel,1,1,0)
#         self.conv1 = nn.Conv2d(out_Channel , out_Channel, 3, 1, 1)
#         # self.weight0 = nn.Parameter(torch.Tensor([0]))
#         # self.weight1 = nn.Parameter(torch.Tensor([0]))
#     def forward(self,x):
#         # y = self.relu(self.path1(x)+x)
#         # y1 = self.path2(y)
#         # # x = self.se_ptah(x)
#         # out = self.relu(torch.cat([x,y1],dim=1))
#         # out = self.conv1X1(out)
#         # out = self.conv1(self.relu(out))
#         y0 = self.path1(x)
#         y = torch.cat([x,y0],dim=1)
#         y = self.relu(self.conv1X1(y))
#         y1 = self.path2(y)
#         out = y1+y+x
#         out = self.conv1(out)
#         return out
# class Res_in_res_blk_E(nn.Module):
#     def __init__(self, in_Channel, out_Channel):
#         super(Res_in_res_blk_E, self).__init__()
#         self.path1 = nn.Sequential(
#                     nn.Conv2d(in_Channel,in_Channel*3,3,1,1),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(in_Channel*3, out_Channel, 1, 1, 0),
#                     )
#         self.path2= nn.Sequential(
#             nn.Conv2d(out_Channel, out_Channel*3, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_Channel*3, out_Channel, 1, 1, 0),
#         )
#         self.relu = nn.ReLU(inplace=True)
#         # self.se_ptah = SELayer(out_Channel)
#         self.conv1X1 = nn.Conv2d(out_Channel*2,out_Channel,1,1,0)
#         self.conv1X1_0 = nn.Conv2d(out_Channel * 2, out_Channel, 1, 1, 0)
#         self.conv1X1_1 = nn.Conv2d(out_Channel * 2, out_Channel, 1, 1, 0)
#         self.weight0 = nn.Parameter(torch.Tensor([0]))
#         self.weight1 = nn.Parameter(torch.Tensor([0]))
#     def forward(self,x):
#         y = self.relu(self.path1(x)+x)
#         y1 = self.path2(y)
#         # x = self.se_ptah(x)
#         out0 = self.relu(torch.cat([y,y1],dim=1))
#         out0 = self.conv1X1(out0)
#
#         z0 = self.path1(x)
#         z0 = self.relu(torch.cat([x,z0],dim=1))
#         z0 = self.conv1X1_0(z0)
#         z1 = self.path2(z0)
#         out1 = z1+z0
#         out = self.relu(torch.cat([out0, out1], dim=1))
#         out = self.conv1X1_1(out)
#         return out
#
#
# class Res_in_res_blk_E2(nn.Module):
#     def __init__(self, in_Channel, out_Channel,k_size=3,pad=1,stride=1):
#         super(Res_in_res_blk_E2,self).__init__()
#         self.path1 = nn.Sequential(
#                     nn.Conv2d(in_Channel,in_Channel*6,k_size,1,pad,groups=64),
#                     nn.ReLU(inplace=True),
#                     nn.Conv2d(in_Channel*6, out_Channel, 1, 1, 0),
#                     )
#         self.path2 = nn.Sequential(
#             nn.Conv2d(in_Channel, in_Channel * 6, k_size, 1, pad, groups=64),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_Channel * 6, out_Channel, 1, 1, 0),
#         )
#         self.relu = nn.ReLU(inplace=True)
#         self.se_ptah = SELayer(out_Channel)
#         self.conv1X1 = nn.Conv2d(out_Channel*2,out_Channel,1,1,0)
#
#     def forward(self,x):
#         z0 = self.path1(x)
#         z0 = self.relu(torch.cat([x,z0],dim=1))
#         z0 = self.conv1X1(z0)
#         z1 = self.path2(self.relu(z0))
#         z1 = self.se_ptah(z1)
#         out = z1+z0
#         return out
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
# class Upsample_blk2(nn.Module):
#     def __init__(self,inplanes,outplanes,scale=2,size=None):
#         super(Upsample_blk2, self).__init__()
#         self.up_op = nn.Upsample(size,scale_factor=2,mode='bicubic',align_corners=True)
#         self.conv1 = DepthWiseConv2d(inplanes,outplanes,5)
#         self.conv1X1 = nn.Conv2d(outplanes, outplanes, 3, 1, 1)
#         self.relu = nn.ReLU(inplace=True)
#     def forward(self,x):
#         out = self.up_op(x)
#         out0 = self.conv1(out)
#         out = self.conv1X1(self.relu(out0))
#         return out+out0
#
# class Pixelshuffle_blk(nn.Module):
#     def __init__(self,inplanes,outplanes,scale=2):
#         super(Pixelshuffle_blk, self).__init__()
#         # self.conv1X1 = nn.Conv2d(inplanes, inplanes^scale, 1, 1, 0)
#         self.up_op = nn.PixelShuffle(scale)
#         self.conv1 = nn.Conv2d(inplanes,outplanes,3,1,1)
#         self.relu = nn.ReLU(inplace=True)
#     def forward(self,x):
#         # out = self.conv1X1(x)
#         out = self.up_op(x)
#         out = self.conv1(self.relu(out))
#         return out
#
#
# class MixedFeature_blk(nn.Module):
#     def __init__(self,inplanes,outplanes,scale=2):
#         super(MixedFeature_blk, self).__init__()
#         self.convd1 = nn.Conv2d(inplanes,inplanes*2,4,2,1)
#         self.convd2 = nn.Conv2d(inplanes*2, inplanes * 4, 4, 2, 1)
#         self.convt1 = nn.ConvTranspose2d(inplanes*4,inplanes*2,4,2,1)
#         # self.conv0 = nn.Conv2d(inplanes*2, inplanes*2, 3, 1, 1)
#         self.convt2 = nn.ConvTranspose2d(inplanes * 2, inplanes , 4, 2, 1)
#         # self.conv02 = nn.Conv2d(inplanes , inplanes, 3, 1, 1)
#
#         self.conv1 = nn.Conv2d(inplanes,inplanes,3,1,1)
#         self.conv2 = nn.Conv2d(inplanes, outplanes, 3, 1, 1)
#         self.conv_mixed = nn.Conv2d(inplanes*2, outplanes, 1, 1, 0)
#         self.relu = nn.ReLU(inplace=True)
#     def forward(self,x):
#         out_d0 = self.convd1(x)
#         out_d1 = self.convd2(self.relu(out_d0))
#         out_t1 = self.convt1(self.relu(out_d1))
#         # out_t2 = self.conv0(self.relu(out_t1))
#         out_t1 = out_t1+out_d0
#         out_t2 = self.convt2(self.relu(out_t1))
#         out_t2 = out_t2+x
#
#
#         out_r = self.conv1(x)
#         out_r = self.conv2(self.relu(out_r))
#         out_r = out_r+x
#
#         out = torch.cat([out_r,out_t2],dim=1)
#         out = self.conv_mixed(self.relu(out))
#         return out
# class RACB_blk(nn.Module):
#     def __init__(self,inplanes,outplanes,scale=2):
#         super(RACB_blk, self).__init__()
#         self.conv1 = RCAB(default_conv,inplanes,3,16)
#         self.conv2 = RCAB(default_conv,inplanes,3,16)
#         self.conv3 = RCAB(default_conv,outplanes,3,16)
#         self.conv4 = RCAB(default_conv, outplanes, 3, 16)
#         self.conv5 = RCAB(default_conv, outplanes, 3, 16)
#         self.conv6 = RCAB(default_conv, outplanes, 3, 16)
#         self.relu = nn.ReLU(inplace=False)
#     def forward(self,x):
#         out0 = self.conv1(self.relu(x))
#         out1 = self.conv2(self.relu(x))
#         x = x+out0+out1
#         out2 = self.conv3(self.relu(x))
#         out3 = self.conv4(self.relu(x))
#         x = x + out2 + out3
#         out4 = self.conv5(self.relu(x))
#         out5 = self.conv6(self.relu(x))
#         out = x + out4 + out5
#         return out
#
#
#
# class HSBlock(nn.Module):
#     '''
#     替代3x3卷积
#     '''
#     def __init__(self, in_ch, s=8):
#         '''
#         特征大小不改变
#         :param in_ch: 输入通道
#         :param s: 分组数
#         '''
#         super(HSBlock, self).__init__()
#         self.s = s
#         self.module_list = nn.ModuleList()
#         # 避免无法整除通道数
#         in_ch, in_ch_last = (in_ch // s, in_ch // s) if in_ch % s == 0 else (in_ch // s + 1, in_ch % s)
#         self.module_list.append(nn.Sequential())
#         acc_channels = 0
#         for i in range(1,self.s):
#             if i == 1:
#                 channels=in_ch
#                 acc_channels=channels//2
#             elif i == s - 1:
#                 channels = in_ch_last + acc_channels
#             else:
#                 channels=in_ch+acc_channels
#                 acc_channels=channels//2
#             self.module_list.append(self.conv_bn_relu(in_ch=channels, out_ch=channels))
#         self.initialize_weights()
#
#     def conv_bn_relu(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
#         conv_bn_relu = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#         return conv_bn_relu
#
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def forward(self, x):
#         x = list(x.chunk(chunks=self.s, dim=1))
#         for i in range(1, len(self.module_list)):
#             y = self.module_list[i](x[i])
#             if i == len(self.module_list) - 1:
#                 x[0] = torch.cat((x[0], y), 1)
#             else:
#                 y1, y2 = y.chunk(chunks=2, dim=1)
#                 x[0] = torch.cat((x[0], y1), 1)
#                 x[i + 1] = torch.cat((x[i + 1], y2), 1)
#         return x[0]
#
# # //GCANet-WACV2019
# class ShareSepConv(nn.Module):
#     def __init__(self, kernel_size):
#         super(ShareSepConv, self).__init__()
#         assert kernel_size % 2 == 1, 'kernel size should be odd'
#         self.padding = (kernel_size - 1)//2
#         weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
#         weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
#         self.weight = nn.Parameter(weight_tensor)
#         self.kernel_size = kernel_size
#
#     def forward(self, x):
#         inc = x.size(1)
#         expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
#         return F.conv2d(x, expand_weight,
#                         None, 1, self.padding, 1, inc)
#
#
# class SmoothDilatedResidualBlock(nn.Module):
#     def __init__(self, channel_num, dilation=1, group=1):
#         super(SmoothDilatedResidualBlock, self).__init__()
#         self.pre_conv1 = ShareSepConv(dilation*2-1)
#         self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
#         self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
#         self.pre_conv2 = ShareSepConv(dilation*2-1)
#         self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
#         self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)
#
#     def forward(self, x):
#         y = F.relu(self.norm1(self.conv1(self.pre_conv1(x))))
#         y = self.norm2(self.conv2(self.pre_conv2(y)))
#         return F.relu(x+y)
#
# class make_dense(nn.Module):
#   def __init__(self, nChannels, growthRate, kernel_size=3):
#     super(make_dense, self).__init__()
#     self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
#   def forward(self, x):
#     out = F.relu(self.conv(x))
#     out = torch.cat((x, out), 1)
#     return out
#
# # Residual dense block (RDB) architecture Residual Dense Network for Image Super-Resolution
# class RDB(nn.Module):
#   def __init__(self, nChannels, nDenselayer, growthRate):
#     super(RDB, self).__init__()
#     nChannels_ = nChannels
#     modules = []
#     for i in range(nDenselayer):
#         modules.append(make_dense(nChannels_, growthRate))
#         nChannels_ += growthRate
#     self.dense_layers = nn.Sequential(*modules)
#     self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
#   def forward(self, x):
#     out = self.dense_layers(x)
#     out = self.conv_1x1(out)
#     out = out + x
#     return out
#
# #ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
# class CALayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(CALayer, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#                 nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#                 nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y
#
# ## Residual Channel Attention Block (RCAB)
#
# class RCAB(nn.Module):
#     def __init__(
#         self, conv, n_feat, kernel_size, reduction,
#         bias=False, bn=False, act=nn.ReLU(True), res_scale=1):
#
#         super(RCAB, self).__init__()
#         modules_body = []
#         for i in range(2):
#             modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#             if bn: modules_body.append(nn.BatchNorm2d(n_feat))
#             if i == 0: modules_body.append(act)
#         # modules_body.append(CALayer(n_feat, reduction))
#         modules_body.append(GALayer(n_feat))
#         self.body = nn.Sequential(*modules_body)
#         self.res_scale = res_scale
#
#     def forward(self, x):
#         res = self.body(x)
#         #res = self.body(x).mul(self.res_scale)
#         # res += x
#         return res
#
# ## Residual Group (RG)
# class ResidualGroup(nn.Module):
#     def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
#         super(ResidualGroup, self).__init__()
#         modules_body = []
#         modules_body = [
#             RCAB(
#                 conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
#             for _ in range(n_resblocks)]
#         modules_body.append(conv(n_feat, n_feat, kernel_size))
#         self.body = nn.Sequential(*modules_body)
#
#     def forward(self, x):
#         res = self.body(x)
#         res += x
#         return res
#
#
#
# class Disout(nn.Module):
#     """
#     Beyond Dropout: Feature Map Distortion to Regularize Deep Neural Networks
#     https://arxiv.org/abs/2002.11022
#     Args:
#         dist_prob (float): probability of an element to be distorted.
#         block_size (int): size of the block to be distorted.
#         alpha: the intensity of distortion.
#     Shape:
#         - Input: `(N, C, H, W)`
#         - Output: `(N, C, H, W)`
#     """
#
#     def __init__(self, dist_prob, block_size=6, alpha=1.0):
#         super(Disout, self).__init__()
#
#         self.dist_prob = dist_prob
#         self.weight_behind = None
#
#         self.alpha = alpha
#         self.block_size = block_size
#
#     def forward(self, x):
#
#         if not self.training:
#             return x
#         else:
#             x = x.clone()
#             if x.dim() == 4:
#                 width = x.size(2)
#                 height = x.size(3)
#
#                 seed_drop_rate = self.dist_prob * (width * height) / self.block_size ** 2 / (
#                             (width - self.block_size + 1) * (height - self.block_size + 1))
#
#                 valid_block_center = torch.zeros(width, height, device=x.device).float()
#                 valid_block_center[int(self.block_size // 2):(width - (self.block_size - 1) // 2),
#                 int(self.block_size // 2):(height - (self.block_size - 1) // 2)] = 1.0
#
#                 valid_block_center = valid_block_center.unsqueeze(0).unsqueeze(0)
#
#                 randdist = torch.rand(x.shape, device=x.device)
#
#                 block_pattern = ((1 - valid_block_center + float(1 - seed_drop_rate) + randdist) >= 1).float()
#
#                 if self.block_size == width and self.block_size == height:
#                     block_pattern = \
#                     torch.min(block_pattern.view(x.size(0), x.size(1), x.size(2) * x.size(3)), dim=2)[0].unsqueeze(
#                         -1).unsqueeze(-1)
#                 else:
#                     block_pattern = -F.max_pool2d(input=-block_pattern,
#                                                   kernel_size=(self.block_size, self.block_size), stride=(1, 1),
#                                                   padding=self.block_size // 2)
#
#                 if self.block_size % 2 == 0:
#                     block_pattern = block_pattern[:, :, :-1, :-1]
#                 percent_ones = block_pattern.sum() / float(block_pattern.numel())
#
#                 if not (self.weight_behind is None) and not (len(self.weight_behind) == 0):
#                     wtsize = self.weight_behind.size(3)
#                     weight_max = self.weight_behind.max(dim=0, keepdim=True)[0]
#                     sig = torch.ones(weight_max.size(), device=weight_max.device)
#                     sig[torch.rand(weight_max.size(), device=sig.device) < 0.5] = -1
#                     weight_max = weight_max * sig
#                     weight_mean = weight_max.mean(dim=(2, 3), keepdim=True)
#                     if wtsize == 1:
#                         weight_mean = 0.1 * weight_mean
#                     # print(weight_mean)
#                 mean = torch.mean(x).clone().detach()
#                 var = torch.var(x).clone().detach()
#
#                 if not (self.weight_behind is None) and not (len(self.weight_behind) == 0):
#                     dist = self.alpha * weight_mean * (var ** 0.5) * torch.randn(*x.shape, device=x.device)
#                 else:
#                     dist = self.alpha * 0.01 * (var ** 0.5) * torch.randn(*x.shape, device=x.device)
#
#             x = x * block_pattern
#             dist = dist * (1 - block_pattern)
#             x = x + dist
#             x = x / percent_ones
#             return x
#
# class LinearScheduler(nn.Module):
#     def __init__(self, disout, start_value, stop_value, nr_steps):
#         super(LinearScheduler, self).__init__()
#         self.disout = disout
#         self.i = 0
#         self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)
#
#     def forward(self, x):
#         return self.disout(x)
#
#     def step(self):
#         if self.i < len(self.drop_values):
#             self.disout.dist_prob = self.drop_values[self.i]
#         self.i += 1
#
# class _UpProjection(nn.Sequential):
#
#     def __init__(self, num_input_features, num_output_features):
#         super(_UpProjection, self).__init__()
#
#         self.conv1 = nn.Conv2d(num_input_features, num_output_features,
#                                kernel_size=5, stride=1, padding=2, bias=False)
#         # self.bn1 = nn.BatchNorm2d(num_output_features)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
#                                  kernel_size=3, stride=1, padding=1, bias=False)
#         # self.bn1_2 = nn.BatchNorm2d(num_output_features)
#
#         self.conv2 = nn.Conv2d(num_input_features, num_output_features,
#                                kernel_size=5, stride=1, padding=2, bias=False)
#         # self.bn2 = nn.BatchNorm2d(num_output_features)
#
#     def forward(self, x, size):
#         x = F.upsample(x, size=size, mode='bilinear')
#         x_conv1 = self.relu(self.conv1(x))
#         bran1 = self.conv1_2(x_conv1)
#         bran2 = self.conv2(x)
#
#         out = self.relu(bran1 + bran2)
#
#         return out
#
#
#
#
#
# class _routing(nn.Module):
#
#     def __init__(self, in_channels, num_experts, dropout_rate):
#         super(_routing, self).__init__()
#
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc = nn.Linear(in_channels, num_experts)
#
#     def forward(self, x):
#         x = torch.flatten(x)
#         x = self.dropout(x)
#         x = self.fc(x)
#         return F.sigmoid(x)
#
#
# class CondConv2D(_ConvNd):
#     r"""Learn specialized convolutional kernels for each example.
#     As described in the paper
#     `CondConv: Conditionally Parameterized Convolutions for Efficient Inference`_ ,
#     conditionally parameterized convolutions (CondConv),
#     which challenge the paradigm of static convolutional kernels
#     by computing convolutional kernels as a function of the input.
#     Args:
#         in_channels (int): Number of channels in the input image
#         out_channels (int): Number of channels produced by the convolution
#         kernel_size (int or tuple): Size of the convolving kernel
#         stride (int or tuple, optional): Stride of the convolution. Default: 1
#         padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
#         padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
#         dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
#         groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
#         bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
#         num_experts (int): Number of experts per layer
#     Shape:
#         - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
#         - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
#           .. math::
#               H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
#                         \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
#           .. math::
#               W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
#                         \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
#     Attributes:
#         weight (Tensor): the learnable weights of the module of shape
#                          :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
#                          :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
#                          The values of these weights are sampled from
#                          :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
#                          :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
#         bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
#                          then the values of these weights are
#                          sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
#                          :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
#     .. _CondConv: Conditionally Parameterized Convolutions for Efficient Inference:
#        https://arxiv.org/abs/1904.04971
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1,
#                  bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
#         kernel_size = _pair(kernel_size)
#         stride = _pair(stride)
#         padding = _pair(padding)
#         dilation = _pair(dilation)
#         super(CondConv2D, self).__init__(
#             in_channels, out_channels, kernel_size, stride, padding, dilation,
#             False, _pair(0), groups, bias, padding_mode)
#
#         self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
#         self._routing_fn = _routing(in_channels, num_experts, dropout_rate)
#
#         self.weight = Parameter(torch.Tensor(
#             num_experts, out_channels, in_channels // groups, *kernel_size))
#
#         self.reset_parameters()
#
#     def _conv_forward(self, input, weight):
#         if self.padding_mode != 'zeros':
#             return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
#                             weight, self.bias, self.stride,
#                             _pair(0), self.dilation, self.groups)
#         return F.conv2d(input, weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)
#
#     def forward(self, inputs):
#         b, _, _, _ = inputs.size()
#         res = []
#         for input in inputs:
#             input = input.unsqueeze(0)
#             pooled_inputs = self._avg_pooling(input)
#             routing_weights = self._routing_fn(pooled_inputs)
#             kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
#             out = self._conv_forward(input, kernels)
#             res.append(out)
#         return torch.cat(res, dim=0)
#
# class SPADE(nn.Module):
#     def __init__(self, config_text, norm_nc, label_nc,ks =3):
#         super().__init__()
#
#         assert config_text.startswith('spade')
#         parsed = re.search('spade(\D+)(\d)x\d', config_text)
#         param_free_norm_type = str(parsed.group(1))
#         ks = int(parsed.group(2))
#
#         if param_free_norm_type == 'instance':
#             self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
#         # elif param_free_norm_type == 'syncbatch':
#         #     self.param_free_norm = SynchronizedBatchNorm2d(norm_nc, affine=False)
#         elif param_free_norm_type == 'batch':
#             self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
#         else:
#             raise ValueError('%s is not a recognized param-free norm type in SPADE'
#                              % param_free_norm_type)
#
#         # The dimension of the intermediate embedding space. Yes, hardcoded.
#         nhidden = 128
#
#         pw = ks // 2
#         self.mlp_shared = nn.Sequential(
#             nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
#             nn.ReLU()
#         )
#         self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
#         self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
#
#     def forward(self, x, segmap):
#
#         # Part 1. generate parameter-free normalized activations
#         normalized = self.param_free_norm(x)
#
#         # Part 2. produce scaling and bias conditioned on semantic map
#         segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
#         actv = self.mlp_shared(segmap)
#         gamma = self.mlp_gamma(actv)
#         beta = self.mlp_beta(actv)
#
#         # apply scale and bias
#         out = normalized * (1 + gamma) + beta
#
#         return out
#
# class LayerNorm_R(nn.Module):
#     def __init__(self, num_features, eps=1e-5, affine=True):
#         super(LayerNorm_R, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps
#
#         if self.affine:
#             self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
#             self.beta = nn.Parameter(torch.zeros(num_features))
#
#     def forward(self, x):
#         shape = [-1] + [1] * (x.dim() - 1)
#         # print(x.size())
#         if x.size(0) == 1:
#             # These two lines run much faster in pytorch 0.4 than the two lines listed below.
#             mean = x.view(-1).mean().view(*shape)
#             std = x.view(-1).std().view(*shape)
#         else:
#             mean = x.view(x.size(0), -1).mean(1).view(*shape)
#             std = x.view(x.size(0), -1).std(1).view(*shape)
#
#         x = (x - mean) / (std + self.eps)
#
#         if self.affine:
#             shape = [1, -1] + [1] * (x.dim() - 2)
#             x = x * self.gamma.view(*shape) + self.beta.view(*shape)
#         return x
#
# class LayerNorm(nn.Module):
#     r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
#     """
#
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape))
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError
#         self.normalized_shape = (normalized_shape,)
#
#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             u = x.mean(1, keepdim=True)
#             s = (x - u).pow(2).mean(1, keepdim=True)
#             x = (x - u) / torch.sqrt(s + self.eps)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x
#
# class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
#     def __init__(self, dim_in, dim_out, x=8, y=8):
#         super().__init__()
#
#         c_dim_in = dim_in // 4
#         k_size = 3
#         pad = (k_size - 1) // 2
#
#         self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
#         nn.init.ones_(self.params_xy)
#         self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
#                                      nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))
#
#         self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
#         nn.init.ones_(self.params_zx)
#         self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
#                                      nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))
#
#         self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
#         nn.init.ones_(self.params_zy)
#         self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
#                                      nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))
#
#         self.dw = nn.Sequential(
#             nn.Conv2d(c_dim_in, c_dim_in, 1),
#             nn.GELU(),
#             nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
#         )
#
#         self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
#         self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
#
#         self.ldw = nn.Sequential(
#             nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
#             nn.GELU(),
#             nn.Conv2d(dim_in, dim_out, 1),
#         )
#
#     def forward(self, x):
#         x = self.norm1(x)
#         x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
#         B, C, H, W = x1.size()
#         # ----------xy----------#
#         params_xy = self.params_xy
#         x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4], mode='bilinear', align_corners=True))
#         # ----------zx----------#
#         x2 = x2.permute(0, 3, 1, 2)
#         params_zx = self.params_zx
#         x2 = x2 * self.conv_zx(
#             F.interpolate(params_zx, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
#         x2 = x2.permute(0, 2, 3, 1)
#         # ----------zy----------#
#         x3 = x3.permute(0, 2, 1, 3)
#         params_zy = self.params_zy
#         x3 = x3 * self.conv_zy(
#             F.interpolate(params_zy, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
#         x3 = x3.permute(0, 2, 1, 3)
#         # ----------dw----------#
#         x4 = self.dw(x4)
#         # ----------concat----------#
#         x = torch.cat([x1, x2, x3, x4], dim=1)
#         # ----------ldw----------#
#         x = self.norm2(x)
#         x = self.ldw(x)
#         return x
# #Global filter networks for image classification,NIPS2021
# class GlobalLocalFilter(nn.Module):
#     def __init__(self, dim, h=14, w=8):
#         super().__init__()
#         self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
#         self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
#         trunc_normal_(self.complex_weight, std=.02)
#         self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
#         self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
#
#     def forward(self, x):
#         x = self.pre_norm(x)
#         x1, x2 = torch.chunk(x, 2, dim=1)
#         x1 = self.dw(x1)
#
#         x2 = x2.to(torch.float32)
#         B, C, a, b = x2.shape
#         x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')
#
#         weight = self.complex_weight
#         if not weight.shape[1:3] == x2.shape[2:4]:
#             weight = F.interpolate(weight.permute(3,0,1,2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1,2,3,0)
#
#         weight = torch.view_as_complex(weight.contiguous())
#
#         x2 = x2 * weight
#         x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')
#
#         x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
#         x = self.post_norm(x)
#         return x
# class GlobalFilter_old(nn.Module):
#
#     def __init__(self, dim, h=14, w=8):
#         super().__init__()
#         self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
#
#     def forward(self, x):
#
#         B, H, W, C = x.shape
#         x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
#         weight = torch.view_as_complex(self.complex_weight)
#         x = x * weight
#         x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')
#         return x
# def get_dwconv(dim, kernel, bias):
#     return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)
#
# class gnconv(nn.Module):
#     def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
#         super().__init__()
#         self.order = order
#         self.dims = [dim // 2 ** i for i in range(order)]
#         self.dims.reverse()
#         self.proj_in = nn.Conv2d(dim, 2 * dim, 1)
#
#         if gflayer is None:
#             self.dwconv = get_dwconv(sum(self.dims), 7, True)
#         else:
#             self.dwconv = gflayer(sum(self.dims), h=h, w=w)
#
#         self.proj_out = nn.Conv2d(dim, dim, 1)
#
#         self.pws = nn.ModuleList(
#             [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
#         )
#
#         self.scale = s
#         print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)
#
#     def forward(self, x, mask=None, dummy=False):
#         B, C, H, W = x.shape
#
#         fused_x = self.proj_in(x)
#         pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
#
#         dw_abc = self.dwconv(abc) * self.scale
#
#         dw_list = torch.split(dw_abc, self.dims, dim=1)
#         x = pwa * dw_list[0]
#
#         for i in range(self.order - 1):
#             x = self.pws[i](x) * dw_list[i + 1]
#
#         x = self.proj_out(x)
#
#         return x