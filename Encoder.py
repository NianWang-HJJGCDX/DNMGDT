
from Blocks import *


# class Encoder_Semi(nn.Module):
#     def __init__(self, channel,conv=default_conv, res_blokcs=16):
#         super(Encoder_Semi, self).__init__()
#         self.channels = channel
#         self.conv_d1 = nn.Conv2d(3, self.channels, 7, 1, 3)
#         self.conv_d2 = nn.Conv2d(self.channels, self.channels*2, 4, 2, 1)
#         self.conv_d3 = nn.Conv2d(self.channels*2, self.channels * 4, 4, 2, 1)
#         self.conv_d4 = nn.Conv2d(self.channels*4, self.channels * 8, 4, 2, 1)
#         self.act = nn.ReLU()
#         m_body = [ResBlock(conv, self.channels*8, 3, act=self.act, res_scale=1
#                            ) for _ in range(res_blokcs)
#                   ]
#         self.body = nn.Sequential(*m_body)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         encoder_layer = []
#         x = self.conv_d1(x)
#         encoder_layer.append(x)
#         x = self.conv_d2(self.relu(x))
#         encoder_layer.append(x)
#         x = self.conv_d3(self.relu(x))
#         encoder_layer.append(x)
#         x = self.conv_d4(self.relu(x))
#         # encoder_layer.append(x)
#         out = self.body(self.relu(x))
#         out = out+x
#         encoder_layer.append(out)
#         return encoder_layer

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

