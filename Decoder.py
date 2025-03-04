

from Blocks import *

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
