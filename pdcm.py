import torch
import torch.nn as nn

class PDCM(nn.Module):
    def __init__(self, c_in):
        super(PDCM, self).__init__()
        self.rate = 1
        self.conv = nn.Conv2d(c_in, c_in, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation  = self.rate*1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(c_in, c_in, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation   = self.rate*2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(c_in, c_in, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation   = self.rate*4 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(c_in, c_in, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)
        dilation   = self.rate*6 if self.rate >= 1 else 1
        self.conv4 = nn.Conv2d(c_in, c_in, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu4 = nn.ReLU(inplace=True)

        self._initialize_weights()
    def forward(self, x):
        o   = self.relu(self.conv(x))
        o1  = self.relu1(self.conv1(o))
        o2  = self.relu2(self.conv2(o))
        o3  = self.relu3(self.conv3(o))
        o4  = self.relu4(self.conv4(o))

        out = o + o1 + o2 + o3 + o4

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
