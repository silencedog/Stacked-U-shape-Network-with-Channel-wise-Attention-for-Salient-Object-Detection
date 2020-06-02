import torch
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn as nn

class _ConvBatchNormReLU(nn.Sequential):
    def __init__(self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            "bn",
            nn.BatchNorm2d(out_channels),
        )

        if relu:
            self.add_module("relu", nn.ReLU())

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)

class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels):
        super(_ASPPModule, self).__init__()
        output_stride = 16
        if output_stride == 8:
            pyramids = [12, 24, 36]
        elif output_stride == 16:
            pyramids = [6, 12, 18]
        self.stages = nn.Module()
        self.stages.add_module(
            "c0", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)
        )
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding, dilation),
            )
        self.imagepool = nn.Sequential(
            OrderedDict(
                [
                    ("pool", nn.AdaptiveAvgPool2d((1,1))),
                    ("conv", _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)),
                ]
            )
        )
        self.fire = nn.Sequential(
            OrderedDict(
                [
                    ("conv", _ConvBatchNormReLU(out_channels * 5, out_channels, 3, 1, 1, 1)),
                    ("dropout", nn.Dropout2d(0.1))
                ]
            )
        )

    def forward(self, x):
        h = self.imagepool(x)
        h = [F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)]
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        h = self.fire(h)
        return h

def crop(data1, data2, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    _, _, h2, w2 = data2.size()
    #assert(h2 <= h1 and w2 <= w1)
    data = data1[:, :, crop_h:crop_h+h2, crop_w:crop_w+w2]
    return data


