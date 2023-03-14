""" Parts of the U-Net model """

from e2cnn import gspaces                                         
from e2cnn import nn                                               
import torch
import torch.nn.functional as F

def conv7x7(in_type: nn.FieldType, out_type: nn.FieldType, stride=1, padding=3,
            dilation=1, bias=False):
    """7x7 convolution with padding"""
    return nn.R2Conv(in_type, out_type, 7,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


def conv5x5(in_type: nn.FieldType, out_type: nn.FieldType, stride=1, padding=2,
            dilation=1, bias=False):
    """5x5 convolution with padding"""
    return nn.R2Conv(in_type, out_type, 5,
                      stride=stride,
                      padding=padding, 
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


def conv3x3(in_type: nn.FieldType, out_type: nn.FieldType, stride=1, padding=1,
            dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


def conv1x1(in_type: nn.FieldType, out_type: nn.FieldType, stride=1, padding=0,
            dilation=1, bias=False):
    """1x1 convolution with padding"""
    return nn.R2Conv(in_type, out_type, 1,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )


class e2DoubleConv(torch.nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        """
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        """
        self.r2_act = gspaces.Rot2dOnR2(N=8) 
        self.in_type = nn.FieldType(self.r2_act,  in_channels*[self.r2_act.trivial_repr]) 
        self.mid_type = nn.FieldType(self.r2_act, (mid_channels//8)*[self.r2_act.regular_repr]) 
        self.out_type = nn.FieldType(self.r2_act, (out_channels//8)*[self.r2_act.regular_repr]) 
        self.double_conv = nn.SequentialModule(
            conv3x3(self.in_type, self.mid_type),
            nn.InnerBatchNorm(self.mid_type),
            nn.ReLU(self.mid_type),
            conv3x3(self.mid_type, self.out_type),
            nn.InnerBatchNorm(self.out_type),
            nn.ReLU(self.out_type)
            )
    def forward(self, x):
        x = nn.GeometricTensor(x, self.in_type)
        return self.double_conv(x).tensor


class e2Down(torch.nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.max_pool = torch.nn.MaxPool2d(2)
        self.conv = e2DoubleConv(in_channels, out_channels)
       

    def forward(self, x):
        x = self.max_pool(x)
        x = self.conv(x)
        return x


class e2Up(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = e2DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class e2OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(e2OutConv, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        r2_act = gspaces.Rot2dOnR2(N=8)   
        self.in_type=nn.FieldType(r2_act,  in_channels*[r2_act.trivial_repr])
        self.out_type=nn.FieldType(r2_act, (out_channels//8)*[r2_act.regular_repr])  
        self.conv = conv1x1(self.in_type, self.out_type)

    def forward(self, x):
        x = nn.GeometricTensor(x, self.in_type)
        return self.conv(x).tensor

class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)