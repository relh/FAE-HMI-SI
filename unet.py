import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn

# --- UNet --- #
class UNet(nn.Module):
    def __init__(self, in_channels, out_im_channels, batchnorm, dropout=0.3, regression=False, bins=80, bc=64):
        super(UNet, self).__init__()

        self.inc = inconv(in_channels, bc*1, batchnorm)
        self.down1 = down(bc*1, bc*2, batchnorm, dropout=dropout)
        self.down2 = down(bc*2, bc*4, batchnorm, dropout=dropout)
        self.down3 = down(bc*4, bc*8, batchnorm, dropout=dropout)
        self.down4 = down(bc*8, bc*8, batchnorm, dropout=dropout)
        self.up1 = up(bc*16, bc*4, batchnorm, dropout=dropout)
        self.up2 = up(bc*8, bc*2, batchnorm, dropout=dropout)
        self.up3 = up(bc*4, bc*1, batchnorm, dropout=dropout)
        self.up4 = up(bc*2, bc*2, batchnorm, dropout=dropout)
        self.outc = outconv(bc*2, out_im_channels, regression, bins)

    def forward(self, x):
        #x0 = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

# --- helper modules --- #
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
      nn.ReLU(inplace=True),
    )


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, batchnorm=True):
        super(double_conv, self).__init__()
        if batchnorm:
            self.conv = nn.Sequential(
              nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
              nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(
              nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
              nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, batchnorm):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, batchnorm)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, batchnorm, dropout=None):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch, batchnorm))

        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.mpconv(x)

        if self.dropout:
            x = self.dropout(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, batchnorm, method='conv', dropout=None):
        super(up, self).__init__()

        if method == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif method == 'conv':
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        elif method == 'upconv':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2, padding=0),
            )
        elif method == 'none':
            self.up = nn.Identity()

        self.conv = double_conv(in_ch, out_ch, batchnorm)

        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # up conv here

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        if self.dropout:
            x = self.dropout(x)

        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, regression, bins=80):
        super(outconv, self).__init__()
        if regression:
            self.conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch*bins, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
