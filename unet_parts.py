import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        '''
            nn.Upsample()
                    scale_factor： 指定输出的尺寸是输入尺寸的倍数;
                    mode: 上采样的算法可选 ‘nearest’, ‘linear’, ‘bilinear’, ‘bicubic’，‘trilinear’.  最近邻、线性、双线性插值算法
                    align_corners：为True，则输入和输出张量的角像素对齐，从而保留这些像素的值        
        '''
        # 在Unet结构中设置了bilinear=False
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels=in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # print(f"x1.size:{x1.size()}, x2.size:{x2.size()}")
        x1 = self.up(x1)  # 这里的尺寸应该是[batch, channel, height, weight]
        # x1的size比x2的小
        # 填充x1
        if x1.size() != x2.size():
            diffY = x2.size()[2] - x1.size()[2]  # 在height上的差距
            diffX = x2.size()[3] - x1.size()[3]  # 在weight上的差距
            x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])   # 填充操作是对x1的四个边（左、右、上、下）进行的

        # 裁剪 X2， X2的size比X1的大；
        # x2 = x2[:, :, :x1.shape[2], :x1.shape[3]]
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.out_conv(x)


