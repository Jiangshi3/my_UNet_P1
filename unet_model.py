'''
11月22号：（把CBAM结构加在了①DoubleConv后面；②Unet的最底层的conv后面；  CBAM不改变size）
    此Unet+CBAM结构：
        input：3×256×256  -(DoubleConv)> 16×256×256    -(CBAM)->16×256×256  -(Down)>32×128×128  -(Down)>64×64×64
        -(Down)>128×32×32   -(Down)>256×16×16  -(Down)>512×8×8  +(CBAM)-512×8×8  -(Up)>        -(output)>3×256×256
'''

from unet_parts import *


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=32):  # ratio 原始为16
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层
        # 这两个操作使得尺寸： (batch_size, channels, height, width) -> (batch_size, channels, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化层

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):  # 这里kernel_size等于3或者7； 看自己传入尺寸大小，如果height和weight比较小，就令k=3
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# class CBAM(nn.Module):
#     def __init__(self, planes):
#         super(CBAM, self).__init__()
#         self.ca = ChannelAttention(planes)  # planes是feature map的通道个数
#         self.sa = SpatialAttention()
#
#     def forward(self, x):
#         x = self.ca(x) * x  # 广播机制
#         x = self.sa(x) * x  # 广播机制
#         return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.channel1 = ChannelAttention(in_planes=16, ratio=4)  # 这里传入[1, 16, 256, 256] ;so 我的radio小一些，kernel_size大一些
        self.spatial1 = SpatialAttention(kernel_size=7)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.down5 = Down(256, 512)
        self.channel2 = ChannelAttention(in_planes=512, ratio=16)  # 这里传入[1, 512, 8, 8] ;so 我的radio大一些，kernel_size小一些
        self.spatial2 = SpatialAttention(kernel_size=3)
        # self.down5 = Down(512, 1024)

        # self.up1 = Up(1024, 512, bilinear=bilinear)
        self.up1 = Up(512, 256, bilinear=bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.up5 = Up(32, 16, bilinear)

        self.out = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x_channel1 = self.channel1(x1) * x1
        x_cbam1 = self.spatial1(x_channel1) * x_channel1
        x2 = self.down1(x_cbam1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x_channel2 = self.channel2(x6) * x6
        x_cbam2 = self.spatial2(x_channel2) * x_channel2

        y1 = self.up1(x_cbam2, x5)
        # y1 = self.up1(x6, x5)
        y2 = self.up2(y1, x4)
        y3 = self.up3(y2, x3)
        y4 = self.up4(y3, x2)
        y5 = self.up5(y4, x1)
        return self.out(y5)





'''
# 梯度检查点是一种内存优化策略，可以在神经网络训练过程中减少GPU内存的使用。
# 在PyTorch中，torch.utils.checkpoint函数可以将一个模块（在这个例子中是self.down1和self.down2）转换为一个使用梯度检查点的模块。
# 这意味着在前向传播过程中，这些模块的中间输出将不会被保存，而是在反向传播过程中重新计算。这可以大大减少GPU内存的使用，但是会增加一些计算开销。

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
'''

