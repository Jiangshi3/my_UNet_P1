'''
11月23号：(把CBAM放在解码器的后面)
    此Unet+CBAM结构：
    input：3×256×256  -(DoubleConv)> 16×256×256  ＋          -(Up5+cat+conv)-> 16×256×256  -(OutConv)->3×256×256
        -(DenseNet1) ->32×128×128                  ＋          -(Up4+cat+conv)-> 32×128×128 ↑  +CBAM
         -(DN2) ->64×64×64                  ＋          -(Up3+cat+conv)-> 64×64×64 ↑   +CBAM
          -(DN3) ->128×32×32               ＋          -(Up2+cat+conv)-> 128×32×32 ↑  +CBAM
           -(DN4) ->256×16×16            ＋          -(Up1+cat+conv)-> 256×16×16 ↑  +CBAM
            -(DN5) ->512×8×8             -----------→↑
'''
from unet_parts import *
from densenet_2 import *


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
        self.DN1 = DenseNet(in_channel=16, growth_rate=8, block_config=(2, 2), bn_size=4, drop_rate=0)  # output: (32, 128, 128)
        # self.down1 = Down(16, 32)
        self.DN2 = DenseNet(in_channel=32, growth_rate=12, block_config=(2, 3), bn_size=4, drop_rate=0)
        # self.down2 = Down(32, 64)
        self.DN3 = DenseNet(in_channel=64, growth_rate=16, block_config=(2, 5), bn_size=4, drop_rate=0)
        # self.down3 = Down(64, 128)
        self.DN4 = DenseNet(in_channel=128, growth_rate=24, block_config=(2, 7), bn_size=4, drop_rate=0)
        # self.down4 = Down(128, 256)
        self.DN5 = DenseNet(in_channel=256, growth_rate=24, block_config=(4, 14), bn_size=4, drop_rate=0)
        # self.down5 = Down(256, 512)

        # self.up1 = Up(1024, 512, bilinear=bilinear)
        self.up1 = Up(512, 256, bilinear=bilinear)
        self.channel_up1 = ChannelAttention(in_planes=256, ratio=16)
        self.spatial_up1 = SpatialAttention(kernel_size=3)
        self.up2 = Up(256, 128, bilinear)
        self.channel_up2 = ChannelAttention(in_planes=128, ratio=8)
        self.spatial_up2 = SpatialAttention(kernel_size=3)
        self.up3 = Up(128, 64, bilinear)
        self.channel_up3 = ChannelAttention(in_planes=64, ratio=4)
        self.spatial_up3 = SpatialAttention(kernel_size=7)
        self.up4 = Up(64, 32, bilinear)
        self.channel_up4 = ChannelAttention(in_planes=32, ratio=4)
        self.spatial_up4 = SpatialAttention(kernel_size=7)
        self.up5 = Up(32, 16, bilinear)

        self.out = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.DN1(x1)
        # x2 = self.down1(x1)
        x3 = self.DN2(x2)
        # x3 = self.down2(x2)
        x4 = self.DN3(x3)
        # x4 = self.down3(x3)
        x5 = self.DN4(x4)
        # x5 = self.down4(x4)
        x6 = self.DN5(x5)
        # x6 = self.down5(x5)

        # y1 = self.up1(x_cbam2, x5)
        y1 = self.up1(x6, x5)
        y1_channel_up1 = self.channel_up1(y1) * y1
        y1_cbam_up1 = self.spatial_up1(y1_channel_up1) * y1_channel_up1

        y2 = self.up2(y1_cbam_up1, x4)
        y2_channel_up2 = self.channel_up2(y2) * y2
        y2_cbam_up2 = self.spatial_up2(y2_channel_up2) * y2_channel_up2

        y3 = self.up3(y2_cbam_up2, x3)
        y3_channel_up3 = self.channel_up3(y3) * y3
        y3_cbam_up3 = self.spatial_up3(y3_channel_up3) * y3_channel_up3

        y4 = self.up4(y3_cbam_up3, x2)
        y4_channel_up4 = self.channel_up4(y4) * y4
        y4_cbam_up4 = self.spatial_up4(y4_channel_up4) * y4_channel_up4

        y5 = self.up5(y4_cbam_up4, x1)
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

