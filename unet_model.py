'''
11月15号：
    此Unet结构：
        input：3×256×256  -(DoubleConv)> 16×256×256   -(Down)>32×128×128  -(Down)>64×64×64  -(Down)>128×32×32
        -(Down)>256×16×16  -(Down)>512×8×8    -(Up)>        -(output)>3×256×256
'''

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.down5 = Down(256, 512)
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
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        y1 = self.up1(x6, x5)
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

