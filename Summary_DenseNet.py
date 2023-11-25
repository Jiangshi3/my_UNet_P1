import torch

from unet_cbam_densenet import *
from densenet_2 import *
from torchsummary import summary


if __name__ == "__main__":
    print("----DenseNet------")
    # 创建模型实例
    model = DenseNet(in_channel=256, growth_rate=24, block_config=(4, 14), bn_size=4).cuda()
    input_size = (256, 16, 16)
    # input_size = (128, 32, 32)
    # input_size = (64, 64, 64)
    # input_size = (32, 128, 128)
    # input_size = (16, 256, 256)
    batch_size = 1
    # 打印模型的详细信息
    summary(model, input_size=input_size, batch_size=batch_size, device="cuda")
    # print(model)

'''
找到了公式：设growth_rate=x；block_config=(m, n)； 只要满足：(inChannel+x*m)/2+x*n=outChannel，这里的x, (m, n)超参数即可得到自己想要的size
input:(16×256×256)  超参数DenseNet(in_channel=16, growth_rate=8, block_config=(2, 2), bn_size=4)； output：(32, 128, 128)
input:(32, 128, 128)  超参数DenseNet(in_channel=32, growth_rate=12, block_config=(2, 3), bn_size=4)； output：(64, 64, 64)
input:(64, 64, 64)  超参数DenseNet(in_channel=64, growth_rate=16, block_config=(2, 5), bn_size=4)； output：(128, 32, 32)
input:(128, 32, 32) 超参数DenseNet(in_channel=128, growth_rate=24, block_config=(2, 7), bn_size=4)； output：(256, 16, 16) // 若growth_rate=24,则block_config=(m, n) 这里只要满足m+2n=16即可
input:(256, 16, 16) 超参数DenseNet(in_channel=256, growth_rate=24, block_config=(4, 14), bn_size=4)； output：(512, 8, 8) // 若growth_rate=24,则block_config=(m, n) 这里只要满足m+2n=32即可

'''


