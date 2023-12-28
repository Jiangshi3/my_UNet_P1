import torch

# from unet_model import *
from unet_cbam_densenet_2 import *
from torchsummary import summary


if __name__ == "__main__":
    print("----Unet_P1------")
    # 创建模型实例
    model = UNet(3, 3).cuda()
    input_size = (3, 256, 256)      # 最底层的size[4, 512, 8, 8]
    # input_size = (3, 512, 512)    # 最底层的size[4, 512, 16, 16]
    batch_size = 2
    # 打印模型的详细信息
    summary(model, input_size=input_size, batch_size=batch_size, device="cuda")
    # print(model)


