import torch

from unet_model import *
from torchsummary import summary


if __name__ == "__main__":
    print("----Unet_P1------")
    # 创建模型实例
    model = UNet(3, 3).cuda()
    input_size = (3, 256, 256)
    batch_size = 1
    # 打印模型的详细信息
    summary(model, input_size=input_size, batch_size=batch_size, device="cuda")
    # print(model)


