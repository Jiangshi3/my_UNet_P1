import torch
import torch.nn as nn
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# 创建解析器
parser = argparse.ArgumentParser(description='这是一个示例程序')

# 添加参数
parser.add_argument('--param', type=int, default=1, help='这是一个整数参数')
parser.add_argument('--flag', action='store_true', help='这是一个布尔标志')
parser.add_argument('--string', type=str, choices=['a', 'b', 'c'], help='这是一个字符串选择参数')

# 解析命令行参数
args = parser.parse_args()

print('param:', args.param)
print('flag:', args.flag)
print('string:', args.string)


print(torch.cuda.device_count())


#
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.p1 = torch.nn.Sigmoid()
#
#     def forward(self, x):
#         return self.p1(x)
#
#
# mymodel = Net()
# path = 'D:\\dataset\\UIEB_Dataset\\challenging-60/2.png'
# img = Image.open(path)
# tensor = torchvision.transforms.ToTensor()
# # img_new = mymodel(img)
# img_tensor = tensor(img)
# img_output = mymodel(img_tensor)
# img.show()
#
# # plt.imshow(img)
# # plt.imshow(img_tensor)
# # plt.show(img_output)
# print("img.shape : {}", img_tensor.shape)
# print("img.shape : {}", img_output.shape)
# # print("img_new.shape : {}", img_new.shape)
#
