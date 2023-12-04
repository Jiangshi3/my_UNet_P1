import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from SSIM_2 import ssim, ms_ssim, SSIM, MS_SSIM
from torch.autograd import Variable
import os

# ssim_module = SSIM(data_range=255, size_average=True, channel=3)  # channel=1 for灰度图像
ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)  # channel=1 for灰度图像

# 定义图像转换
transform = transforms.Compose([
    transforms.ToTensor()
])


def calculate_ssim(img1_path, img2_path):
    # 读取图像
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    # 对图像进行转换，将其转换为PyTorch的张量
    img1_tensor = transform(img1).unsqueeze(0)
    img2_tensor = transform(img2).unsqueeze(0)

    # loss = mse_loss(img1_tensor, img2_tensor)
    loss_ssim = 1 - ms_ssim_module(img1_tensor, img2_tensor)  # 返回一个标量

    return loss_ssim


def calculate_ssim_for_folder(raw_folder, reference_folder):
    # 获取文件夹中的所有图像文件
    raw_images = os.listdir(raw_folder)
    reference_images = os.listdir(reference_folder)

    # 按照文件名排序确保对应图像匹配
    raw_images.sort()
    reference_images.sort()

    for raw_img, ref_img in zip(raw_images, reference_images):
        raw_img_path = os.path.join(raw_folder, raw_img)
        ref_img_path = os.path.join(reference_folder, ref_img)

        # 计算均方误差损失
        ssim_value = calculate_ssim(raw_img_path, ref_img_path)

        # 打印结果
        print(f"ssim between {raw_img} and {ref_img}: {ssim_value}")


if __name__ == '__main__':
    raw_folder = 'D:\\dataset\\UIEB_Dataset\\raw-10-test'
    reference_folder = 'D:\\dataset\\UIEB_Dataset\\reference-10-test'
    calculate_ssim_for_folder(raw_folder, reference_folder)


'''
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
# X: (N,3,H,W) a batch of non-negative RGB images (0~255)
# Y: (N,3,H,W)  

# calculate ssim & ms-ssim for each image
ssim_val = ssim( X, Y, data_range=255, size_average=False) # return (N,)
ms_ssim_val = ms_ssim( X, Y, data_range=255, size_average=False ) #(N,)

# set 'size_average=True' to get a scalar value as loss. see tests/tests_loss.py for more details
ssim_loss = 1 - ssim( X, Y, data_range=255, size_average=True) # return a scalar
ms_ssim_loss = 1 - ms_ssim( X, Y, data_range=255, size_average=True )

# reuse the gaussian kernel with SSIM & MS_SSIM. 
ssim_module = SSIM(data_range=255, size_average=True, channel=3) # channel=1 for grayscale images
ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)

ssim_loss = 1 - ssim_module(X, Y)
ms_ssim_loss = 1 - ms_ssim_module(X, Y)
'''


