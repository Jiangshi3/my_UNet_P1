# SSIM具有对称性，即SSIM(x,y)=SSIM(y,x)
# SSIM是一个0到1之间的数，越大表示输出图像和无失真图像的差距越小，即图像质量越好。当两幅图像一模一样时，SSIM=1；

import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ssim_2 import ssim, ms_ssim, SSIM, MS_SSIM
from torch.autograd import Variable


ssim_module = SSIM(data_range=255, size_average=True, channel=3)  # channel=1 for灰度图像
# ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)  # channel=1 for灰度图像

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
    loss_ssim = 1 - ssim_module(img1_tensor, img2_tensor)  # 返回一个标量
    print(f"ssim between\n{img1_path}\n{img2_path}\n : {loss_ssim}")
    # return loss_ssim


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
    # raw_path = 'D:\\dataset\\UIEB_Dataset\\raw-10\\5554.png'  # 0.00104522705078125
    # raw_path = 'D:\\dataset\\UIEB_Dataset\\raw-10-out-P4-Resize256\\result_5554.png'  # 0.0006222724914550781
    raw_path = 'D:\\dataset\\UIEB_Dataset\\raw-10-out-P5-Resize512\\result_5554.png'  # 0.0017660260200500488
    reference_path = 'D:\\dataset\\UIEB_Dataset\\reference-10\\5554.png'
    calculate_ssim(raw_path, reference_path)


# from skimage.metrics import structural_similarity as ssim
# from PIL import Image
# import numpy as np
#
# raw_path = 'D:\\dataset\\UIEB_Dataset\\raw-10\\5554.png'  # 0.00104522705078125  0.9511327778686033
# # raw_path = 'D:\\dataset\\UIEB_Dataset\\raw-10-out-P4-Resize256\\result_5554.png'  # 0.0006222724914550781  0.895319857109255
# # raw_path = 'D:\\dataset\\UIEB_Dataset\\raw-10-out-P5-Resize512\\result_5554.png'  # 0.0017660260200500488  0.8821506560478949
# reference_path = 'D:\\dataset\\UIEB_Dataset\\reference-10\\5554.png'
#
# img1 = np.array(Image.open(raw_path))
# img2 = np.array(Image.open(reference_path))
#
#
# if __name__ == "__main__":
# 	# If the input is a multichannel (color) image, set multichannel=True.
#     print(ssim(img1, img2, multichannel=True))
