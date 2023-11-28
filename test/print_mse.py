import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor
from torch.nn.functional import mse_loss


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('RGB')  # 确保图像为RGB格式
        img.filename = filename  # 添加文件名属性
        images.append(img)
    return images


# 两个文件夹的路径
folder1_path = 'D:\\dataset\\UIEB_Dataset\\raw-90'
folder2_path = 'D:\\dataset\\UIEB_Dataset\\reference-90'

# 从文件夹中加载图像
images_folder1 = load_images_from_folder(folder1_path)
images_folder2 = load_images_from_folder(folder2_path)

# 检查两个文件夹中的图像数是否相同
if len(images_folder1) != len(images_folder2):
    raise ValueError("The number of images in the two folders must be the same.")

# 将图像转换为 PyTorch 张量
to_tensor = ToTensor()

# 计算均方误差（MSE）并打印结果
for img1, img2 in zip(images_folder1, images_folder2):
    # 转换图像为 PyTorch 张量
    img1_tensor = to_tensor(img1).unsqueeze(0)
    img2_tensor = to_tensor(img2).unsqueeze(0)

    # 计算 MSE
    mse = mse_loss(img1_tensor, img2_tensor)

    # 打印结果
    print(f"MSE between {img1.filename} and {img2.filename}: {mse.item()}")
