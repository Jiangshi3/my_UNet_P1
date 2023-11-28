import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# from unet_model import *
from unet_cbam_densenet_2 import *
import argparse
from utils.data_utils import *
# from SSIM.SSIM_1 import *
from SSIM.SSIM_2 import ssim, ms_ssim, SSIM, MS_SSIM

print("-----train.py-----")


parser = argparse.ArgumentParser(description="this is a demo")
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.002, help="adam: learning rate")
args = parser.parse_args()

# training params
epoch = args.epoch
num_epochs = args.num_epochs
batch_size = args.batch_size
lr = args.lr
# dataset_raw_path = "D:\\dataset\\UIEB_Dataset\\raw-890"
# dataset_ref_path = "D:\\dataset\\UIEB_Dataset\\reference-890"
root_dir = "D:\\dataset\\UIEB_Dataset"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_UNet = UNet(3, 3)  # (in_channel, out_channel)
model_UNet.to(device)

loss_function = nn.MSELoss().to(device)

optimizer_UNet = torch.optim.Adam(model_UNet.parameters(), lr=lr)
# optimizer_UNet = optimizer_UNet.to(device)

# 创建数据转换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 后面看一下这个数据集论文中，他们是怎样对此数据集UIEB进行预处理的？？（或者其他论文有用到此数据集如何处理的）
    # transforms.RandomCrop((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# dataset = PairedImageDataset(root_dir, transform=transform)
paired_dataset = PairedImageDataset_2(root_dir, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(paired_dataset))
val_size = len(paired_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(paired_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# paired_dataloader = DataLoader(dataset=paired_dataset, batch_size=batch_size, shuffle=True)

# 定义损失函数权重
weight_mse = 1  # 权重1，用于 MSE 损失
weight_ssim = 1  # 权重2，用于 SSIM 损失

for epoch in range(epoch, num_epochs):
    model_UNet.train()
    running_loss_train = 0.0
    sum_ssim_loss_train = 0.0
    # 初始化 SSIM 模块
    ssim_module = SSIM(data_range=255, size_average=True, channel=3)  # channel=1 for灰度图像
    print(f'-------------Epoch {epoch+1} begin training-----------')
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # torch.Size([8, 3, 256, 256])
        optimizer_UNet.zero_grad()
        outputs = model_UNet(inputs)
        loss_mse = loss_function(outputs, targets)  # mse损失
        # 计算 SSIM 损失
        loss_ssim = 1 - ssim_module(outputs, targets)  # 返回一个标量
        sum_ssim_loss_train += loss_ssim

        # 计算总体损失
        total_loss = weight_mse * loss_mse + weight_ssim * loss_ssim
        running_loss_train += total_loss.item()

        total_loss.backward()
        optimizer_UNet.step()
    print("len(train_loader)=", len(train_loader))
    # print("running_loss_train=", running_loss_train)
    # avg_train_loss = running_loss_train / len(train_loader)  # avg_train_loss太小了；不打算除以len(train_loader)了
    # print(f"Epoch [{epoch + 1}/{num_epochs}], Train_Loss: {avg_train_loss}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], sum_ssim_loss_train: {sum_ssim_loss_train}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train_Loss_total: {running_loss_train}")

    print(f'***********Epoch {epoch + 1} begin Validation***********')
    # Validation
    model_UNet.eval()
    with torch.no_grad():
        val_loss = 0.0
        sum_ssim_loss_val = 0.0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # print(f"inputs.size():{inputs.size()}--targets.size():{targets.size()}")
            outputs = model_UNet(inputs)
            # print(f"outputs.size():{outputs.size()}")

            loss_mse = loss_function(outputs, targets)  # mse损失
            # 计算 SSIM 损失
            loss_ssim = 1 - ssim_module(outputs, targets)  # 返回一个标量
            sum_ssim_loss_val += loss_ssim

            # 计算总体损失
            total_loss = weight_mse * loss_mse + weight_ssim * loss_ssim
            val_loss += total_loss.item()
        print("len(val_loader)=", len(val_loader))
        print(f"Epoch [{epoch + 1}/{num_epochs}], sum_ssim_loss_val: {sum_ssim_loss_val}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val_Loss_total: {val_loss} \n")

# Save the trained model if needed
torch.save(model_UNet.state_dict(), "trained_model.pth")

print('Finished Training')

'''  
for i, batch in enumerate(dataloader):
    raw_img, ref_img = batch
    raw_img, ref_img = raw_img.to(device), ref_img.to(device)
    raw_output = model_UNet(raw_img)
    loss = loss_function(raw_output, ref_img)

    optimizer_UNet.zero_grad()
    loss.backward()
    optimizer_UNet.step()
    running_loss += loss
print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(dataloader)}')
'''




