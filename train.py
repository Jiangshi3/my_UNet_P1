import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from unet_cbam_densenet_2 import *
import argparse
from utils.data_utils import *
from SSIM.SSIM_2 import ssim, ms_ssim, SSIM, MS_SSIM
from torch.utils.tensorboard import SummaryWriter

print("-----train.py-----")


parser = argparse.ArgumentParser(description="this is a demo")
parser.add_argument("--epoch", type=int, default=30, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=40, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument("--m_weight", default=[1.0, 0.0], help="weight: [L1/L2, MS_SSIM]")
parser.add_argument("--m_resize", default=(512, 512), help="resize images to this size")
parser.add_argument("--m_save_path", default="pth/trained_model_12-23-resume-1.pth", help="save path")
parser.add_argument("--m_resume_path", default="pth/trained_model_12-23-resume-1.pth", help="resume path")
parser.add_argument("--m_Resume", type=bool, default=False, help="continue training from")
args = parser.parse_args()

# training params
epoch = args.epoch
num_epochs = args.num_epochs
batch_size = args.batch_size
lr = args.lr
m_weight = args.m_weight
m_resize = args.m_resize
m_save_path = args.m_save_path
m_resume_path = args.m_resume_path
m_Resume = args.m_Resume

root_dir = "D:\\dataset\\UIEB_Dataset"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print("device_count :{}".format(torch.cuda.device_count()))

model_UNet = UNet(3, 3)  # (in_channel, out_channel)
model_UNet.to(device)

if m_Resume:
    path_checkpoint = m_resume_path
    # checkpoint = torch.load(path_checkpoint, map_location=torch.device('gpu'))
    checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu'))
    model_UNet.load_state_dict(checkpoint)

# 创建数据转换
transform = transforms.Compose([
    # transforms.Resize((512, 512)),  # 后面看一下这个数据集论文中，他们是怎样对此数据集UIEB进行预处理的？？（或者其他论文有用到此数据集如何处理的）
    transforms.Resize(m_resize),
    # transforms.Resize((256, 256)),
    # transforms.RandomCrop((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

paired_dataset = PairedImageDataset_2(root_dir, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(paired_dataset))
val_size = len(paired_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(paired_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# paired_dataloader = DataLoader(dataset=paired_dataset, batch_size=batch_size, shuffle=True)

l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()
ms_ssim_module = MS_SSIM(data_range=255, size_average=True, channel=3)  # 初始化ms_ssim模块
# ssim_module = SSIM(data_range=255, size_average=True, channel=3)  # 初始化ssim模块
# weight = [0.16, 0.84]  # (1-0.84)*L1 + 0.84*MS_SSIM    # 论文中说是0.84，但论文的代码是用的0.025？
# weight = [1.0, 1.0]


def custom_loss(output, target, weights):
    # l1 = weights[0] * l1_loss(output, target)
    # ms_ssim_loss = weights[1] * (1 - ms_ssim_module(output, target))
    l2 = weights[0] * l2_loss(output, target)
    # print(f"L1Loss: {l1.item()}, MS_SSIM: {ms_ssim_loss}\n")
    '''  【挖坑】
        查看L1和ms_ssim损失的值（排除一方太大一方太小导致没有效果）
        本实验结果：L1Loss比MS_SSIM在一个batch中大了20倍左右； 需要调整？（提高MS_SSIM的权重？）   需要让两个损失贡献大致相同吗？？？
            L1Loss: 0.08015632629394531, MS_SSIM: 0.004331073723733425
    '''
    # total_loss = l1 + ms_ssim_loss
    return l2


optimizer_UNet = torch.optim.Adam(model_UNet.parameters(), lr=lr)
# optimizer_UNet = optimizer_UNet.to(device)

# 定义损失函数权重
# weight_mse = 1  # 权重1，用于 MSE 损失
# weight_ssim = 1  # 权重2，用于 SSIM 损失

writer = SummaryWriter("./logs")

best_val_loss = float('inf')
for epoch in range(epoch, num_epochs):
    model_UNet.train()
    running_loss_train = 0.0
    train_custom_loss = 0.0

    print(f'-------------Epoch {epoch+1} begin training-----------')
    for inputs, targets in train_loader:
        # print(f"input_size:{inputs.size()}, target_size:{targets.size()}")  # ----print test----
        inputs, targets = inputs.to(device), targets.to(device)
        # torch.Size([8, 3, 256, 256])
        optimizer_UNet.zero_grad()
        outputs = model_UNet(inputs)
        writer.add_image("train_img", outputs, global_step=epoch, dataformats='NCHW')  # add_image
        # loss_mse = loss_function(outputs, targets)  # mse损失
        # 计算 SSIM 损失
        # loss_ssim = 1 - ssim_module(outputs, targets)  # 返回一个标量
        # 计算 MS_SSIM 损失
        # loss_ms_ssim = 1 - ms_ssim_module(outputs, targets)
        # sum_ssim_loss_train += loss_ms_ssim
        train_custom_loss = custom_loss(outputs, targets, m_weight)
        running_loss_train += train_custom_loss.item()

        train_custom_loss.backward()
        optimizer_UNet.step()
    writer.add_scalar("train_loss", running_loss_train, epoch+1)
    if epoch == 0:
        print("len(train_loader)=", len(train_loader))
    # print("running_loss_train=", running_loss_train)
    # avg_train_loss = running_loss_train / len(train_loader)  # avg_train_loss太小了；不打算除以len(train_loader)了
    # print(f"Epoch [{epoch + 1}/{num_epochs}], Train_Loss: {avg_train_loss}")
    # print(f"Epoch [{epoch + 1}/{num_epochs}], sum_ms_ssim_loss_train: {sum_ssim_loss_train}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train_Loss_total: {running_loss_train}")

    print(f'***********Epoch {epoch + 1} begin Validation***********')
    # Validation
    model_UNet.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_custom_loss = 0.0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # print(f"inputs.size():{inputs.size()}--targets.size():{targets.size()}")
            outputs = model_UNet(inputs)
            # print(f"outputs.size():{outputs.size()}")
            val_custom_loss = custom_loss(outputs, targets, m_weight)
            val_loss += val_custom_loss.item()
        # 判断当前损失值是否更好
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model_UNet.state_dict(), "pth/trained_model_12-16-2.pth")
            torch.save(model_UNet.state_dict(), m_save_path)
        writer.add_scalar("Validation_loss", val_loss, epoch + 1)
        if epoch == 0:
            print("len(val_loader)=", len(val_loader))
        # print(f"Epoch [{epoch + 1}/{num_epochs}], sum_ssim_loss_val: {sum_ssim_loss_val}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val_Loss_total: {val_loss} \n")

# Save the trained model if needed
# torch.save(model_UNet.state_dict(), "pth/trained_model_12-15.pth")
writer.close()
print('Finished Training')

# 训练完成后执行关机命令 /s 表示shutdown /t 表示1秒
# os.system('shutdown /s /t 1')

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




