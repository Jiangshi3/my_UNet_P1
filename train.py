from unet_model import *
import argparse
from utils.data_utils import *

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms


print("-----train.py-----")


parser = argparse.ArgumentParser(description="this is a demo")
parser.add_argument("--epoch", type=int, default=0, help="which epoch to start from")
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
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
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
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

for epoch in range(epoch, num_epochs):
    model_UNet.train()
    running_loss = 0.0
    print(f'-------------Epoch {epoch+1} begin training-----------')
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # torch.Size([8, 3, 256, 256])
        optimizer_UNet.zero_grad()
        outputs = model_UNet(inputs)
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer_UNet.step()

    print(f'***********Epoch {epoch + 1} begin Validation***********')
    # Validation
    model_UNet.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # print(f"inputs.size():{inputs.size()}--targets.size():{targets.size()}")
            outputs = model_UNet(inputs)
            # print(f"outputs.size():{outputs.size()}")
            val_loss += loss_function(outputs, targets)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_val_loss.item()}")

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




