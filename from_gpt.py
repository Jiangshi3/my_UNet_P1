import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from utils.data_utils import PairedImageDataset_2


# # Define the dataset class
# class PairedImageDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.raw_folder = os.path.join(root_dir, "raw-890")
#         self.reference_folder = os.path.join(root_dir, "reference-890")
#         self.raw_dataset = ImageFolder(self.raw_folder, transform=transform)
#         self.reference_dataset = ImageFolder(self.reference_folder, transform=transform)
#
#     def __len__(self):
#         return len(self.raw_dataset)
#
#     def __getitem__(self, idx):
#         raw_image, _ = self.raw_dataset[idx]
#         reference_image, _ = self.reference_dataset[idx]
#         return raw_image, reference_image


# Define the neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # Define your model architecture here
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


# Set up data transformation and create datasets
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
root_dir = r"D:\dataset\UIEB_Dataset"
paired_dataset = PairedImageDataset_2(root_dir=root_dir, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(paired_dataset))
val_size = len(paired_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(paired_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Instantiate the model, loss function, and optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, targets)

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_val_loss.item()}")

# Save the trained model if needed
torch.save(model.state_dict(), "trained_model.pth")


'''
-----train.py-----
cuda
Epoch [1/50], Loss: 0.09484001249074936
Epoch [2/50], Loss: 0.06908601522445679
Epoch [3/50], Loss: 0.05737699568271637
Epoch [4/50], Loss: 0.05695175379514694
Epoch [5/50], Loss: 0.06292032450437546
Epoch [6/50], Loss: 0.05125832185149193
Epoch [7/50], Loss: 0.04688762500882149
Epoch [8/50], Loss: 0.06257044523954391
Epoch [9/50], Loss: 0.05046875774860382
Epoch [10/50], Loss: 0.044312380254268646
'''
