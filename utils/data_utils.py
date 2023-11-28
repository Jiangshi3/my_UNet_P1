import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image


class PairedImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.raw_dataset = ImageFolder(os.path.join(root_dir, "raw-890"), transform=self.transform)
        self.reference_dataset = ImageFolder(os.path.join(root_dir, "reference-890"), transform=self.transform)

        # 确保两个文件夹中的图像数量一致
        assert len(self.raw_dataset) == len(self.reference_dataset)

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        raw_image, _ = self.raw_dataset[idx]
        reference_image, _ = self.reference_dataset[idx]
        return raw_image, reference_image


class PairedImageDataset_2(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.raw_folder = os.path.join(root_dir, "raw-90")
        self.reference_folder = os.path.join(root_dir, "reference-90")
        self.raw_images = os.listdir(self.raw_folder)
        self.reference_images = os.listdir(self.reference_folder)
        self.root_dir = root_dir
        self.transform = transform
        # 确保两个文件夹中的图像数量一致
        # assert len(self.raw_dataset) == len(self.reference_dataset)

    def __len__(self):
        return len(self.raw_images)

    def __getitem__(self, idx):
        raw_image_path = os.path.join(self.raw_folder, self.raw_images[idx])
        reference_image_path = os.path.join(self.reference_folder, self.reference_images[idx])

        raw_image = Image.open(raw_image_path).convert("RGB")
        reference_image = Image.open(reference_image_path).convert("RGB")

        if self.transform:
            raw_image = self.transform(raw_image)
            reference_image = self.transform(reference_image)

        return raw_image, reference_image

# # Example usage
# root_dir = r"D:\dataset\UIEB_Dataset"
#
# # Assuming you have a transform defined
# transform = transforms.Compose([transforms.ToTensor()])
#
# # Creating the PairedImageDataset
# paired_dataset = PairedImageDataset(root_dir=root_dir, transform=transform)
#
# # Example of accessing the dataset
# for i in range(len(paired_dataset)):
#     raw_image, reference_image = paired_dataset[i]
#     # Use raw_image and reference_image as needed

