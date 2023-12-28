import os
import torch
from torchvision import transforms
from PIL import Image

# 1. 加载模型和测试数据
from unet_cbam_densenet_2 import UNet

model = UNet(3, 3)
# model.load_state_dict(torch.load('../pth/trained_model_12-04.pth'))  # resize(256,256)
# model.load_state_dict(torch.load('../pth/trained_model_12-05.pth'))  # resize(512,512); batchsize=2; epoch=20;
# model.load_state_dict(torch.load('../pth/trained_model_12-14.pth'))  # resize(512,512); batchsize=2; epoch=40; lr=0.00001;修改了loss
model.load_state_dict(torch.load('../pth/trained_model_12-23-resume-1.pth'))  # resize(512,512); batchsize=2; epoch=40; lr=0.00001;修改了loss
model.eval()

# 设置测试数据集文件夹路径
# test_data_folder = 'D:\\dataset\\UIEB_Dataset\\raw-10'
test_data_folder = 'D:\\dataset\\UIEB_Dataset\\challenging-10'

# 2. 预处理测试数据
transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 3. 进行推断和保存结果
# output_folder = 'D:\\dataset\\UIEB_Dataset\\raw-10-out-12-24-1'
# output_folder = 'D:\\dataset\\UIEB_Dataset\\raw-10-out-12-23-resume-1-2'
# output_folder = 'D:\\dataset\\UIEB_Dataset\\raw-10-out-P4-Resize256-1'
# output_folder = 'D:\\dataset\\UIEB_Dataset\\challenging-10-out-P5-Resize512'
output_folder = 'D:\\dataset\\UIEB_Dataset\\challenging-10-out-12-23-resume-1-2'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历测试数据集文件夹中的图片
for image_name in os.listdir(test_data_folder):
    # 构建图像路径
    image_path = os.path.join(test_data_folder, image_name)

    # 读取图像
    image = Image.open(image_path).convert("RGB")

    # 预处理
    input_data = transform(image).unsqueeze(0)  # 添加批次维度
    # 将输入数据移动到GPU上（如果模型在GPU上）
    input_data = input_data.cpu()

    # 进行推断
    with torch.no_grad():
        output = model(input_data)

    output_image = transforms.ToPILImage()(output[0].cpu())  # 输出是RGB格式

    # 保存结果图片
    output_image_path = os.path.join(output_folder, f'result_{image_name}')
    output_image.save(output_image_path)

    print(f"Processed: {image_name}, Result saved at: {output_image_path}")

