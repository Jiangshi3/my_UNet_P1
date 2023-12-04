import torch
import torch.nn as nn
from unet_cbam_densenet_2 import *
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('../logs')
myUNet = UNet(3, 3)
input_size = torch.randn([1, 3, 256, 256])
writer.add_graph(myUNet, input_size)
writer.close()

