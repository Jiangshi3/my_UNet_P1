import torch
from unet_cbam_densenet_2 import *
import hiddenlayer as h
from torchviz import make_dot

Mymodule = UNet(n_channels=3, n_classes=3)

torch.save(Mymodule, "E:\\py_workplace\\myProject\\model.pth")


# x = torch.randn(1, 3, 256, 256).requires_grad_(True)  # 定义一个网络的输入值
# y = Mymodule(x)  # 获取网络的预测值
# MyConvNetVis = make_dot(y, params=dict(list(Mymodule.named_parameters()) + [('x', x)]))
# MyConvNetVis.format = "png"
# # 指定文件生成的文件夹
# MyConvNetVis.directory = "data"
# # 生成文件
# MyConvNetVis.view()


# from torchvision.models import resnet18  # 以 resnet18 为例
#
# myNet = resnet18()  # 实例化 resnet18
# x = torch.randn(16, 3, 64, 64)  # 随机生成一个输入
# myNetGraph = h.build_graph(myNet, x)  # 建立网络模型图
# # myNetGraph.theme = h.graph.THEMES['blue']  # blue 和 basic 两种颜色，可以不要
# myNetGraph.save(path='./demoModel.png', format='png')  # 保存网络模型图，可以设置 png 和 pdf 等





# method 2
# x = torch.randn(1, 3, 256, 256).requires_grad_(True)  # 定义一个网络的输入值
# y = Mymodule(x)  # 获取网络的预测值
# MyConvNetVis = make_dot(y, params=dict(list(Mymodule.named_parameters()) + [('x', x)]))
# MyConvNetVis.format = "png"
# # 指定文件生成的文件夹
# MyConvNetVis.directory = "out"
# # 生成文件
# MyConvNetVis.view()

# vis_graph = h.build_graph(Mynet, torch.zeros([1, 3, 256, 256]))  # 获取绘制图像的对象
# vis_graph.display()
# vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
# vis_graph.save("./demo1.png")  # 保存图像的路径

# # 输入示例数据的维度，这里假设输入维度为 (batch_size, input_size)
# input_tensor = torch.randn((1, 3, 256, 256))
# # 使用HiddenLayer构建图形
# graph, _, _ = h.build_graph(Mynet, input_tensor)

