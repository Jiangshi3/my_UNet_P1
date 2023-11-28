import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale


def ssim_loss(img1, img2, window_size=11, size_average=True):
    img1_gray = rgb_to_grayscale(img1)
    img2_gray = rgb_to_grayscale(img2)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    window = create_window(window_size, img1_gray.device)

    mu1 = F.conv2d(img1_gray, window, padding=window_size // 2, groups=img1_gray.shape[0])
    mu2 = F.conv2d(img2_gray, window, padding=window_size // 2, groups=img2_gray.shape[0])

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1_gray ** 2, window, padding=window_size // 2, groups=img1_gray.shape[0]) - mu1_sq
    sigma2_sq = F.conv2d(img2_gray ** 2, window, padding=window_size // 2, groups=img2_gray.shape[0]) - mu2_sq
    sigma12 = F.conv2d(img1_gray * img2_gray, window, padding=window_size // 2, groups=img1_gray.shape[0]) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return torch.mean(ssim_map)
    else:
        return torch.mean(ssim_map, dim=(1, 2, 3))


def create_window(window_size, channel):
    sigma = 1.5
    coords = torch.arange(window_size).float()
    coords -= window_size // 2

    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()

    g = g.view(1, 1, -1, 1)
    g = g.expand(1, 1, -1, -1).contiguous()

    window = g.expand(1, channel, -1, -1).contiguous()
    return window
