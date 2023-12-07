#  全参考标准：PSNR；  MSE越小，PSNR越大，PSNR越大越好
from PIL import Image
import numpy as np

# original_path = 'D:\\dataset\\UIEB_Dataset\\raw-10\\5554.png'  # 21.368982004963343
# original_path = 'D:\\dataset\\UIEB_Dataset\\raw-10-out-P5-Resize512\\result_5554.png'  # 19.01174241515821
# original_path = 'D:\\dataset\\UIEB_Dataset\\raw-10-out-P4-Resize256\\result_5554.png'  # 23.479541106586318
# compress_path = 'D:\\dataset\\UIEB_Dataset\\reference-10\\5554.png'

# original_path = 'D:\\dataset\\UIEB_Dataset\\raw-10\\15045.png'  # 17.446286692117226
# original_path = 'D:\\dataset\\UIEB_Dataset\\raw-10-out-P4-Resize256\\result_15045.png'  # 23.51785063435742
original_path = 'D:\\dataset\\UIEB_Dataset\\raw-10-out-P5-Resize512\\result_15045.png'  # 25.915912462419804
compress_path = 'D:\\dataset\\UIEB_Dataset\\reference-10\\15045.png'

img1 = np.array(Image.open(original_path)).astype(np.float64)
img2 = np.array(Image.open(compress_path)).astype(np.float64)


def psnr(img1, img2):
    mse = np.mean((img1-img2)**2)
    if mse == 0:
        return float('inf')
    else:
        return 20*np.log10(255/np.sqrt(mse))


if __name__ == "__main__":
    print(psnr(img1, img2))


# from skimage.metrics import peak_signal_noise_ratio as psnr
# from PIL import Image
# import numpy as np
#
# original_path = 'D:\\dataset\\UIEB_Dataset\\raw-10\\5554.png'  # 21.368982004963343    21.368982004963343 结果一致
# compress_path = 'D:\\dataset\\UIEB_Dataset\\reference-10\\5554.png'
# img1 = np.array(Image.open(original_path))
# img2 = np.array(Image.open(compress_path))
#
#
# if __name__ == "__main__":
#     print(psnr(img1, img2))

