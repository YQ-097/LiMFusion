import os
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.autograd import Variable
from args_fusion import args
#from scipy.misc import imread, imsave, imresize
import matplotlib as mpl
import cv2
import torch.nn.functional as F
from torchvision import datasets, transforms
from guided_filter import GuidedFilter
import fusion_strategy
from skimage.feature import hog
from skimage import exposure
from scipy.ndimage import gaussian_filter

# 转换函数，将PIL图像转换为Tensor
transform = transforms.ToTensor()

def get_hop_weight_map2(image_gray):
    image_gray_np = image_gray.cpu().numpy()

    # 创建一个空列表来存储每个通道的 HOG 特征
    hog_features_list = []

    # 针对每个通道计算 HOG 特征
    for batch in range(image_gray_np.shape[0]):  # 遍历每个批次
        hog_features_batch = []
        for channel in range(image_gray_np.shape[1]):  # 更新通道循环
            # 计算 HOG 特征
            _, hog_image = hog(image_gray_np[batch, channel], orientations=8, pixels_per_cell=(4, 4),
                               cells_per_block=(2, 2),
                               visualize=True, block_norm='L2-Hys')

            # 转换为 PyTorch tensor
            hog_features = torch.from_numpy(hog_image).float()

            # 添加到通道列表
            hog_features_batch.append(hog_features.unsqueeze(0))  # 添加批处理维度

        # 合并每个通道的 HOG 特征
        hog_features_batch = torch.cat(hog_features_batch, dim=0)  # 沿通道维度合并

        hog_features_list.append(hog_features_batch.unsqueeze(0))  # 添加批处理维度

    # 合并每个批次的 HOG 特征
    hog_features = torch.cat(hog_features_list, dim=0)  # 沿批处理维度合并

    # 获取图像尺寸
    image_height, image_width = image_gray_np.shape[2], image_gray_np.shape[3]

    # 创建一个与图像大小相同的全零矩阵
    gradient_sum_image = torch.zeros_like(hog_features)

    # 计算每个 cell 中梯度的总值，并将整个 cell 区域赋予该总值
    cell_height, cell_width = (4, 4)
    for y in range(0, image_height, cell_height):
        for x in range(0, image_width, cell_width):
            # 计算当前 cell 内的梯度总和
            cell_gradient_sum = torch.sum(hog_features[:, :, y:y + cell_height, x:x + cell_width], dim=(2, 3))
            # 将当前 cell 区域内的像素都赋予梯度总值
            gradient_sum_image[:, :, y:y + cell_height, x:x + cell_width] = cell_gradient_sum[:, :, None, None]

    # 归一化处理
    normalized_image = (gradient_sum_image - gradient_sum_image.min()) / (
                gradient_sum_image.max() - gradient_sum_image.min())
    hog_image = (hog_image - hog_image.min()) / (
            hog_image.max() - hog_image.min())
    # 进行形状调整并放回原来的设备
    # = normalized_image.unsqueeze(0)  # 添加批处理维度
    normalized_image = normalized_image.to(device=image_gray.device)  # 将张量放回原来的设备上
    #print(normalized_image.shape)
    return normalized_image,hog_image



def fpde(I, T=15):
    I = I * 255.0
    I1 = I.clone()
    dt = 0.9
    k = 4.0
    for t in range(T):
        Ix, Iy = torch.gradient(I1, axis=(2, 3))
        Ixx, Iyt = torch.gradient(Ix, axis=(2, 3))
        Ixt, Iyy = torch.gradient(Iy, axis=(2, 3))

        c = 1.0 / (1.0 + (torch.sqrt(Ixx**2 + Iyy**2) / k)**2)

        div1, divt1 = torch.gradient(c * Ixx, axis=(2, 3))
        divt2, div2 = torch.gradient(c * Iyy, axis=(2, 3))
        div11, divt3 = torch.gradient(div1, axis=(2, 3))
        divt4, div22 = torch.gradient(div2, axis=(2, 3))

        div = div11 + div22
        I2 = I1 - dt * div
        I1 = I2

    #frth = I1.byte()
    #frth = (I1 - torch.min(I1)) / (torch.max(I1) - torch.min(I1))
    frth = I1/255.0
    return frth
# images = []
# image_path = r'D:\qianyao\code\parer4\论文\返修\hog_show\ir.jpg'
# img = Image.open(image_path).convert('L')
# img = np.array(img)/255.0
# img = np.reshape(img, [1, img.shape[0], img.shape[1]])
# images.append(img)
# img = np.stack(images, axis=0)
# img = torch.from_numpy(img).float()
# print(img.shape)
# #img_tensor = transform(img)
# #img_tensor = img_tensor.unsqueeze(0)
# img_tensor = fpde(img)
# img_tensor = img - img_tensor
# img_pil = transforms.ToPILImage()(img_tensor.squeeze(0))  # 取消添加的维度
# # enhancer = ImageEnhance.Brightness(img_pil)
# # img_pil = enhancer.enhance(1.5)  # 调整亮度，这里的1.5可以调整
# # enhancer = ImageEnhance.Contrast(img_pil)
# # img_pil = enhancer.enhance(2)  # 调整对比度，这里的2可以调整
#
# img_pil.save(r'D:\qianyao\code\parer4\论文\返修\hog_show\ir_LLVIP_fpde.png')

# 文件夹1和文件夹2路径
folder1_path = r'D:\qianyao\code\parer4\论文\返修\hog_show\ir_H'
folder2_path = r'D:\qianyao\code\parer4\论文\返修\hog_show'

# 确保文件夹2存在，如果不存在则创建
if not os.path.exists(folder2_path):
    os.makedirs(folder2_path)

# 遍历文件夹1中的图像文件
for filename in os.listdir(folder1_path):
    file_path = os.path.join(folder1_path, filename)

    # 确保文件是图像文件
    if os.path.isfile(file_path) and any(file_path.endswith(ext) for ext in ['.jpg', '.png', '.jpeg']):
        # 使用PIL库打开图像
        img = Image.open(file_path)

        # 将图像转换为Tensor
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        print(img_tensor.shape)

        # 进行图像处理，例如应用模型、转换等
        img_tensor,hog_img = get_hop_weight_map2(img_tensor)
        img_tensor = torch.nn.functional.avg_pool2d(img_tensor, kernel_size=19, stride=1, padding=9)
        img_tensor = torch.nn.functional.avg_pool2d(img_tensor, kernel_size=19, stride=1, padding=9)

        # 在文件夹2中保存处理后的图像
        save_path = os.path.join(folder2_path, filename)
        # 将Tensor转换回PIL图像并保存
        img_pil = transforms.ToPILImage()(img_tensor.squeeze(0))  # 取消添加的维度
        img_pil.save(save_path)

        hog_img = transforms.ToPILImage()(np.uint8(hog_img*255.0))  # 取消添加的维度
        enhancer = ImageEnhance.Brightness(hog_img)
        hog_img = enhancer.enhance(1.5)  # 调整亮度，这里的1.5可以调整
        enhancer = ImageEnhance.Contrast(hog_img)
        hog_img = enhancer.enhance(2)  # 调整对比度，这里的2可以调整
        hog_img.save(r'D:\qianyao\code\parer4\论文\返修\hog_show\hog.png')
        print(f"Saved {filename} to {folder2_path}")