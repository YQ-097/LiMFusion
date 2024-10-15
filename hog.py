import os
from PIL import Image
import torch
from torchvision import transforms
import utils

# 转换函数，将PIL图像转换为Tensor
transform = transforms.ToTensor()

# 文件夹1和文件夹2路径
folder1_path = r'D:\qianyao\code\parer4\FPDE\output\VIS_H'
folder2_path = r'D:\qianyao\code\parer4\FPDE\output\VIS_hog'

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
        img_tensor = utils.get_hop_weight_map2(img_tensor)

        # 在文件夹2中保存处理后的图像
        save_path = os.path.join(folder2_path, filename)
        # 将Tensor转换回PIL图像并保存
        img_pil = transforms.ToPILImage()(img_tensor.squeeze(0))  # 取消添加的维度
        img_pil.save(save_path)
        print(f"Saved {filename} to {folder2_path}")