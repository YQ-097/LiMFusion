from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random
import numpy as np
import torch
from PIL import Image
import matplotlib as mpl
import torch.nn.functional as F
from torchvision import datasets, transforms
from skimage.feature import hog
from scipy.ndimage import gaussian_filter

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=True):
    if cuda:
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U,D,V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()


# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = Image.open(path).convert('L')
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')
    elif mode == 'YCbCr':
        img = Image.open(path).convert('YCbCr')
        image, _, _ = img.split()


    if height is not None and width is not None:
        image = image.resize((width, height), resample=Image.NEAREST)
    image = np.array(image)

    return image


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y

def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode) / 255.0
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images
    
def get_train_images(paths, height=256, width=256, flag=False):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, flag)
        if flag is True:
            image = np.transpose(image, (2, 0, 1))
        else:
            image = np.reshape(image, [1, height, width])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


def get_test_images(paths, height=None, width=None, mode='RGB'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode) / 255.0
        if mode == 'L' :#or mode == 'YCbCr':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = ImageToTensor(image).float().numpy()
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


# colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)


def save_images(path, data):
    if data.shape[2] == 1:
        data = data.reshape([data.shape[0], data.shape[1]])
    image = Image.fromarray(data)
    image = image.convert('L')
    image.save(path)


def PixelIntensityDecision(latlrr_image,ir_image,vi_image):
    mask = torch.where(latlrr_image > 30, 1, 0)
    vi_mask = vi_image * mask
    ir_mask = ir_image * mask
    max_input_pixel_mask = torch.max(vi_mask, ir_mask)
    max_input_pixel = vi_image - vi_mask + max_input_pixel_mask
    return max_input_pixel,mask



#FPDE
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

    frth = I1/255.0
    return frth


def get_hop_weight_map(image_gray):
    image_gray_np = image_gray.cpu().numpy()

    hog_features_list = []

    for batch in range(image_gray_np.shape[0]):
        hog_features_batch = []
        for channel in range(image_gray_np.shape[1]):
            _, hog_image = hog(image_gray_np[batch, channel], orientations=8, pixels_per_cell=(4, 4),
                               cells_per_block=(2, 2),
                               visualize=True, block_norm='L2-Hys')

            hog_features = torch.from_numpy(hog_image).float()

            hog_features_batch.append(hog_features.unsqueeze(0))

        hog_features_batch = torch.cat(hog_features_batch, dim=0)

        hog_features_list.append(hog_features_batch.unsqueeze(0))

    hog_features = torch.cat(hog_features_list, dim=0)

    image_height, image_width = image_gray_np.shape[2], image_gray_np.shape[3]

    gradient_sum_image = torch.zeros_like(hog_features)

    cell_height, cell_width = (4, 4)
    for y in range(0, image_height, cell_height):
        for x in range(0, image_width, cell_width):
            cell_gradient_sum = torch.sum(hog_features[:, :, y:y + cell_height, x:x + cell_width], dim=(2, 3))
            gradient_sum_image[:, :, y:y + cell_height, x:x + cell_width] = cell_gradient_sum[:, :, None, None]

    normalized_image = (gradient_sum_image - gradient_sum_image.min()) / (
                gradient_sum_image.max() - gradient_sum_image.min())

    normalized_image = normalized_image.to(device=image_gray.device)  # 将张量放回原来的设备上

    return normalized_image


def gradient_loss(predicted, target1, target2):
    gradient_predicted_x = torch.abs(F.conv2d(predicted, torch.tensor([[-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().cuda(), padding=1))
    gradient_predicted_y = torch.abs(F.conv2d(predicted, torch.tensor([[-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().cuda().transpose(2, 3),padding=1))

    gradient_target1_x = torch.abs(F.conv2d(target1, torch.tensor([[-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().cuda(), padding=1))
    gradient_target1_y = torch.abs(F.conv2d(target1, torch.tensor([[-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().cuda().transpose(2, 3),padding=1))

    gradient_target2_x = torch.abs(F.conv2d(target2, torch.tensor([[-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().cuda(), padding=1))
    gradient_target2_y = torch.abs(F.conv2d(target2, torch.tensor([[-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().cuda().transpose(2, 3),padding=1))

    loss = F.l1_loss(gradient_predicted_x, torch.max(gradient_target1_x,gradient_target2_x)) + F.l1_loss(gradient_predicted_y, torch.max(gradient_target1_y,gradient_target2_y))

    return loss

def median_filter_torch(image_tensor, kernel_size):

    image_np = image_tensor.cpu().squeeze(0).squeeze(0).numpy()

    image_np_uint8 = (image_np * 255).astype(np.uint8)

    img_height, img_width = image_np_uint8.shape
    pad = kernel_size // 2
    filtered_image = np.zeros_like(image_np_uint8)

    for i in range(pad, img_height - pad):
        for j in range(pad, img_width - pad):
            window = image_np_uint8[i - pad:i + pad + 1, j - pad:j + pad + 1]
            median_val = np.median(window)
            filtered_image[i, j] = median_val

    filtered_tensor = torch.from_numpy(filtered_image / 255.0).float().unsqueeze(0).unsqueeze(0).to(image_tensor.device)
    return filtered_tensor

class L_spa(torch.nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = torch.nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = torch.nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = torch.nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = torch.nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = torch.nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)

        return E





