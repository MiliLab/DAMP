import os
import random
import numpy as np

from torch.utils.data import Dataset
import torch

from degradation_utils import Degradation
import h5py
from PIL import Image

import glob



class HSI2Tensor(object):
    """
    Transform a numpy array with shape (C, H, W)
    into torch 4D Tensor (1, C, H, W) or (C, H, W)
    """
    def __init__(self, use_2dconv):
        self.use_2dconv = use_2dconv

    def __call__(self, hsi):
        
        if self.use_2dconv:
            img = torch.from_numpy(hsi)
        else:
            img = torch.from_numpy(hsi[None])

        return img


def data_augmentation(image, mode=None):
    """
    Args:
        image: np.ndarray, shape: C X H X W
    """
    axes = (-2, -1)
    flipud = lambda x: x[:, ::-1, :] 
    if mode is None:
        mode = random.randint(0, 7)
    if mode == 0:
        # original
        image = image
    elif mode == 1:
        # flip up and down
        image = flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        image = np.rot90(image, axes=axes)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image, axes=axes)
        image = flipud(image)
    elif mode == 4:
        # rotate 180 degree
        image = np.rot90(image, k=2, axes=axes)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2, axes=axes)
        image = flipud(image)
    elif mode == 6:
        # rotate 270 degree
        image = np.rot90(image, k=3, axes=axes)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3, axes=axes)
        image = flipud(image)
    
    return np.ascontiguousarray(image)


def random_augmentation(*args):
    out = []
    flag_aug = random.randint(1, 7)
    for data in args:
        out.append(data_augmentation(data, flag_aug).copy())
    return out

class ImageTransformDataset(Dataset):
    def __init__(self, root = None, mode = 'train',crop_size=64):
        super(ImageTransformDataset, self).__init__()
        self.mode = mode
        self.D = Degradation()
        self.to_tensor = HSI2Tensor(use_2dconv=True)

        self.file_list = [os.path.basename(f) for f in glob.glob(os.path.join(root, "*.mat"))]
        self.root = root
        self.root_rgb = root[:-4] + '_RGB'
        self.file_list_rgb = os.listdir(self.root_rgb)
        self.length = len(self.file_list)
        self.crop_size = crop_size

        self.deg_type = ["gaussianN","sr"]
        self.deg_range = {"gaussianN": [(30,70)],"sr": [(2,)]}


    def __len__(self):
        return self.length * 2

    def __getitem__(self, idx):   
        idx = idx % (self.length)     
        name = self.file_list[idx]

        with h5py.File(os.path.join(self.root, name), 'r') as f:
            img = np.transpose(np.array(f['cube'])[:,0:448,0:448], (0, 2, 1))
        img_rgb = np.array(Image.open(os.path.join(self.root_rgb, name[:-4]+".jpg")))
        img_rgb = np.transpose(img_rgb, (2, 0, 1))[:,0:448,0:448]/255.0
        clean_patch = img.copy()

        deg = random.choice(self.deg_type)

        degrad_patch,_ = self.D.single_degrade(clean_patch.copy(), deg, self.deg_range[deg], name)


        if self.mode == 'train':
            degrad_patch, clean_patch, img_rgb = random_augmentation(*(degrad_patch, clean_patch, img_rgb))
            h, w = degrad_patch.shape[1], degrad_patch.shape[2]
            # 确保图像足够大
            if h < self.crop_size or w < self.crop_size:
                raise ValueError(f"Image size ({h}x{w}) is smaller than 64x64")
            
            # 随机选择裁剪起点
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            
            # 对三个图像进行相同位置的裁剪
            degrad_patch = degrad_patch[:,top:top+self.crop_size, left:left+self.crop_size]
            clean_patch = clean_patch[:,top:top+self.crop_size, left:left+self.crop_size]
            img_rgb = img_rgb[:,top:top+self.crop_size, left:left+self.crop_size]

        clean_patch = self.to_tensor(clean_patch)
        degrad_patch = self.to_tensor(degrad_patch)
        img_rgb = self.to_tensor(img_rgb).float()
        #print(name, degrad_patch.shape, clean_patch.shape, img_rgb.shape)
        return name, degrad_patch, clean_patch, img_rgb


class ImageTransformDataset_SR(Dataset):
    def __init__(self, root = None, mode = 'train',crop_size=64):
        super(ImageTransformDataset_SR, self).__init__()
        self.mode = mode
        self.D = Degradation()
        self.to_tensor = HSI2Tensor(use_2dconv=True)

        self.file_list = [os.path.basename(f) for f in glob.glob(os.path.join(root, "*.mat"))]
        self.root = root
        self.root_rgb = root[:-4] + '_RGB'
        self.file_list_rgb = os.listdir(self.root_rgb)
        self.length = len(self.file_list)
        self.crop_size = crop_size

        self.deg_type = ["sr"]
        self.deg_range = {"sr": [(4,)]}


    def __len__(self):
        return self.length

    def __getitem__(self, idx):   
        idx = idx % (self.length)     
        name = self.file_list[idx]

        with h5py.File(os.path.join(self.root, name), 'r') as f:
            img = np.transpose(np.array(f['cube'])[:,0:448,0:448], (0, 2, 1))
        img_rgb = np.array(Image.open(os.path.join(self.root_rgb, name[:-4]+".jpg")))
        img_rgb = np.transpose(img_rgb, (2, 0, 1))[:,0:448,0:448]/255.0
        clean_patch = img.copy()

        deg = random.choice(self.deg_type)

        degrad_patch,_ = self.D.single_degrade(clean_patch.copy(), deg, self.deg_range[deg], name)


        if self.mode == 'train':
            degrad_patch, clean_patch, img_rgb = random_augmentation(*(degrad_patch, clean_patch, img_rgb))
            h, w = degrad_patch.shape[1], degrad_patch.shape[2]
            # 确保图像足够大
            if h < self.crop_size or w < self.crop_size:
                raise ValueError(f"Image size ({h}x{w}) is smaller than 64x64")
            
            # 随机选择裁剪起点
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            
            # 对三个图像进行相同位置的裁剪
            degrad_patch = degrad_patch[:,top:top+self.crop_size, left:left+self.crop_size]
            clean_patch = clean_patch[:,top:top+self.crop_size, left:left+self.crop_size]
            img_rgb = img_rgb[:,top:top+self.crop_size, left:left+self.crop_size]

        clean_patch = self.to_tensor(clean_patch)
        degrad_patch = self.to_tensor(degrad_patch)
        img_rgb = self.to_tensor(img_rgb).float()
        #print(name, degrad_patch.shape, clean_patch.shape, img_rgb.shape)
        return name, degrad_patch, clean_patch, img_rgb



class ImageTransformDataset(Dataset):
    def __init__(self, root = None, mode = 'train',crop_size=64, deg_index = -1):
        super(ImageTransformDataset, self).__init__()
        self.mode = mode
        self.D = Degradation()
        self.to_tensor = HSI2Tensor(use_2dconv=True)

        self.file_list = [os.path.basename(f) for f in glob.glob(os.path.join(root, "*.mat"))]
        self.root = root
        self.root_rgb = root[:-4] + '_RGB'
        self.file_list_rgb = os.listdir(self.root_rgb)
        self.length = len(self.file_list)
        self.crop_size = crop_size

        self.deg_type = ['gaussianN', 'blur', 'sr', 'inpaint', 'bandmiss']
        self.deg_range = {'gaussianN': [(30, 70)], 'blur':[(9, 15)], 'sr': [(2, 4)], 'inpaint':[(0.7, 0.8)], 'bandmiss':[(0.1,0.2)]}

        if deg_index >= 0 and deg_index < len(self.deg_type):
            self.deg_type = [self.deg_type[deg_index]]
            print(f"Using fixed degradation type: {self.deg_type[0]}")
 

    def __len__(self):
        return self.length * len(self.deg_type)

    def __getitem__(self, idx):   
        idx = idx % (self.length)     
        name = self.file_list[idx]

        with h5py.File(os.path.join(self.root, name), 'r') as f:
            img = np.transpose(np.array(f['cube'])[:,0:448,0:448], (0, 2, 1))
        img_rgb = np.array(Image.open(os.path.join(self.root_rgb, name[:-4]+".jpg")))
        img_rgb = np.transpose(img_rgb, (2, 0, 1))[:,0:448,0:448]/255.0
        clean_patch = img.copy()

        deg = random.choice(self.deg_type)
        deg_index = random.randint(0, len(self.deg_type) - 1)
        deg = self.deg_type[deg_index]
        degrad_patch,_ = self.D.single_degrade(clean_patch.copy(), deg, self.deg_range[deg], name)


        if self.mode == 'train':
            degrad_patch, clean_patch, img_rgb = random_augmentation(*(degrad_patch, clean_patch, img_rgb))
            h, w = degrad_patch.shape[1], degrad_patch.shape[2]
            # 确保图像足够大
            if h < self.crop_size or w < self.crop_size:
                raise ValueError(f"Image size ({h}x{w}) is smaller than 64x64")
            
            # 随机选择裁剪起点
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            
            # 对三个图像进行相同位置的裁剪
            degrad_patch = degrad_patch[:,top:top+self.crop_size, left:left+self.crop_size]
            clean_patch = clean_patch[:,top:top+self.crop_size, left:left+self.crop_size]
            img_rgb = img_rgb[:,top:top+self.crop_size, left:left+self.crop_size]

        clean_patch = self.to_tensor(clean_patch)
        degrad_patch = self.to_tensor(degrad_patch)
        img_rgb = self.to_tensor(img_rgb).float()
        #print(name, degrad_patch.shape, clean_patch.shape, img_rgb.shape)
        return name, degrad_patch, clean_patch, img_rgb, deg_index


class ImageTransformDataset_Noise(Dataset):
    def __init__(self, root = None, mode = 'train',crop_size=64):
        super(ImageTransformDataset_Noise, self).__init__()
        self.mode = mode
        self.D = Degradation()
        self.to_tensor = HSI2Tensor(use_2dconv=True)

        self.file_list = [os.path.basename(f) for f in glob.glob(os.path.join(root, "*.mat"))]
        self.root = root
        self.root_rgb = root[:-4] + '_RGB'
        self.file_list_rgb = os.listdir(self.root_rgb)
        self.length = len(self.file_list)
        self.crop_size = crop_size

        self.deg_type = ["gaussianN"]
        self.deg_range = {"gaussianN": [(30,70)]}


    def __len__(self):
        return self.length

    def __getitem__(self, idx):   
        idx = idx % (self.length)     
        name = self.file_list[idx]

        with h5py.File(os.path.join(self.root, name), 'r') as f:
            img = np.transpose(np.array(f['cube'])[:,0:448,0:448], (0, 2, 1))
        img_rgb = np.array(Image.open(os.path.join(self.root_rgb, name[:-4]+".jpg")))
        img_rgb = np.transpose(img_rgb, (2, 0, 1))[:,0:448,0:448]/255.0
        clean_patch = img.copy()

        #deg = random.choice(self.deg_type)

        deg_index = random.randint(0, len(self.deg_type) - 1)
        deg = self.deg_type[deg_index]

        degrad_patch,_ = self.D.single_degrade(clean_patch.copy(), deg, self.deg_range[deg], name)


        if self.mode == 'train':
            degrad_patch, clean_patch, img_rgb = random_augmentation(*(degrad_patch, clean_patch, img_rgb))
            h, w = degrad_patch.shape[1], degrad_patch.shape[2]
            # 确保图像足够大
            if h < self.crop_size or w < self.crop_size:
                raise ValueError(f"Image size ({h}x{w}) is smaller than 64x64")
            
            # 随机选择裁剪起点
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            
            # 对三个图像进行相同位置的裁剪
            degrad_patch = degrad_patch[:,top:top+self.crop_size, left:left+self.crop_size]
            clean_patch = clean_patch[:,top:top+self.crop_size, left:left+self.crop_size]
            img_rgb = img_rgb[:,top:top+self.crop_size, left:left+self.crop_size]

        clean_patch = self.to_tensor(clean_patch)
        degrad_patch = self.to_tensor(degrad_patch)
        img_rgb = self.to_tensor(img_rgb).float()
        #print(name, degrad_patch.shape, clean_patch.shape, img_rgb.shape)
        return name, degrad_patch, clean_patch, img_rgb, deg_index



class ImageTransformDataset_test(Dataset):
    def __init__(self, root = None, mode = 'test', crop_size = 64, deg_idx = 0):
        super(ImageTransformDataset_test, self).__init__()
        self.D = Degradation()
        self.to_tensor = HSI2Tensor(use_2dconv=True)

        #self.file_list = os.listdir(root)
        self.file_list = [os.path.basename(f) for f in glob.glob(os.path.join(root, "*.mat"))]
        self.root = root
        self.root_rgb = root[:-4] + '_RGB'
        self.file_list_rgb = os.listdir(self.root_rgb)
        self.length = len(self.file_list)
        self.crop_size = crop_size

        self.degrad_patchs = []
        self.clean_patchs = []
        self.img_rgbs = []

        self.deg_type = ["gaussianN","sr"]
        self.deg_range = {"gaussianN": [(30,70)],"sr": [(4,)]}

        self.deg_type = ['gaussianN', 'blur', 'sr', 'inpaint', 'bandmiss']
        self.deg_range = {'gaussianN': [(30, 70)], 'blur':[(9, 15)], 'sr': [(2, 4)], 'inpaint':[(0.7, 0.8)], 'bandmiss':[(0.1,0.2)]}


        self.deg_index = deg_idx
        
        
        deg = self.deg_type[deg_idx]

        seed = 2025
        # 为 numpy 设置随机种子
        np.random.seed(seed)

        # 为 Python 内置的 random 模块设置随机种子
        random.seed(seed)


        for idx in range(self.length):
            name = self.file_list[idx]

            with h5py.File(os.path.join(self.root, name), 'r') as f:
                img = np.transpose(np.array(f['cube'])[:,0:448,0:448], (0, 2, 1))
            img_rgb = np.array(Image.open(os.path.join(self.root_rgb, name[:-4]+".jpg")))
            img_rgb = np.transpose(img_rgb, (2, 0, 1))[:,0:448,0:448]/255.0
            clean_patch = img.copy()
            degrad_patch,_ = self.D.single_degrade(clean_patch.copy(), deg, self.deg_range[deg], name)

            clean_patch = self.to_tensor(clean_patch)
            degrad_patch = self.to_tensor(degrad_patch)
            img_rgb = self.to_tensor(img_rgb).float()

            self.degrad_patchs.append(degrad_patch)
            self.clean_patchs.append(clean_patch)
            self.img_rgbs.append(img_rgb)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):       
        return self.file_list[idx], self.degrad_patchs[idx], self.clean_patchs[idx], self.img_rgbs[idx], self.deg_index