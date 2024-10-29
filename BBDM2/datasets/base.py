from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import os
import numpy as np

import torch

class ImagePathDataset(Dataset):
    def __init__(self, image_paths, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size
        self.image_paths = image_paths
        self._length = len(image_paths)
        self.flip = flip
        self.to_normal = to_normal # 是否归一化到[-1, 1]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return image, image_name


class ImageIndicesDataset(Dataset):
    def __init__(self, root_folder, indices, data_list, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size
        self.indices = indices
        self.root_folder = root_folder
        self.data_list = data_list
        self._length = len(indices)
        self.flip = flip
        self.to_normal = to_normal # 是否归一化到[-1, 1]

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        item_infor = self.data_list[self.indices[index]]
        img_path = os.path.join(self.root_folder, f"{item_infor['crop_index']}.png")
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return image, image_name
    
class ImageDiscreteIndicesDataset(Dataset):
    def __init__(self, root_folder, indices, data_list, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size
        self.indices = indices
        self.root_folder = root_folder
        self.data_list = data_list
        self._length = len(indices)
        self.flip = flip
        self.to_normal = to_normal # 是否归一化到[-1, 1]
        
        self.color_map = [
            [255, 255, 255],    # background
            [0, 128, 0],    # Viable tumor
            [255, 143, 204],    # Necrosis
            [255, 0, 0],    # Fibrosis/Hyalination
            [0, 0, 0],  # Hemorrhage/ Cystic change
            [165, 42, 42],  # Inflammatory
            [0, 0, 255]]    # Non-tumor tissue

    def apply_threshold_mapping(self, image):
        # Create masks for pixels that are closer to green or pink
        # Initialize the output image with the original image
        tolerance = 50
        
        output = torch.zeros((len(self.color_map), image.shape[1], image.shape[2]))
        masks = 0.
        for idx, color in enumerate(self.color_map):
            color = torch.tensor(color).reshape(-1, 1, 1)
            mask = torch.all(torch.abs(image - color) < tolerance, axis=0)
            mask = mask.unsqueeze(0) # 1xHxW
            # print(mask.shape)
            # print(output.shape)
            # output[mask] = color
            class_mask = torch.zeros_like(output)
            class_mask[idx, :, :]=1.0
            output = output * (~mask) + class_mask * mask 
            masks += mask.float().mean()
        # print(masks)


        return output

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])
        
        item_infor = self.data_list[self.indices[index]]
        img_path = os.path.join(self.root_folder, f"{item_infor['crop_index']}.png")
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image) * 255
        image = self.apply_threshold_mapping(image)
        

        # if self.to_normal:
        #     image = (image - 0.5) * 2.
        #     image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return image, image_name

class ImageVectorDiscreteIndicesDataset(Dataset):
    def __init__(self, root_folder, indices, data_list, image_size=(256, 256), flip=False, to_normal=False):
        self.image_size = image_size
        self.indices = indices
        self.root_folder = root_folder
        self.data_list = data_list
        self._length = len(indices)
        self.flip = False
        self.to_normal = to_normal # 是否归一化到[-1, 1]
        
        self.color_map = [
            [255, 255, 255],    # background
            [0, 128, 0],    # Viable tumor
            [255, 143, 204],    # Necrosis
            [255, 0, 0],    # Fibrosis/Hyalination
            [0, 0, 0],  # Hemorrhage/ Cystic change
            [165, 42, 42],  # Inflammatory
            [0, 0, 255]]    # Non-tumor tissue

    def apply_threshold_mapping(self, image):
        # Create masks for pixels that are closer to green or pink
        # Initialize the output image with the original image
        tolerance = 50
        
        output = torch.zeros((len(self.color_map), image.shape[1], image.shape[2]))
        masks = 0.
        for idx, color in enumerate(self.color_map):
            color = torch.tensor(color).reshape(-1, 1, 1)
            mask = torch.all(torch.abs(image - color) < tolerance, axis=0)
            mask = mask.unsqueeze(0) # 1xHxW
            # print(mask.shape)
            # print(output.shape)
            # output[mask] = color
            class_mask = torch.zeros_like(output)
            class_mask[idx, :, :]=1.0
            output = output * (~mask) + class_mask * mask 
            masks += mask.float().mean()
        # print(masks)


        return output

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.0
        if index >= self._length:
            index = index - self._length
            p = 1.0

        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            # transforms.ToTensor()
        ])
        
        item_infor = self.data_list[self.indices[index]]
        img_path = os.path.join(self.root_folder, f"{item_infor['crop_index']}.npy")
        image = None
        try:
            image = np.load(img_path)
        except BaseException as e:
            print(img_path)

        # if not image.mode == 'RGB':
        #     image = image.convert('RGB')
        image = torch.tensor(image).permute(2,0,1)
        # print(image.shape)
        image = transform(image)
        # print(image.shape)
        # image = self.apply_threshold_mapping(image)
        

        # if self.to_normal:
        #     image = (image - 0.5) * 2.
        #     image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return image, image_name