"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
import scipy.io as sio

from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def get_id_list(root):
    train_list = os.path.join(root,'splits/train-19zl.csv')
    
    id_list = []
    id_idx_list = []
    with open(train_list, 'r') as file:
        idx = 0
        for line in file:
            data = line.split(',')
            pano_id = (data[0].split('/')[-1]).split('.')[0]
            # satellite filename, streetview filename, pano_id
            id_list.append(pano_id)
            id_idx_list.append(idx)
            idx += 1
    data_size = len(id_list)
    print('CVUSA: load', train_list, ' data_size =', data_size)
    return id_list

def make_dataset_cvusa(dir,root="./data/CVUSA/"):
    train_list = get_id_list(root)
    images = []
    fnames = sorted(os.listdir(dir))
    suffix = None
    for fname in sorted(fnames):
        for train_sample in train_list:
            if train_sample in fname: 
                suffix = fname.replace(train_sample,"") 
                break
        if suffix is not None:
            break
    for  train_sample in train_list:
        if has_file_allowed_extension(train_sample + suffix, IMG_EXTENSIONS):
            path = os.path.join(dir, train_sample + suffix)
            if os.path.exists(path):
                images.append(path)
    # for test all images
    # images = []
    # fnames = sorted(os.listdir(dir))
    # for sample in fnames:
    #     path = os.path.join(dir, sample)
    #     if os.path.exists(path):
    #         images.append(path)
    
    return images


def default_loader(path):
    return Image.open(path).convert('RGB')

