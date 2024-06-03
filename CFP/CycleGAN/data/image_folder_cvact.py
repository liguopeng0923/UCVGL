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
    anuData = sio.loadmat(os.path.join(root, 'ACT_data.mat'))
    id_all_list = []
    for i in range(0, len(anuData['panoIds'])):
        fname = anuData['panoIds'][i]
        id_all_list.append(fname)

    id_list = []
    training_inds = anuData['trainSet']['trainInd'][0][0] - 1
    trainNum = len(training_inds)

    for k in range(trainNum):
        fname = id_all_list[training_inds[k][0]]
        id_list.append(id_all_list[training_inds[k][0]])

    # val_inds = anuData['valSet']['valInd'][0][0] - 1
    # valNum = len(val_inds)

    # id_test_list = []
    # for k in range(valNum):
    #     fname = id_all_list[val_inds[k][0]]
    #     id_test_list.append(id_all_list[val_inds[k][0]])
    
    return id_list

def make_dataset_cvact(dir,root="./data/CVACT/"):
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
        
    for train_sample in train_list:
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


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        train_list = get_id_list(os.path.dirname(root))
        imgs = make_dataset(root, IMG_EXTENSIONS,train_list)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
