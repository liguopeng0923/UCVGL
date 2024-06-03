import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import io
import random
import sys
import cv2
import matplotlib.pyplot as plt 
from torch_geometry import euler_angles_to_matrix, get_perspective_transform

from sklearn.decomposition import PCA
import time
import math

def gps2distance(Lat_A,Lng_A,Lat_B,Lng_B):
    # https://en.wikipedia.org/wiki/Great-circle_distance
    lat_A = torch.deg2rad(Lat_A.double()) #Lat_A * torch.pi/180.
    lat_B = torch.deg2rad(Lat_B.double())
    lng_A = torch.deg2rad(Lng_A.double())
    lng_B = torch.deg2rad(Lng_B.double())
    R = torch.tensor(6371004.).cuda()  # Earth's radius in meters
    C = torch.sin(lat_A) * torch.sin(lat_B) + torch.cos(lat_A) * torch.cos(lat_B) * torch.cos(lng_A - lng_B)
    C = torch.clamp(C, min=-1.0, max=1.0)
    distance = R * torch.acos(C)
    return distance
    
def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the real distance between two locations based on GPS latitude and longitude values.
    :param lat1: Latitude of location 1
    :param lon1: Longitude of location 1
    :param lat2: Latitude of location 2
    :param lon2: Longitude of location 2
    :return: The real distance between the two locations (in meters)
    """
    # Convert latitude and longitude to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Use the Haversine formula to calculate the distance between two locations
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371393 # Earth's radius (in meters)
    distance = c * r

    return distance

def get_feature_show(feature_map):
    # Load the feature map, inputs.shape = [batch, channel, H, W]
    feature_map = feature_map.squeeze(0).permute(1, 2, 0)
    feature_map = feature_map.cpu().detach().numpy()

    # Flatten the feature map into a one-dimensional array
    feature_vector = feature_map.reshape(-1, feature_map.shape[-1])

    # Create a PCA (Principal Component Analysis) object and set the desired reduced dimensionality
    pca = PCA(n_components=3)

    # Reduce the dimensionality of the feature vectors
    feature_reduced = pca.fit_transform(feature_vector)

    # Reshape the dimension-reduced feature vectors back to the shape of the feature map
    feature_reduced = feature_reduced.reshape(feature_map.shape[:-1] + (3,))

    # Clip pixel values to be between 0 and 1
    feature_reduced = np.clip(feature_reduced, 0, 1)
    return feature_reduced

def get_BEV_projection(img, Ho, Wo, Fov=170, dty=-20, dx=0, dy=0, device = 'cpu',delta=(0,0)):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = device

    Hp, Wp = img.shape[0], img.shape[1]  # Panorama image dimensions

    Fov = Fov * torch.pi / 180  # Field of View in radians
    center = torch.tensor([Wp / 2 + dx, Hp + dy]).to(device)  # Overhead view center

    anglex = torch.tensor(dx).to(device) * 2 * torch.pi / Wp
    angley = -torch.tensor(dy).to(device) * torch.pi / Hp
    anglez = torch.tensor(0).to(device)

    # Euler angles
    euler_angles = (anglex, angley, anglez)
    euler_angles = torch.stack(euler_angles, -1)

    # Calculate the rotation matrix
    R02 = euler_angles_to_matrix(euler_angles, "XYZ")
    R20 = torch.inverse(R02)

    f = Wo / 2 / torch.tan(torch.tensor(Fov / 2))
    out = torch.zeros((Wo, Ho, 2)).to(device)
    f0 = torch.zeros((Wo, Ho, 3)).to(device)
    f0[:, :, 0] = Ho / 2 - (torch.ones((Ho, Wo)).to(device) * (torch.arange(Ho)).to(device)).T + delta[0]
    f0[:, :, 1] = Wo / 2 - torch.ones((Ho, Wo)).to(device) * torch.arange(Wo).to(device) - delta[1]
    f0[:, :, 2] = -torch.ones((Wo, Ho)).to(device) * f
    f1 = R20 @ f0.reshape((-1, 3)).T  # x, y, z (3, N)
    f1_0 = torch.sqrt(torch.sum(f1**2, 0))
    f1_1 = torch.sqrt(torch.sum(f1[:2, :]**2, 0))
    theta = torch.arctan2(f1[2, :], f1_1) + torch.pi / 2  # [-pi/2, pi/2] => [0, pi]
    phi = torch.arctan2(f1[1, :], f1[0, :])  # [-pi, pi]
    phi = phi + torch.pi  # [0, 2pi]

    i_p = 1 - theta / torch.pi  # [0, 1]
    j_p = 1 - phi / (2 * torch.pi)  # [0, 1]
    out[:, :, 0] = j_p.reshape((Ho, Wo))
    out[:, :, 1] = i_p.reshape((Ho, Wo))
    out[:, :, 0] = (out[:, :, 0] - 0.5) / 0.5  # [-1, 1]
    out[:, :, 1] = (out[:, :, 1] - 0.5) / 0.5  # [-1, 1]

    return out


def get_BEV_tensor(img, Ho, Wo, Fov=170, dty=-20, dx=0, dy=0, dataset=False, out=None, device = 'cpu',delta=(0,0)):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = device

    t0 = time.time()
    Hp, Wp = img.shape[0], img.shape[1]  # Panorama image dimensions
    if dty != 0 or Wp != 2 * Hp:
        ty = (Wp / 2 - Hp) / 2 + dty  # Non-standard panorama image completion
        matrix_K = np.array([[1, 0, 0], [0, 1, ty], [0, 0, 1]])
        img = cv2.warpPerspective(img, matrix_K, (int(Wp), int(Hp + (Wp / 2 - Hp))))
    ######################
    t1 = time.time()
    # frame = torch.from_numpy(img.astype(np.float32)).to(device)
    frame = torch.from_numpy(img.copy()).to(device)
    t2 = time.time()

    if out is None:
        Fov = Fov * torch.pi / 180  # Field of View in radians
        center = torch.tensor([Wp / 2 + dx, Hp + dy]).to(device)  # Overhead view center

        anglex = torch.tensor(dx).to(device) * 2 * torch.pi / Wp
        angley = -torch.tensor(dy).to(device) * torch.pi / Hp
        anglez = torch.tensor(0).to(device)

        # Euler angles
        euler_angles = (anglex, angley, anglez)
        euler_angles = torch.stack(euler_angles, -1)

        # Calculate the rotation matrix
        R02 = euler_angles_to_matrix(euler_angles, "XYZ")
        R20 = torch.inverse(R02)

        f = Wo / 2 / torch.tan(torch.tensor(Fov / 2))
        out = torch.zeros((Wo, Ho, 2)).to(device)
        f0 = torch.zeros((Wo, Ho, 3)).to(device)
        f0[:, :, 0] = Ho / 2 - (torch.ones((Ho, Wo)).to(device) * (torch.arange(Ho)).to(device)).T + delta[0]
        f0[:, :, 1] = Wo / 2 - torch.ones((Ho, Wo)).to(device) * torch.arange(Wo).to(device) - delta[1]
        f0[:, :, 2] = -torch.ones((Wo, Ho)).to(device) * f
        f1 = R20 @ f0.reshape((-1, 3)).T  # x, y, z (3, N)
        # f1 = f0.reshape((-1, 3)).T
        f1_0 = torch.sqrt(torch.sum(f1**2, 0))
        f1_1 = torch.sqrt(torch.sum(f1[:2, :]**2, 0))
        theta = torch.arctan2(f1[2, :], f1_1) + torch.pi / 2  # [-pi/2, pi/2] => [0, pi]
        phi = torch.arctan2(f1[1, :], f1[0, :])  # [-pi, pi]
        phi = phi + torch.pi  # [0, 2pi]

        i_p = 1 - theta / torch.pi  # [0, 1]
        j_p = 1 - phi / (2 * torch.pi)  # [0, 1]
        out[:, :, 0] = j_p.reshape((Ho, Wo))
        out[:, :, 1] = i_p.reshape((Ho, Wo))
        out[:, :, 0] = (out[:, :, 0] - 0.5) / 0.5  # [-1, 1]
        out[:, :, 1] = (out[:, :, 1] - 0.5) / 0.5  # [-1, 1]
    # else:
    #     out = out.to(device)
    t3 = time.time()

    BEV = F.grid_sample(frame.permute(2, 0, 1).unsqueeze(0).float(), out.unsqueeze(0), align_corners=True)
    t4 = time.time()
    # print("Read image ues {:.2f} ms, warpPerspective image use {:.2f} ms, Get matrix ues {:.2f} ms, Get out ues {:.2f} ms, All out ues {:.2f} ms.".format((t1-t0)*1000,(t2-t1)*1000, (t3-t2)*1000,(t4-t3)*1000,(t4-t0)*1000))
    if dataset:
        return BEV.squeeze(0)
    else:
        return BEV.permute(0, 2, 3, 1).squeeze(0).int()



def show_overlap(img1, img2, H, show=True, g_u=0, g_v=0):
    # Display overlap from img1 to img2 (np.array)
    image1 = img1.copy()
    image0 = img2.copy()
    h, w = image0.shape[0], image0.shape[1]
    h_, w_ = image1.shape[0], image1.shape[1]

    result = cv2.warpPerspective(image1, H, (w + w + image1.shape[1], h))  # result placing image1 warped image on the left
    mask_temp = result[:, 0:w] > 1
    temp2 = result[:, 0:w, :].copy()
    frame = image0.astype(np.uint8)
    roi = frame[mask_temp]
    frame[mask_temp] = (0.5 * temp2.astype(np.uint8)[mask_temp] + 0.5 * roi).astype(np.uint8)
    result[0:image0.shape[0], image1.shape[1]:image0.shape[1] + image1.shape[1]] = frame  # result placing overlap in the middle
    result[0:h, image1.shape[1] + w:w + w + image1.shape[1]] = image0  # result placing image0 on the right
    pts = np.float32([[0, 0], [0, h_], [w_, h_], [w_, 0]]).reshape(-1, 1, 2)
    center = np.float32([w_ / 2, h_ / 2]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    dst_center = cv2.perspectiveTransform(center, H).reshape(-1, 2)
    # Adding offsets
    for i in range(4):
        dst[i][0] += w_ + w
    dst_center[0][0] += w_ + w
    cv2.polylines(result, [np.int32(dst)], True, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.circle(result, (int(dst_center[0][0]), int(dst_center[0][1])), 15, (0, 255, 0), 1)
    cv2.circle(temp2, (int(dst_center[0][0] - w_ - w), int(dst_center[0][1])), 15, (0, 255, 0), 1)
    if show:
        plt.subplot(1, 3, 1)
        plt.imshow(temp2.astype(np.uint8))
        plt.subplot(1, 3, 2)
        plt.imshow(result[:, w_:w_ + w].astype(np.uint8))
        plt.subplot(1, 3, 3)
        if g_u != 0 and g_v != 0:
            result = draw_markers(np.ascontiguousarray(result), [g_u + w + w_, g_v], size=4, thickness=2, color=(255, 0, 0), shape=2)
        plt.imshow(result[:, w + w_:].astype(np.uint8))
    return result


def get_homograpy(four_point, sz, k = 1):
    """
    four_point: four corner flow
    sz: image size
    k: scale
    Shape:
        - Input: :four_point:`(B, 2, 2, 2)` and :sz:`(B, C, H, W)`
        - Output: :math:`(B, 3, 3)`
    """
    N,_,h,w = sz
    h, w = h//k, w//k
    four_point = four_point / k 
    four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
    four_point_org[:, 0, 0] = torch.Tensor([0, 0])
    four_point_org[:, 0, 1] = torch.Tensor([w-1, 0])
    four_point_org[:, 1, 0] = torch.Tensor([0, h-1])
    four_point_org[:, 1, 1] = torch.Tensor([w -1, h-1])

    four_point_org = four_point_org.unsqueeze(0)
    four_point_org = four_point_org.repeat(N, 1, 1, 1)
    four_point_new = torch.autograd.Variable(four_point_org) + four_point
    four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
    four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
    # H = tgm.get_perspective_transform(four_point_org, four_point_new)
    H = get_perspective_transform(four_point_org, four_point_new)
    return H


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    # [1024, 1, 32, 32]  [1024, 9, 9, 2] https://www.jb51.net/article/273930.htm
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)  # [1024, 9, 9, 2]
    img = F.grid_sample(img, grid, align_corners=True) # [1024, 1, 9, 9]

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd),indexing='ij')
    coords = torch.stack(coords[::-1], dim=0).float() 

    return coords[None].expand(batch, -1, -1, -1)

def save_img(img, path):
    npimg = img.detach().cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    npimg = npimg.astype(np.uint8)
    io.imsave(path, npimg)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_params(model):
    Total_params = 0
    Trainable_params = 0
    NonTrainable_params = 0

    for param in model.parameters():
        mulValue = np.prod(param.size())  
        Total_params += mulValue  
        if param.requires_grad:
            Trainable_params += mulValue  
        else:
            NonTrainable_params += mulValue 

    print(f'Total params: {Total_params}')
    print(f'Trainable params: {Trainable_params}')
    print(f'Non-trainable params: {NonTrainable_params}')

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask

def warp_coor(x, coor):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    coor: [B, 2, H, W] coor2 after H
    """
    B, C, H, W = x.size()
    # mesh grid
    vgrid = torch.autograd.Variable(coor)

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask

class Logger_(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

class Logger:
    def __init__(self, args):
        self.args = args
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_mace_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}

    def _print_training_status(self):
        metrics_data = [np.nanmean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}] ".format(self.total_steps)
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        # print the training status
        print(training_str + metrics_str)
        # print([np.nanmedian(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())])
        # logging running loss to total loss
        # for key in self.running_loss_dict:
        #     self.running_loss_dict[key] = []

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []
            self.running_loss_dict[key].append(metrics[key])
            if np.isnan(metrics[key]):
                print('\033[1;91m'+"There is a nan at {}!\033[0m".format(len(self.running_loss_dict[key])))

def draw_markers(img, point, size=10, thickness=2, color=(0, 0, 255), shape=0):
    if shape == 0:
        # Draw a triangle
        pts = np.array([[point[0] - size, point[1] + size], [point[0], point[1] - size], [point[0] + size, point[1] + size]], np.int32)
        cv2.polylines(img, [pts], True, color, thickness=thickness)
    elif shape == 1:
        # Draw a five-pointed star
        size //= 2
        pts = np.array([[point[0] - size, point[1] + size], [point[0] + size, point[1] - size], [point[0] - size, point[1] - size], [point[0] + size, point[1] + size], [point[0], point[1] - 2 * size]], np.int32)
        cv2.polylines(img, [pts], True, color, thickness=thickness)
    elif shape == 2:
        # Draw a five-pointed star
        size //= 2
        pts = np.array([[point[0] - size, point[1] - size], [point[0] + size, point[1] + size]], np.int32)
        cv2.polylines(img, [pts], False, color, thickness=thickness)
        pts = np.array([[point[0] - size, point[1] + size], [point[0] + size, point[1] - size]], np.int32)
        cv2.polylines(img, [pts], False, color, thickness=thickness)
    elif shape == 3:
        # Draw a circle
        cv2.circle(img, point, size, color, thickness=thickness)

    return img


