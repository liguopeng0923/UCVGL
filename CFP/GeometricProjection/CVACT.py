import os
import cv2
import numpy as np
import scipy.io as sio
from utils import get_BEV_projection, get_BEV_tensor

root = "./data/CVACT"
save_path = os.path.join(root,"g2a")

anuData = sio.loadmat(os.path.join(root, 'ACT_data.mat'))

ids = anuData['panoIds']

train_ids = ids[anuData['trainSet'][0][0][1]-1]


for i,idx in enumerate(train_ids.squeeze()):
    if i % 100 == 0:
        print(i)
        
    grd = os.path.join(root,f'streetview/{idx}_grdView.jpg')
    image = cv2.imread(grd)
    out = get_BEV_projection(image,1200,1200,Fov = 170, dty = 0, dx = 0, dy = 0)
    BEV = get_BEV_tensor(image,1200,1200,Fov = 170, dty = 0, dx = 0, dy = 0, out = out).cpu().numpy().astype(np.uint8)
    cv2.imwrite(os.path.join(save_path,f'{idx}_grdView.png'),BEV)
    
    
    
    
    
