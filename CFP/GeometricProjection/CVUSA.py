import os
import cv2
import numpy as np
import scipy.io as sio
from utils import get_BEV_projection, get_BEV_tensor
from PIL import Image
import pandas as pd


root = "./data/CVUSA"
save_path = os.path.join(root,"g2a")

df = pd.read_csv(f'{root}/splits/train-19zl.csv', header=None)
        
df = df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})

df["idx"] = df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))

idx2sat = dict(zip(df.idx, df.sat))
idx2ground = dict(zip(df.idx, df.ground))

pairs = list(zip(df.idx, df.sat, df.ground))

idx2pair = dict()
train_ids_list = list()

# for shuffle pool
for pair in pairs:
    idx = pair[0]
    idx2pair[idx] = pair
    train_ids_list.append(idx)


for i,idx in enumerate(train_ids_list):
    if i % 100 == 0:
        print(i)
    _, _, ground = idx2pair[idx]
    
    grd = os.path.join(root,ground)
    
    image = cv2.imread(grd)
    Hp, Wp = image.shape[0], image.shape[1] # pano shape

    ty = (Wp/2-Hp)/2 - 20                  # completing pano to the correct proportion

    matrix_K = np.array([[1,0,0],[0,1,ty],[0,0,1]])

    image = cv2.warpPerspective(image,matrix_K,(int(Wp),int(Hp+(Wp/2-Hp))))
        
    out = get_BEV_projection(image,750,750,Fov = 175, dty = 0, dx = 0, dy = 0)
    BEV = get_BEV_tensor(image,750,750,Fov = 175, dty = 0, dx = 0, dy = 0, out = out).cpu().numpy().astype(np.uint8)
    cv2.imwrite(os.path.join(save_path,ground.replace("streetview/panos/","")).replace(".jpg",".png"),BEV)

    
    
    
    
    
