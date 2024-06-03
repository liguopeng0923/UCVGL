import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import copy
import random
import albumentations as A
import cv2
import numpy as np
import torchvision.transforms.functional as F
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
import pandas as pd

def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])


# Same loader from VIGOR, modified for pytorch
class VIGORTrain(torch.utils.data.Dataset):
    def __init__(self, args):
        super(VIGORTrain, self).__init__()

        super().__init__()
 
        self.data_folder = args.data_folder
        self.grd_size = args.grd_size
        self.sat_size = args.sat_size
        self.mean = args.mean
        self.std = args.std
        
        self.transform_scan_val_grd = A.Compose([A.Resize(self.grd_size[0],self.grd_size[1]),
                                        A.Normalize(),
                                        ToTensorV2()
        ])
        
        self.transform_scan_val_sat = A.Compose([A.Resize(self.sat_size[0],self.sat_size[1]),
                                        A.Normalize(),
                                        ToTensorV2()
        ])
        
        self.mode = "train"
        
        self.transform_sat = A.Compose([
                                         A.Resize(self.sat_size[0],self.sat_size[1]),
                                         A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0,random_offset=True),
                                               A.CoarseDropout(max_holes=25,
                                                               max_height=int(0.2*self.sat_size[0]),
                                                               max_width=int(0.2*self.sat_size[1]),
                                                               min_holes=10,
                                                               min_height=int(0.1*self.sat_size[0]),
                                                               min_width=int(0.1*self.sat_size[1]),
                                                               p=1.0),
                                              ], p=0.5),
                                         A.ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.1,rotate_limit=0,value=0,border_mode=cv2.BORDER_CONSTANT),
                                         A.ColorJitter(0.4,0.4,0.4,0.1,p=0.8),
                                         A.ToGray(p=0.2),
                                         A.GaussianBlur(blur_limit=(3,3)),
                                         A.Normalize(mean=self.mean,std=self.std),
                                         ToTensorV2(),
                                         ]
                                         )
        self.transform_grd = A.Compose([
                                         A.Resize(self.grd_size[0],self.grd_size[1]),
                                         A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0,random_offset=True),
                                            A.CoarseDropout(max_holes=25,
                                                            max_height=int(0.2*self.grd_size[0]),
                                                            max_width=int(0.2*self.grd_size[1]),
                                                            min_holes=10,
                                                            min_height=int(0.1*self.grd_size[0]),
                                                            min_width=int(0.1*self.grd_size[1]),
                                                            p=1.0),
                                           ], p=0.5),
                                         A.ColorJitter(0.4,0.4,0.4,0.1,p=0.8),
                                         A.ToGray(p=0.2),
                                         A.GaussianBlur(blur_limit=(3,3)),
                                         A.Normalize(mean=self.mean,std=self.std),
                                         ToTensorV2()
                                         ])
        
        if args.same_area:
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle'][:args.cities]
        else:
            self.cities = ['NewYork', 'Seattle'] 

        # load sat list 
        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(f'{self.data_folder}/splits/{city}/satellite_list.txt', header=None, delim_whitespace=True)
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(lambda x: f'{self.data_folder}/{city}/satellite/{x.sat}', axis=1)
            sat_list.append(df_tmp)
        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)
        
        # idx for complete train and test independent of mode = train or test
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))
        
        
        # ground dependent on mode 'train' or 'test'
        ground_list = []
        for city in self.cities:

            if args.same_area:
                df_tmp = pd.read_csv(f'{self.data_folder}/splits/{city}/same_area_balanced_train.txt', header=None, delim_whitespace=True)
            else:
                df_tmp = pd.read_csv(f'{self.data_folder}/splits/{city}/pano_label_balanced.txt', header=None, delim_whitespace=True)
            
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(columns={0:  "ground",
                                                                     1:  "sat",
                                                                     4:  "sat_np1",
                                                                     7:  "sat_np2",
                                                                     10: "sat_np3"})
            
            df_tmp["path_ground"] = df_tmp.apply(lambda x: f'{self.data_folder}/{city}/panorama/{x.ground}', axis=1)
            df_tmp["path_sat"] = df_tmp.apply(lambda x: f'{self.data_folder}/{city}/satellite/{x.sat}', axis=1)
            
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[f'{sat_n}'] = df_tmp[f'{sat_n}'].map(sat2idx)
                
            ground_list.append(df_tmp) 
        self.ground_list = ground_list
        self.df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)
        
        # idx for split train or test dependent on mode = train or test
        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_path = dict(zip(self.df_ground.index, self.df_ground.path_ground))
                
      
        self.pairs = list(zip(self.df_ground.index, self.df_ground.sat))
        self.idx2pairs = defaultdict(list)
        
        # for a unique sat_id we can have 1 or 2 ground views as gt
        for pair in self.pairs:      
            self.idx2pairs[pair[1]].append(pair)
            
            
        self.label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values 
        
        self.samples = copy.deepcopy(self.pairs)
        
        random.seed(42 if args.seed is None else args.seed)
        self.shuffle_samples = np.arange(len(self.samples))
        self.gt_ratio = args.gt_ratio
        if self.gt_ratio > 0.:
            shuffle_ids = 0
            for city in ground_list:
                value = city.shape[0]
                remaining_labels = (shuffle_ids+np.arange(value))[int(value*self.gt_ratio):value]
                random.shuffle(remaining_labels)
                self.shuffle_samples[shuffle_ids+int(value*self.gt_ratio):shuffle_ids+value] = remaining_labels
                shuffle_ids += value
        print('VIGOR: load train',' data_size =', len(self.samples))
                
    def __getitem__(self, index):
        
        idx_ground, _ = self.samples[index]
        
        # load query -> ground image
        grd = cv2.imread(self.idx2ground_path[idx_ground])
        grd = cv2.cvtColor(grd, cv2.COLOR_BGR2RGB)
        
        _, idx_sat = self.samples[self.shuffle_samples[index]]
        # load reference -> satellite image
        sat = cv2.imread(self.idx2sat_path[idx_sat])
        sat = cv2.cvtColor(sat, cv2.COLOR_BGR2RGB)
        
        if self.mode == "scan":
            return  self.transform_scan_val_grd(image=grd)["image"], self.transform_scan_val_sat(image=sat)["image"], index
            
        # Flip simultaneously query and reference
        if np.random.random() < 0.5:
            grd = cv2.flip(grd, 1)
            sat = cv2.flip(sat, 1) 
        
        # image transforms
        if self.transform_grd is not None:
            grd = self.transform_grd(image=grd)['image']
            
        if self.transform_sat is not None:
            sat = self.transform_sat(image=sat)['image']
                
        # Rotate simultaneously query and reference
        if np.random.random() < 0.5:
        
            r = np.random.choice([1,2,3])
            
            # rotate sat img 90 or 180 or 270
            sat = torch.rot90(sat, k=r, dims=(1, 2)) 
            
            # use roll for ground view if rotate sat view
            c, h, w = grd.shape
            shifts = - w//4 * r
            grd = torch.roll(grd, shifts=shifts, dims=2)   
                   
        return grd, sat,index
    
    def __len__(self):
        return len(self.samples)

class VigorVal(torch.utils.data.Dataset):
    
    def __init__(self,
                 args
                 ):
        
        super().__init__()
 
        self.data_folder = args.data_folder
        self.img_type = "grd"
        self.grd_size = args.grd_size
        self.sat_size = args.sat_size
        self.mean = args.mean
        self.std = args.std
        
        print(f"Validate grd size = {self.grd_size}, sat size = {self.sat_size}, mean = {self.mean}, std = {self.std}")
        
        self.transform_grd = A.Compose([A.Resize(self.grd_size[0],self.grd_size[1]),
                                        A.Normalize(),
                                        ToTensorV2()
        ])
        
        self.transform_sat = A.Compose([A.Resize(self.sat_size[0],self.sat_size[1]),
                                        A.Normalize(),
                                        ToTensorV2()
        ])
        
            
        if args.same_area:
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle'][:args.cities]
        else:
            self.cities = ['Chicago', 'SanFrancisco'] 
               
        # load sat list 
        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(f'{self.data_folder}/splits/{city}/satellite_list.txt', header=None, delim_whitespace=True)
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(lambda x: f'{self.data_folder}/{city}/satellite/{x.sat}', axis=1)
            sat_list.append(df_tmp)
        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)
        
        # idx for complete train and test independent of mode = train or test
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))
        
        
        # ground dependent on mode 'train' or 'test'
        ground_list = []
        for city in self.cities:
            
            if args.same_area:
                df_tmp = pd.read_csv(f'{self.data_folder}/splits/{city}/same_area_balanced_test.txt', header=None, delim_whitespace=True)
            else:
                df_tmp = pd.read_csv(f'{self.data_folder}/splits/{city}/pano_label_balanced.txt', header=None, delim_whitespace=True)
  
            
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(columns={0:  "ground",
                                                                     1:  "sat",
                                                                     4:  "sat_np1",
                                                                     7:  "sat_np2",
                                                                     10: "sat_np3"})
            
            df_tmp["path_ground"] = df_tmp.apply(lambda x: f'{self.data_folder}/{city}/panorama/{x.ground}', axis=1)
            df_tmp["path_sat"] = df_tmp.apply(lambda x: f'{self.data_folder}/{city}/satellite/{x.sat}', axis=1)
            
            df_tmp["path_sat_np1"] = df_tmp.apply(lambda x: f'{self.data_folder}/{city}/satellite/{x.sat_np1}', axis=1)
            df_tmp["path_sat_np2"] = df_tmp.apply(lambda x: f'{self.data_folder}/{city}/satellite/{x.sat_np2}', axis=1)
            df_tmp["path_sat_np3"] = df_tmp.apply(lambda x: f'{self.data_folder}/{city}/satellite/{x.sat_np3}', axis=1)
            
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[f'{sat_n}'] = df_tmp[f'{sat_n}'].map(sat2idx)
                
            ground_list.append(df_tmp) 
        self.df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)
        
        # idx for split train or test dependent on mode = train or test
        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_path = dict(zip(self.df_ground.index, self.df_ground.path_ground))
        print('VIGOR: load val',' sat data_size =', len(self.df_sat["path"].values))
        print('VIGOR: load val',' grd data_size =', len(self.df_ground["path_ground"].values))
        

    def __getitem__(self, index):
        
        # image transforms
        if self.img_type == "sat":
            img_path = self.df_sat["path"].values[index]
            label = self.df_sat.index.values[index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform_sat(image=img)['image']
        else:
            img_path = self.df_ground["path_ground"].values[index]
            label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values [index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform_grd(image=img)['image']
            
        label = torch.tensor(label, dtype=torch.long)

        return img, label

    def __len__(self):
        if self.img_type == "sat":
            return len(self.df_sat["path"].values)
        else:
            return len(self.df_ground["path_ground"].values)
            

            
