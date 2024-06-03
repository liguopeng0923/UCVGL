import torch
import numpy as np
import os
import random
import scipy.io as sio
import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
import copy
import pandas as pd

# pytorch implementation of CVUSA loader
class CVUSATrainIntra(torch.utils.data.Dataset):
    def __init__(self, args):
        super(CVUSATrainIntra, self).__init__()
        
        self.data_folder = args.data_folder
        self.grd_size = args.grd_size
        self.sat_size = args.sat_size
        self.mean = args.mean
        self.std = args.std

        print(f"Train grd size = {self.grd_size}, sat size = {self.sat_size}, mean = {self.mean}, std = {self.std}")
        
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
        
        self.ShiftScaleRotate = A.ShiftScaleRotate(shift_limit=0.2,scale_limit=0.2,rotate_limit=15,value=0,border_mode=cv2.BORDER_CONSTANT)
        
        self.df = pd.read_csv(f'{self.data_folder}/splits/train-19zl.csv', header=None)
        
        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})
        
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))
        

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
        self.pairs = list(zip(self.df.idx, self.df.sat, self.df.ground))
        
        self.idx2pair = dict()
        train_ids_list = list()
        
        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)
            
        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)
        
        random.seed(42 if args.seed is None else args.seed)
        self.gt_ratio = args.gt_ratio
        self.shuffle_samples = np.arange(len(self.samples))
        if self.gt_ratio > 0.:
            nums = len(self.shuffle_samples)
            shuffle_labels = self.shuffle_samples[int(nums*self.gt_ratio):]
            random.shuffle(shuffle_labels)
            self.shuffle_samples[int(nums*self.gt_ratio):] = shuffle_labels
        else:
            random.shuffle(self.shuffle_samples)
        
        print('CVUSA: load train',' data_size =', len(self.samples))
         

    def __getitem__(self, index):
        idx, _, ground = self.idx2pair[self.samples[index]]
        idx_sat, sat, _ = self.idx2pair[self.samples[self.shuffle_samples[index]]]
        
        # load query -> ground image
        grd1 = cv2.imread(f'{self.data_folder}/{ground}')
        grd1 = cv2.cvtColor(grd1, cv2.COLOR_BGR2RGB)
        grd2 = copy.deepcopy(grd1)
        
        # load reference -> satellite image
        sat1 = cv2.imread(f'{self.data_folder}/{sat}')
        sat1 = cv2.cvtColor(sat1, cv2.COLOR_BGR2RGB)
        sat2 = copy.deepcopy(sat1)
        
        grd1 = self.transform_grd(image=grd1)["image"]
        grd2 = self.transform_grd(image=grd2)["image"]
        
        sat2 = self.ShiftScaleRotate(image=sat2)["image"]
        sat1 = self.transform_sat(image=sat1)["image"]
        sat2 = self.transform_sat(image=sat2)["image"]
        
        if np.random.random() < 0.5:
            grd1 = torch.flip(grd1, [2])
            sat1 = torch.flip(sat1, [2])
            
        return grd1, grd2, sat1, sat2

    def __len__(self):
        return len(self.samples)

class CVUSATrainFake(CVUSATrainIntra):
    def __init__(self, args):
        super(CVUSATrainFake, self).__init__(args)
        
    def __getitem__(self, index):
        idx, _, ground = self.idx2pair[self.samples[index]]

        # load query -> ground image
        grd = cv2.imread(f'{self.data_folder}/{ground}')
        grd = cv2.cvtColor(grd, cv2.COLOR_BGR2RGB)
        
        fake = ground.replace("streetview/panos","g2a_sat")
        fake = fake.replace(".jpg",".png")
        
        # load reference -> satellite image
        fake = cv2.imread(f'{self.data_folder}/{fake}')
        fake = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)
        
        grd = self.transform_grd(image=grd)["image"]
        fake = self.transform_sat(image=fake)["image"]
        
        if np.random.random() < 0.5:
            grd = torch.flip(grd, [2])
            fake = torch.flip(fake, [2])
            
        if np.random.random() < 0.5:
        
            r = np.random.choice([1,2,3])
            
            # use roll for ground view if rotate sat view
            c, h, w = grd.shape
            shifts = - w//4 * r
            grd = torch.roll(grd, shifts=shifts, dims=2)  
            
            # rotate sat img 90 or 180 or 270
            fake = torch.rot90(fake, k=r, dims=(1, 2)) 
        
        return grd,fake


class CVUSATrainSat(CVUSATrainIntra):
    def __init__(self, args):
        super(CVUSATrainSat, self).__init__(args)
        random.seed(42 if args.seed is None else args.seed)
        self.gt_ratio = args.gt_ratio
        self.shuffle_samples = np.arange(len(self.samples))
        if self.gt_ratio > 0.:
            nums = len(self.shuffle_samples)
            shuffle_labels = self.shuffle_samples[int(nums*self.gt_ratio):]
            random.shuffle(shuffle_labels)
            self.shuffle_samples[int(nums*self.gt_ratio):] = shuffle_labels
        else:
            random.shuffle(self.shuffle_samples)
        
        self.transform_scan_val_grd = A.Compose([A.Resize(self.grd_size[0],self.grd_size[1]),
                                        A.Normalize(),
                                        ToTensorV2()
        ])
        
        self.transform_scan_val_sat = A.Compose([A.Resize(self.sat_size[0],self.sat_size[1]),
                                        A.Normalize(),
                                        ToTensorV2()
        ])
        
        self.mode = "train"
        
    def __getitem__(self, index):
        idx, _, ground = self.idx2pair[self.samples[index]]
        idx_sat, sat, _ = self.idx2pair[self.samples[self.shuffle_samples[index]]]
        
        # load query -> ground image
        grd = cv2.imread(f'{self.data_folder}/{ground}')
        grd = cv2.cvtColor(grd, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        sat = cv2.imread(f'{self.data_folder}/{sat}')
        sat = cv2.cvtColor(sat, cv2.COLOR_BGR2RGB)
        
        if self.mode == "scan":
            return  self.transform_scan_val_grd(image=grd)["image"], self.transform_scan_val_sat(image=sat)["image"], index
        
        grd = self.transform_grd(image=grd)["image"]
        sat = self.transform_sat(image=sat)["image"]
        
        
        if np.random.random() < 0.5:
            grd = torch.flip(grd, [2])
            sat = torch.flip(sat, [2])
            
        if np.random.random() < 0.5:
        
            r = np.random.choice([1,2,3])
            
            # use roll for ground view if rotate sat view
            c, h, w = grd.shape
            shifts = - w//4 * r
            grd = torch.roll(grd, shifts=shifts, dims=2)  
            
            # rotate sat img 90 or 180 or 270
            sat = torch.rot90(sat, k=r, dims=(1, 2)) 
        
        return grd,sat,index
       
class CVUSAVal(torch.utils.data.Dataset):
    def __init__(self,args):
        super(CVUSAVal, self).__init__()
        
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

        self.df = pd.read_csv(f'{self.data_folder}/splits/val-19zl.csv', header=None)
        
        self.df = self.df.rename(columns={0:"sat", 1:"ground", 2:"ground_anno"})
        
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
    
        self.sat_images = self.df.sat.values
        self.grd_images = self.df.ground.values
        self.label = self.df.idx.values 
        
                
        print('CVUSA: load val',' data_size =', len(self.sat_images))
       

    def __getitem__(self, index):
        if self.img_type == "sat":
            img = cv2.imread(f'{self.data_folder}/{self.sat_images[index]}')
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform_sat(image=img)['image']
            
        elif self.img_type == "grd":
            img = cv2.imread(f'{self.data_folder}/{self.grd_images[index]}')
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform_grd(image=img)['image']
            
        label = torch.tensor(self.label[index], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.sat_images)
        
