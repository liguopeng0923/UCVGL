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

# pytorch implementation of CVACT loader
class CVACTTrainIntra(torch.utils.data.Dataset):
    def __init__(self, args):
        super(CVACTTrainIntra, self).__init__()
        
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
        
        anuData = sio.loadmat(os.path.join(self.data_folder, 'ACT_data.mat'))

        ids = anuData['panoIds']

        train_ids = ids[anuData['trainSet'][0][0][1]-1]
        
        train_ids_list = []
        train_idsnum_list = []
        self.idx2numidx = dict()
        self.numidx2idx = dict()
        self.idx_ignor = set()
        i = 0
        
        for i,idx in enumerate(train_ids.squeeze()):
            idx = str(idx)
            
            grd_path = f'streetview/{idx}_grdView.jpg'
            fake_path = f'g2a_sat/{idx}_grdView.png'
            sat_path = f'satview_polish/{idx}_satView_polish.jpg'
            
            if not os.path.exists(f'{self.data_folder}/{grd_path}') or not os.path.exists(f'{self.data_folder}/{fake_path}') or not os.path.exists(f'{self.data_folder}/{sat_path}'):
                self.idx_ignor.add(idx)
            else:
                self.idx2numidx[idx] = i
                self.numidx2idx[i] = idx
                train_ids_list.append(idx)
                train_idsnum_list.append(i)
                i+=1
        
        print("IDs not found in grd/fake/sat images:", self.idx_ignor)
        
        self.train_ids = train_ids_list
        self.train_idsnum = train_idsnum_list
        self.samples = copy.deepcopy(self.train_idsnum)
        
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
            
        
        print('CVACT: load train',' data_size =', len(self.samples))
         

    def __getitem__(self, index):
        idnum = self.samples[index]
        
        idx = self.numidx2idx[idnum]
        
        # load query -> ground image
        grd1 = cv2.imread(f'{self.data_folder}/streetview/{idx}_grdView.jpg')
        grd1 = cv2.cvtColor(grd1, cv2.COLOR_BGR2RGB)
        grd2 = copy.deepcopy(grd1)
        
        # load reference -> satellite image
        sat1 = cv2.imread(f'{self.data_folder}/satview_polish/{idx}_satView_polish.jpg')
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

class CVACTTrainFake(CVACTTrainIntra):
    def __init__(self, args):
        super(CVACTTrainFake, self).__init__(args)
        
    def __getitem__(self, index):
        idnum = self.samples[index]
        
        idx = self.numidx2idx[idnum]
        
        # load query -> ground image
        grd = cv2.imread(f'{self.data_folder}/streetview/{idx}_grdView.jpg')
        grd = cv2.cvtColor(grd, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        fake = cv2.imread(f'{self.data_folder}/g2a_sat/{idx}_grdView.png')
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


class CVACTTrainSat(CVACTTrainIntra):
    def __init__(self, args):
        super(CVACTTrainSat, self).__init__(args)
        
        
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
        idnum = self.samples[index]
        
        idx = self.numidx2idx[idnum]
        
        idnum_sat = self.samples[self.shuffle_samples[index]]
        idx_sat = self.numidx2idx[idnum_sat]
        
        # load query -> ground image
        grd = cv2.imread(f'{self.data_folder}/streetview/{idx}_grdView.jpg')
        grd = cv2.cvtColor(grd, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        sat = cv2.imread(f'{self.data_folder}/satview_polish/{idx_sat}_satView_polish.jpg')
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
       
class CVACTVal(torch.utils.data.Dataset):
    def __init__(self,args):
        super(CVACTVal, self).__init__()
        
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

        anuData = sio.loadmat(os.path.join(self.data_folder, 'ACT_data.mat'))

        ids = anuData['panoIds']
        
        ids = ids[anuData[f'valSet'][0][0][1]-1]
        
        ids_list = []
       
        self.idx2label = dict()
        self.idx_ignor = set()
        
        i = 0
        
        for idx in ids.squeeze():
            
            idx = str(idx)
            
            grd_path = f'{self.data_folder}/streetview/{idx}_grdView.jpg'
            sat_path = f'{self.data_folder}/satview_polish/{idx}_satView_polish.jpg'
   
            if not os.path.exists(grd_path) or not os.path.exists(sat_path):
                self.idx_ignor.add(idx)
            else:
                self.idx2label[idx] = i
                ids_list.append(idx)
                i+=1
        
        self.samples = ids_list
        print('CVACT: load val',' data_size =', len(self.samples))
       
        

    def __getitem__(self, index):
        idx = self.samples[index]
        
        if self.img_type == "sat":
            path = f'{self.data_folder}/satview_polish/{idx}_satView_polish.jpg'
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform_sat(image=img)['image']
            
        elif self.img_type == "grd":
            path = f'{self.data_folder}/streetview/{idx}_grdView.jpg'
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform_grd(image=img)['image']
            
        label = torch.tensor(self.idx2label[idx], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.samples)
        
