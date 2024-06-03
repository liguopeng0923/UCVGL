import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from .SAFA import SAFA
from torch.cuda.amp import autocast
from .ConvNext import convnext_base

class Sample4Geo(nn.Module):
    """
    Simple Siamese baseline with avgpool
    """
    def __init__(self, args,mode="cross"):
        """
        self_dim: feature dimension (default: 1024)
        """
        self.mode = mode
        super(Sample4Geo, self).__init__()
        kwargs = {'num_classes' : None, 'drop_path_rate': 0.2}
            
        self.grd_size = args.grd_size
        self.sat_size = args.sat_size
        
        self.cross_query_model = SAFA(convnext_base(**kwargs),in_channel=(self.grd_size[0] // 32) * (self.grd_size[1] // 32))
        self.cross_reference_model = SAFA(convnext_base(**kwargs),in_channel=(self.sat_size[0] // 32) * (self.sat_size[1] // 32))      

    def forward(self, im_q=None, im_k=None,mode=None,delta=None, atten=None, indexes=None):
        mode = self.mode if mode==None else mode
        if mode=="cross":
            emb_q = self.cross_query_model(im_q)
            emb_k = self.cross_reference_model(im_k)
            return F.normalize(emb_q,dim=-1), F.normalize(emb_k,dim=-1)
        elif mode=="grd":
            emb_q = self.cross_query_model(im_q)
            return F.normalize(emb_q,dim=-1)
        elif mode=="sat":
            emb_k = self.cross_reference_model(im_q)
            return F.normalize(emb_k,dim=-1)
        else:
            print(f"the forward mode {mode} is not implemented")
            
    