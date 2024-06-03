import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

# this is equivalent to the loss function in CVMNet with alpha=10, here we simplify it with cosine similarity
class InfoNCELoss(nn.Module):
    def __init__(self, args,):
        super(InfoNCELoss, self).__init__()
        self.device = args.gpu if args.gpu is not None else "cuda"
        self.temperature = torch.nn.Parameter(torch.ones([]) * np.log(1/0.07))

    def forward(self, inputs_q, inputs_k,GT=None,label_smoothing=0.3):
        if GT is None:
            device = self.device
            n = inputs_q.size(0)

            normalized_inputs_q = inputs_q / torch.norm(inputs_q, dim=1, keepdim=True)
            normalized_inputs_k = inputs_k / torch.norm(inputs_k, dim=1, keepdim=True)
            
            # Compute similarity matrix
            sim_mat = torch.matmul(normalized_inputs_q, normalized_inputs_k.t())*torch.exp(self.temperature)

            labels = torch.arange(n).to(device)
            loss_i = F.cross_entropy(sim_mat,labels,label_smoothing=label_smoothing)
            loss_t = F.cross_entropy(sim_mat.T,labels,label_smoothing=label_smoothing)
            loss = loss_i / 2 + loss_t / 2
            return loss
        else:
            device = self.device
            n = inputs_q.size(0)

            normalized_inputs_q = inputs_q / torch.norm(inputs_q, dim=1, keepdim=True)
            normalized_inputs_k = inputs_k / torch.norm(inputs_k, dim=1, keepdim=True)
            
            # Compute similarity matrix
            org_sim_mat = torch.matmul(normalized_inputs_q, normalized_inputs_k.t())*torch.exp(self.temperature)
            
            zeros = torch.zeros(n)[:,None].to(device)
            
            labels = n*torch.ones(n).long()
            labels[GT] = torch.arange(n)[GT]
            labels = labels.to(device)
            
            loss_i = F.cross_entropy(torch.cat([org_sim_mat,zeros],dim=1),labels,label_smoothing=label_smoothing)
            loss_t = F.cross_entropy(torch.cat([org_sim_mat.T,zeros],dim=1),labels,label_smoothing=label_smoothing)
            loss = loss_i / 2 + loss_t / 2
            return loss

        
        