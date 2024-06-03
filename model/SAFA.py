import torch
import torch.nn as nn
class SpatialAware(nn.Module):
    def __init__(self, in_channel, d=8):
        super().__init__()
        c = in_channel
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, c // 2),
                nn.Linear(c // 2, c),
            ) for _ in range(d)
        ])

    def forward(self, x):
        # channel dim
        x,_ = torch.max(x, dim=1)
        x = [b(x) for b in self.fc]
        x = torch.stack(x, -1)
        return x

class SAFA(nn.Module):
    def __init__(self,model=None,in_channel=64):
        super().__init__()
        self.features = model
        self.spatial_aware = SpatialAware(in_channel, d=8)
        
    def forward(self, x):
        x = self.features.forward_features(x)
        x = torch.flatten(x, start_dim=-2, end_dim=-1)
        x_sa = self.spatial_aware(x)
        # b c h*w @ b h*w d = b c d
        x = x @ x_sa
        x = torch.transpose(x, -1, -2).flatten(-2, -1) # [B, 8*384]
        return x
    