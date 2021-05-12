import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F

class Net_convo(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, 1, groups=2)
        self.norm = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, groups=2)
        self.lin1 = nn.Linear(3200, 10)
        self.lin2 = nn.Linear(10, 1)
        self.lin3 = nn.Linear(2, 1)
        
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # (25, 2, 14, 14)
        # We use batch normalization for covariate shift
        x = F.relu(self.conv1(x))
        x = self.norm(x)
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 1)
        x = x.view(25, 2, 32, 10, 10)
        x = torch.flatten(x, 2)
        # (25, 2, 3200)
        x = F.relu(self.lin1(x))
        # (25, 2, 10)
        output1 = self.logsoft(x)
        # (25, 2, 10)
        x = F.relu(self.lin2(x))
        print(x.size())
        # (25, 2, 1)
        x = torch.flatten(x, 1)
        # (25, 2)
        output2 = F.relu(self.lin3(x))
        # (25, 1)
        return output1, output2
