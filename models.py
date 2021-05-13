import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F

class Net_convo(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(2, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU())
        
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.dropout = nn.Dropout(0.2)
        self.lin1 = nn.Linear(3200, 10)
        self.lin2 = nn.Linear(10, 1)
        self.lin3 = nn.Linear(2, 1)
        
        self.logsoft = nn.LogSoftmax(dim=2)

    def forward(self, x):
        
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        
        x = x.view(25, 2, -1)
        
        # (25, 2, 3200)
        x = F.relu(self.lin1(x))
        # (25, 2, 10)
        output1 = self.logsoft(x)
        
        # (25, 2, 10)
        x = F.relu(self.lin2(x))
        
        # (25, 2, 1)
        x = torch.flatten(x, 1)
        
        # (25, 2)
        output2 = torch.sigmoid(self.lin3(x))
        # (25, 1)
        return output1, output2
    
    
class Net_FC(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Flatten()
        self.fc1 = nn.Linear(196, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc4 = nn.Linear(32, 10)
        self.fc5 = nn.Linear(10, 1)
        self.fc6 = nn.Linear(2, 1)
        
        self.logsoft = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        # INPUT IS A (2, 14, 14) TENSOR, OUTPUT IS (1)
        x = self.fc0(x)
        # (2, 196)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        output1 = self.logsoft(x)
        
        x = F.relu(self.fc5(x))
        x = x.view(-1)
        output2 = self.fc6(x)
        
        return output1, output2
