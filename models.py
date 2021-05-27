import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F

class Net_Convo_AuxLosses(nn.Module):
    
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
        self.lin1 = nn.Linear(3200, 50)
        self.lin12 = nn.Linear(50, 10)
        self.lin2 = nn.Linear(10, 1)
        self.lin3 = nn.Linear(2, 1)
        
        self.logsoft = nn.LogSoftmax(dim=2)

    def forward(self, x):
        
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        
        x = x.view(25, 2, -1)
        
        # (25, 2, 3200)
        x = self.dropout(x)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin12(x))
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
    
# optimal parameters:
# lr = 1e-3, batch_size = 10
class Net_FC(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Flatten(2)
        self.fc1 = nn.Linear(196, 256)
        self.fc2 = nn.Linear(256, 400)
        self.fc3 = nn.Linear(400, 256)
        self.fc4 = nn.Linear(256, 100)
        self.fc5 = nn.Linear(100, 20)
        self.fc6 = nn.Linear(20, 10)
        self.fc7 = nn.Linear(10, 1)
        self.fc8 = nn.Linear(2, 1)
        
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(num_features=2)
        self.logsoft = nn.LogSoftmax(dim=1) 
    
    
    def forward(self, x):
        # INPUT IS A (2, 14, 14) TENSOR, OUTPUT IS (1)
        x = self.fc0(x)
        # (2, 196)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn1(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn1(self.fc3(x)))
        x = F.relu(self.bn1(self.fc4(x)))
        x = F.relu(self.bn1(self.fc5(x)))
        x = F.relu(self.bn1(self.fc6(x)))
        output1 = self.logsoft(x)
        
        x = F.relu(self.bn1(self.fc7(x)))
        x = x.view(-1, 2)
        output2 = torch.sigmoid(self.fc8(x))
        
        return output1, output2


class Net_Conv_Classification(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(2, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2))
        
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.lin1 = nn.Linear(6400, 3200)
        self.lin2 = nn.Linear(3200, 10)
        self.lin3 = nn.Linear(10, 1)
        
        self.logsoft = nn.LogSoftmax(dim=2)

    def forward(self, x):
        
        x = self.convlayer1(x)
        x = self.convlayer2(x)
        
        x = x.view(25, -1)
    
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        output = self.lin3(x)
        # (25, 1)
        return output


class Net_Convo_Logic(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.convlayer1 = nn.Sequential(
            nn.Conv2d(2, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2))
        
        self.convlayer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
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
        x = self.logsoft(x)

        output1 = x

        output2 = torch.empty(25)
        probs = torch.exp(x)
        _, preds = torch.max(probs,dim=2)
        for i in range(0, len(preds)):
            if preds[i][0] > preds[i][1]:
                output2[i] = 0.
            else:
                output2[i] = 1.
            # (25, 1)
        return output1, output2
        
