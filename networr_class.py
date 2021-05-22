from dlc_practical_prologue import *
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F


train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(
    1000)


train_input_norm = (train_input - torch.min(train_input)) / (torch.max(train_input))
test_input_norm = (test_input - torch.min(train_input)) / (torch.max(train_input))

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def compute_err_digit_recog(model, test_input, test_classes, mini_batch_size = 25, batches = False):

    correct_count_equal, all_count_equal = 0, 0
    
    if not batches:
    
    
        for img, target, i in zip(test_input, test_target, range(len(test_classes))):   
            with torch.no_grad():
                probs_equality = model(img)

            if((torch.sigmoid(probs_equality) > 0.5 and target == 1) or 
                       (torch.sigmoid(probs_equality) <= 0.5 and target == 0)):
                correct_count_equal += 1
            all_count_equal +=1
            
            
    else:

        for i in range(0, len(test_input), mini_batch_size):
        
            with torch.no_grad():
                probs_equality = model(test_input.narrow(0, b, mini_batch_size))
            
            targets = test_target.narrow(0, b, mini_batch_size)
               
            for prob_equality, target in zip(probs_equality.view(-1), targets):
                if((torch.sigmoid(prob_equality) >= 0.5 and target == 1) or 
                           (torch.sigmoid(prob_equality) < 0.5 and target == 0)):
                    correct_count_equal += 1
                all_count_equal +=1

    print("Number Of Inequalities tested =", all_count_equal)
    print("\nModel Accuracy =", (correct_count_equal/all_count_equal))

class Net_convo(nn.Module):
    
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
    
get_n_params(Net_convo())

model = Net_convo()

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.006, momentum=0.8)
epochs = 25
mini_batch_size = 25

for e in range(epochs):
    time0 = time()
    running_loss = 0
    for b in range(0, train_input.size(0), mini_batch_size):


        optimizer.zero_grad()
        output= model(train_input.narrow(0, b, mini_batch_size))
        
        loss = criterion(output.view(-1), train_target.narrow(0, b, mini_batch_size).to(torch.float32))

        loss.backward()
        optimizer.step()

        running_loss += loss

    print("Epoch {} - Training loss: {}".format(e+1, running_loss/len(train_input)))
    print("\nTraining Time =", round(time()-time0, 2), "seconds")
compute_err_digit_recog(model, test_input, test_classes, batches = True)