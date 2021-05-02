import numpy as np
from dlc_practical_prologue import *
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F


train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(
    500)

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

class Net2(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Flatten()
        self.fc1 = nn.Linear(196, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.fc5 = nn.Linear(20, 1)
        
        self.logsoft = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        # INPUT IS A (2, 14, 14) TENSOR, OUTPUT IS (1)
        x = self.fc0(x)
        # (2, 196)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        output1 = self.logsoft(x)
        #x = F.relu(self.fc5(torch.exp(x)))
        #print(x.size())
        x = torch.flatten(x, 0)
        #print(x.size())
        output2 = F.relu(self.fc5(x))
        #print(output2)
        return output1, output2


class Net3(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size = 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size= 2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.fc4 = nn.Linear(20, 1)

        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self, x):
        
        print(x.size())
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 2))
        print(x.size())
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 2))
        print(x.size())
        x = x.view(-1, 64*2*2)
        print(x.size())
        x = F.relu(self.fc1(x))
        print(x.size())
        x = F.relu(self.fc2(x))
        print(x.size())
        x = F.relu(self.fc3(x))
        print(x.size())
        output1 = self.logsoft(x)
        x = x.view(25, 20)
        print(x.size())
        output2 = F.relu(self.fc4(x))

        return output1, output2


model = Net3()

criterion1 = nn.NLLLoss()
criterion2 = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 25

#train_input_flat = torch.flatten(train_input_norm, 0, 1)
#train_classes_flat = torch.flatten(train_classes, 0, 1)
mini_batch_size = 25

for e in range(epochs):
    running_loss = 0
    for i in range(0, len(train_input_norm), mini_batch_size):
        
        optimizer.zero_grad()
        #optimizer_aux.zero_grad()
        #print(torch.flatten(train_input_flat.narrow(0, b, mini_batch_size), 1, 2).size())
        
        output1, output2 = model(train_input_norm.narrow(0, i, mini_batch_size))
        

        
        loss1 = criterion1(output1, train_classes.narrow(0, i, mini_batch_size).flatten())
        print(output2)
        temp = train_target.narrow(0, i, mini_batch_size).to(torch.float32).view(25, -1)
        print(temp)
        loss2 = criterion2(output2, temp)
        #print(loss2)
        
        loss = loss1 + loss2 * 0.15
        loss.backward()
        optimizer.step()
        
        
        
        running_loss += loss1.item() + loss2.item()
    
    print("Epoch {} - Training loss: {}".format(e+1, running_loss/len(train_input)))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)


def compute_err_digit_recog(model, test_input_norm, test_classes): # CHANGER LES NOMS SVP

    correct_count_digit, all_count_digit = 0, 0
    correct_count_equal, all_count_equal = 0, 0
    
    for i in range(0, test_input_norm.size(), mini_batch_size):   

        with torch.no_grad():
            log_probs_digits, probs_equality = model(test_input_normnarrow(0, i, mini_batch_size))
        
        #print(torch.sigmoid(probs_equality), test_target[i])


        probs = torch.exp(log_probs_digits)
        _, preds = torch.max(probs,dim=1)
        
        for b in range(mini_batch_size):

            true_labels = test_classes[b]

            for predicted, groundtruth in zip(preds, true_labels):
                if(predicted == groundtruth):
                    correct_count_digit += 1
                #print(predicted, groundtruth)
                all_count_digit += 1
            
            
            
            if((torch.sigmoid(probs_equality) >= 0.5 and test_target[i] == 1) or (torch.sigmoid(probs_equality) < 0.5 and test_target[i] == 0)):
                correct_count_equal += 1
            all_count_equal +=1
        

    print("Number Of Images Tested =", all_count_digit)
    print("\nModel Accuracy =", (correct_count_digit/all_count_digit))
    print("Number Of Images Tested =", all_count_equal)
    print("\nModel Accuracy =", (correct_count_equal/all_count_equal))
    
compute_err_digit_recog(model, test_input_norm, test_classes)