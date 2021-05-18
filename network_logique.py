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

def compute_err_digit_recog(model, test_input, test_classes, batches = False):

    correct_count_digit, all_count_digit = 0, 0
    correct_count_equal, all_count_equal = 0, 0
    
    if not batches:
    
    
        for img, label, target, i in zip(test_input, test_classes, 
                                                test_target, range(len(test_classes))):   
            with torch.no_grad():
                log_probs_digits, probs_equality = model(img)

            probs = torch.exp(log_probs_digits)
            _, preds = torch.max(probs,dim=1)
            true_labels = label

            for predicted, groundtruth in zip(preds, true_labels):
                if(predicted == groundtruth):
                    correct_count_digit += 1
                all_count_digit += 1


            if((torch.sigmoid(probs_equality) > 0.5 and target == 1) or 
                       (torch.sigmoid(probs_equality) <= 0.5 and target == 0)):
                correct_count_equal += 1
            all_count_equal +=1
            
            
    else:

        for i in range(0, len(test_input), mini_batch_size):
        
            with torch.no_grad():
                log_probs_digits, probs_equality = model(test_input.narrow(0, b, mini_batch_size))
            probs = torch.exp(log_probs_digits)
            _, preds = torch.max(probs,dim=2)
            true_labels = test_classes.narrow(0, b, mini_batch_size)
            targets = test_target.narrow(0, b, mini_batch_size)
            
            
            
            for predicted, groundtruth in zip(preds, true_labels):
                if(predicted[0] == groundtruth[0]):
                    correct_count_digit += 1
                if(predicted[1] == groundtruth[1]):
                    correct_count_digit += 1
                all_count_digit += 2
               
            for prob_equality, target in zip(probs_equality.view(-1), targets):
                if((torch.sigmoid(prob_equality) >= 0.5 and target == 1) or 
                           (torch.sigmoid(prob_equality) < 0.5 and target == 0)):
                    correct_count_equal += 1
                all_count_equal +=1
            
            
        
    print("Number Of Images Tested =", all_count_digit)
    print("\nModel Accuracy =", (correct_count_digit/all_count_digit), '\n\n')
    print("Number Of Inequalities tested =", all_count_equal)
    print("\nModel Accuracy =", (correct_count_equal/all_count_equal))

class ConvoLogic(nn.Module):

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
            if preds[i][0] <= preds[i][1]:
                output2[i] = 0
            else:
                output2[i] = 1
            # (25, 1)
        return output1, output2

print(get_n_params(ConvoLogic()))


model = ConvoLogic()

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
epochs = 25

mini_batch_size = 25

for e in range(epochs):
    time0 = time()
    running_loss = 0
    for b in range(0, train_input.size(0), mini_batch_size):


        optimizer.zero_grad()
        output1, output2 = model(train_input.narrow(0, b, mini_batch_size))
        print(output2)
        
        loss = criterion(output1.view(-1, 10), train_classes.narrow(0, b, mini_batch_size).view(-1))

        loss.backward()
        optimizer.step()

        running_loss += loss

    print("Epoch {} - Training loss: {}".format(e+1, running_loss/len(train_input)))
    print("\nTraining Time =", round(time()-time0, 2), "seconds")
compute_err_digit_recog(model, test_input, test_classes, batches = True)