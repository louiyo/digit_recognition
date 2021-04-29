import torch
import math

from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import Tensor

from time import time

import dlc_practical_prologue as prologue

train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)

train_input_norm = (train_input - torch.min(train_input)) / (torch.max(train_input))
test_input_norm = (test_input - torch.min(train_input)) / (torch.max(train_input))

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size = 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size= 2)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):

        #print(x.size())
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 3))
        #print(x.size())
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 3))
        #print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        #print(x.size())
        x = F.relu(self.fc1(x))
        #print(x.size())
        x = F.relu(self.fc2(x))
        #print(x.size())
        x = self.fc3(x)
        #print(x.size())
        return x


#Attention il faudra modifier ça y'apaldroit
    def num_flat_features(self, x):

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net2(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Flatten()
        self.fc1 = nn.Linear(196, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.fc5 = nn.Linear(10, 8)
        self.fc6 = nn.Linear(8, 1)
        self.logsoft = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        # INPUT IS A (2, 14, 14) TENSOR, OUTPUT IS (1)
        x = self.fc0(x)
        # (2, 196)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.logsoft(x)
        output1 = x
        x = F.relu(self.fc5(x))
        print(x)
        output2 = F.relu(self.fc6(x))[0]
        return output1, output2
    


model = Net2()

criterion1 = nn.NLLLoss()
criterion2 = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
time0 = time()
epochs = 25

#train_input_flat = torch.flatten(train_input_norm, 0, 1)
#train_classes_flat = torch.flatten(train_classes, 0, 1)
mini_batch_size = 25

for e in range(epochs):
    running_loss = 0
    for i in range(len(train_input_norm)):
        
        optimizer.zero_grad()
        #optimizer_aux.zero_grad()
        #print(torch.flatten(train_input_flat.narrow(0, b, mini_batch_size), 1, 2).size())
        output1, output2 = model(train_input_norm[i])
        
        #print(labels.size())
        loss1 = criterion1(output1, train_classes[i])
        loss2 = criterion2(output2, train_target[i].reshape(1).to(torch.float32))
        
        loss = loss1 + loss2 * 0.6
        loss.backward()
        optimizer.step()
        
        
        
        running_loss += loss1.item() + loss2.item()
    
    print("Epoch {} - Training loss: {}".format(e+1, running_loss/len(train_input)))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)





def train_model(model, train_input, train_target, mini_batch_size = 25):
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1)
    nb_epochs = 100

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):

            output = model(train_input.narrow(0, b, mini_batch_size))
            temp = output.flatten()
            loss = criterion(temp, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()



correct_count, all_count = 0, 0
test_input_flat = torch.flatten(torch.flatten(test_input_norm, 0, 1), 1, 2)
test_classes_flat = torch.flatten(test_classes, 0, 1)

for img, label, i in zip(test_input_flat, test_classes_flat, range(len(test_classes_flat))):   
    
    with torch.no_grad():
        logps = model(img[None, ...])


    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    true_label = test_classes_flat.numpy()[i]
    print(pred_label, true_label)
    if(true_label == pred_label):
        correct_count += 1
    all_count += 1


print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

"""
print(train_input.size(0))
model = Net()
train_model(model, train_input.float(), train_target.float())


def compute_nb_errors(model, data_input, data_target, mini_batch_size = 25):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors



print('train_error {:.02f}% test_error {:.02f}%'.format(

    compute_nb_errors(model, train_input, train_target) / train_input.size(0) * 100,
    compute_nb_errors(model, test_input, test_target) / test_input.size(0) * 100
)
)
"""