import torch
import math

from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import Tensor

import dlc_practical_prologue as prologue

train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)

print(train_input[1])
print(train_target[1])
print(train_classes[1])


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 6, kernel_size = 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size= 2)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):

        
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 3))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    
    def num_flat_features(self, x):

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def train_model(model, train_input, train_target, mini_batch_size = 25):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1)
    nb_epochs = 250

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            print(output)
            temp = output.flatten()
            print(temp)
            print(train_target.narrow(0, b, mini_batch_size))
            loss = criterion(temp, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()


print(train_input.size(0))
model = Net()
train_model(model, train_input, train_target)


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