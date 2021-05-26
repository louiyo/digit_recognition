from dlc_practical_prologue import *
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from models import *
from utility import *


train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(
    1000)

"""
model = Net_FC()
criterion1 = nn.NLLLoss()
criterion2 = nn.BCELoss()
compute_performances_auxilliary(model, criterion1, criterion2, train_input, train_classes, 
                                    train_target, test_input, test_classes, test_target, 
                                    batches = True, mini_batch_size = 10, lr = 7.5e-3)


# ConvNet with learnt comparison
model = Net_Convo_AuxLosses()
print("Convolutional model with auxiliary losses")
criterion1 = nn.NLLLoss()
criterion2 = nn.BCELoss()
compute_performances_auxilliary(model, criterion1, criterion2, train_input, train_classes, 
                                    train_target, test_input, test_classes, test_target, 
                                    batches = True, mini_batch_size = 25, lr = 1e-3)

"""
model = Net_Convo_Logic()
criterion1 = nn.NLLLoss()
compute_performances(model, criterion1, train_input, train_classes, 
                        train_target, test_input, test_classes, test_target,
                        logic = True, batches = True, mini_batch_size = 25, lr = 1e-3, mom = 0.95)


"""

model = Net_Conv_Classification()
criterion1 = nn.BCEWithLogitsLoss()
compute_performances(model, criterion1, train_input, train_classes, 
                        train_target, test_input, test_classes, test_target,
                        logic = False, batches = True, mini_batch_size = 25, lr = 1e-3, mom = 0.95)

"""