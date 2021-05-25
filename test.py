from utility import *
from models import *
from dlc_practical_prologue import *

import torch
import matplotlib.pyplot as plt
from time import time
from torch import nn, optim
from torch.nn import functional as F

train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(
    1000)

train_input_norm = (train_input - torch.min(train_input)) / (torch.max(train_input))
test_input_norm = (test_input - torch.min(train_input)) / (torch.max(train_input))

model = Net_FC()

criterion1 = nn.NLLLoss()
criterion2 = nn.BCEWithLogitsLoss()

compute_performances_auxilliary(model, criterion1, criterion2, train_input_norm, train_classes, train_target, test_input_norm, test_classes, test_target, batches = True)

