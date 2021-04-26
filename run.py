import numpy as np
from dlc_practical_prologue import *
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

train_input,   train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(
    100)

print(train_input.size)
