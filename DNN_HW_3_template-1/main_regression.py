from models.MLP_regressor import MLP_regressor
import time
import numpy as np
from utils import ConcreteData, HouseData, MSE
from models.MLP_classifier import MLP_classifier
import matplotlib.pyplot as plt
import copy

np.random.seed(123)
np.tanh = lambda x: x


# =============================== EDIT HERE ===============================

"""
    Build model Architecture and do experiment.
"""

# concrete / house
dataset = 'house'

# Hyper-parameters
num_epochs = 1000
learning_rate = 0.001
reg_lambda = 0.01
print_every = 10

batch_size = 1000

# =========================================================================
assert dataset in ['concrete', 'house']

# Dataset
if dataset == 'concrete':
    x_train, y_train = ConcreteData('./data/concrete', 'train.csv')
    x_test, y_test = ConcreteData('./data/concrete', 'test.csv')
else:
    x_train, y_train = HouseData('./data/house', 'train.csv')
    x_test, y_test = HouseData('./data/house', 'test.csv')

# Random 10% of train data as valid data
num_train = len(x_train)
perm = np.random.permutation(num_train)
num_valid = int(len(x_train) * 0.1)

valid_idx = perm[:num_valid]
train_idx = perm[num_valid:]

x_valid, y_valid = x_train[valid_idx], y_train[valid_idx]
x_train, y_train = x_train[train_idx], y_train[train_idx]

num_train, num_features = x_train.shape
print('# of Training data : ', num_train)

model = MLP_regressor(num_features, learning_rate)

print('Training Starts...')
model.train(x_train, y_train, x_valid, y_valid, num_epochs, batch_size, print_every)

# TEST ACCURACY
model.restore()
pred = model.predict(x_test)
mse = MSE(pred, y_test)

print('MSE at Best Epoch : %.2f' % (mse))

model.plot_accuracy()
