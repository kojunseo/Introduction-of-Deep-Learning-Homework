import time
import numpy as np
from utils import load_mnist, load_fashion_mnist
from models.MLP_classifier import MLP_classifier
import matplotlib.pyplot as plt
import copy

np.random.seed(123)
np.tanh = lambda x: x


# =============================== EDIT HERE ===============================

"""
    Build model Architecture and do experiment.
"""

# mnist / fashion_mnist
dataset = 'mnist'

# Hyper-parameters
num_epochs = 200
learning_rate = 0.01
reg_lambda = 0.01
print_every = 10

batch_size = 1000

# =========================================================================
assert dataset in ['mnist', 'fashion_mnist']

# Dataset
if dataset == 'mnist':
    x_train, y_train, x_test, y_test = load_mnist('./data')
else:
    x_train, y_train, x_test, y_test = load_fashion_mnist('./data')

x_train, x_test = np.squeeze(x_train), np.squeeze(x_test)

# Random 10% of train data as valid data
num_train = len(x_train)
perm = np.random.permutation(num_train)
num_valid = int(len(x_train) * 0.1)

valid_idx = perm[:num_valid]
train_idx = perm[num_valid:]

x_valid, y_valid = x_train[valid_idx], y_train[valid_idx]
x_train, y_train = x_train[train_idx], y_train[train_idx]

num_train, height, width = x_train.shape
num_class = y_train.shape[1]
print('# of Training data : ', num_train)

# MLP_classifier __init__ function
model = MLP_classifier(height*width, num_class, learning_rate)

print('Training Starts...')
# MLP_classifier train function
model.train(x_train.reshape(-1, height*width), y_train, x_valid.reshape(-1, height*width), y_valid, num_epochs, batch_size, print_every)

# TEST ACCURACY
# MLP_classifier restore function
model.restore()
# MLP_classifier predict function
pred = model.predict(x_test.reshape(-1, height*width))
true = np.argmax(y_test, -1).astype(int)

correct = len(np.where(pred == true)[0])
total = len(true)
test_acc = correct / total

print('Test Accuracy at Best Epoch : %.2f' % (test_acc))

model.plot_accuracy()
