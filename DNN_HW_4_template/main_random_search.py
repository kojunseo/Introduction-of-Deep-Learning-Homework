import numpy as np
from utils import load_mnist, load_fashion_mnist, set_random_seed
from models.LeNet_5 import LeNet_5
from models.AlexNet import AlexNet
from models.ResNet import ResNet
import torch
import random

set_random_seed(123)

# =============================== EDIT HERE ===============================
"""
    Build model Architecture and do experiment.
"""
# lenet / alexnet / resnet
model_name = 'lenet'
# mnist / fashion_mnist
dataset = 'fashion_mnist'

# Hyper-parameters to search
num_epochs_list = [1, 2, 3, 4, 5]
learning_rate_list = [0.1, 0.2, 0.3]
reg_lambda_list = [0.1, 0.2, 0.3]
batch_size_list = [100]
num_search = 10

test_every = 1
print_every = 5000
# =========================================================================
assert dataset in ['mnist', 'fashion_mnist']

# Dataset
if dataset == 'mnist':
    x_train, y_train, x_test, y_test = load_mnist('./data')
else:
    x_train, y_train, x_test, y_test = load_fashion_mnist('./data')

y_train, y_test = np.argmax(y_train, -1).astype(int), np.argmax(y_test, -1).astype(int)

# Random 10% of train data as valid data
num_train = len(x_train)
perm = np.random.permutation(num_train)
num_valid = int(len(x_train) * 0.1)

valid_idx = perm[:num_valid]
train_idx = perm[num_valid:]

x_valid, y_valid = x_train[valid_idx], y_train[valid_idx]
x_train, y_train = x_train[train_idx], y_train[train_idx]

num_train, channel, height, width = x_train.shape
num_class = len(np.unique(y_test))
print('# of Training data : ', num_train)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device : ', device)

print('Search Starts...')
best_acc = -1
best_hyper_params = []
for search_cnt in range(num_search):
    num_epochs = random.choice(num_epochs_list)
    learning_rate = random.choice(learning_rate_list)
    reg_lambda = random.choice(reg_lambda_list)
    batch_size = random.choice(batch_size_list)

    if model_name == 'lenet':
        model = LeNet_5(x_train.shape[1], num_class, learning_rate, reg_lambda, device)
    elif model_name == 'alexnet':
        model = AlexNet(x_train.shape[1], num_class, learning_rate, reg_lambda, device)
    elif model_name == 'resnet':
        model = ResNet(x_train.shape[1], num_class, learning_rate, reg_lambda, device)
    model = model.to(device)

    cur_acc = model.train(x_train, y_train, x_valid, y_valid, num_epochs, batch_size, test_every, print_every)
    if cur_acc > best_acc:
        best_acc = cur_acc
        best_hyper_params = [num_epochs, learning_rate, reg_lambda, batch_size]
    print(f'search count: {search_cnt}, cur_valid_acc: {cur_acc},  best_valid_acc: {best_acc}')
print(f'best_valid_acc: {best_acc}, best_hyper_params: {best_hyper_params} (num_epochs, learning_rate, reg_lambda, batch_size)')
