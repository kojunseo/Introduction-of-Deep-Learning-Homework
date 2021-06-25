import os
import numpy as np
import torch
import random


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def load_fashion_mnist(data_path):
    mnist_path = os.path.join(data_path, 'fashion_mnist')

    x_train = np.load(os.path.join(mnist_path, 'train_images_full.npy'))
    y_train = np.load(os.path.join(mnist_path, 'train_labels_full.npy'))
    x_test = np.load(os.path.join(mnist_path, 'test_images_full.npy'))
    y_test = np.load(os.path.join(mnist_path, 'test_labels_full.npy'))

    x_train = x_train.reshape(len(x_train), 1, 28, 28) / 255
    x_test = x_test.reshape(len(x_test), 1, 28, 28) / 255

    # Y as one-hot
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return x_train, y_train, x_test, y_test

def load_mnist(data_path):
    mnist_path = os.path.join(data_path, 'mnist')

    x_train = np.load(os.path.join(mnist_path, 'mnist_train_x.npy'))
    y_train = np.load(os.path.join(mnist_path, 'mnist_train_y.npy'))
    x_test = np.load(os.path.join(mnist_path, 'mnist_test_x.npy'))
    y_test = np.load(os.path.join(mnist_path, 'mnist_test_y.npy'))

    x_train = x_train.reshape(len(x_train), 1, 28, 28)
    x_test = x_test.reshape(len(x_test), 1, 28, 28)

    # Y as one-hot
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return x_train, y_train, x_test, y_test