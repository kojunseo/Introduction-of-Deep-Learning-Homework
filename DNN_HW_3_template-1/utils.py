import os
import numpy as np

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


def rel_error(x, y):
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def load_reg_data(path, filename, target_at_front, normalize=False, shuffle=False):
    fullpath = os.path.join(path, filename)

    with open(fullpath, 'r') as f:
        lines = f.readlines()
    lines = [s.strip().split(',') for s in lines]

    header = lines[0]
    data = lines[1:]

    data = np.array([[float(f) for f in d] for d in data], dtype=np.float32)
    if target_at_front:
        x, y = data[:, 1:], data[:, 0]
    else:
        x, y = data[:, :-1], data[:, -1]

    num_data = x.shape[0]
    if normalize:
        mins = np.expand_dims(np.min(x, axis=0), 0).repeat(num_data, 0)
        maxs = np.expand_dims(np.max(x, axis=0), 0).repeat(num_data, 0)
        x = (x - mins) / maxs

    # Add 1 column for bias
    bias = np.ones((x.shape[0], 1), dtype=np.float32)
    x = np.concatenate((bias, x), axis=1)

    if shuffle:
        perm = np.random.permutation(num_data)
        x = x[perm]
        y = y[perm]

    return x, y

def ConcreteData(path, filename):
    return load_reg_data(path, filename, target_at_front=False, normalize=True)

def GraduateData(path, filename):
    return load_reg_data(path, filename, target_at_front=False, normalize=True)

def HouseData(path, filename):
    return load_reg_data(path, filename, target_at_front=False, normalize=True)

def MSE(h, y):
    if len(h.shape) > 1:
        h = h.squeeze()
    se = np.square(h - y)
    mse = np.mean(se)
    return mse/2