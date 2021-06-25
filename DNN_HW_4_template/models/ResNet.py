
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
from tqdm import tqdm

# W10 Modern ConvNets.pdf - 23 page
# https://pytorch.org/assets/images/resnet.png

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        if in_channels == out_channels:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, input_channel, output_dim, learning_rate, reg_lambda, device):
        super(ResNet, self).__init__()

        self.output_dim = output_dim
        self.device = device
        self.loss_function = None
        self.optimizer = None

        self.CONV1 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=(7, 7), stride=2, padding=3)
        self.POOL1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        
        # You can implement ResNet-18 more simply using BasicBlock Module. 
        # =============================== EDIT HERE ===============================
       
        # =============================== EDIT HERE ===============================

    def forward(self, x):
        out = torch.zeros((x.shape[0], self.output_dim))

        # =============================== EDIT HERE ===============================
      
        # =============================== EDIT HERE ===============================

        return out

    def predict(self, x):
        pred_y = np.zeros((x.shape[0], ))
        pred_y = []
        x_tenser = torch.tensor(x, dtype=torch.float, device=self.device)
        data_loader = DataLoader(x_tenser, batch_size=self.batch_size)
        with torch.no_grad():
            for batch_data in data_loader:
                batch_x = batch_data
                batch_x = resize(batch_x, (224, 224))
                batch_pred = self.forward(batch_x).argmax(axis=1)
                pred_y.append(batch_pred.cpu().numpy())
        pred_y = np.concatenate(pred_y, axis=0)
        return pred_y

    def train(self, train_x, train_y, valid_x, valid_y, num_epochs, batch_size, test_every=10, print_every=10):
        self.train_accuracy = []
        self.valid_accuracy = []
        best_epoch = -1
        best_acc = -1
        self.num_epochs = num_epochs
        self.test_every = test_every

        # transfrom numpy data to torch data and make torch dataset
        x_tenser = torch.tensor(train_x, dtype=torch.float, device=self.device)
        y_tenser = torch.tensor(train_y, dtype=torch.long, device=self.device)
        dataset = TensorDataset(x_tenser, y_tenser)

        data_loader = DataLoader(dataset, batch_size=batch_size)
        self.batch_size = batch_size

        for epoch in range(1, num_epochs+1):
            start = time.time()
            epoch_loss = 0.0
            # model Train
            for b, batch_data in enumerate(data_loader):
                batch_x, batch_y = batch_data
                batch_x = resize(batch_x, (224, 224))
                pred_y = self.forward(batch_x)

                loss = self.loss_function(pred_y, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss

            epoch_loss /= len(data_loader)
            end = time.time()
            lapsed_time = end - start

            if epoch % print_every == 0:
                print(f'Epoch {epoch} took {lapsed_time} seconds\n')
                print('[EPOCH %d] Loss = %.5f' % (epoch, epoch_loss))

            if epoch % test_every == 0:
                # TRAIN ACCURACY
                pred = self.predict(train_x)
                correct = len(np.where(pred == train_y)[0])
                total = len(train_y)
                train_acc = correct / total
                self.train_accuracy.append(train_acc)

                # VAL ACCURACY
                pred = self.predict(valid_x)
                correct = len(np.where(pred == valid_y)[0])
                total = len(valid_y)
                valid_acc = correct / total
                self.valid_accuracy.append(valid_acc)

                if best_acc < valid_acc:
                    best_acc = valid_acc
                    best_epoch = epoch
                    torch.save(self.state_dict(), './best_model/ResNet.pt')
                if epoch % print_every == 0:
                    print('Train Accuracy = %.3f' % train_acc + ' // ' + 'Valid Accuracy = %.3f' % valid_acc)
                    if best_acc < valid_acc:
                        print('Best Accuracy updated (%.4f => %.4f)' % (best_acc, valid_acc))
        print('Training Finished...!!')
        print('Best Valid acc : %.2f at epoch %d' % (best_acc, best_epoch))
        
        return best_acc

    def restore(self):
        with open(os.path.join('./best_model/ResNet.pt'), 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)

    def plot_accuracy(self):
        """
            Draw a plot of train/valid accuracy.
            X-axis : Epoch
            Y-axis : train_accuracy & valid_accuracy
            Draw train_acc-epoch, valid_acc-epoch graph in 'one' plot.
        """
        epochs = list(np.arange(1, self.num_epochs+1, self.print_every))

        plt.plot(epochs, self.train_accuracy, label='Train Acc.')
        plt.plot(epochs, self.valid_accuracy, label='Valid Acc.')

        plt.title('Epoch - Train/Valid Acc.')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()
