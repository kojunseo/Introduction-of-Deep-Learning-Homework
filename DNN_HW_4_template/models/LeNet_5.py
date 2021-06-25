import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
from tqdm import tqdm

# W09 Convolutional Neural Networks (CNNs).pdf - 4 page
class LeNet_5(nn.Module):
    def __init__(self, input_channel, output_dim, learning_rate, reg_lambda, device):
        super(LeNet_5, self).__init__()

        self.output_dim = output_dim
        self.device = device

        # convolution layers
        self.CONV1 = nn.Conv2d(in_channels=input_channel, out_channels=6, kernel_size=(5, 5), stride=1)
        self.CONV2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=1)
        # pooling layers
        self.POOL1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.POOL2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        # Fully-connected layers
        self.FC1 = nn.Linear(400, 120)
        self.FC2 = nn.Linear(120, 84)
        self.FC3 = nn.Linear(84, output_dim)
    
        # For simplicity, we can use multiple modules as a single module
        self.Conv_layers = nn.Sequential(self.CONV1, nn.ReLU(), self.POOL1, self.CONV2, nn.ReLU(), self.POOL2)
        self.FC_layers = nn.Sequential(self.FC1, nn.ReLU(), self.FC2, nn.ReLU(), self.FC3)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=reg_lambda)

    def forward(self, x):
        out = torch.zeros((x.shape[0], self.output_dim))

        # Version 1 and Version 2 is same.
        # Version 1 ---------------------------------------------
        h = self.Conv_layers(x)
        stretched_h = h.reshape(x.shape[0], -1)
        out = self.FC_layers(stretched_h)
        # Version 1 ---------------------------------------------

        # Version 2 ---------------------------------------------
        # h = self.CONV1(x)
        # h = torch.relu(h)
        # h = self.POOL1(h)
        # h = self.CONV2(h)
        # h = torch.relu(h)
        # h = self.POOL2(h)
        # stretched_h = h.reshape(x.shape[0], -1)
        # out = self.FC1(stretched_h)
        # out = torch.relu(out)
        # out = self.FC2(out)
        # out = torch.relu(out)
        # out = self.FC3(out)
        # Version 2 ---------------------------------------------

        return out

    def predict(self, x):
        pred_y = np.zeros((x.shape[0], ))
        pred_y = []
        x_tenser = torch.tensor(x, dtype=torch.float, device = self.device)
        data_loader = DataLoader(x_tenser, batch_size=self.batch_size)
        with torch.no_grad():
            for batch_data in data_loader:
                batch_x = batch_data
                batch_x = resize(batch_x, (32, 32))
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
        x_tenser = torch.tensor(train_x, dtype=torch.float, device = self.device)
        y_tenser = torch.tensor(train_y, dtype=torch.long, device = self.device)
        dataset = TensorDataset(x_tenser, y_tenser)

        data_loader = DataLoader(dataset, batch_size=batch_size)
        self.batch_size = batch_size

        for epoch in range(1, num_epochs+1):
            start = time.time()
            epoch_loss = 0.0
            # model Train
            for b, batch_data in enumerate(data_loader):
                batch_x, batch_y = batch_data
                batch_x = resize(batch_x, (32, 32))
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
                    torch.save(self.state_dict(), './best_model/LeNet_5.pt')
                if epoch % print_every == 0:
                    print('Train Accuracy = %.3f' % train_acc + ' // ' + 'Valid Accuracy = %.3f' % valid_acc)
                    if best_acc < valid_acc:
                        print('Best Accuracy updated (%.4f => %.4f)' % (best_acc, valid_acc))
        print('Training Finished...!!')
        print('Best Valid acc : %.2f at epoch %d' % (best_acc, best_epoch))
        
        return best_acc

    def restore(self):
        with open(os.path.join('./best_model/LeNet_5.pt'), 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)

    def plot_accuracy(self):
        """
            Draw a plot of train/valid accuracy.
            X-axis : Epoch
            Y-axis : train_accuracy & valid_accuracy
            Draw train_acc-epoch, valid_acc-epoch graph in 'one' plot.
        """
        epochs = list(np.arange(1, self.num_epochs+1, self.test_every))

        plt.plot(epochs, self.train_accuracy, label='Train Acc.')
        plt.plot(epochs, self.valid_accuracy, label='Valid Acc.')

        plt.title('Epoch - Train/Valid Acc.')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()