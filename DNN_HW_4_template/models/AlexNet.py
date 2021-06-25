
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
from tqdm import tqdm

# W09 Convolutional Neural Networks (CNNs).pdf - 10 page
class AlexNet(nn.Module):
    def __init__(self, input_channel, output_dim, learning_rate, reg_lambda, device):
        super(AlexNet, self).__init__()

        self.output_dim = output_dim
        self.device = device
        self.loss_function = None
        self.optimizer = None
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
        x_tenser = torch.tensor(x, dtype=torch.float, device = self.device)
        data_loader = DataLoader(x_tenser, batch_size=self.batch_size)
        with torch.no_grad():
            for batch_data in data_loader:
                batch_x = batch_data
                batch_x = resize(batch_x, (227, 227))
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
                batch_x = resize(batch_x, (227, 227))
                pred_y = self.forward(batch_x)

                if self.loss_function is not None:
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
                    torch.save(self.state_dict(), './best_model/AlexNet.pt')
                if epoch % print_every == 0:
                    print('Train Accuracy = %.3f' % train_acc + ' // ' + 'Valid Accuracy = %.3f' % valid_acc)
                    if best_acc < valid_acc:
                        print('Best Accuracy updated (%.4f => %.4f)' % (best_acc, valid_acc))
        print('Training Finished...!!')
        print('Best Valid acc : %.2f at epoch %d' % (best_acc, best_epoch))
        
        return best_acc

    def restore(self):
        with open(os.path.join('./best_model/AlexNet.pt'), 'rb') as f:
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