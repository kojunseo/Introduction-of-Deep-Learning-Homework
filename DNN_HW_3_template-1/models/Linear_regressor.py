import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import TensorDataset, DataLoader


class Linear_regressor(nn.Module):
    def __init__(self, num_features, learning_rate=1e-3):
        super(Linear_regressor, self).__init__()
        '''
        Define model parameters, loss, and optimizer in here.
        '''
        self.linear = nn.Linear(in_features=num_features, out_features=1, bias=True)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

    def forward(self, x):
        '''
        Define model in here.
        '''
        y_pred = self.linear(x)
        return y_pred

    def predict(self, x):
        '''
        Define model predict function.
        Since there is no backpropagation going on, there is no need to calculate the gradient.
        So, you have to use "torch.no_grad()"
        By using "DataLoader", you can predict with mini-batch.
        '''

        # transfrom numpy data to torch data and make torch dataset
        x_tenser = torch.tensor(x)
        data_loader = DataLoader(x_tenser, batch_size=self.batch_size)

        # predict y with mini-batch
        pred_y = []
        with torch.no_grad():
            for batch_data in data_loader:
                batch_x = batch_data
                batch_pred_y = self.forward(batch_x)
                pred_y.append(batch_pred_y.numpy())
        pred_y = np.concatenate(pred_y, axis=0)

        return pred_y

    def train(self, x, y, num_epochs, batch_size, print_every=100):
        '''
        Calculate loss and update model using optimizer.
        You can easily use mini-batch using "TensorDataset" and "DataLoader".
        '''
        self.batch_size = batch_size

        # transfrom numpy data to torch data and make torch dataset
        x_tenser = torch.tensor(x)
        y_tenser = torch.tensor(y)
        dataset = TensorDataset(x_tenser, y_tenser)
        data_loader = DataLoader(dataset, batch_size=batch_size)

        # train the model with mini-batch
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_data in data_loader:
                batch_x, batch_y = batch_data
                pred_y = self.forward(batch_x)

                # calcuate the loss
                loss = self.loss_function(pred_y.reshape(-1), batch_y)

                # model update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss
            epoch_loss /= len(data_loader)

            if epoch % print_every == 0:
                print(f'epoch={epoch}, loss={epoch_loss}')

        return epoch_loss
