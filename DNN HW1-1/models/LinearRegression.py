import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.zeros((self.num_features, 1))

    def train(self, x, y, epochs, batch_size, lr, optim):
        final_loss = None   # loss of final epoch
        # Train should be done for 'epochs' times with minibatch size of 'batch_size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.
        # ========================= EDIT HERE ========================   
        for epoch in range(epochs):
            loss_mean = []
            if len(y)%batch_size:
                to_go = len(y)//batch_size+1
            else:
                to_go = len(y)//batch_size

            for idx in range(0, to_go):
                # 이번 배치에 해당하는 x, y를 this_x, this_y에 저장
                this_y = y[idx*batch_size:(idx+1)*batch_size] # 이번 배치의 y값 리스트
                this_x = x[idx*batch_size:(idx+1)*batch_size] # 이번 배치의 x값 리스트

                # forward후 gradient를 구하는 코드 작성
                y_hat = self.forward(this_x)
                y_err = this_y - y_hat.squeeze()
                grad = np.multiply(-this_x.T, y_err).T

                # 배치평균 연산 후 업데이트 
                mini_batch_mean = np.mean(grad, axis = 0)
                self.W = optim.update(self.W, mini_batch_mean.reshape(self.num_features,1), lr)

                # 로스 function을 계산하여 에포크 로스를 구하기 위해서 loss_mean에 더해줌
                loss_mean.append(np.mean(np.square(y_err), axis = 0)*1/2)

            final_loss = np.mean(loss_mean)
            loss_mean = []
            if not epoch % 250:  # 250 Epochs마다 Epoch와 loss를 찍어서 보여줌
                print(f"Epoch: {epoch} || Loss: {final_loss}")
        # ============================================================
        return final_loss

    def forward(self, x):
        y_predicted = None
        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'

        # ========================= EDIT HERE ========================
        y_predicted = np.dot(x, self.W)
        # ============================================================
        return y_predicted