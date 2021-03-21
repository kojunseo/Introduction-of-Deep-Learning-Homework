import numpy as np
import sklearn

class LogisticRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.random.rand(self.num_features, 1)

    def train(self, x, y, epochs, batch_size, lr, optim):
        loss = None   # loss of final epoch

        # Train should be done for 'epochs' times with minibatch size of 'batch size'
        # The function 'train' should return the loss of final epoch
        # Loss of an epoch is calculated as an average of minibatch losses
        # Weights are updated through the optimizer, not directly within 'train' function.

        # Tip : log computation may cause some error, so try to solve it by adding an epsilon(small value) within log term.
        epsilon = 0.01
        # ========================= EDIT HERE ========================
        self.history = {
            "loss": [], 
            "predicts": [],
            "epoch": []
        }
        for epoch in range(epochs):
            loss_mean = []
            if len(y)%batch_size:
                    to_go = len(y)//batch_size+1
            else:
                to_go = len(y)//batch_size
                
            for idx in range(0, to_go):
                # 배치에 해당하는 x, y값을 this_x, this_y로 처리
                this_y = y[idx*batch_size:(idx+1)*batch_size] # 이번 배치의 y값 리스트
                this_x = x[idx*batch_size:(idx+1)*batch_size] # 이번 배치의 x값 리스트
                
                # forward를 쓰면 0, 1로 이진값이 나오기 때문에, sigmoid만 처리한 값을 얻기 위해 따로 구성
                y_hat = self._sigmoid(np.dot(this_x, self.W)).squeeze()
                
                # Caculate Cost Function and get Loss
                cost = np.where(this_y==1, -np.log(y_hat+epsilon), -np.log(1-y_hat+epsilon))
                L = np.mean(cost)

                # 에러를 가지고 gradient계산
                y_err = this_y- y_hat
                mini_batch_grad = -np.dot(this_x.T, y_err)/len(this_y) # 이번 배치의 gradient평균 연산
                self.W = optim.update(self.W, mini_batch_grad.reshape(-1,1), lr) # 배치 전체의 gradient평균을 연산하여 그것을 기준으로 업데이트
                
                # 에포크 로스를 구하기 위해 미니배치의 로스를 모두 추가해놓음
                loss_mean.append(L)

            loss = np.mean(loss_mean)
            self.history["loss"].append(loss)
            self.history["predicts"].append(self.forward(x))
            self.history["epoch"].append(epoch)
            if not epoch % 30 :
                print(f"Epoch: {epoch} || Loss: {loss}")
            loss_mean = [] # 에포크 loss를 구하기 위한 리스트 초기화
        # ============================================================
        return loss

    def forward(self, x):
        threshold = 0.5
        y_predicted = None

        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the probability is greater or equal to 'threshold'
        # Otherwise, it predicts as 0

        # ========================= EDIT HERE ========================
        y_predicted = np.dot(x, self.W)
        y_predicted = self._sigmoid(y_predicted)
        y_predicted = np.where(y_predicted > threshold, 1, 0)
        # ============================================================

        return y_predicted

    def _sigmoid(self, x):
        sigmoid = None

        # Sigmoid Function
        # The function returns the sigmoid of 'x'

        # ========================= EDIT HERE ========================
        sigmoid = 1 / (1 +np.exp(-x))
        # ============================================================
        return sigmoid