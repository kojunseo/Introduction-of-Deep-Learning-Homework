import numpy as np

class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.W = np.random.rand(self.num_features, 1)

    def train(self, x, y, epochs, batch_size, lr, optim):
        loss = None   # loss of final epoch
        # ========================= Example ==========================

        # Train should be done for 'epochs' times.
        # The function 'train' should return the loss of final epoch.
        # Weights are updated through the optimizer, not directly within 'train' function.
        y = y.reshape(x.shape[0], 1)
        w = self.W
        print("x.shape {}, y.shape{}".format(x.shape,y.shape))
        print(int(x.shape[0]/batch_size))

        for i in range(epochs):
            loss = 0
            wd = np.zeros_like(self.W)
            for j in range(x.shape[0]):
                y_predicted = self.forward(x[j])

                # Perceptron updates only with samples whose predictions are wrong.
                if y_predicted != y[j]:
                    loss += - y_predicted * y[j]
                    # accumulate gradients
                    wd += -(x[j] * y[j]).reshape(self.num_features, 1)

            # update Perceptron's weights with accumulated gradients
            w = optim.update(w, wd, lr)

        self.W = w
        print ("cost {}, batch_size {}, epoch {}".format(loss,batch_size,epochs))
        # ============================================================
        return loss

    def forward(self, x):
        y_predicted = None
        # ========================= Example ========================
        # Evaluation Function
        # Given the input 'x', the function should return prediction for 'x'
        # The model predicts the label as 1 if the output is positive or equal to 0.
        # Otherwise, it predicts as -1
        y_predicted = np.dot(x, self.W)
        y_predicted = Perceptron._sign(self, y_predicted)
        # ==========================================================

        return y_predicted

    def _sign(self, x):
        # Sign Function
        # The function returns the sign of 'x'
        # ========================= Example ========================
        x[x>=0] = 1
        x[x<0] = -1
        sign = x
        # ==========================================================
        return sign
