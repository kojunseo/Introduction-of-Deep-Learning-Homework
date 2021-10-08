from collections import OrderedDict
import numpy as np

def softmax(z):
    # We provide numerically stable softmax.
    z = z - np.max(z, axis=1, keepdims=True)
    _exp = np.exp(z)

    _sum = np.sum(_exp, axis=1, keepdims=True)
    sm = _exp / _sum

    return sm


class ReLU:
    """
    ReLU Function. ReLU(x) = max(0, x)
    Implement forward & backward path of ReLU.

    ReLU(x) = x if x > 0.
              0 otherwise.
    Be careful. It's '>', not '>='.
    """

    def __init__(self):
        # 1 (True) if ReLU input <= 0
        self.zero_mask = None
        self.out = None

    def forward(self, z):
        """
        ReLU Forward.
        ReLU(x) = max(0, x)

        z --> (ReLU) --> out

        [Inputs]
            z : ReLU input in any shape.

        [Outputs]
            self.out : Values applied elementwise ReLU function on input 'z'.
        """
        self.out = None
        # =============================== EDIT HERE ===============================
        self.out = np.where(z > 0, z, 0)
        
        # =========================================================================
        return self.out

    def backward(self, d_prev, reg_lambda):
        """
        ReLU Backward.

        z --> (ReLU) --> out
        dz <-- (dReLU) <-- d_prev(dL/dout)

        [Inputs]
            d_prev : Gradients flow from upper layer.
                - d_prev = dL/dk, where k = ReLU(z).
            reg_lambda: L2 regularization weight. (Not used in activation function)
        [Outputs]
            dz : Gradients w.r.t. ReLU input z.
        """
        dz = None
        # =============================== EDIT HERE ===============================
        dz = np.where(self.out>0, 1, 0)*d_prev + (reg_lambda/2)*np.sum(d_prev**2)
        
        # =========================================================================
        return dz

    def update(self, learning_rate):
        # NOT USED IN ReLU
        pass

    def summary(self):
        return 'ReLU Activation'


class Sigmoid:
    """
    Sigmoid Function.
    Implement forward & backward path of Sigmoid.
    """

    def __init__(self):
        self.out = None

    def forward(self, z):
        """
        Sigmoid Forward.

        z --> (Sigmoid) --> self.out

        [Inputs]
            z : Sigmoid input in any shape.

        [Outputs]
            self.out : Values applied elementwise sigmoid function on input 'z'.
        """
        self.out = None
        # =============== EDIT HERE ===============
        self.out = 1 / (1 +np.exp(-z))

        # =========================================
        return self.out

    def backward(self, d_prev, reg_lambda):
        """
        Sigmoid Backward.

        z --> (Sigmoid) --> self.out
        dz <-- (dSigmoid) <-- d_prev(dL/d self.out)

        [Inputs]
            d_prev : Gradients flow from upper layer.
            reg_lambda: L2 regularization weight. (Not used in activation function)

        [Outputs]
            dz : Gradients w.r.t. Sigmoid input z .
        """
        dz = None
        # =============== EDIT HERE ===============

        dz = self.out*(1-self.out) * d_prev + (reg_lambda/2)*np.sum(d_prev**2)
        # =========================================
        return dz

    def update(self, learning_rate):
        # NOT USED IN Sigmoid
        pass

    def summary(self):
        return 'Sigmoid Activation'


class Tanh:
    """
    Hyperbolic Tangent Function(Tanh).
    Implement forward & backward path of Tanh.
    """

    def __init__(self):
        self.out = None

    def forward(self, z):
        """
        Hyperbolic Tangent Forward.

        z --> (Tanh) --> self.out

        [Inputs]
            z : Tanh input in any shape.

        [Outputs]
            self.out : Values applied elementwise tanh function on input 'z'.

        =====CAUTION!=====
        You are not allowed to use np.tanh function!
        """
        self.out = None
        # =============== EDIT HERE ===============
        self.out = 2/(1+np.exp(-2*z)) -1
        # =========================================
        return self.out

    def backward(self, d_prev, reg_lambda):
        """
        Hyperbolic Tangent Backward.

        z --> (Tanh) --> self.out
        dz <-- (dTanh) <-- d_prev(dL/d self.out)

        [Inputs]
            d_prev : Gradients flow from upper layer.
            reg_lambda: L2 regularization weight. (Not used in activation function)

        [Outputs]
            dz : Gradients w.r.t. Tanh input z .
            In other words, the derivative of tanh should be reflected on d_prev.
        """
        dz = None
        # =============== EDIT HERE ===============
        dz = (1-self.out**2) * d_prev + (reg_lambda/2)*np.sum(d_prev**2)

        # =========================================
        return dz

    def update(self, learning_rate):
        # NOT USED IN Tanh
        pass

    def summary(self):
        return 'Tanh Activation'


"""
    ** Fully-Connected Layer **
    Single Fully-Connected Layer.

    Given input features,
    FC layer linearly transforms the input with weights (self.W) & bias (self.b).

    You need to implement forward and backward pass.
"""

class FCLayer:
    def __init__(self, input_dim, output_dim):
        # Weight Initialization
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim / 2)
        self.b = np.zeros(output_dim)

    def forward(self, x):
        """
        FC Layer Forward.
        Use variables : self.x, self.W, self.b

        [Input]
        x: Input features.
        - Shape : (batch size, In Channel, Height, Width)
        or
        - Shape : (batch size, input_dim)

        [Output]
        self.out : fc result
        - Shape : (batch size, output_dim)

        Tip : you do 'not' need to implement L2 regularization here. It is already implemented in ClassifierModel.forward()
        """
        # Flatten input if needed.
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)

        self.x = x
        self.out = None
        # =============================== EDIT HERE ===============================
        self.out = np.dot(x, self.W)+self.b
        # =========================================================================
        return self.out

    def backward(self, d_prev, reg_lambda):
        """
        FC Layer Backward.
        Use variables : self.x, self.W

        [Input]
        d_prev: Gradients value so far in back-propagation process.
        reg_lambda: L2 regularization weight. (Not used in activation function)

        [Output]
        dx : Gradients w.r.t input x
        - Shape : (batch_size, input_dim) - same shape as input x

        Tip : you should implement backpropagation of L2 regularization here.
        """
        dx = None           # Gradient w.r.t. input x
        self.dW = None      # Gradient w.r.t. weight (self.W)
        self.db = None      # Gradient w.r.t. bias (self.b)
        # =============================== EDIT HERE ===============================
        dx =  np.dot(d_prev, self.W.T)
        self.dW = np.dot(self.x.T, d_prev) + reg_lambda * self.W
        self.db = np.sum(d_prev, axis =0)
        # =========================================================================
        return dx

    def update(self, learning_rate):
        self.W -= self.dW * learning_rate
        self.b -= self.db * learning_rate

    def summary(self):
        return 'Input -> Hidden : %d -> %d ' % (self.W.shape[0], self.W.shape[1])

"""
    ** Softmax Layer **
    Softmax Layer applies softmax (WITHOUT any weights or bias)

    Given an score,
    'SoftmaxLayer' applies softmax to make probability distribution. (Not log softmax or else...)

    You need to implement forward and backward pass.
    (This is NOT an entire model.)
"""

class SoftmaxLayer:
    def __init__(self):
        # No parameters
        pass

    def forward(self, x):
        """
        Softmax Layer Forward.
        Apply softmax (not log softmax or others...) on axis-1

        Use 'softmax' function above in this file.
        We recommend you see the function.

        [Input]
        x: Score to apply softmax
        - Shape: (batch_size, # of class)

        [Output]
        y_hat: Softmax probability distribution.
        - Shape: (batch_size, # of class)
        """
        self.y_hat = None
        # =============================== EDIT HERE ===============================
        exp_a = np.exp(x)
        sum_exp_a = np.sum(exp_a, axis = -1)
        self.y_hat = exp_a / sum_exp_a[:, np.newaxis]
        # =========================================================================
        return self.y_hat

    def backward(self, d_prev=1, reg_lambda=0):
        """
        Softmax Layer Backward.
        Gradients w.r.t input score.

        That is,
        Forward  : softmax prob = softmax(score)
        Backward : dL / dscore => 'dx'

        Compute dx (dL / dscore).
        Check loss function in HW3 word file.

        [Input]
        d_prev : Gradients flow from upper layer.

        [Output]
        dx: Gradients of softmax layer input 'x'
        """
        batch_size = self.y.shape[0]
        dx = None
        # =============================== EDIT HERE ===============================
        dx =  (self.y_hat - self.y)/batch_size
        # =========================================================================
        return dx


    def ce_loss(self, y_hat, y):
        """
        Compute Cross-entropy Loss.
        Use epsilon (eps) for numerical stability in log.

        Check loss function in HW2 word file.

        [Input]
        y_hat: Probability after softmax.
        - Shape : (batch_size, # of class)

        y: One-hot true label
        - Shape : (batch_size, # of class)

        [Output]
        self.loss : cross-entropy loss
        - float
        """
        self.loss = None
        eps = 1e-10
        self.y_hat = y_hat
        self.y = y
        # =============================== EDIT HERE ===============================
        self.loss = -np.sum(self.y * np.log(self.y_hat+eps)) / self.y.shape[0]
        # =========================================================================
        return self.loss

    def update(self, learning_rate):
        # Not used in softmax layer.
        pass

    def summary(self):
        return 'Softmax layer'


"""
    ** Classifier Model **
    This is an class for entire Classifier Model.
    All the functions and variables are already implemented.
    Look at the codes below and see how the codes work.

    <<< DO NOT CHANGE ANYTHING HERE >>>
"""

class ClassifierModel:
    def __init__(self,):
        self.layers = OrderedDict()
        self.softmax_layer = None
        self.loss = None
        self.pred = None

    def predict(self, x):
        # Outputs model softmax score
        for name, layer in self.layers.items():
            x = layer.forward(x)
        x = self.softmax_layer.forward(x)
        return x

    def forward(self, x, y, reg_lambda):
        # Predicts and Compute CE Loss
        reg_loss = 0
        self.pred = self.predict(x)
        ce_loss = self.softmax_layer.ce_loss(self.pred, y)

        for name, layer in self.layers.items():
            if isinstance(layer, FCLayer):
                norm = np.linalg.norm(layer.W, 2)
                reg_loss += 0.5 * reg_lambda *  norm * norm

        self.loss = ce_loss + reg_loss

        return self.loss

    def backward(self, reg_lambda):
        # Back-propagation
        d_prev = 1
        d_prev = self.softmax_layer.backward(d_prev, reg_lambda)
        for name, layer in list(self.layers.items())[::-1]:
            d_prev = layer.backward(d_prev, reg_lambda)

    def update(self, learning_rate):
        # Update weights in every layer with dW, db
        for name, layer in self.layers.items():
            layer.update(learning_rate)

    def add_layer(self, name, layer):
        # Add Neural Net layer with name.
        if isinstance(layer, SoftmaxLayer):
            if self.softmax_layer is None:
                self.softmax_layer = layer
            else:
                raise ValueError('Softmax Layer already exists!')
        else:
            self.layers[name] = layer

    def summary(self):
        # Print model architecture.
        print('======= Model Summary =======')
        for name, layer in self.layers.items():
            print('[%s] ' % name + layer.summary())
        print('[Softmax Layer] ' + self.softmax_layer.summary())
        print()
