import numpy as np
from sklearn import metrics
from utils import _initialize, optimizer

np.random.seed(428)

# ========================= EDIT HERE ========================
# 1. Choose DATA : Titanic / Digit
# 2. Adjust Hyperparameters
# 3. Choose Optimizer : SGD

# DATA
DATA_NAME = 'Titanic'

# HYPERPARAMETERS
batch_size = 10
num_epochs = 300
learning_rate = 0.0005

# ============================================================
epsilon = 0.01 # not for SGD
gamma = 0.9 # not for SGD

# OPTIMIZER
OPTIMIZER = 'SGD'

assert DATA_NAME in ['Titanic', 'Digit', 'Basic_coordinates']
assert OPTIMIZER in ['SGD']

# Load dataset, model and evaluation metric
train_data, test_data, logistic_regression, metric = _initialize(DATA_NAME)
train_x, train_y = train_data

num_data, num_features = train_x.shape
print('# of Training data : ', num_data)

# Make model & optimizer
model = logistic_regression(num_features)
optim = optimizer(OPTIMIZER, gamma=gamma, epsilon=epsilon)

# TRAIN
loss = model.train(train_x, train_y, num_epochs, batch_size, learning_rate, optim)
print('Training Loss at the last epoch: %.2f' % loss)

# EVALUATION
test_x, test_y = test_data
pred = model.forward(test_x)
ACC = metric(pred, test_y)

print(OPTIMIZER, ' ACC on Test Data : %.3f' % ACC)

# PLOTING
preds_history = model.history["predicts"]
loss_history = model.history["loss"]
epoch_history = model.history["epoch"]
acc_history = []
for pred_h in preds_history:
    acc_history.append(metric(pred_h, train_y))

import matplotlib.pyplot as plt
plt.plot(acc_history)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig("./ModelAccuracy.png",)
plt.close()

plt.plot(loss_history)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig("./ModelLoss.png")