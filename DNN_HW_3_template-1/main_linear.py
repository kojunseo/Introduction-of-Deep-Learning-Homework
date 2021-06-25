import numpy as np
from utils import ConcreteData, MSE
from models.Linear_regressor import Linear_regressor

np.random.seed(428)

# DATA
DATA_NAME = 'Concrete'

# HYPERPARAMETERS
batch_size = 10
num_epochs = 1000
learning_rate = 0.001

# Load dataset, model and evaluation metric
train_x, train_y = ConcreteData('./data/concrete', 'train.csv')

num_data, num_features = train_x.shape
print('# of Training data : ', num_data)

# =============================== PyTorch ===============================
# Make model & optimizer
model = Linear_regressor(num_features, learning_rate)

# TRAIN
loss = model.train(train_x, train_y, num_epochs, batch_size)
print('Training Loss at the last epoch: %.2f' % loss)

# EVALUATION
test_x, test_y = ConcreteData('./data/concrete', 'test.csv')
pred = model.predict(test_x)
mse = MSE(pred, test_y)

print(' MSE on Test Data (Torch Linear Regression) : %.2f' % mse)
# =========================================================================

# =============================== Sklearn ===============================
from sklearn.linear_model import LinearRegression
# Make model & optimizer
model = LinearRegression()

# TRAIN
model.fit(train_x, train_y)

# EVALUATION
pred = model.predict(test_x)
ACC = MSE(pred, test_y)

print('MSE on Test Data (Sklearn) : %.2f ' % ACC)
# =========================================================================
