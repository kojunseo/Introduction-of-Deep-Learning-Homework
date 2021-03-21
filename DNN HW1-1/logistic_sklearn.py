from utils import _initialize, optimizer
import sklearn
from sklearn.linear_model import LogisticRegression

# 1. Choose DATA : Titanic / Digit
# 2. Adjust Hyperparameters
# ========================= EDIT HERE ========================
# DATA
DATA_NAME = 'Digit'

# HYPERPARAMETERS
num_epochs = 30
# ============================================================

assert DATA_NAME in ['Titanic', 'Digit','Basic_coordinates']

# Load dataset, model and evaluation metric
train_data, test_data, _, metric = _initialize(DATA_NAME)
train_x, train_y = train_data

num_data, num_features = train_x.shape
print('# of Training data : ', num_data)
ACC = 0.0
# ========================= EDIT HERE ========================
# Make model & optimizer
LR = LogisticRegression(max_iter=num_epochs)

# TRAIN
LR.fit(train_x, train_y)

# EVALUATION
test_x, test_y = test_data
pred = LR.predict(test_x)
ACC = metric(pred.reshape(-1, 1), test_y)
# ============================================================

print('ACC on Test Data : %.2f ' % ACC)
