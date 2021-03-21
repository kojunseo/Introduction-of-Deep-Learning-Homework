from models.LogisticRegression import LogisticRegression
from utils import _initialize, optimizer
import sklearn
from sklearn.linear_model import LinearRegression

# 1. Choose DATA : Titanic / Digit
# ========================= EDIT HERE ========================
# DATA
DATA_NAME = 'Graduate'
# ============================================================

assert DATA_NAME in ['Concrete', 'Graduate']

# Load dataset, model and evaluation metric
train_data, test_data, _, metric = _initialize(DATA_NAME)
train_x, train_y = train_data

num_data, num_features = train_x.shape
print('# of Training data : ', num_data)
ACC = 0.0
# ========================= EDIT HERE ========================
# Make model & optimizer
LR = LinearRegression()
# TRAIN
LR.fit(train_x, train_y)
# EVALUATION
ACC = metric(test_data[1], LR.predict(test_data[0]))
# ============================================================

print('MSE on Test Data : %.2f ' % ACC)
