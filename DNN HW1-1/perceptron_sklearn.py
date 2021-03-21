from utils import _initialize, optimizer
import sklearn
from sklearn.linear_model import Perceptron

# 1. Choose DATA : Titanic / Digit
# 2. Adjust Hyperparameters

# DATA
DATA_NAME = 'Basic_coordinates'

# HYPERPARAMETERS
num_epochs = 300

assert DATA_NAME in ['Titanic', 'Digit','Basic_coordinates']

# Load dataset, model and evaluation metric
train_data, test_data, _, metric = _initialize(DATA_NAME)
train_x, train_y = train_data
test_x, test_y = test_data

num_data, num_features = train_x.shape
print('# of Training data : ', num_data)
ACC= 0.0
# ========================= Example ========================
# Make model & optimizer
model = Perceptron(max_iter=num_epochs, random_state=0)

# TRAIN
model.fit(train_x, train_y)

# EVALUATION
pred = model.predict(test_x)
pred = pred.reshape(-1,1)
ACC = metric(pred, test_y)
# ============================================================
print('ACC on Test Data : %.2f ' % ACC)
