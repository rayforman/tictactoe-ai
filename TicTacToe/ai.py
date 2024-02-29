# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold

# Data global variables
multi_X_train = None
multi_X_test = None
multi_y_train = None
multi_y_test = None

# Get data and make 
def load_data(shuffle=False):
    # Load in intermediate multi-label board dataset for training
    multi_data = np.loadtxt("./datasets/tictac_multi.txt")
    # Shuffle data
    if (shuffle):
        np.random.shuffle(multi_data)
    # Split intermediate multi-label data into features and lables
    multi_features = multi_data[:,:-9]
    multi_labels = multi_data[:,-9:]
    # Split intermediate multi-label data into traning and testing data
    multi_X_train, multi_X_test, multi_y_train, multi_y_test = \
    train_test_split( multi_features, multi_labels, test_size=0.2, random_state=42)
    split_data = (multi_X_train, multi_X_test, multi_y_train, multi_y_test)
    return split_data   # tuple containing splitting


# TRAIN OPTIMAL MODEL 
# - An MLP with 3 hidden layers of 400 neurons each, using a sigmoid (logistic) activation function
def train_model():
    # Load in the data
    split_data = load_data()
    multi_X_train = split_data[0]
    multi_X_test  = split_data[1]
    multi_y_train = split_data[2]
    multi_y_test  = split_data[3]

    # Optimal Model Training which we perform on the 3-layer 400 dim network
    mlp_clf_optimal = MLPClassifier(hidden_layer_sizes = (400,400,400), activation='logistic', random_state = 0)
    mlp_clf_optimal.fit(multi_X_train, multi_y_train)
    mlp_pred_multi_y = mlp_clf_optimal.predict(multi_X_test)
    # Print Optimal Results
    print("MLP (3 hidden layers of size 400) Accuracy : ", metrics.accuracy_score(multi_y_test, mlp_pred_multi_y))
    return mlp_clf_optimal

