import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize


def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_features + 1, 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    X = np.c_[np.ones(n_data), train_data]
    
    # initialWeight was flattened when using minimize. Thus we need to reshape
    # the initialWeight to a 2D (D+1, 1) array.
    w = initialWeights.reshape(n_features + 1, 1)
    
    # Matrix multiplication for arrays should be np.dot
    h = sigmoid(np.dot(X, w))
    
    # using 1D-array to compute error then the error could be int
    error = -(1/n_data) * (np.dot(labeli.flatten(), np.log(h).flatten())\
              + np.dot((1.0 - labeli).flatten(), np.log(1.0 - h).flatten()))
    
    # The partial derative of the cost function respect to  Wk in matrix-wise form 
    error_grad = (1/n_data) * np.dot(X.T, h - labeli)
    
    # in order to use minimize, the error_grad should be flattened 1D array
    error_grad = error_grad.flatten()
    
    print (error)
    return error, error_grad


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    
    # getting the number of examples, the number of row
    m = data.shape[0]
    
    # After the optimization of Wk, W the weight is assembled column by column
    # Thus each column W[:, K] of W is the weight respect to the class k (Ck)
    # The binary logistic regression compute the probability of each test example
    # respect to each class Wk. Thus the row of h is the probabities of one example 
    # respect to each class. Thus h is a N x K matrix. The maximum of each row
    # is the most possible class respect to its column index.
    h = sigmoid(np.dot(np.c_[np.ones(m), data], W))
    col_index = np.argmax(h, axis = 1)
    
    # The col_index is a 1-D array. In order to compare with the test label
    # we need to reshaple it respect to the shape of the test label.
    label = col_index.reshape(data.shape[0], 1)
    
    return label

def softmax(z):
    # soft maximization function without considering outflow
    # in this case the col# represent the Class#, thus we need to sum the columns
    # into one columns. The np.sum(a, axis=1) gets a 1-D array, thus we need 
    # to reshape it to fit the dimension of the column of the input.
    return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0], 1)
    

def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    train_data, Y = args
    
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    X = np.c_[np.ones(n_data), train_data]
    
    # For multi-class logistic regression, the weights for each class are generated
    # simultaneously. Then using the softmax to compute the probabilities of each
    # training example respect to each class. Obviously, this method is much faster
    # than one vs all, because we need not to generate ten classifiers and do the 
    # optimization ten times.
    w = params.reshape(n_feature + 1, n_class)
    h = softmax(np.dot(X, w))  # N x K
    error = -np.sum(Y * np.log(h)) # arrays multiplied element by element
    error_grad = np.dot(X.T, h - Y)
    error_grad = np.array(error_grad).flatten()
    
    print(error)
    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    m = data.shape[0]
    h = sigmoid(np.dot(np.c_[np.ones(m), data], W))
    col_index = np.argmax(h, axis = 1)
    label = col_index.reshape(data.shape[0], 1)
    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()


# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')



"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

from sklearn.svm import SVC
import time
from datetime import timedelta

start_time1 = time.monotonic()
clf = SVC(kernel='linear')
clf.fit(train_data, train_label)
print('\n\n using linear kernel and all other parameters as default\n\n')
print('\n Training set Accuracy: ' + str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' + str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' + str(clf.score(test_data, test_label)*100) + '%')
end_time1 = time.monotonic()
print('Linear Kernel:' + str(timedelta(seconds=end_time1 - start_time1)))


start_time2 = time.monotonic()
clf = SVC(kernel='rbf', gamma=1.0)
clf.fit(train_data, train_label)
print('\n\n using rbf kernel with value of gamma setting to 1 and all other parameters kept default\n\n')
print('\n Training set Accuracy: ' + str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' + str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' + str(clf.score(test_data, test_label)*100) + '%')
end_time2 = time.monotonic()
print('rbf, gamma=1:' + str(timedelta(seconds=end_time2 - start_time2)))


start_time3 = time.monotonic()
clf = SVC(kernel='rbf')
clf.fit(train_data, train_label)
print('\n\n using rbf kernel and all other parameters as default\n\n')
print('\n Training set Accuracy: ' + str(clf.score(train_data, train_label)*100) + '%')
print('\n Validation set Accuracy: ' + str(clf.score(validation_data, validation_label)*100) + '%')
print('\n Testing set Accuracy: ' + str(clf.score(test_data, test_label)*100) + '%')
end_time3 = time.monotonic()
print('rbf, all defaults:' + str(timedelta(seconds=end_time3 - start_time3)))


start_time4 = time.monotonic()
C_max = 100
step = 10
length = 1 + C_max / step
trainAcc_history = np.zeros((1, length))
validationAcc_history = np.zeros((1, length))
testAcc_history = np.zeros((1, length))
m = 0

for i in range(1, 111, 10):
    clf = SVC(kernel='rbf', C=i)
    clf.fit(train_data, train_label)
    trainAcc_history[:, m] = clf.score(train_data, train_label)
    validationAcc_history[:, m] = clf.score(validation_data, validation_label)
    testAcc_history[:, m] = clf.score(test_data, test_label)
    m += 1
     
end_time4 = time.monotonic()
print('rbf, different C:' + str(timedelta(seconds=end_time4 - start_time4)))

print('\n\n End of project.')




"""
Script for Extra Credit Part
"""

# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')

import pickle
f1 = open('params.pickle', 'wb') 
pickle.dump(W, f1) 
f1.close()

f2 = open('params_bonus.pickle', 'wb')
pickle.dump(W_b, f2)
f2.close()