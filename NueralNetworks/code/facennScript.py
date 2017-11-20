'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from math import sqrt
from scipy.optimize import minimize

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return  1.0 / (1.0 + np.exp(-z))

def sigmoidGradient(z):
    return np.multiply(sigmoid(z), 1.0 - sigmoid(z))

# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    
    # Feedward the neural network and check the J
    m = training_data.shape[0]
    
    a1 = np.c_[np.ones(m), training_data]
    
    a2 = np.dot(a1, w1.T)
    z2 = sigmoid(a2)
    z2 = np.c_[np.ones(z2.shape[0]), z2]
    
    a3 = np.dot(z2, w2.T)
    z3 = sigmoid(a3)
    
    label = np.zeros([m, n_class])
    
    for i in range(m):
        label[i][training_label[i]] = 1
        
    obj_val = (1/m) * np.sum((-1.0 * label * np.log(z3) - (1.0 - label) * np.log(1.0 - z3)))
    
    regularator = (lambdaval / 2 / m) * ((w1[:,1:] ** 2).sum() + (w2[:,1:] ** 2).sum())
 
    obj_val = obj_val + regularator

    
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    
    m = training_data.shape[0]
    w1_grad = np.zeros(w1.shape)
    w2_grad = np.zeros(w2.shape)
    for t in range(m):
        a1 = np.vstack([1, training_data[t, :].reshape(n_input, 1)]) 
        a2 = np.dot(w1, a1)   
        z2 = sigmoid(a2)        
        z2 = np.vstack([1, z2])  
        a3 = np.dot(w2, z2)     
        z3 = sigmoid(a3)       
        yy = np.arange(n_class).reshape(n_class, 1) 
        yy = np.double(yy == training_label[t])   
        delta3 = z3 - yy   
        delta2 = np.multiply(np.dot(w2.T, delta3), np.vstack([1, sigmoidGradient(a2)])) 
        delta2 = delta2[1:]     
        
        w2_grad = w2_grad + delta3 * z2.reshape(1, n_hidden + 1) 
        w1_grad = w1_grad + delta2 * a1.reshape(1, n_input + 1)      
        
    w2_grad = 1/m * w2_grad + lambdaval/m * w2
    w1_grad = 1/m * w1_grad + lambdaval/m * w1
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate((w1_grad.flatten(), w2_grad.flatten()),0)
    print(obj_val)
    return (obj_val, obj_grad)

# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    m = data.shape[0]
    h1 = sigmoid(np.dot(np.c_[np.ones(m), data], w1.transpose()))
    h2 = sigmoid(np.dot(np.c_[np.ones(m), h1], w2.transpose()))
    col_index = np.argmax(h2, axis = 1)
    Predected_labels = col_index
    # Your code here
    return Predected_labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
# nn_params = fmin_cg(nnObjFunction, initialWeights, fprime = nnObjGradient, args = args, maxiter=50)
#params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
