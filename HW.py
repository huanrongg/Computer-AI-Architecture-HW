# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:48:16 2019

@author: 曾焕荣
"""
import numpy as np
import pandas
from sklearn.preprocessing import OneHotEncoder

def relu(Z):
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache

def softmax(Z):
    Z_shift = Z - np.max(Z, axis = 0)
    A = np.exp(Z_shift)/ np.sum(np.exp(Z_shift), axis=0)
    
    cache = Z_shift
    
    return A, cache

def initialize_parameters(n_x, n_h, n_y):    
    W1 = np.random.randn(n_h, n_x) *  0.01   # weight matrix随机初始化
    b1 = np.zeros((n_h, 1))                 # bias vector零初始化
    W2 = np.random.randn(n_y, n_h) *  0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters    

def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(np.sum(Y * np.log(AL))) / float(m)
    #cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / float(m)
    db = np.sum(dZ, axis=1, keepdims=True) / float(m)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0
    
    return dZ

def softmax_backward(Y, cache):
    """
    合并求导dL/dZ = （dL/dA） * （dA/dZ）
    """
    Z = cache  #注意cache中的Z实际是Z_shift
    
    s = np.exp(Z)/ np.sum(np.exp(Z), axis=0)
    dZ = s - Y
    
    return dZ

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
 
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    
    for l in range(1,L+1):
        parameters['W'+str(l)] -= learning_rate * grads['dW'+str(l)]
        parameters['b'+str(l)] -= learning_rate * grads['db'+str(l)]
    
    return parameters

def two_layer_model(X, Y, layers_dims, learning_rate = 0.05, num_iterations = 15000, print_cost=True):  
    grads = {}
    costs = []                             
    (n_x, n_h, n_y) = layers_dims
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations+1):
        A1, cache1 = linear_activation_forward(X, W1, b1, activation='relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation='softmax')
        
        #print(W1)
        #print(X)
        #print(A1.shape)
        
        cost = compute_cost(A2, Y)
                
        dA1, dW2, db2 = linear_activation_backward(Y, cache2, activation='softmax')
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation='relu')
        
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    return parameters

def predict(X, y, parameters, set):
    m = X.shape[1]
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Forward propagation
    A1, _ = linear_activation_forward(X, W1, b1, activation='relu')
    probs, _ = linear_activation_forward(A1, W2, b2, activation='softmax')
    
    # print(probs)
    # convert probas to 0-9 predictions
    prediction = np.argmax(probs, axis=0) 
    
    prediction = prediction.reshape(1, -1)
    print(set + " Accuracy: "  + str(np.sum((prediction == y)/float(m))))
    
    #print("prediction " + str(prediction))
    #print("y " + str(y))
    
    return prediction

if __name__ == '__main__':
    dataset = pandas.read_csv('dataset.txt', sep='\s+')
    
    trainset = dataset[0:400]
    y_train = trainset.pop('y')
    train_x = trainset.values
    train_y = y_train.values.reshape(1, -1) - 1  #注意这里一定要执行reshape操作
    
    testset = dataset[400:500]
    y_test = testset.pop('y')
    test_x = testset.values
    test_y = y_test.values.reshape(1, -1) - 1 
    
    
    enc = OneHotEncoder()
    train_y_oh = enc.fit_transform(train_y.T).toarray().T
    
    #print(train_y)
    #print(train_x.T.shape)  #(8, 400)
    #print(train_y.shape)    #(1, 400)
    
    
    parameters = two_layer_model(train_x.T, train_y_oh, layers_dims = (8, 9, 4))
    
    #注意在测试集中y不需要转换成one-hot
    prediction = predict(train_x.T, train_y, parameters, 'TrainSet')
    prediction = predict(test_x.T, test_y, parameters, 'TestSet')
    print(y_test.values.reshape(1, -1))
    print(prediction+1)
    

    
    
    
    