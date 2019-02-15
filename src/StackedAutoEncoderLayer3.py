
import numpy as np
from load_mnist import mnist
import matplotlib.pyplot as plt
import pdb
import sys, ast
import pickle
import os


epsilon = 0.0000000
alpha = 1

def relu(Z):
    '''
    computes relu activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    
    A = np.maximum(0,Z)
    cache = {}
    cache["Z"] = Z
    return A, cache

def relu_der(dA, cache):
    '''
    computes derivative of relu activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z<0] = 0
    #dZ[Z>0] = 1
    return dZ

def tanh(Z):
    '''
    computes tanh activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.tanh(Z*alpha)
    cache = {}
    cache["Z"] = Z
    return A, cache

def tanh_der(dA, cache):
    '''
    computes derivative of tanh activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    A, cache = tanh(cache["Z"])

    dZ = dA * (1 - A * A)*alpha


    return dZ

def sigmoid(Z):
    '''
    computes sigmoid activation of Z

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    
    A = 1/(1+np.exp(-Z*alpha))
    cache = {}
    cache["Z"] = Z
    return A, cache

def sigmoid_der(dA, cache):
    '''
    computes derivative of sigmoid activation

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    A, cache = sigmoid(cache["Z"])
    dZ = dA * A * (1 - A) * alpha
    return dZ

def initialize_2layer_weights(n_in, n_h, n_fin):
    '''
    Initializes the weights of the 2 layer network

    Inputs: 
        n_in input dimensions (first layer)
        n_h hidden layer dimensions
        n_fin final layer dimensions

    Returns:
        dictionary of parameters
    '''
    # initialize network parameters
    
    np.random.seed(81)
    W1 = np.random.rand(n_h, n_in)*0.01
    b1 = np.random.rand(n_h, 1)*0.01
    W2 = np.random.rand(n_fin, n_h)*0.01
    b2 = np.random.rand(n_fin, 1)*0.01


    parameters = {}
    parameters["W1"] = W1
    parameters["b1"] = b1
    parameters["W2"] = W2
    parameters["b2"] = b2

    return parameters

def linear_forward(A, W, b):
    '''
    Input A propagates through the layer 
    Z = WA + b is the output of this layer. 

    Inputs: 
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A, W and b
        to be used for derivative
    '''

    Z = np.dot(W,A) + b

    cache = {}
    cache["A"] = A
    return Z, cache

def layer_forward(A_prev, W, b, activation):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs: 
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    Z, lin_cache = linear_forward(A_prev, W, b) #lin_cache stores this A_prev
    if activation == "sigmoid":
        A, act_cache = sigmoid(Z) #act_cache stores the Z
    elif activation == "tanh":
        A, act_cache = tanh(Z)
    elif activation == "relu":
        A, act_cache = relu(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache

    return A, cache



def linear_backward(dZ, cache, W, b):
    '''
    Backward propagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz 
        cache - a dictionary containing the inputs A
            where Z = WA + b,    
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)  
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''

    dA_prev = np.dot(W.T,dZ)
    dW = np.dot(dZ, cache["A"].T)
    db = np.sum(dZ, axis=1, keepdims=True)

    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        W - numpy.ndarray (n,p)  
        b - numpy.ndarray (n, 1)
    
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"] #contains the A that comes to this layer
    act_cache = cache["act_cache"] #contains the Z that is calculated in this layer

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    elif activation == "relu":
        dZ = relu_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db



def quadratic_loss_layer(AL, Y):
    """
    The last loss layer for calculating the quadratic loss and returning the derivative
    
    Inputs:
        A: the output of the last layer
        Y: the original Y required
    
    """
    k , m = AL.shape
    l = AL-Y
    L = (l*l)/(2*m*1.0)
    cost = np.sum(L)
    
    dAL = l/(m)
    
    return dAL, cost
    


def denoise(X, parameters, activations):
    """
    Input:
        X: an array of shape [n, m], containing the noisy images to be de-noised
        parameter: a dictionary containing the weights and biases for the different layers
                a dictionary like {"W1":[...], "b1":[...]..}
    Output:
        A2: the denoised version as predicted by the autoencoder
    """
    A0=X
    A1, cache_ = layer_forward(A_prev = A0, W = parameters["W1"], b = parameters["b1"], activation=activations[0] )
    A2, cache_ = layer_forward(A_prev = A1, W = parameters["W2"], b = parameters["b2"], activation=activations[1] )
    
    return A2
    

def two_layer_network(X, Y, X_validation, Y_validation, net_dims, num_iterations=20, learning_rate=0.1,
                      activations=["tanh", "sigmoid"],
                      parameter_file = None
                     ):
    '''
    Creates the 2 layer network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent
    
    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''
    n_in, n_h, n_fin = net_dims
    if parameter_file == None:
        parameters = initialize_2layer_weights(n_in, n_h, n_fin)
    else:
        with open(parameter_file, "rb") as fp:
            parameters = pickle.load(fp)

    
    
    
    A0 = X
    costs = []
    validation_costs = []
    logs = []
    for ii in range(num_iterations):
        # Forward propagation
        A1, cache1 = layer_forward(A_prev = A0, W = parameters["W1"], b = parameters["b1"], activation=activations[0] )
        A2, cache2 = layer_forward(A_prev = A1, W = parameters["W2"], b = parameters["b2"], activation=activations[1] )
        
        A1_validation, cache_validation = layer_forward(A_prev = X_validation,
                                                         W = parameters["W1"],
                                                         b = parameters["b1"],
                                                         activation=activations[0] )
        A2_validation, cache_validation = layer_forward(A_prev = A1_validation,
                                                         W = parameters["W2"],
                                                         b = parameters["b2"],
                                                         activation=activations[1] )
        
        
        

        # cost estimation
        dA2, cost = quadratic_loss_layer(AL=A2, Y=Y)
        dA2_validation, validation_cost = quadratic_loss_layer(AL = A2_validation , Y=Y_validation)
        #print("at i = ",ii, "cost = ", cost)
        
        # Backward Propagation
        dA1, dW2, db2 = layer_backward(dA2,
                                       cache2,
                                       W = parameters["W2"],
                                       b = parameters["b2"],
                                       activation = activations[1]
                                      )
        dA0, dW1, db1 = layer_backward(dA1,
                                       cache1,
                                       W = parameters["W1"],
                                       b = parameters["b1"],
                                       activation = activations[0]
                                      )

        #update parameters

        parameters["W1"] -= learning_rate*dW1
        parameters["b1"] -= learning_rate*db1
        parameters["W2"] -= learning_rate*dW2
        parameters["b2"] -= learning_rate*db2
        
        
        


        if ii % 10 == 0:
            costs.append(cost)
            validation_costs.append(validation_cost)
        if ii % 10 == 0:
            str1 = "Cost at iteration %i is: %f" %(ii, cost)
            str2 = "Validation Cost at iteration %i is: %f" %(ii, validation_cost)
            print(str1)
            print(str2)
            logs.append(str1)
            logs.append(str2)
    
    return costs, validation_costs, parameters, logs

def main():
    
    net_dims3=[200, 50, 200]  
    
    
    data_file = "fashion_train_validation_test_split_data_noTr_60000_noTs_10000_digits_[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_noise_0_2.txt"
    data = {}
    with open(data_file, "rb") as fp:
        data = pickle.load(fp)
        
    print("Data Loading done")
        
    noisy_train_data = data["noisy_train_data"]
    train_data = data["train_data"]
    train_label = data["train_label"]
    noisy_validation_data = data["noisy_validation_data"]
    validation_data = data["validation_data"]
    validation_label = data["validation_label"]
    noisy_test_data = data["noisy_test_data"]
    test_data = data["test_data"]
    test_label = data["test_label"]
    
    digit_range = [0,1,2,3,4,5,6,7,8,9]
    
    unique_test_image_index = []
    for ll in digit_range:
        idx = np.where(test_label == ll)
        unique_test_image_index.append(idx[1][0])
    
    
    
        
    n_in, m = noisy_train_data.shape
    
    n_h = net_dims3[1]
    n_fin = net_dims3[2]
    learning_rate3 = 0.04
    num_iterations3 = 501
    activations3 = ["tanh","tanh"]
    
    Layer1parameter_file = "StackedLayer1parameters_[784, 500, 784]_activations_['tanh', 'sigmoid']_lr_0_04_digits_10_ni_702_train_size_55000.params"
    Layer2parameter_file = "StackedLayer2parameters_[500, 200, 500]_activations_['relu', 'tanh']_lr_0_01_digits_10_ni_601_train_size_55000.params"
    
    with open(Layer1parameter_file, "rb") as fp:
        layer1parameters = pickle.load(fp)
        
    with open(Layer2parameter_file, "rb") as fp:
        layer2parameters = pickle.load(fp)
        
    layer2_train_data, _ = layer_forward(A_prev = train_data,
                                         W = layer1parameters["W1"],
                                         b = layer1parameters["b1"],
                                         activation="tanh" 
                                        )
    layer2_validation_data, _ = layer_forward(A_prev = validation_data,
                                         W = layer1parameters["W1"],
                                         b = layer1parameters["b1"],
                                         activation="tanh" )
    
    layer3_train_data, _ = layer_forward(A_prev = layer2_train_data,
                                         W = layer2parameters["W1"],
                                         b = layer2parameters["b1"],
                                         activation="tanh" 
                                        )
    layer3_validation_data, _ = layer_forward(A_prev = layer2_validation_data,
                                         W = layer2parameters["W1"],
                                         b = layer2parameters["b1"],
                                         activation="tanh" )
    
                
    
    print("network dimensions = ", net_dims3)

    costs3, validation_costs3, parameters3, logs3 = two_layer_network(X=layer3_train_data,
                                                            Y=layer3_train_data,
                                                            X_validation=layer3_validation_data,
                                                            Y_validation=layer3_validation_data,
                                                            net_dims=net_dims3,
                                                            num_iterations=num_iterations3,
                                                            learning_rate=learning_rate3,
                                                            activations=activations3,
                                                            parameter_file = None
                                                           )
                                         
    parameters3["num_iterations"] = num_iterations3
    parameters3["activation"] = activations3
    #num_iterations += 601
    
    # compute the accuracy for training set and testing set
                                         
    A0 = train_data
    A1, _ = layer_forward(A_prev = A0,
                          W = layer1parameters["W1"],
                          b = layer1parameters["b1"],
                          activation="tanh" 
                         )
    A2, _ = layer_forward(A_prev = A1,
                          W = layer2parameters["W1"],
                          b = layer2parameters["b1"],
                          activation="tanh" 
                         )
    A3, _ = layer_forward(A_prev = A2,
                          W = parameters3["W1"],
                          b = parameters3["b1"],
                          activation=activations3[0] 
                         )
    A4, _ = layer_forward(A_prev = A3,
                          W = parameters3["W2"],
                          b = parameters3["b2"],
                          activation=activations3[1]  
                         )
    A5, _ = layer_forward(A_prev = A4,
                          W = layer2parameters["W2"],
                          b = layer2parameters["b2"],
                          activation="tanh" 
                         )
    A6, _ = layer_forward(A_prev = A5,
                          W = layer1parameters["W2"],
                          b = layer1parameters["b2"],
                          activation="sigmoid" 
                         )
    train_Pred = A6
    
    A0 = test_data
    A1, _ = layer_forward(A_prev = A0,
                          W = layer1parameters["W1"],
                          b = layer1parameters["b1"],
                          activation="tanh" 
                         )
    A2, _ = layer_forward(A_prev = A1,
                          W = layer2parameters["W1"],
                          b = layer2parameters["b1"],
                          activation="tanh" 
                         )
    A3, _ = layer_forward(A_prev = A2,
                          W = parameters3["W1"],
                          b = parameters3["b1"],
                          activation=activations3[0]  
                         )
    A4, _ = layer_forward(A_prev = A3,
                          W = parameters3["W2"],
                          b = parameters3["b2"],
                          activation=activations3[1] 
                         )
    A5, _ = layer_forward(A_prev = A4,
                          W = layer2parameters["W2"],
                          b = layer2parameters["b2"],
                          activation="tanh" 
                         )
    A6, _ = layer_forward(A_prev = A5,
                          W = layer1parameters["W2"],
                          b = layer1parameters["b2"],
                          activation="sigmoid" 
                         )
    test_Pred = A6
    
                
    
    dA_, train_cost = quadratic_loss_layer(AL=train_Pred, Y=train_data)
    dA_, test_cost = quadratic_loss_layer(AL=test_Pred, Y=test_data)
    
    print("train_pred shape:",train_Pred.shape)
    print("test_pred shape:",test_Pred.shape)
    
    
    
    print("Cost for training set is {0:0.3f} ".format(train_cost))
    print("Cost for testing set is {0:0.3f} ".format(test_cost))
    root = ""
    fig_name3 = "StackedLayer3figure_"+str(net_dims3)+"_activations_"+str(activations3)+\
                    "_lr_"+str(learning_rate3)+\
                    "_digits_"+str(len(digit_range))+\
                    "_ni_"+str(num_iterations3)+\
                    "_train_size_"+str(train_data.shape[1])
    fig_name3 = root + fig_name3
    
    parameter_file3 = "StackedLayer3parameters_"+str(net_dims3)+"_activations_"+\
                        str(activations3)+"_lr_"+str(learning_rate3)+\
                        "_digits_"+str(len(digit_range))+\
                        "_ni_"+str(num_iterations3)+\
                    "_train_size_"+str(train_data.shape[1])
    parameter_file3 = root + parameter_file3
    
    log_file3 = "StackedLayer3logs_"+str(net_dims3)+"_activations_"+str(activations3)+"_lr_"+str(learning_rate3)+\
                        "_digits_"+str(len(digit_range))+\
                        "_ni_"+str(num_iterations3)+\
                    "_train_size_"+str(train_data.shape[1])
    log_file3 =  root +  log_file3
    fig_name3 = fig_name3.replace(".","_")
    parameter_file3 = parameter_file3.replace(".", "_")
    log_file3 = log_file3.replace(".","_")
    parameter_file3 += ".params"
    log_file3 += ".log"
    
    with open(parameter_file3, "wb") as fpp:
        pickle.dump(parameters3, fpp)
    
    with open(log_file3, "w") as fp:
        for i in logs3:
            fp.write(i)
    
    
    
    count = 1
    num_digits = len(digit_range)
    plt.figure( figsize=(9, 30) )
    
    for i in unique_test_image_index:
        plt.subplot(num_digits,2,count)
        plt.imshow(test_data[:,i].reshape(28, -1), cmap="gray")
        count += 1
        plt.subplot(num_digits,2,count)
        plt.imshow(test_Pred[:,i].reshape(28, -1), cmap="gray")
        count += 1
        
    plt.tight_layout()
    plt.savefig(fig_name3)
    plt.show()
    


if __name__ == "__main__":
    main()

