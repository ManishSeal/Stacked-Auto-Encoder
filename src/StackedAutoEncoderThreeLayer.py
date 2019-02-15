
import numpy as np
from load_mnist import mnist
import matplotlib.pyplot as plt
import pdb
import sys, ast
import pickle

epsilon = 1e-11
a_const = 1

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
    A = np.tanh(Z*a_const)
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
    #print("inside tanh_der A:")
    #print(A)

    dZ = dA * (1 - A * A)*a_const


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
    
    A = 1/(1+np.exp(-Z*a_const))
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
    ### CODE HERE
    A, cache = sigmoid(cache["Z"])
    dZ = dA * A * (1 - A) * a_const
    return dZ

def linear(Z):
    '''
    computes linear activation of Z
    This function is implemented for completeness

    Inputs: 
        Z is a numpy.ndarray (n, m)

    Returns: 
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = Z
    cache = {}
    return A, cache

def linear_der(dA, cache):
    '''
    computes derivative of linear activation
    This function is implemented for completeness

    Inputs: 
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input 
        to the activation layer during forward propagation

    Returns: 
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    return dZ

def softmax_cross_entropy_loss(Z, Y=np.array([]) ):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs: 
        Z - numpy.ndarray (n, m)
        Y - numpy.ndarray (1, m) of labels
            when y=[] loss is set to []
    
    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction
    '''
    
    A = (np.array(Z))
    m = A.shape[1]
    
    for i in range(m):
        zm = max(A[ : , i])
        A[ : , i] = A[ : , i] - zm
    
    A = np.exp(A) 
    
    for i in range(m):
        s = sum(A[ : , i])
        A[ : , i ] = A[ : , i ]/s
        
    
    
    
    # Calculating the loss:
    
    loss = 0.0
    m = A.shape[1]
    
    if len(Y) != 0:    
        for i in range(m):
            label = int(Y[0][i])
            #print("label = ",label)
            loss = loss + (np.log(A[label , i]))/(-m)
    #loss = loss/(-m)
            
        
    cache = {}
    cache['A'] = A    
    
    return A, cache, loss

def softmax_cross_entropy_loss_der(Y, cache):
    '''
    Computes the derivative of softmax activation and cross entropy loss

    Inputs: 
        Y - numpy.ndarray (1, m) of labels
        cache -  a dictionary with cached activations A of size (n,m)

    Returns:
        dZ - numpy.ndarray (n, m) derivative for the previous layer
    '''
    
    dZ = cache['A']
    m = dZ.shape[1]
    for i in range(m):
        label = int(Y[0][i])
        dZ[label, i] = dZ[label, i] - 1
        
    #multiply by (1/m) ??
    dZ = dZ /(m)
    #print("dZ = \n", dZ[:,[400,600,700,1200]])

    return dZ

def initialize_multilayer_weights(net_dims):
    '''
    Initializes the weights of the multilayer network

    Inputs: 
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''
    np.random.seed(0)
    numLayers = len(net_dims)
    parameters = {}
    for l in range(numLayers-1):
        parameters["W"+str(l+1)] = np.random.randn(net_dims[l+1], net_dims[l]) * np.sqrt(2./net_dims[l+1])#CODE HERE
        parameters["b"+str(l+1)] = np.random.randn(net_dims[l+1], 1) #* np.sqrt(2./net_dims[l+1])#CODE HERE
    return parameters

def initialize_multilayer_weights_from_file(file_name):
    '''
    Initializes the weights of the multilayer network by reading from a file

    Inputs: 
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''
    with open(file_name, "rb") as fp:
        parameters=pickle.load(fp)
        
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
        cache - a dictionary containing the inputs A
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
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)
    elif activation == "tanh":
        A, act_cache = tanh(Z)
    
    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache

def multi_layer_forward(X, parameters, activations):
    '''
    Forward propgation through the layers of the network

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
    Returns:
        AL - numpy.ndarray (c,m)  - outputs of the last fully connected layer before softmax
            where c is number of categories and m is number of samples in the batch
        caches - a dictionary of associated caches of parameters and network inputs
    '''
    L = len(parameters)//2  #we have 6/2 = 3
    AL = X
    caches = []
    for ll in range(1,L+1):  # since there is no W0 and b0        
        AL, cache = layer_forward(AL, parameters["W"+str(ll)], parameters["b"+str(ll)], activations[ll-1])
        caches.append(cache)

#     AL, cache = layer_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "linear")
#     caches.append(cache)
    return AL, caches

def linear_backward(dZ, cache, W, b):
    '''
    Backward prpagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz 
        cache - a dictionary containing the inputs A, for the linear layer
            where Z = WA + b,    
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    A_prev = cache["A"]
    ## CODE HERE
    
    dA_prev = np.dot(W.T,dZ)
    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis=1, keepdims=True)
    
    return dA_prev, dW, db

def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)
    
    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W 
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    elif activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db

def multi_layer_backward(dAL, caches, parameters, activations):
    '''
    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately

    Inputs: 
        dAL - numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches - a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}

    Returns:
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
    '''
    L = len(caches)  # with one hidden layer, L = 2
    gradients = {}
    dA = dAL
    #activation = "linear"
    for l in reversed(range(1,L+1)):
        dA, gradients["dW"+str(l)], gradients["db"+str(l)] = \
                    layer_backward(dA, caches[l-1], \
                    parameters["W"+str(l)],parameters["b"+str(l)],\
                    activations[l-1])
        #activation = activations
    return gradients

def classify(X, parameters, activations):
    '''
    Network prediction for inputs X

    Inputs: 
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    # Forward propagate X using multi_layer_forward
    
    AL, caches = multi_layer_forward(X, parameters, activations)
    
    # Get predictions using softmax_cross_entropy_loss
    A, cache, loss = softmax_cross_entropy_loss(AL)
    
    Ypred = np.argmax(A, axis=0).reshape(1,X.shape[1])
    
    # Estimate the class labels using predictions
    return Ypred

def update_parameters(parameters, gradients, epoch, learning_rates, decay_rate=0.0):
    '''
    Updates the network parameters with gradient descent

    Inputs:
        parameters - dictionary of network parameters 
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients - dictionary of gradient of network parameters 
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch - epoch number
        learning_rate - step size for learning
        decay_rate - rate of decay of step size - not necessary - in case you want to use
    '''
    alphas = learning_rates*(1/(1+decay_rate*epoch))
    #alpha = learning_rate
    L = len(parameters)//2
    
    for i in range(1, L+1):
        parameters["W"+str(i)] = parameters["W"+str(i)] - alphas[i-1]*gradients["dW"+str(i)]
        parameters["b"+str(i)] = parameters["b"+str(i)] - alphas[i-1]*gradients["db"+str(i)]
    
    
    return parameters, alphas

def multi_layer_network(X, Y, X_validation, Y_validation, net_dims,
                        learning_rates,
                        activations,
                        parameters_file,
                        num_iterations=500,
                        decay_rate=0.01,
                        parameters = None
                       ):
    '''
    Creates the multilayer network and trains the network

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
    if parameters == None:
        parameters = initialize_multilayer_weights_from_file(parameters_file)
    A0 = X
    costs = []
    costs_validation = []
    texts = []
    #activation = "tanh"
    
    for ii in range(num_iterations):
        # Forward Prop
        ## call to multi_layer_forward to get activations
        AL, caches = multi_layer_forward(X=A0, parameters = parameters, activations=activations)
        AL_validation, caches_validation = multi_layer_forward(X=X_validation, parameters=parameters, 
                                                               activations=activations
                                                              )
        
        ## call to softmax cross entropy loss       
        A_softmax , cache_softmax, loss = softmax_cross_entropy_loss(AL, Y)
        A_softmax_validation , cache_softmax_validation, loss_validation = softmax_cross_entropy_loss(AL_validation, Y_validation)
        #print("loss = ", loss)
        #print("A = ",cache_softmax["A"][:,[300,400,600,700,1200]])

        # Backward Prop
        ## call to softmax cross entropy loss der
        dZ = softmax_cross_entropy_loss_der(Y, cache_softmax)
        
        ## call to multi_layer_backward to get gradients
        gradients = multi_layer_backward(dZ, caches, parameters, activations=activations)
        
        ## call to update the parameters
        parameters, alphas = update_parameters(parameters, gradients, ii, learning_rates, decay_rate)
        
        if ii % 10 == 0:
            costs.append(loss)
            costs_validation.append(loss_validation)
        if ii % 10 == 0:
            str1 = "Cost at iteration %i is: %.05f, learning rate: " %(ii, loss) + str(alphas) + "\n"
            str2 = "Validation Cost at iteration %i is: %.05f, learning rate: " %(ii, loss) + str(alphas) + "\n"
            print(str1)
            print(str2)
            texts.append(str1)
            texts.append(str2)
    
    
    
    return costs, costs_validation, parameters, texts

def main():
    
    net_dims = [784, 500, 200, 50]
    net_dims.append(10) # Adding the digits layer with dimensionality = 10 
    # finally net_dims = [784, 500, 200, 50, 10]
    print("Network dimensions are:" + str(net_dims))
    
    
    
    data_file = "fashion_train_validation_test_split_data_noTr_80_noTs_10000_digits_[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_noise_0_4.txt"
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
    
    
    
    
    # initialize learning rate and num_iterations
    learning_rates = np.array([0.000001, 0.000001, 0.0152, 0.512])
    num_iterations = 181
    activations = ["tanh", "relu", "relu", "linear"]
    
    layer1parameter_file = "StackedLayer1parameters_[784, 500, 784]_activations_['tanh', 'sigmoid']_lr_0_04_digits_10_ni_702_train_size_55000.params"
    layer2parameter_file = "StackedLayer2parameters_[500, 200, 500]_activations_['relu', 'tanh']_lr_0_01_digits_10_ni_601_train_size_55000.params"
    layer3parameter_file = "StackedLayer3parameters_[200, 50, 200]_activations_['relu', 'tanh']_lr_0_04_digits_10_ni_501_train_size_55000.params"
    
    parameter_files = [layer1parameter_file, layer2parameter_file, layer3parameter_file]
    kk = 1
    stackedae_parameters = dict()
    for file_name in parameter_files:
        with open(file_name, "rb") as fp:
            parameter = pickle.load(fp)
        stackedae_parameters["W"+str(kk)] = parameter["W1"]
        stackedae_parameters["b"+str(kk)] = parameter["b1"]
        kk += 1
    
    np.random.seed(81)
    stackedae_parameters["W"+str(kk)] = np.random.randn(net_dims[kk], net_dims[kk-1]) * np.sqrt(2./net_dims[kk])
    stackedae_parameters["b"+str(kk)] = np.random.randn(net_dims[kk], 1) * np.sqrt(2./net_dims[kk])
    
    

    costs, costs_validation, parameters, logs = multi_layer_network(X = train_data,
                                                                    Y = train_label,
                                                                    X_validation = validation_data, 
                                                                    Y_validation = validation_label,
                                                                    net_dims = net_dims,
                                                                    num_iterations=num_iterations,
                                                                    learning_rates=learning_rates,
                                                                    decay_rate=0.01,
                                                                    activations=activations,
                                                                    parameters_file=None,
                                                                    parameters = stackedae_parameters
                                                                   )
    
    
    file_prefix = "StackedAllLayers_"
    
    log_file = "logs_"+str(net_dims)+"_activations_"+str(activations)+"_lr_"+\
                        "_digits_"+str(len(digit_range))+ \
                        "_ni_"+str(num_iterations)+ \
                    "_train_size_"+str(train_data.shape[1])
    log_file =  file_prefix +  log_file
    
    """file_out = "output/"+str(net_dims)+"_"+str(learning_rates)+"_"+str(num_iterations)+"_"+str(activations)
    file_out = file_out.replace(".", "_")
    file_out = file_out + ".txt"
    """
#     compute the accuracy for training set and testing set
    train_Pred = classify(train_data, parameters, activations=activations)
    test_Pred = classify(test_data, parameters, activations=activations)

    trAcc = 0
    teAcc = 0
    #print("train_Pred shape = ", train_Pred.shape)
    for i in range(train_Pred.shape[1]):
        if train_Pred[0][i] == train_label[0][i]:
            trAcc += 1
    trAcc = trAcc/train_Pred.shape[1]
    
    for i in range(test_Pred.shape[1]):
        if test_Pred[0][i] == test_label[0][i]:
            teAcc += 1
    teAcc = teAcc/test_Pred.shape[1]
    
    str1 = "Accuracy for training set is {0:0.3f} %".format(trAcc*100)
    str2 = "Accuracy for testing set is {0:0.3f} %".format(teAcc*100)
    print(str1)
    print(str2)
    logs.append(str1)
    logs.append(str2)
    
    with open(log_file,"w") as fp:
        for i in logs:
            fp.write((i+"\n"))
    
    plt.figure(figsize = (20,10))
    plot1, = plt.plot(costs_validation, "o-", color = "r")
    plot2, = plt.plot(costs, "o-", color='g')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.legend([plot1, plot2],["Validation Cost", "Training Cost"])
    plt.show()


if __name__ == "__main__":
    main()