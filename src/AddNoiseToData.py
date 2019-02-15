import numpy as np
import os
import pdb
import matplotlib.pyplot as plt

#datasets_dir = '/Users/cubic/hemanth/S2018/cse591/miniProjects/data/'
datasets_dir = './data/'


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def mnist(noTrSamples=1000, noTsSamples=100, \
                        digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
                        noTrPerClass=100, noTsPerClass=10):
    assert noTrSamples==noTrPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    assert noTsSamples==noTsPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    data_dir = os.path.join(datasets_dir, 'mnist/')
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)

    trData = trData/255.
    tsData = tsData/255.

    tsX = np.zeros((noTsSamples, 28*28))
    trX = np.zeros((noTrSamples, 28*28))
    tsY = np.zeros(noTsSamples)
    trY = np.zeros(noTrSamples)

    count = 0
    for ll in digit_range:
        # Train data
        idl = np.where(trLabels == ll)
        idl = idl[0][: noTrPerClass]
        idx = list(range(count*noTrPerClass, (count+1)*noTrPerClass))
        trX[idx, :] = trData[idl, :]
        trY[idx] = trLabels[idl]
        # Test data
        idl = np.where(tsLabels == ll)
        idl = idl[0][: noTsPerClass]
        idx = list(range(count*noTsPerClass, (count+1)*noTsPerClass))
        tsX[idx, :] = tsData[idl, :]
        tsY[idx] = tsLabels[idl]
        count += 1
    
    np.random.seed(1)
    test_idx = np.random.permutation(tsX.shape[0])
    tsX = tsX[test_idx,:]
    tsY = tsY[test_idx]

    trX = trX.T
    tsX = tsX.T
    trY = trY.reshape(1, -1)
    tsY = tsY.reshape(1, -1)
    return trX, trY, tsX, tsY

def mnist_fashion(noTrSamples=1000, noTsSamples=100, \
                        digit_range=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], \
                        noTrPerClass=100, noTsPerClass=10):
    assert noTrSamples==noTrPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    assert noTsSamples==noTsPerClass*len(digit_range), 'noTrSamples and noTrPerClass mismatch'
    data_dir = os.path.join(datasets_dir, 'fashion/')
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trData = loaded[16:].reshape((60000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trLabels = loaded[8:].reshape((60000)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsData = loaded[16:].reshape((10000, 28*28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    tsLabels = loaded[8:].reshape((10000)).astype(float)

    trData = trData/255.
    tsData = tsData/255.

    tsX = np.zeros((noTsSamples, 28*28))
    trX = np.zeros((noTrSamples, 28*28))
    tsY = np.zeros(noTsSamples)
    trY = np.zeros(noTrSamples)

    count = 0
    for ll in digit_range:
        # Train data
        idl = np.where(trLabels == ll)
        idl = idl[0][: noTrPerClass]
        idx = list(range(count*noTrPerClass, (count+1)*noTrPerClass))
        trX[idx, :] = trData[idl, :]
        trY[idx] = trLabels[idl]
        # Test data
        idl = np.where(tsLabels == ll)
        idl = idl[0][: noTsPerClass]
        idx = list(range(count*noTsPerClass, (count+1)*noTsPerClass))
        tsX[idx, :] = tsData[idl, :]
        tsY[idx] = tsLabels[idl]
        count += 1
    
    np.random.seed(1)
    test_idx = np.random.permutation(tsX.shape[0])
    tsX = tsX[test_idx,:]
    tsY = tsY[test_idx]

    trX = trX.T
    tsX = tsX.T
    trY = trY.reshape(1, -1)
    tsY = tsY.reshape(1, -1)
    return trX, trY, tsX, tsY

noise_levels = [0.15, 0.17, 0.2, 0.3, 0.4, 0.5, .6, 0.7, 0.75, 0.80, 0.85, 0.90 ]
for noise_probability in noise_levels:
    #noise_probability = float(input("Enter noise probability"))
    noTrPerClass = 5500
    noValPerClass = 500
    noTsPerClass = 1000
    noTrSamples= 60000
    noTsSamples= 10000
    digit_range = [0,1,2,3,4,5,6,7,8,9]
    output_file  = "fashion_train_validation_test_split_data"+\
                    "_noTr_"+str(noTrSamples)+\
                    "_noTs_"+str(noTsSamples)+\
                    "_digits_"+str(digit_range)+\
                    "_noise_"+str(noise_probability)
    output_file = output_file.replace(".","_")
    output_file += ".txt"


    train_data, train_label, test_data, test_label = \
            mnist_fashion(noTrSamples=noTrSamples,noTsSamples=noTsSamples,\
            digit_range=digit_range,\
            noTrPerClass=noTrPerClass+noValPerClass, noTsPerClass=noTsPerClass)

    print("loading data from mnist done")
    print("now adding noise")

    ## Code to create validation data
    train_labels = train_label.reshape((train_label.shape[1]))




    noisy_train_data = np.array(train_data)
    noisy_test_data = np.array(test_data)

    for i in range(noisy_train_data.shape[1]):
        for j in range(noisy_train_data.shape[0]):
            r = np.random.rand(1)
            if r < noise_probability :
                noisy_train_data[j, i] = np.random.rand(1)

    for i in range(noisy_test_data.shape[1]):
        for j in range(noisy_test_data.shape[0]):
            r = np.random.rand(1)
            if r < noise_probability :
                noisy_test_data[j, i] = np.random.rand(1)

    print("noise adding done")


    idVal = []
    idTr = []

    for ll in digit_range:
        idx = np.where(train_labels == ll)
        #print(idx)
        for ii in idx[0][noTrPerClass: ]:
            idVal.append(ii)
        for ii in idx[0][ : noTrPerClass]:
            idTr.append(ii)

    idVal = np.array(idVal)
    idTr = np.array(idTr)

    print("index gathering done")



    
    val_noisyX = noisy_train_data[ : , idVal]
    val_originalX = train_data[ : , idVal]
    valY = train_label[ : , idVal]
    tr_noisyX = noisy_train_data[: , idTr]
    tr_originalX = train_data[: , idTr]
    trY = train_label[:, idTr]

    
    output = {}
    output["train_data"] = tr_originalX
    output["noisy_train_data"] = tr_noisyX
    output["train_label"] = trY
    output["validation_data"] = val_originalX
    output["noisy_validation_data"] = val_noisyX
    output["validation_label"] = valY
    output["test_data"] = test_data
    output["noisy_test_data"] = noisy_test_data
    output["test_label"] = test_label

    with open(output_file, "wb") as fp:
        pickle.dump(output, fp)


