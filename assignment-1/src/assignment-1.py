import time
import numpy as np
import pickle
import os
import copy
import matplotlib.pyplot as plt
from matplotlib import pyplot
np.random.seed(400)

def LoadBatch(filename):
    """
    LoadBatch is a function that loads the data and returns an object.
    """
    class parse_file():
        def __init__(self, data, labels, y):
            self.data = data
            self.labels = labels
            self.y = y
    
    with open(filename, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')   
    
    X = (dict_[b'data']).astype('float32')/255
    Y =np.zeros((len(dict_[b'labels']),len(np.unique(dict_[b'labels']))))   
    Y[np.arange(len(dict_[b'labels'])), dict_[b'labels']] = 1    
    y= np.array(dict_[b'labels'])    
    
    return parse_file(X,Y.transpose(), y)        

class params():
    def __init__(self, n_batch, eta, n_epochs):
        self.n_batch = n_batch
        self.eta = eta
        self.n_epochs = n_epochs

def ComputeAccuracy(X_, y_, W_, b_):
    """
    :param X_:  N x d matrix,  trainning images.
    :param y_:  N x 1 vector containing the trainning set true class.
    :param W_: K x d matrix. Weights.
    :param b_:  K x 1 vector. Bias.
    :return acc: scalar.
    """
    P_ = np.argmax(EvaluateClassifier(X_, W_ ,b_),axis=1)
    acc = np.count_nonzero(y_== P_) / float(len(P_))
    return acc


def EvaluateClassifier(X_, W_, b_):
    """
    EvaluateClassifier is a function that computes the SOFTMAX, complete the forward pass.
    :param X_:  N x d matrix,  trainning images.
    :param W_: K x d matrix. Weights.
    :param  b_: K x 1 vector. Bias.
    :return P_: K x N matrix. SOFTMAX.
    """
    s = np.matmul(X_, W_.transpose()) + b_.transpose()
    exp_s = np.exp(s)
    exp_sum = np.sum(exp_s, axis=1)
    P = exp_s.transpose() / exp_sum 
    
    return P.transpose()

def ComputeCost(X_, Y_, W_, b_, lambda_=0):
    """
    :param X_:  N x d matrix,  trainning images.
    :param Y_: K x N matrix, Labels for the trainning data set images. one-hot representation.
    :param W_: K x d matrix. Weights.
    :param b_:  K x 1 vector. Bias.
    :param lambda_: regularization parameter.
    :return: scalar. Cost function.
    """
    Npts = X_.shape[0] if len(X_.shape) == 2 else 1
    P_ = EvaluateClassifier(X_, W_, b_)
    
    l_cross = -np.log(np.sum(np.multiply(Y_, P_),axis=1))
    J = np.sum(l_cross)/Npts + lambda_*np.linalg.norm(W_)

    return J

def ComputeGradients(X_, Y_, P_, W_, b_, lambda_=0):
    """
    ComputeGradients is a function that computes the analytic gradient.
    :param X_:  N x d matrix,  trainning images.
    :param Y_: K x N matrix, Labels for the trainning data set images. one-hot representation.
    :param P_: K x N matrix. SOFTMAX.
    :param W_: K x d matrix. Weights.
    :return grad_W:  Kxd matrix, is the gradient matrix of the cost J relative to W.
    :return grad_b: Kx1, is the gradient vector of the cost J relative to b.
    """
    X_ = X_ if len(X_.shape) == 2 else X_.reshape(1, -1)
      
    Npts_, k_  = Y_.shape if len(Y_.shape) == 2 else (1, Y_.shape[0])

    d_ = X_.shape[-1]
    
    #initialize gradients at zero:
    grad_W = np.zeros((k_,d_))
    grad_b = np.zeros((k_,1))
    
    # G_batch:
    G_ = -(Y_ - P_)
    
    #compute gradients of the cost L:
    grad_W_L = np.matmul(G_.transpose(), X_) / float(Npts_)
    grad_b_L = np.sum(G_,axis=0) / float(Npts_) 
    
    #update gradients of the cost J:
    grad_W = grad_W_L + 2*(np.multiply(lambda_, W_))
    grad_b = grad_b_L

    return grad_W, grad_b.reshape(-1,1)

def ComputeGradsNum(X_, Y_, W_, b_, h_=1e-6, lambda_=0):
    """
    ComputeGradsNum is a function that computes the numerical gradient 
    using Centered difference formula.
    :param X_: N x d matrix,  data set images.
    :param Y_: K x N matrix, Labels for the data set images. one-hot representation.
    :param h_: h represents a small change in x, and it can be either positive or negative.
    :param lambda_: regularization parameter.
    :return W_: K x d matrix. Weights.
    :return b_: K x 1 vector. Bias.
    """
    no = W_.shape[0]
    d = W_.shape[1] 

    grad_W = np.zeros((no,d))
    grad_b = np.zeros((no,1))

    c = ComputeCost(X_, Y_, W_, b_, lambda_)

    for i in range(len(b_)):
        b_try = copy.deepcopy(b_)
        b_try[i] = b_try[i] + h_
        c2 = ComputeCost(X_, Y_, W_, b_try, lambda_)
        grad_b[i] = (c2-c) / h_

    for i in range(no):
        for j in range(d):
            W_try = copy.deepcopy(W_)
            W_try[i][j] = W_try[i][j] + h_
            c2 = ComputeCost(X_, Y_, W_try, b_, lambda_)
            grad_W[i][j] = (c2-c) / h_

    return grad_W, grad_b 

def ComputeGradsNumSlow(X_, Y_, W_, b_, lambda_=0,h_=1e-6):
    """
    ComputeGradsNumSlow is a function that computes the numerical gradient
    using the finite difference method.
    :param X_: N x d matrix,  data set images.
    :param Y_: K x N matrix, Labels for the data set images. one-hot representation.
    :return W_: K x d matrix. Weights.
    :return b_: K x 1 vector. Bias.
    """
    no = W_.shape[0]
    d = W_.shape[1] 

    grad_W = np.zeros((no,d))
    grad_b = np.zeros((no,1))
    
    for i in range(len(b_)):
        b_try = copy.deepcopy(b_)
        b_try[i] = b_try[i] - h_
        c1 = ComputeCost(X_, Y_, W_, b_try, lambda_)
        b_try = copy.deepcopy(b_)
        b_try[i] = b_try[i] + h_
        c2 = ComputeCost(X_, Y_, W_, b_try, lambda_)
        grad_b[i] = (c2-c1) / (2*h_)

    for i in range(no):
        for j in range(d):
            W_try = copy.deepcopy(W_)
            W_try[i][j] = W_try[i][j] - h_
            c1 = ComputeCost(X_, Y_, W_try, b_, lambda_)
            W_try =copy.deepcopy(W_)
            W_try[i][j] = W_try[i][j] + h_
            c2 = ComputeCost(X_, Y_, W_try, b_, lambda_)
            grad_W[i][j]= (c2-c1) / (2*h_)

    return grad_W, grad_b

def MaxRelativeError(g_a, g_n):
    """
    :returns max_relative_error: highest relative error among all gradients.
    After computing the gradients Analitically and numerically.
    :g_a and g_n should be the same dimension.
    """
    max_value = np.zeros(g_a.shape)
    eps = np.finfo(np.float32).eps
    absolute_v = np.absolute(g_a) + np.absolute(g_n)
    
    for i in range(g_a.shape[0]):
        for j in range(g_a.shape[1]):
            max_value[i][j] = max(eps, absolute_v[i][j])
    
    max_relative_error = np.amax(np.absolute(g_a-g_n) / max_value)
    
    return max_relative_error


def w_b_random_initiation(X_, Y_, mu, sigma):
    """
    w_b_random_initiation is a function that initializes each entry to have Gaussian random values.
    :param X_: N x d matrix,  data set images.
    :param Y_: K x N matrix, Labels for the data set images. one-hot representation.
    :param mu: mean.
    :param sigma: deviance.
    :return W_: K x d matrix. Weights.
    :return b_: K x 1 vector. Bias.
    """
    no = Y_.shape[0]
    d = X_.shape[1]
    Npts = X_.shape[0]
    W_ = np.random.normal(mu, sigma, (no, d))
    b_ = np.random.normal(mu, sigma, (no,1))
    
    return W_, b_


def MiniBatchGD(X_, X_val, Y_, y_, Y_val, y_val, GD_params, lambda_=0):
    """
    :param X_: trainning images.
    :param Y_: Labels for the trainning images
    :param GDparams: object containing the parameter values:
        *'n_batch': size of the mini batch
        *'eta': learning rate and
        *'n_epochs': number of runs through the whole trainning set.
    :param lambda_: regularization factor in the cost function.
    :return Wstar:
    :return bstar:
    """
    Npts = X_.shape[0]
    cf_train = []
    cf_val = []
    cl_train = []
    cl_val = []
    acc_train = []
    acc_val = []

    W_, b_ = w_b_random_initiation(X_, Y_, mu, sigma)

    for epoch in range(GD_params.n_epochs):
        for j in range(Npts // GD_params.n_batch):
            j_start = j * GD_params.n_batch
            j_end = (j + 1) * GD_params.n_batch
            Xbatch = X_[j_start:j_end]
            Ybatch = Y_[:, j_start:j_end].transpose()

            P_ = EvaluateClassifier(Xbatch, W_, b_)
            grad_w, grad_b = ComputeGradients(Xbatch, Ybatch, P_, W_, b_, lambda_)

            W_ = W_ - GD_params.eta * grad_w
            b_ = b_ - GD_params.eta * grad_b

        # cost per epoch
        trainning_cost = ComputeCost(X_, Y_.transpose(), W_, b_, lambda_)
        validation_cost = ComputeCost(X_val, Y_val.transpose(), W_, b_, lambda_)

        # loss per epoch
        trainning_loss = ComputeCost(X_, Y_.transpose(), W_, b_, 0)
        validation_loss = ComputeCost(X_val, Y_val.transpose(), W_, b_, 0)

        cf_train.append(trainning_cost)
        cf_val.append(validation_cost)

        cl_train.append(trainning_loss)
        cl_val.append(validation_loss)

        # accuracy per epoch
        acc_train_ = ComputeAccuracy(X_, y_, W_, b_)
        acc_val_ = ComputeAccuracy(X_val, y_val, W_, b_)

        acc_train.append(acc_train_)
        acc_val.append(acc_val_)

    return W_, b_, cf_train, cf_val, cl_train, cl_val, acc_train, acc_val

def plot_validation_trainning_cf_acc(
                                    validation_list_cf_,
                                    trainning_list_cf_,
                                    acc_train_,
                                    acc_val_,
                                    cl_train_,
                                    cl_val_,
                                    GD_params_,
                                    lambda_,
                                    out_filename='none'
                                    ):
    """
    plot_validation_trainning_cf_acc plots the loss and accuracy for both, validation and trainning
    set for a given number of epochs.
    """
    t=range(len(validation_list_cf_))
    pyplot.figure(figsize=(13,5))
    pyplot.subplots_adjust(wspace=0.3)
    pyplot.suptitle(f'batch_n = {GD_params_.n_batch}, eta = {GD_params_.eta},lambda = {lambda_}', size =16)
    pyplot.style.use('seaborn-darkgrid')
    # sp1
    pyplot.subplot(131)
    pyplot.plot(t, validation_list_cf_, '#4363d8', label = 'validation loss')
    pyplot.plot(t, trainning_list_cf_, '#3cb44b', label = 'trainning loss')
    pyplot.legend(loc='best')
    pyplot.xlabel('epoch', size = 13.5)
    pyplot.ylabel('Cost', size = 13.5)
    pyplot.title('Cost',size = 14)
    # sp2
    pyplot.subplot(132)
    pyplot.plot(t, cl_val_, '#4363d8', label = 'validation cost')
    pyplot.plot(t, cl_train_, '#3cb44b', label = 'trainning cost')
    pyplot.legend(loc='best')
    pyplot.xlabel('epoch', size = 13.5)
    pyplot.ylabel('Loss', size = 13.5)
    pyplot.title('Loss', size = 14)
    # sp3
    pyplot.subplot(133)
    pyplot.plot(acc_train_,'#4363d8',  label= 'trainning acc')
    pyplot.plot(acc_val_, '#3cb44b', label = 'validation acc')
    pyplot.legend(loc='best')
    pyplot.xlabel('epoch', size = 13.5)
    pyplot.ylabel('Accuracy', size = 13.5)
    pyplot.title('Accuracy', size = 14)
    if out_filename=='none':
        pyplot.show()
    else:
        pyplot.savefig(out_filename)
        pyplot.show()


def display_class_template_image(W_star_,out_filename='none' ):
    """
    :param W_star_: k x d matrix.
    :param out_filename:
    :return: displays the image class template the network has learnt.
    """

    fig, axarr = plt.subplots(nrows=1, ncols=10, figsize=(20, 20))
    for i, arr in zip((range(len(W_star_))), axarr):
        image = W_star_[i].reshape(3, 32,32)
        s_im = ((image - image.min()) / (image.max() - image.min())).astype('float32')
        s_im = np.transpose(s_im, (1, 2, 0))
        arr.axis('off')
        arr.grid(b=None)
        arr.imshow(s_im)
    if out_filename=='none':
        plt.show() 
    else:
        plt.savefig(out_filename)
        plt.show()  



