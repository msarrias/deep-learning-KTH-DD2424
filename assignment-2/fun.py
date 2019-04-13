import time
import math
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
        def __init__(self, data, labels, y, raw_data):
            self.data = data
            self.labels = labels
            self.y = y
            self.raw_data = raw_data     
    
    with open(filename, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')   
    
    X = ((dict_[b'data']).astype('float32')/255).transpose()
    mean_X = np.mean(X, axis=1).reshape(-1,1)
    X_ = X - mean_X
    Y =np.zeros((len(dict_[b'labels']),len(np.unique(dict_[b'labels']))))   
    Y[np.arange(len(dict_[b'labels'])), dict_[b'labels']] = 1    
    y= np.array(dict_[b'labels'])      
         
    return parse_file(X_, Y.transpose(), y, X.transpose())       

class params():
    def __init__(self, n_batch, eta, n_epochs):
        self.n_batch = n_batch
        self.eta = eta
        self.n_epochs = n_epochs
        
def init_two_layers_w_b_param(X_, Y_, m_):
    """
    :param X_: dxN matrix, dataset images .
    :param Y_: kxN matrix, labels for the dataset images.
    :param m: number of nodes in the hidden layer.
    :return W_1_: mxd matrix.
    :return b_1_:mx1 matrix.
    :return W_2_:kxm matrix.
    :return b_2_:kx1 matrix.
    """
    d = X_.shape[0]
    k = Y_.shape[0]
    
    mu = 0
    std_1st_layer_= 1/math.sqrt(d)
    std_2nd_layer_= 1/math.sqrt(m_)

    W_1_ = np.random.normal(mu, std_1st_layer_, (m_, d))
    b_1_ = np.zeros((m_, 1))

    W_2_ = np.random.normal(mu, std_2nd_layer_, (k, m_))
    b_2_ = np.zeros((k, 1))
    
    return W_1_, b_1_, W_2_, b_2_

def ReLU(X_, W_1_, b_1_):
    
    ind_size_n = np.ones((X_.shape[1],1))
    S_1_ = np.matmul(W_1_, X_) + np.matmul(b_1_, ind_size_n.T)
    return np.maximum(S_1_, 0)

def SOFTMAX(X_, W_1_, W_2_, b_1_, b_2_):
    """ 
    :param s: kxN matrix
    :return p: kxN softmax
    """
    ind_size_n = np.ones((X_.shape[1],1))
    H_ = ReLU(X_, W_1_, b_1_)
    S_2_ = np.matmul(W_2_, H_) + np.matmul(b_2_, ind_size_n.T)
    p_ =  np.exp(S_2_) / np.matmul(np.ones((1, S_2_.shape[0])), np.exp(S_2_))
    
    #same as dividing by np.exp(s_)/ (np.sum(np.exp(s_), axis=0))
    return p_ , H_

def cross_entropy_loss(Y_, p_):
    """
    :param Y_:
    :param p_:
    :return l_cross_:
    """
    l_cross_ = -np.log(np.sum((np.multiply(Y_, p_)),axis=0))
    return l_cross_

def cost_function(Y_, p_, W_1_, W_2_, lambda_):
    """
    :param Y_: kxN matrix, labels for the dataset images.
    :param p_: kxN matrix softmax
    :param W_1_: mxd matrix.
    :param W_2_: kxm matrix.
    :param lambda_: regularization term.
    """
    J_ = np.sum(cross_entropy_loss(Y_, p_)) / Y_.shape[1] + lambda_*(np.linalg.norm(W_1_) + np.linalg.norm(W_2_))
    return J_
    
def ComputeCost(X_, Y_, W_1_, W_2_, b_1_, b_2_, lambda_):
    
    P_, _ = SOFTMAX(X_, W_1_, W_2_, b_1_, b_2_)
    c = cost_function(Y_, P_, W_1_, W_2_, lambda_) 
    return c

def SOFTMAX(X_, W_1_, W_2_, b_1_, b_2_):
    """ 
    :param s: kxN matrix
    :return p: kxN softmax
    """
    ind_size_n = np.ones((X_.shape[1],1))
    H_ = ReLU(X_, W_1_, b_1_)
    S_2_ = np.matmul(W_2_, H_) + np.matmul(b_2_, ind_size_n.T)
    p_ =  np.exp(S_2_) / np.matmul(np.ones((1, S_2_.shape[0])), np.exp(S_2_))
    
    #same as dividing by np.exp(s_)/ (np.sum(np.exp(s_), axis=0))
    return p_ , H_

def ComputeAccuracy(X_, y_, W_1_, W_2_, b_1_, b_2_):
    """
    :param X_:  N x d matrix,  trainning images.
    :param y_:  N x 1 vector containing the trainning set true class.
    :param W_: K x d matrix. Weights.
    :param b_:  K x 1 vector. Bias.
    :return acc: scalar.
    """
    ind_size_n = np.ones((X_.shape[1],1))
    P_, _ = SOFTMAX(X_, W_1_, W_2_, b_1_, b_2_)
    P = np.argmax(P_,axis=0)
    acc = np.count_nonzero(y_== P) / float(len(P))
    
    return acc

def ComputeGradsAnalt(X_, Y_, b_1_, b_2_, W_1_, W_2_, lambda_):

    Npts = X_.shape[1]
    ind_size_n = np.ones((Npts,1))
    #forward pass:
    P, H = SOFTMAX(X_, W_1_, W_2_, b_1_, b_2_)
    G = -(Y_ - P)

    grad_W_2_L = np.matmul(G, H.T) / float(Npts)
    grad_b_2_L = np.matmul(G, ind_size_n) / float(Npts)

    #Propagate the gradient back through the second layer
    # Backward pass:
    # G = mxN
    G = np.matmul(W_2_.T, G)

    #Compute indicator fuction:
    indicator_f_H = np.sign(H)

    G= G*indicator_f_H
    G[G == -0.0] = 0.0

    grad_W_1_L = np.matmul(G,X_.T) / float(Npts)
    grad_b_1_L = np.matmul(G, ind_size_n) / float(Npts)

    #Compute cost function gradients:
    grad_W_1_J = grad_W_1_L + 2*(np.multiply(lambda_, W_1_))
    grad_b_1_J = grad_b_1_L

    grad_W_2_J = grad_W_2_L  + 2*(np.multiply(lambda_, W_2_))
    grad_b_2_J = grad_b_2_L
    
    return grad_b_1_J, grad_b_2_J, grad_W_1_J, grad_W_2_J
        
    
def ComputeGradsNum(X_, Y_, W_1_, W_2_, b_1_, b_2_, h_=1e-5, lambda_=0):
    """
    ComputeGradsNum is a function that computes the numerical gradient 
    using Centered difference formula.
    """
    d, Npts = X_.shape
    m,_ = W_1_.shape
    k,_ = W_2_.shape
    grad_W_1 = np.zeros(W_1_.shape)
    grad_b_1 = np.zeros(b_1_.shape)
    
    grad_W_2 = np.zeros(W_2_.shape)
    grad_b_2 = np.zeros(b_2_.shape)

    c = ComputeCost(X_, Y_, W_1_, W_2_, b_1_, b_2_, lambda_=0)

    for i in range(len(b_1_)):
        b_try = copy.deepcopy(b_1_)
        b_try[i] = b_try[i] + h_
        c2 = ComputeCost(X_, Y_, W_1_, W_2_, b_try, b_2_, lambda_=0)

        grad_b_1[i] = (c2-c) / h_
    
    for i in range(len(b_2_)):
        b_try = copy.deepcopy(b_2_)
        b_try[i] = b_try[i] + h_
        c2 = ComputeCost(X_, Y_, W_1_, W_2_, b_1_, b_try, lambda_=0)

        grad_b_2[i] = (c2-c) / h_
        
    for i in range(m):
        for j in range(d):
            W_try = copy.deepcopy(W_1_)
            W_try[i][j] = W_try[i][j] + h_
            c2 = ComputeCost(X_, Y_, W_try, W_2_, b_1_, b_2_, lambda_=0)
            grad_W_1[i][j] = (c2-c) / h_
    
    for i in range(k):
        for j in range(m):
            W_try = copy.deepcopy(W_2_)
            W_try[i][j] = W_try[i][j] + h_
            c2 = ComputeCost(X_, Y_, W_1_, W_try, b_1_, b_2_, lambda_=0)
            grad_W_2[i][j] = (c2-c) / h_
            

    return grad_b_1, grad_b_2, grad_W_1, grad_W_2

def ComputeGradsNumSlow(X_, Y_, W_1_, W_2_, b_1_, b_2_, h_=1e-6, lambda_=0):

    d, Npts = X_.shape
    m,_ = W_1_.shape
    k,_ = W_2_.shape
    
    grad_W_1 = np.zeros(W_1_.shape)
    grad_b_1 = np.zeros(b_1_.shape)
    
    grad_W_2 = np.zeros(W_2_.shape)
    grad_b_2 = np.zeros(b_2_.shape)
    
    
    for i in range(len(b_1_)):
        b_try = copy.deepcopy(b_1_)
        b_try[i] = b_try[i] - h_
        c1 = ComputeCost(X_, Y_, W_1_, W_2_, b_try, b_2_, lambda_=0)
        b_try = copy.deepcopy(b_1_)
        b_try[i] = b_try[i] + h_
        c2 = ComputeCost(X_, Y_, W_1_, W_2_, b_try, b_2_, lambda_=0)
        grad_b_1[i] = (c2-c1) / (2*h_)

    for i in range(len(b_2_)):
        b_try = copy.deepcopy(b_2_)
        b_try[i] = b_try[i] - h_
        c1 = ComputeCost(X_, Y_, W_1_, W_2_, b_1_, b_try, lambda_=0)
        b_try = copy.deepcopy(b_2_)
        b_try[i] = b_try[i] + h_
        c2 = ComputeCost(X_, Y_, W_1_, W_2_, b_1_, b_try, lambda_=0)
        grad_b_2[i] = (c2-c1) / (2*h_)
        
    for i in range(m):
        for j in range(d):
            W_try = copy.deepcopy(W_1_)
            W_try[i][j] = W_try[i][j] - h_
            c1 = ComputeCost(X_, Y_, W_try, W_2_, b_1_, b_2_, lambda_=0)
            W_try = copy.deepcopy(W_1_)
            W_try[i][j] = W_try[i][j] + h_
            c2 = ComputeCost(X_, Y_, W_try, W_2_, b_1_, b_2_, lambda_=0)
            grad_W_1[i][j]= (c2-c1) / (2*h_)
    
    for i in range(k):
        for j in range(m):
            W_try = copy.deepcopy(W_2_)
            W_try[i][j] = W_try[i][j] - h_
            c1 = ComputeCost(X_, Y_, W_1_, W_try, b_1_, b_2_, lambda_=0)
            W_try =copy.deepcopy(W_2_)
            W_try[i][j] = W_try[i][j] + h_
            c2 = ComputeCost(X_, Y_, W_1_, W_try, b_1_, b_2_, lambda_=0)
            grad_W_2[i][j]= (c2-c1) / (2*h_)

    return grad_b_1, grad_b_2, grad_W_1, grad_W_2


