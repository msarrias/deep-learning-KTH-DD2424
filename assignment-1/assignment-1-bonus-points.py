import numpy as np
import pickle
import os
import copy
import matplotlib.pyplot as plt
from matplotlib import pyplot
import time
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
    b_ = np.random.normal(mu, sigma, (no, 1))

    return W_, b_


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

    l_cross = -np.log(np.sum(np.multiply(Y_, P_), axis=1))
    J = np.sum(l_cross) / Npts + lambda_ * np.linalg.norm(W_)

    return J


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


def ComputeAccuracy(X_, y_, W_):
    """
    :param X_:  N x d matrix,  trainning images.
    :param y_:  N x 1 vector containing the trainning set true class.
    :param W_: K x d matrix. Weights.
    :param b_:  K x 1 vector. Bias.
    :return acc: scalar.
    """
    P_ = np.argmax(EvaluateClassifierSVM(X_, W_), axis=1)
    acc = np.count_nonzero(y_ == P_) / float(len(P_))

    return acc


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

    max_relative_error = np.amax(np.absolute(g_a - g_n) / max_value)

    return max_relative_error


def ComputeGradients(X_, Y_, P_, W_, b_, lambda_=0):
    """
    :return grad_W: is the gradient matrix of the cost J relative to W and has size Kxd
    :return grad_b: is the gradient vector of the cost J relative to b and has size Kx1
    """
    X_ = X_ if len(X_.shape) == 2 else X_.reshape(1, -1)

    Npts_, k_ = Y_.shape if len(Y_.shape) == 2 else (1, Y_.shape[0])

    d_ = X_.shape[-1]

    # initialize gradients at zero:
    grad_W = np.zeros((k_, d_))
    grad_b = np.zeros((k_, 1))

    # G_batch:
    G_ = -(Y_ - P_)

    # compute gradients of the cost L:
    grad_W_L = np.matmul(G_.transpose(), X_) / float(Npts_)
    grad_b_L = np.sum(G_, axis=0) / float(Npts_)

    # update gradients of the cost J:
    grad_W = grad_W_L + 2 * (np.multiply(lambda_, W_))
    grad_b = grad_b_L

    return grad_W, grad_b.reshape(-1, 1)


def unison_shuffled_copies(X_, Y_):
    """
    shuffles the data.
    """
    assert len(X_) == len(Y_)
    p = np.random.permutation(len(X_))
    return X_[p], Y_[p]



def MiniBatchGD_shuffled(X_, X_val, Y_, y_, Y_val, y_val, GD_params, lambda_=0):
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
        shuffle_x, shuffle_y = unison_shuffled_copies(X_, Y_.transpose())
        shuffle_y = shuffle_y.transpose()
        
        for j in range(Npts//GD_params.n_batch):
            j_start = j * GD_params.n_batch
            j_end = (j + 1) * GD_params.n_batch
            Xbatch = shuffle_x[j_start:j_end]
            Ybatch = shuffle_y[:, j_start:j_end].transpose()

            P_ = EvaluateClassifier(Xbatch, W_, b_)
            grad_w, grad_b = ComputeGradients(Xbatch, Ybatch, P_, W_, b_, lambda_)

            W_ = W_ - GD_params.eta*grad_w
            b_ = b_ - GD_params.eta* grad_b
                
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


def MiniBatchGD_eta_decay(X_, X_val, Y_, y_, Y_val, y_val, GD_params, lambda_=0):
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
    k = 5
    W_, b_ = w_b_random_initiation(X_, Y_, mu, sigma)
    
    for epoch in range(GD_params.n_epochs):
        
        if epoch % k == 0:
            GD_params.eta == GD_params.eta / 2 # - (GD_params.eta * 10)
            
        for j in range(Npts//GD_params.n_batch):
            j_start = j * GD_params.n_batch
            j_end = (j + 1) * GD_params.n_batch
            Xbatch = X_[j_start:j_end]
            Ybatch = Y_[:, j_start:j_end].transpose()

            P_ = EvaluateClassifier(Xbatch, W_, b_)
            grad_w, grad_b = ComputeGradients(Xbatch, Ybatch, P_, W_, b_, lambda_)

            W_ = W_ - GD_params.eta*grad_w
            b_ = b_ - GD_params.eta* grad_b
                
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



def bias_trick_score_func(X_, W_, b_):
    """
    bias_trick_score_func is a function that performs the bias trick, in which the bias is 
    added to the weights matrix, so we will end up with only one paramer instead of two.
    :param X_:  N x d matrix,  trainning images.
    :param W_: K x d matrix. Weights.
    :param b_:  K x 1 vector. Bias.
    :return transformed_data: N x (d + 1)
    :return W_prime: K x (d + 1)
    """
    transformed_data= np.hstack((X_, np.ones((X_.shape[0],1))))
    W_prime = np.hstack((W_,b_))
    return transformed_data, W_prime 



def EvaluateClassifierSVM(X_, W_):
    """
    EvaluateClassifierSVM is a function that computes the score function for all images in the data set.
    :param X_:  N x (d + 1) matrix,  trainning images.
    :param W_: K x (d + 1) matrix. Weights.
    :return score_function_: N x K matrix.
    """
    score_function_ = np.matmul(X_, W_.transpose())
    return score_function_


def ComputeCostSVM(X_, W_, y_, lambda_=0):  
    """
    ComputeCostSVM is a function that computes the full Multiclass SVM loss.
    :param X_:  N x (d + 1) matrix,  trainning images.
    :param W_: K x (d + 1) matrix. Weights.
    :param y_: N x 1 vector containing the trainning set true class.
    :param lambda_: regularization factor in the cost function.
    :return x_, boundary term of each vector.
    :return loss: scalar, loss function for each vector.
    """
    delta= 1.
    score_function = EvaluateClassifierSVM(X_, W_)
    
    x_ = score_function[np.arange(score_function.shape[0]), y_]
    x_ = score_function - x_.reshape(-1,1) + delta
    
    max_ = np.maximum(0 , x_)
    max_[np.arange(max_.shape[0]), y_] = 0
    
    filler_array = np.zeros((W_.shape[0],1))
    
    loss = np.sum(max_)/len(X_) + np.sum(lambda_*np.linalg.norm(np.hstack((W_[:, :-1], filler_array))))
    
    return x_ , loss


def ComputeGradientsSVM(X_, y_, x_, W_, lambda_=0):
    """
    ComputeGradientsSVM is a function that computes the analytic gradient.
    :param X_:  N x (d + 1) matrix,  trainning images.
    :param y_: N x 1 vector containing the trainning set true class.
    :return x_, boundary term of each vector.
    :param W_: K x (d + 1) matrix. Weights.
    :param lambda_: regularization factor in the cost function.
    """
    
    grad_W_L = np.zeros((len(X_.data), W_.shape[0], W_.shape[1]))
    x_[np.arange(x_.shape[0]), y_] = 0
    x_ = np.int64(x_ > 0)
    for i in range(len(X_)):
        grad_W_L[i] = x_[i].reshape(-1,1) * X_[i]
        grad_W_L[i][y_[i]] = -np.sum(x_[i]) * (X_[i])
    
    filler_array = np.zeros((W_.shape[0],1))
    
    grad_W = np.mean(grad_W_L, axis=0) + 2 * lambda_ * np.hstack((W_[:, :-1], filler_array))
    return grad_W


def ComputeGradsNumSVM(X_,W_, y_, h_=1e-6, lambda_=0):
    """
    ComputeGradientsSVM is a function that computes the numerical gradient.
    using Centered difference formula.
    :param X_:  N x (d + 1) matrix,  trainning images.
    :param W_: K x (d + 1) matrix. Weights.
    :param y_: N x 1 vector containing the trainning set true class.
    :param h_: h represents a small change in x, and it can be either positive or negative.
    :param lambda_: regularization parameter.
    """
    no = W_.shape[0]
    d = W_.shape[1] 

    grad_W = np.zeros((no,d))
    
    _, c = ComputeCostSVM(X_, W_, y_, lambda_=0)
    
    for i in range(no):
        for j in range(d):
            W_try = copy.deepcopy(W_)
            W_try[i][j] = W_try[i][j] + h_
            _, c2 = ComputeCostSVM(X_, W_try, y_, lambda_=0)
            grad_W[i][j] = (c2 - c) / h_

    return grad_W


def MiniBatch_SVM(X_, X_val, y_, y_val, GD_params, W_,lambda_=0):
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
    delta = 1.
    
    for epoch in range(GD_params.n_epochs):  
        for j in range(Npts//GD_params.n_batch):
            j_start = j * GD_params.n_batch
            j_end = (j + 1) * GD_params.n_batch
            Xbatch = X_[j_start:j_end]
            Ybatch = y_[j_start:j_end]

            P_ = EvaluateClassifierSVM(Xbatch, W_)
            x_ = P_[np.arange(P_.shape[0]), Ybatch]
            x_ = P_ - x_.reshape(-1,1) + delta
            grad_w = ComputeGradientsSVM(Xbatch, Ybatch, x_, W_, lambda_=lambda_)

            W_ = W_ - GD_params.eta*grad_w
            
        # cost per epoch
        _, trainning_cost = ComputeCostSVM(X_, W_, y_, lambda_=lambda_)
        _, validation_cost = ComputeCostSVM(X_val, W_, y_val, lambda_=lambda_)
        
        # loss per epoch
        _, trainning_loss = ComputeCostSVM(X_, W_, y_, lambda_=0)
        _, validation_loss = ComputeCostSVM(X_val, W_, y_val, lambda_=0)
        
        cf_train.append(trainning_cost)
        cf_val.append(validation_cost)
        
        cl_train.append(trainning_cost)
        cl_val.append(validation_cost)
        
        # accuracy per epoch
        acc_train_ = ComputeAccuracy(X_, y_, W_)
        acc_val_ = ComputeAccuracy(X_val, y_val, W_)
        
        acc_train.append(acc_train_)
        acc_val.append(acc_val_)
        
        
    return W_, cf_train, cf_val, cl_train, cl_val, acc_train, acc_val



def grid_search(X_, Y_, y_, V_, v_y, nb_batchs_and_epochs, lambda_list, eta_list):
    """
    perform grid search and train for a longer time.
    """
    result_list = []
    mu, sigma = 0, 0.01
    W, b = w_b_random_initiation(X_, Y_, mu, sigma)
    transf_trainning_data, transf_W, = bias_trick_score_func(X_, W, b)
    transf_validation_data =np.hstack((V_, np.ones((V_.shape[0],1))))
    
    for nb_batch, nb_epoch in reversed(nb_batchs_and_epochs):
        for lambda_ in lambda_list:
            for eta in eta_list:
                GD_params = params(nb_batch, eta, nb_epoch)
                tic = time.time()
                print('start training for parameters:')
                print(f'nb_batch: {nb_batch} nb_epoch:{nb_epoch}')
                print(f'lambda_: {lambda_} eta:{eta}')

                W_star, cf_train, cf_val, acc_train, acc_val = MiniBatch_SVM(
                    transf_trainning_data, transf_validation_data, y_, v_y,
                    GD_params,transf_W, lambda_=lambda_
                )
                tac = time.time()
                print(f'training with these parameters took {tac - tic} seconds')

                min_val_cf, corresponding_acc = min(
                        [cf_and_acc for cf_and_acc in zip(cf_val, acc_val)],
                        key=lambda x: x[0]
                    )
                print(
                    f'training ended with minimum validation cost: {min_val_cf} and '
                    f' corresponding accuracy: {100 * corresponding_acc} %'
                )
                print(' ')
                print(' ')

                result_list.append([W_star, cf_train, cf_val, acc_train, acc_val, GD_params, lambda_])
                
    return result_list
