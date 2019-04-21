import time
import math
import numpy as np
import pickle
import os
import copy
import matplotlib.pyplot as plt
from matplotlib import pyplot
np.random.seed(400)

def LoadBatch(path, training_batch_name, validation_batch_name, test_batch_name):
    """
    LoadBatch is a function that loads the data and returns an object.
    """
    class parse_file():
        def __init__(self, data, labels, y, raw_data):
            self.data = data
            self.labels = labels
            self.y = y
            self.raw_data = raw_data     
    
    def load_data(filename):
        with open(filename, 'rb') as fo:
            dict_ = pickle.load(fo, encoding='bytes')
        X = ((dict_[b'data']).astype('float32')).transpose()
        Y = np.zeros((len(dict_[b'labels']),len(np.unique(dict_[b'labels']))))   
        Y[np.arange(len(dict_[b'labels'])), dict_[b'labels']] = 1    
        y = np.array(dict_[b'labels'])
        return X, Y, y
    
    def normalize_array(array, mean_, std_):
        return (array - mean_) / std_
    
    # training data
    training_path = os.path.join(path, training_batch_name)
    raw_training_X, training_Y, training_y = load_data(training_path)
    
    # validation data
    validation_path = os.path.join(path, validation_batch_name)
    raw_validation_X, validation_Y, validation_y = load_data(validation_path)
    
    # test data
    test_path = os.path.join(path, test_batch_name)
    raw_test_X, test_Y, test_y = load_data(test_path)
    
    # put training data to zero mean and variance 1.
    training_mean_X = np.mean(raw_training_X, axis=1).reshape(-1,1)
    training_std_X = np.std(raw_training_X, axis=1, ddof=1).reshape(-1, 1)
    training_X = normalize_array(raw_training_X, training_mean_X, training_std_X)
    
    # normalize validation and test data with this mean and variance
    validation_X = normalize_array(raw_validation_X, training_mean_X, training_std_X)
    test_X = normalize_array(raw_test_X, training_mean_X, training_std_X)
    
    # build parser objects
    training_data = parse_file(training_X, training_Y.transpose(), training_y, raw_training_X.transpose())
    validation_data = parse_file(validation_X, validation_Y.transpose(), validation_y, raw_validation_X.transpose())
    test_data = parse_file(test_X, test_Y.transpose(), test_y, raw_test_X.transpose())
         
    return training_data, validation_data, test_data


def LoadBatch_join_and_split(
                            path, 
                            training_batch_name,
                            validation_batch_name,
                            test_batch_name,
                            n_validation_ex
                            ):
    
    class parse_file():
        def __init__(self, data, labels, y, raw_data):
            self.data = data
            self.labels = labels
            self.y = y
            self.raw_data = raw_data 
    
    def load_data(filename):
        with open(filename, 'rb') as fo:
            dict_ = pickle.load(fo, encoding='bytes')
        X = ((dict_[b'data']).astype('float32')).transpose()
        Y = np.zeros((len(dict_[b'labels']),len(np.unique(dict_[b'labels']))))   
        Y[np.arange(len(dict_[b'labels'])), dict_[b'labels']] = 1    
        y = np.array(dict_[b'labels'])
        return X, Y, y

    def normalize_array(array, mean_, std_):
        return (array - mean_) / std_
        # training data
    
    training_path = os.path.join(path, training_batch_name)
    raw_training_X, training_Y, training_y = load_data(training_path)
    
    # validation data
    validation_path = os.path.join(path, validation_batch_name)
    raw_validation_X, validation_Y, validation_y = load_data(validation_path)
    
    # test data
    test_path = os.path.join(path, test_batch_name)
    raw_test_X, test_Y, test_y = load_data(test_path)
    
    #stack data:
    training_X = np.hstack((raw_training_X, raw_validation_X, raw_test_X))
    training_Y = (np.hstack((training_Y.T, validation_Y.T, test_Y.T))).T
    training_y = np.hstack((training_y, validation_y, test_y))
    
    
    sel = np.random.choice(training_X.shape[1], size=n_validation_ex, replace=False)
    unsel2 = np.setdiff1d(np.arange(training_X.shape[1]), sel)
    new_training_X = training_X[:,unsel2]
    new_training_Y = training_Y[unsel2,:]
    new_training_y = training_y[unsel2]

    new_val_X = training_X[:,sel]
    new_val_Y = training_Y[sel,:]
    new_val_y = training_y[sel]
    
    # put training data to zero mean and variance 1.
    training_mean_X = np.mean(new_training_X, axis=1).reshape(-1,1)
    training_std_X = np.std(new_training_X, axis=1, ddof=1).reshape(-1, 1)
    training_X = normalize_array(new_training_X, training_mean_X, training_std_X)
    
    # normalize validation data with this mean and variance
    validation_X = normalize_array(new_val_X, training_mean_X, training_std_X)
    
    # build parser objects
    training_data = parse_file(training_X, new_training_Y.transpose(), new_training_y, new_training_X.transpose())
    #   validation_data = parse_file(validation_X, validation_Y.transpose(), validation_y, raw_validation_X.transpose())
    validation_data = parse_file(validation_X, new_val_Y.transpose(), new_val_y, new_val_X.transpose())
         
    return training_data, validation_data

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

def ReLU(s_1):
    h = np.maximum(s_1, 0)
    return h

def SOFTMAX(s_):
    """ 
    :return p: kxN softmax
    """
    p_ =  np.exp(s_) / np.matmul(np.ones((1, s_.shape[0])), np.exp(s_))
    return p_

def cross_entropy_loss(Y_, p_):
    """
    :param Y_:
    :param p_:
    :return l_cross_:
    """
    l_cross_ = -np.log(np.sum((np.multiply(Y_, p_)),axis=0))
    return l_cross_

def l_2_norm_squared(array_):
    return np.inner(array_, array_).trace()

def cost_function(Y_, p_, W_1_, W_2_, lambda_):
    """
    :param Y_: kxN matrix, labels for the dataset images.
    :param p_: kxN matrix softmax
    :param W_1_: mxd matrix.
    :param W_2_: kxm matrix.
    :param lambda_: regularization term.
    """
    J_ = np.sum(cross_entropy_loss(Y_, p_)) / Y_.shape[1] + lambda_ * (l_2_norm_squared(W_1_) + l_2_norm_squared(W_2_))
    return J_

def ComputeCost(s_, Y_, W_1_, W_2_, lambda_):
    
    P_= SOFTMAX(s_)
    c = cost_function(Y_, P_, W_1_, W_2_, lambda_) 
    return c

def forward_pass(X_, W_1_, W_2_, b_1_, b_2_):
    Npts = X_.shape[1]
    ind_size_n = np.ones((Npts,1))
    S_1_ = np.matmul(W_1_, X_) + np.matmul(b_1_, ind_size_n.T)
    H = ReLU(S_1_)
    s = np.matmul(W_2_, H) + np.matmul(b_2_, ind_size_n.T)
    p = SOFTMAX(s)
    return p

def full_forward_and_cost(X_, Y_, W_1_, W_2_, b_1_, b_2_, lambda_):
    ind_size_n = np.ones((X_.shape[1],1))
    S_1_ = np.matmul(W_1_, X_) + np.matmul(b_1_, ind_size_n.T)
    H = ReLU(S_1_)
    s = np.matmul(W_2_, H) + np.matmul(b_2_, ind_size_n.T)
    p = SOFTMAX(s)
    c = cost_function(Y_, p, W_1_, W_2_, lambda_) 
    return c

def ComputeAccuracy(X_, y_, W_1_, W_2_, b_1_, b_2_):
    """
    :param X_:  N x d matrix,  trainning images.
    :param y_:  N x 1 vector containing the trainning set true class.
    :param W_: K x d matrix. Weights.
    :param b_:  K x 1 vector. Bias.
    :return acc: scalar.
    """
    Npts = X_.shape[1]
    ind_size_n = np.ones((Npts,1))
    #S_1 = mxN matrix.
    S_1 = np.matmul(W_1_, X_) + np.matmul(b_1_, ind_size_n.T)
    #H = mxN matrix.
    H = ReLU(S_1)
    #S_2 = kxN matrix.
    S_2 = np.matmul(W_2_, H) + np.matmul(b_2_, ind_size_n.T)
    #P = kxN matrix.
    P = SOFTMAX(S_2)
    P_ = np.argmax(P,axis=0)
    acc = np.count_nonzero(y_== P_) / float(len(P_))
    
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
    
    max_relative_error = np.amax(np.absolute(g_a-g_n) / max_value)
    
    return max_relative_error

def cyclical_learning_rates(eta_min, eta_max, n_s, n_cycles):
    """
    n_s: stepsize per cycle
    n_cycles: number of cycles
    """
    array_eta_val = np.zeros((1))
    cycle_ = np.hstack(
             (np.linspace(eta_min, eta_max, n_s+1),
             (np.linspace(eta_max, eta_min, n_s+1)[1:-1])))     
    for cycle in range(n_cycles):
        array_eta_val = np.hstack((array_eta_val, cycle_))
    array_eta_val = array_eta_val[1:]    
    array_eta_val = np.hstack((array_eta_val, eta_min))
    return array_eta_val

def ComputeGradsAnalt(X_, Y_, b_1_, b_2_, W_1_, W_2_, lambda_):

    Npts = X_.shape[1]
    ind_size_n = np.ones((Npts,1))
    
    #forward pass:
    S_1_ = np.matmul(W_1_, X_) + np.matmul(b_1_, ind_size_n.T)
    H = ReLU(S_1_)
    s = np.matmul(W_2_, H) + np.matmul(b_2_, ind_size_n.T)
    P = SOFTMAX(s)
    
    G = -(Y_ - P)

    grad_W_2_L = np.matmul(G, H.T) / float(Npts)
    grad_b_2_L = np.matmul(G, ind_size_n) / float(Npts)

    #Propagate the gradient back through the second layer
    # Backward pass:
    # G = mxN
    G = np.matmul(W_2_.T, G)

    #Compute indicator fuction:
    indicator_f_H = np.sign(H)

    G = G * indicator_f_H
    G[G == -0.0] = 0.0

    grad_W_1_L = np.matmul(G, X_.T) / float(Npts)
    grad_b_1_L = np.matmul(G, ind_size_n) / float(Npts)

    #Compute cost function gradients:
    grad_W_1_J = grad_W_1_L + 2 * (np.multiply(lambda_, W_1_))
    grad_b_1_J = grad_b_1_L

    grad_W_2_J = grad_W_2_L  + 2 * (np.multiply(lambda_, W_2_))
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
        
    #forward pass:
    c = full_forward_and_cost(X_, Y_, W_1_, W_2_, b_1_, b_2_, lambda_)

    for i in range(len(b_1_)):
        b_try = copy.deepcopy(b_1_)
        b_try[i] = b_try[i] + h_
        c2 = full_forward_and_cost(X_, Y_, W_1_, W_2_, b_try, b_2_, lambda_)
        grad_b_1[i] = (c2-c) / h_
    
    for i in range(len(b_2_)):
        b_try = copy.deepcopy(b_2_)
        b_try[i] = b_try[i] + h_
        c2 = full_forward_and_cost(X_, Y_, W_1_, W_2_, b_1_, b_try, lambda_)
        grad_b_2[i] = (c2-c) / h_
        
    for i in range(m):
        for j in range(d):
            W_try = copy.deepcopy(W_1_)
            W_try[i][j] = W_try[i][j] + h_
            c2 = full_forward_and_cost(X_, Y_, W_try, W_2_, b_1_, b_2_, lambda_)
            grad_W_1[i][j] = (c2-c) / h_
    
    for i in range(k):
        for j in range(m):
            W_try = copy.deepcopy(W_2_)
            W_try[i][j] = W_try[i][j] + h_
            c2 = full_forward_and_cost(X_, Y_, W_1_, W_try, b_1_, b_2_, lambda_)
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
        c1 = full_forward_and_cost(X_, Y_, W_1_, W_2_, b_try, b_2_, lambda_)
        b_try = copy.deepcopy(b_1_)
        b_try[i] = b_try[i] + h_
        c2 = full_forward_and_cost(X_, Y_, W_1_, W_2_, b_try, b_2_, lambda_)
        grad_b_1[i] = (c2-c1) / (2*h_)

    for i in range(len(b_2_)):
        b_try = copy.deepcopy(b_2_)
        b_try[i] = b_try[i] - h_
        c1 =full_forward_and_cost(X_, Y_, W_1_, W_2_, b_1_, b_try, lambda_)
        b_try = copy.deepcopy(b_2_)
        b_try[i] = b_try[i] + h_
        c2 = full_forward_and_cost(X_, Y_, W_1_, W_2_, b_1_, b_try, lambda_)
        grad_b_2[i] = (c2-c1) / (2*h_)
        
    for i in range(m):
        for j in range(d):
            W_try = copy.deepcopy(W_1_)
            W_try[i][j] = W_try[i][j] - h_
            c1 = full_forward_and_cost(X_, Y_, W_try, W_2_, b_1_, b_2_, lambda_)
            W_try = copy.deepcopy(W_1_)
            W_try[i][j] = W_try[i][j] + h_
            c2 = full_forward_and_cost(X_, Y_, W_try, W_2_, b_1_, b_2_, lambda_)
            grad_W_1[i][j]= (c2-c1) / (2*h_)
    
    for i in range(k):
        for j in range(m):
            W_try = copy.deepcopy(W_2_)
            W_try[i][j] = W_try[i][j] - h_
            c1 = full_forward_and_cost(X_, Y_, W_1_, W_try, b_1_, b_2_, lambda_)
            W_try =copy.deepcopy(W_2_)
            W_try[i][j] = W_try[i][j] + h_
            c2 = full_forward_and_cost(X_, Y_, W_1_, W_try, b_1_, b_2_, lambda_)
            grad_W_2[i][j]= (c2-c1) / (2*h_)

    return grad_b_1, grad_b_2, grad_W_1, grad_W_2

def MiniBatchGD(hidden_layer_neuron_nb,
                n_computations,
                eta_min,
                eta_max,
                n_s,
                n_cycles,
                trainning_data_,
                validation_data_,
                n_batch,
                n_epochs,
                lambda_):
    
    
    X_, Y_, y_ = trainning_data_.data, trainning_data_.labels, trainning_data_.y
    X_val, Y_val, y_val = validation_data_.data, validation_data_.labels, validation_data_.y
    GD_params = params(n_batch, 0, n_epochs)
    cf_train = {}
    cf_val = {}
    cl_train = {}
    cl_val = {}
    acc_train = {}
    acc_val = {}
    
    Npts = X_.shape[1]

    array_eta = cyclical_learning_rates(eta_min, eta_max, n_s, n_cycles)
    
    total_n_steps = GD_params.n_epochs * (Npts // GD_params.n_batch)
    W_1, b_1, W_2, b_2 = init_two_layers_w_b_param(X_, Y_, hidden_layer_neuron_nb)
    step_counter = 0
    compute_l_c_a_counter = 0
    compute_l_c_a_init =  int(total_n_steps/n_computations)
   
    for epoch in range(int(GD_params.n_epochs)):
        
        for j in range(Npts // GD_params.n_batch):
        
            GD_params.eta = array_eta[step_counter]
            j_start = j * GD_params.n_batch
            j_end = (j + 1) * GD_params.n_batch
            Xbatch = X_[:, j_start:j_end]
            Ybatch = Y_[:, j_start:j_end]

            grad_b_1_t, grad_b_2_t, grad_W_1_t, grad_W_2_t = ComputeGradsAnalt(
                Xbatch, Ybatch, b_1, b_2, W_1, W_2, lambda_
            )
            W_1 = W_1 - GD_params.eta * grad_W_1_t
            b_1 = b_1 - GD_params.eta * grad_b_1_t
            W_2 = W_2 - GD_params.eta * grad_W_2_t
            b_2 = b_2 - GD_params.eta * grad_b_2_t

            # cost per epoch
        
            if step_counter == compute_l_c_a_counter:
                trainning_cost = full_forward_and_cost(X_, Y_, W_1, W_2, b_1, b_2, lambda_)
                validation_cost =full_forward_and_cost(X_val, Y_val, W_1, W_2, b_1, b_2, lambda_)

                # # loss per epoch
                trainning_loss = full_forward_and_cost(X_, Y_, W_1, W_2, b_1, b_2, 0)
                validation_loss = full_forward_and_cost(X_val, Y_val, W_1, W_2, b_1, b_2, 0)

                cf_train[step_counter] = trainning_cost
                cf_val[step_counter]=validation_cost

                cl_train[step_counter]=trainning_loss
                cl_val[step_counter]=validation_loss

                # accuracy per epoch
                acc_train_ = ComputeAccuracy(X_, y_, W_1, W_2, b_1, b_2)
                acc_val_ = ComputeAccuracy(X_val, y_val, W_1, W_2, b_1, b_2)

                acc_train[step_counter] = acc_train_
                acc_val[step_counter]=acc_val_
                
                compute_l_c_a_counter = compute_l_c_a_counter + compute_l_c_a_init
            
            step_counter = step_counter + 1
    
    return W_1, b_1, W_2, b_2, cf_train, cf_val, cl_train, cl_val, acc_train, acc_val

def coarse_search(training_data_,
                  validation_data_,
                  l_min, l_max,
                  eta_min, eta_max,
                  n_batch, n_cycles,
                  n_computations, k,
                  hidden_layer_neuron_nb=50):
    
    #generate random regularization terms:
    lambdas_list = [10**(l_min + (l_max - l_min)*np.random.rand(1, 1)) for i in range(20)]
    
    #initialize weights and bias:
    W_1, b_1, W_2, b_2 = init_two_layers_w_b_param(training_data_.data,
                                                     training_data_.labels,
                                                     hidden_layer_neuron_nb)
    Npts = training_data_.data.shape[1]
    n_s =  2 * (k * (Npts/n_batch))
    total_n_steps = n_cycles * n_s
    n_epochs = total_n_steps / (Npts/n_batch)
    GD_params = params(n_batch, 0, n_epochs)
    result_list = []    
    for lambda_ in lambdas_list:
        tic = time.time()  
        print('start training for parameters:')
        W_1, b_1, W_2, b_2, cf_train, cf_val, cl_train, cl_val, acc_train, acc_val = MiniBatchGD(
                                                                        hidden_layer_neuron_nb,
                                                                        n_computations,
                                                                        eta_min,
                                                                        eta_max,
                                                                        n_s,
                                                                        n_cycles,
                                                                        training_data_,
                                                                        validation_data_,
                                                                        n_batch,
                                                                        n_epochs,
                                                                        lambda_)

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

        result_list.append(
            [W_1, b_1, W_2, b_2, cf_train, cf_val, cl_train, cl_val, acc_train, acc_val, GD_params, lambda_]
        )

    return result_list


def MiniBatchGD_(hidden_layer_neuron_nb,
                n_computations,
                eta_min,
                eta_max,
                n_s,
                n_cycles,
                trainning_data_,
                validation_data_,
                GD_params,
                lambda_):
    
    
    X_, Y_, y_ = trainning_data_.data, trainning_data_.labels, trainning_data_.y
    X_val, Y_val, y_val = validation_data_.data, validation_data_.labels, validation_data_.y
    
    cf_train = {}
    cf_val = {}
    cl_train = {}
    cl_val = {}
    acc_train = {}
    acc_val = {}
    
    Npts = X_.shape[1]

    array_eta = cyclical_learning_rates(eta_min, eta_max, n_s, n_cycles)
    
#     n_s = k(Npts//GD_params.n_batch)
#     n_cycles = total_n_steps // n_s
#     k = GD_params.n_epochs // n_cycles
    
    total_n_steps = GD_params.n_epochs * (Npts // GD_params.n_batch)
    W_1, b_1, W_2, b_2 = init_two_layers_w_b_param(X_, Y_, hidden_layer_neuron_nb)
    step_counter = 0
    compute_l_c_a_counter = 0
    compute_l_c_a_init =  int(total_n_steps/n_computations)
    for epoch in range(GD_params.n_epochs):
        
        for j in range(Npts // GD_params.n_batch):
        
            GD_params.eta = array_eta[step_counter]
            j_start = j * GD_params.n_batch
            j_end = (j + 1) * GD_params.n_batch
            Xbatch = X_[:, j_start:j_end]
            Ybatch = Y_[:, j_start:j_end]

            grad_b_1_t, grad_b_2_t, grad_W_1_t, grad_W_2_t = ComputeGradsAnalt(
                Xbatch, Ybatch, b_1, b_2, W_1, W_2, lambda_
            )
            W_1 = W_1 - GD_params.eta * grad_W_1_t
            b_1 = b_1 - GD_params.eta * grad_b_1_t
            W_2 = W_2 - GD_params.eta * grad_W_2_t
            b_2 = b_2 - GD_params.eta * grad_b_2_t

            # cost per epoch
        
            if step_counter == compute_l_c_a_counter:
                trainning_cost = full_forward_and_cost(X_, Y_, W_1, W_2, b_1, b_2, lambda_)
                validation_cost =full_forward_and_cost(X_val, Y_val, W_1, W_2, b_1, b_2, lambda_)

                # # loss per epoch
                trainning_loss = full_forward_and_cost(X_, Y_, W_1, W_2, b_1, b_2, 0)
                validation_loss = full_forward_and_cost(X_val, Y_val, W_1, W_2, b_1, b_2, 0)

                cf_train[step_counter] = trainning_cost
                cf_val[step_counter]=validation_cost

                cl_train[step_counter]=trainning_loss
                cl_val[step_counter]=validation_loss

                # accuracy per epoch
                acc_train_ = ComputeAccuracy(X_, y_, W_1, W_2, b_1, b_2)
                acc_val_ = ComputeAccuracy(X_val, y_val, W_1, W_2, b_1, b_2)

                acc_train[step_counter] = acc_train_
                acc_val[step_counter]=acc_val_
                
                compute_l_c_a_counter = compute_l_c_a_counter + compute_l_c_a_init
            
            step_counter = step_counter + 1
    
    return W_1, b_1, W_2, b_2, cf_train, cf_val, cl_train, cl_val, acc_train, acc_val


def plot_cf_loss_acc(
                                    validation_list_cf_,
                                    trainning_list_cf_,
                                    acc_train_,
                                    acc_val_,
                                    cl_train_,
                                    cl_val_,
                                    GD_params_,
                                    lambda_,
                                    out_filename='none',
                                    label1 = 'trainning',
                                    label2 = 'validation'
                                    ):
    """
    plot_validation_trainning_cf_acc plots the loss and accuracy for both, validation and trainning
    set for a given number of epochs.
    """
    t=range(len(validation_list_cf_))
    pyplot.figure(figsize=(15,5))
    pyplot.subplots_adjust(wspace=0.3)
    pyplot.suptitle(f'batch_n = {GD_params_.n_batch}, lambda = {lambda_}', size =16)
    pyplot.style.use('seaborn-darkgrid')
    # sp1
    pyplot.subplot(131)
    pyplot.plot(t, validation_list_cf_, '#4363d8', label = label2+' cost')
    pyplot.plot(t, trainning_list_cf_, '#3cb44b', label = label1+' cost')
    pyplot.legend(loc='best')
    pyplot.xlabel('epoch', size = 13.5)
    pyplot.ylabel('Cost', size = 13.5)
    pyplot.title('Cost',size = 14)
    # sp2
    pyplot.subplot(132)
    pyplot.plot(t, cl_val_, '#4363d8', label = label2+' loss')
    pyplot.plot(t, cl_train_, '#3cb44b', label = label1+' loss')
    pyplot.legend(loc='best')
    pyplot.xlabel('epoch', size = 13.5)
    pyplot.ylabel('Loss', size = 13.5)
    pyplot.title('Loss', size = 14)
    # sp3
    pyplot.subplot(133)
    pyplot.plot(acc_train_,'#3cb44b', label =  label1+' acc')
    pyplot.plot(acc_val_, '#4363d8',  label= label2+' acc')
    pyplot.legend(loc='best')
    pyplot.xlabel('epoch', size = 13.5)
    pyplot.ylabel('Accuracy', size = 13.5)
    pyplot.title('Accuracy', size = 14)
    if out_filename=='none':
        pyplot.show()
    else:
        pyplot.savefig(out_filename)
        pyplot.show()

def plot_cf_loss_acc_vs_step_counter(
                                    validation_list_cf_,
                                    trainning_list_cf_,
                                    acc_train_,
                                    acc_val_,
                                    cl_train_,
                                    cl_val_,
                                    GD_params_,
                                    lambda_,
                                    out_filename='none',
                                    label1 = 'trainning',
                                    label2 = 'validation',
                                    label3 = 'update step',
                                    cycles_n =1
                                    ):
    """
    plot_validation_trainning_cf_acc plots the loss and accuracy for both, validation and trainning
    set for a given number of epochs.
    """
    pyplot.figure(figsize=(15,5))
    pyplot.subplots_adjust(wspace=0.3)
    pyplot.suptitle(f'batch_n = {GD_params_.n_batch},lambda = {lambda_}, cycles = {cycles_n}', size =16,)
    pyplot.style.use('seaborn-darkgrid')
    # sp1
    pyplot.subplot(131)
    pyplot.plot(list(validation_list_cf_.keys()), list(validation_list_cf_.values()) ,'#4363d8', label = label2+' cost')
    pyplot.plot(list(trainning_list_cf_.keys()),list(trainning_list_cf_.values()) ,'#3cb44b', label = label1+' cost')
    pyplot.legend(loc='best')
    pyplot.xlabel(label3, size = 13.5)
    pyplot.ylabel('Cost', size = 13.5)
    pyplot.ylim(0, 4)
    #pyplot.xlim(0.0, 1000)
    pyplot.title('Cost',size = 14)
    # sp2
    pyplot.subplot(132)
    pyplot.plot(list(cl_val_.keys()), list(cl_val_.values()), '#4363d8', label = label2+' loss')
    pyplot.plot(list(cl_train_.keys()),list(cl_train_.values()) , '#3cb44b', label = label1+' loss')
    pyplot.legend(loc='best')
    pyplot.xlabel(label3, size = 13.5)
    pyplot.ylabel('Loss', size = 13.5)
    pyplot.ylim(0.0, 3)
    #pyplot.xlim(0.0, 1000)
    pyplot.title('Loss', size = 14)
    # sp3
    pyplot.subplot(133)
    pyplot.plot(list(acc_train_.keys()), list(acc_train_.values()),'#3cb44b', label =  label1+' acc')
    pyplot.plot(list(acc_val_.keys()), list(acc_val_.values()), '#4363d8',  label= label2+' acc')
    pyplot.legend(loc='best')
    pyplot.xlabel(label3, size = 13.5)
    pyplot.ylabel('Accuracy', size = 13.5)
    #pyplot.ylim(0.0, 0.6)
    pyplot.title('Accuracy', size = 14)
    if out_filename=='none':
        pyplot.show()
    else:
        pyplot.savefig(out_filename)
        pyplot.show()


