import numpy as np
from scipy.special import softmax
import time
import matplotlib.pyplot as plt

def one_hot_encoding(x_vector, K, char_encoding):
    X = np.zeros((K ,len(x_vector)))
    x_encoding = [char_encoding[i] for i in x_vector]
    for idx, x in enumerate(x_encoding):
        X[x,idx]=1
    return X

def init_param(book_data, m, eta, seq_length, sig):
    data = {}
    model_param= {}
    data['book_data'] = book_data
    data['K'] = len(''.join(sorted(list(set(data['book_data'])))))
    model_param['m'] = m
    model_param['eta'] = eta
    model_param['seq_length'] = seq_length
    model_param['sig'] = sig
    h_0 = np.zeros((model_param['m'],1))
    return data, model_param

def init_weights(model_param, data):
    weights = {}
    weights['V'] = np.random.normal(loc=0.0, scale=model_param['sig'],
                                       size=( data['K'], model_param['m']))  #Kxm
    weights['W'] = np.random.normal(loc=0.0, scale= model_param['sig'],
                                       size=(model_param['m'], model_param['m'])) #mxm
    weights['U'] = np.random.normal(loc=0.0, scale= model_param['sig'], 
                                       size=( model_param['m'], data['K'])) #mxK
    weights['c'] = np.zeros((data['K'],1)) #Kx1
    weights['b'] = np.zeros((model_param['m'],1)) #mx1 
    return weights

def synthesize_char_sequence(x0, h_0, weights, n):
    h, x = h_0, x0
    xnext = np.zeros((x0.shape[0], n))
    for i in range(n):
        a =  np.dot(weights['W'], h) + np.dot(weights['U'], x) + weights['b'] #mx1
        h = np.tanh(a) #mx1
        o = np.dot(weights['V'], h) + weights['c'] #Kx1
        p = softmax(o)
        cp = np.cumsum(p) #k,
        a = np.random.uniform(0, 1, 1)
        ixs = np.where(cp - a > 0) #idx list
        ii = ixs[0][0]
        xnext[ii,i] = 1
        x = xnext[:, i].reshape(-1, 1) #Kx1
    return xnext

def compute_loss(weights, context_vect, X, Y):
    ce_loss = 0
    a_t, h_t, o_t, p_t = {}, {}, {}, {}
    h_t[-1] = context_vect
    n_T = X.shape[1]
    for t in range(n_T):
        a_t[t] = np.dot(weights['W'], h_t[t-1]) + np.dot(weights['U'], X[:,t].reshape(-1,1)) + weights['b'] #mx1
        h_t[t] = np.tanh(a_t[t]) #mx1
        o_t[t] = np.dot(weights['V'], h_t[t]) + weights['c'] #Kx1
        p_t[t] = np.exp(o_t[t]) / np.matmul(np.ones((1, o_t[t].shape[0])), np.exp(o_t[t])) #Kx1
        ce_loss += -np.log(np.dot(Y[:, t].reshape(1, -1), p_t[t]))
    return h_t, p_t, ce_loss[0]

def compute_gradients(weights, context_vect, X_, Y_):
    #total number of states t
    n_T = X_.shape[1]
    #compute loss
    h_t, p_t, ce_loss = compute_loss(weights, context_vect, X_, Y_)
    
    # backward pass
    d_b, d_c = np.zeros_like(weights['b']), np.zeros_like(weights['c'])
    d_ot, d_at, d_ht = {}, {}, {}
    grads = {}
    grads['V'] = 0
    grads['W'] = 0
    grads['U'] = 0
    grads['c'] = np.zeros_like(weights['c']) #Kx1
    grads['b'] = np.zeros_like(weights['b'])
    
    #as d_ht at time T = d_ot[t] * V, I initiallize d_a_next (d_at[t+1]) at zero,
    #and later assign d_a_next  = d_at[t] 
    #this as d_ht at time t = d_ot[t] * V + d_at[t+1] * W 
    d_a_next = np.zeros((model_param['m'], 1))
    
    # we need to iterate backwards as for computing d_ht[t] we need d_at[t+1]
    for t in reversed(range(n_T)):
        d_ot[t] = -(Y_[:,t].reshape(-1, 1) - p_t[t]) #Kx1
        grads['V'] += np.dot(d_ot[t], h_t[t].T) #Kxm
        grads['c'] = grads['c'] + d_ot[t] #Kx1
        d_ht[t] = np.dot(d_ot[t].T, weights['V']).T + np.dot(weights['W'].T, d_a_next) #mx1
        d_at[t] = d_ht[t] * (1 - (h_t[t] * h_t[t])) #mx1
        d_a_next = d_at[t]
        grads['b'] =  grads['b'] + d_at[t] #mx1
        grads['W'] += np.dot(d_at[t], h_t[t-1].T) #mxm
        grads['U'] += np.dot(d_at[t], X_[:,t].reshape(1,-1)) #mxk
    new_context_vector = h_t[n_T - 1].copy()
    
    # to avoid the exploding gradient
    for keys, gradients in grads.items():
        gradients = np.clip(gradients, -5, 5, out=gradients)
    
    return ce_loss[0], grads, new_context_vector

def NumericalGradients(weights_,X_,Y_):
    """
    Numerical gradients using Finite difference Formula.
    """
    h_0 = np.zeros((model_param['m'],1))
    h=1e-4
    numerical_gradients = {}
    for key, weight in weights_.items():
        numerical_gradients[key] = np.zeros_like(weight)
        #each weight(key) has a different dimension
        for i in range(weights_[key].shape[0]):
            for j in range(weights_[key].shape[1]):
                # + h, I could have deep copied 
                weights_[key][i][j] += h
                _, _, l2 = compute_loss(weights_, h_0, X_, Y_)
                # as I have previously modified the weights(+h) I need to substract twice - h
                weights_[key][i][j] -= 2*h
                _, _, l1 = compute_loss(weights_, h_0, X_, Y_)
                #initial weights
                weights_[key][i][j] += h
                numerical_gradients[key][i][j] = (l2-l1) / (2*h)
    return numerical_gradients

def VerifyGradients(analytical_grad_dictionary, numerical_grad_dictionary):
    max_relative_error = {}
    eps = np.finfo(np.float32).eps
    for (key_i,analit_weight), (key_j,num_weight) in zip(analytical_grad_dictionary.items(), numerical_grad_dictionary.items()):
        max_value = np.maximum(eps, np.absolute(analit_weight) + np.absolute(num_weight))
        max_relative_error[key_i] = np.amax(np.absolute(analit_weight-num_weight) / max_value)
    return max_relative_error

def train_model(data, model_param, weights, n_epochs = 10, update_step = 10000, n_=200, save_file = 'none'):
    
    book_data = data['book_data']
    seq_length = model_param['seq_length']
    K = data['K']
    m = model_param['m']
    eta = model_param['eta']
    text_book_length = len(book_data)
    iter_per_epoch= text_book_length // seq_length
    total_num_iterations = n_epochs*iter_per_epoch
    
    m_weights = {}
    for weight_key, weight_matrix in weights.items():
        m_weights[weight_key] = np.zeros_like(weight_matrix)

    train_loss = {}
    smooth_loss = {}
    best_model = {}
    synth = {}
    best_smooth_loss = 43
    e = 0
    h_0 = np.zeros((m, 1))
    
    print(f'epochs = {n_epochs}')
    tic = time.time() 
    for iteration in range(total_num_iterations):

        if iteration == 0 or e > (text_book_length - seq_length - 1):
            h_0 = np.zeros((m, 1))
            e = 0

        #prepare data sequence
        X_chars = book_data[e: e + seq_length]
        Y_chars = book_data[e + 1: e + seq_length + 1]

        X = one_hot_encoding(X_chars, K, char_to_int) #Kxseq_length
        Y = one_hot_encoding(Y_chars, K, char_to_int) #Kxseq_length

        #compute loss and grads
        loss, grads, new_h_0 = compute_gradients(weights, h_0, X, Y)

        smooth_loss_ = loss if e == 0 or iteration == 0 else 0.999 * smooth_loss_ + 0.001 * loss
       
        train_loss[iteration] = loss
        smooth_loss[iteration] = smooth_loss_
        
        if smooth_loss_ < best_smooth_loss:
                best_smooth_loss = smooth_loss_
                best_model['loss'] = loss
                best_model['smooth_loss'] = smooth_loss_
                best_model['pred_text'] = xnext
                best_model['grads'] = grads
                best_model['weights'] = weights
                best_model['h_0'] = h_0
                
        if (iteration - 1) % (update_step) == 0:
            print('-------')
            print(f'iteration = {iteration} / {total_num_iterations}')
            print(f'Smooth loss = {float(smooth_loss_)}')
            print('-------')
            xnext = synthesize_char_sequence(X[:,[0]], h_0, weights, n=n_)
            synth[iteration] = xnext
            print(''.join(int_to_char[i] for i in np.argmax(xnext, axis=0)))

        #apply adaGrad Algorithm
        #It adapts the learning rate to the parameters, performing smaller updates
        for key, weight in weights.items():
            m_weights[key] += grads[key]**2
            weights[key] = weights[key] -  eta  * grads[key] / np.sqrt(m_weights[key] + 1e-8)

        e += seq_length

        # new step context update
        h_0 = new_h_0
    tac = time.time() 
    print(f'training took {tac - tic} seconds')
    
    if save_file == 'none':
        return train_loss, smooth_loss, best_model
    
    np.savez( save_file, train_loss=train_loss, smooth_loss=smooth_loss, best_model=best_model)
    return train_loss, smooth_loss, best_model

