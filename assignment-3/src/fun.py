import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools, random, copy, time, os
from collections import Counter
import pandas as pd

class ConvNet:
    def __init__(self, data, model_param, weights_init):
        self.F = {}
        for i in range(len(weights_init['F'])):   
            self.F[i] = np.random.normal(loc=0.0, scale=weights_init['scale'][i], size=weights_init['F'][i])
        self.W = np.random.normal(loc=0.0, scale=weights_init['scale']['W'], size=weights_init['W'])
        
        self.model_param = model_param
        self.data = data
        
#one hot encoding: Nxk, hard labels.
def one_hot_encoding(data_set_idx, labels_list,  K):
    onehot_encoded = np.zeros((len(data_set_idx), K))  
    for idx, value in enumerate(labels_list):
        one_hot = [0] * K
        one_hot[value] = 1
        onehot_encoded[idx] = one_hot
    return onehot_encoded.T

def vectorize_input(names_, d, n_len, char_to_int):
    #X_vectorized dimension: d*lenxN
    #x input: one hot encoding, each input will be a vector.
    names_ = [name.casefold() for name in names_]
    name_encoded = dict((idx, [char_to_int[ch] for ch in name]) for idx, name in enumerate(names_))
    num_names = len(names_)
    X_vectorized = np.zeros((d*n_len, num_names))
    for key_name_idx, name_idx_encoding in name_encoded.items():
        name_matrix = np.zeros((d, n_len))
        for idx, value in enumerate(name_idx_encoding):
            letter = [0] * d
            letter[value] = 1
            name_matrix[:,idx] = letter
        X_vectorized[:,[key_name_idx]] = name_matrix.T.reshape(-1,1) #reshape for flatten
    return X_vectorized

def set_weights_parameters(avg_names_length = avg_names_length,
                           n_1=5, k_1=5, n_len=n_len, n_2=5, k_2=5, 
                           eta=0.001, rho=.9, batch_size=100, 
                           epochs=10, n_updates=500): 
    
    """
    n_1: number of filters first layer.
    k_1: width of the filter first layer.
    n_len: len of longer name max(names_length).
    n_2: number of filters second layer.
    k_2: width of the filter second layer.
    """
    #1 filter: number and width of filters first layer:
    #filter size = dxk_1 = 28*5
    n_len_1 = n_len - k_1 + 1
    sig_1 = np.sqrt(2. / avg_names_length) #He initialization 
    
    #2 filter: number and width of filters second layer:
    #filter size = n_1*k_2 = 5*5
    n_len_2 = n_len_1-k_2 + 1
    sig_2 = np.sqrt(2. / n_1 * k_2) #He initialization

    #Weight matrix for fully connected layer:
    f_size = n_2 * n_len_2
    sig_3 = np.sqrt(2. / f_size) #He initialization

    F = {}
    weights_init = {'F' :{0: [d, k_1, n_1], 1: [n_1, k_2, n_2] },
                    'W':[K, f_size],
                   'scale': {0:sig_1, 1:sig_2, 'W':sig_3 }}

    model_param = {'eta': eta, #learning rate
                   'rho': rho, #momentum
                   'batch_size': batch_size,
                   'epochs': epochs,
                   'n_updates': n_updates}
    return weights_init, model_param

def build_matrix_from_filters(filters_array, nb_columns_inputs):
    
    nb_rows, filter_width, nb_filters = filters_array.shape
    nb_columns_after_conv = nb_columns_inputs - filter_width + 1
    
    # as numpy.reshape stacks rows by rows from left to right, need to transpose to stack columns by columns
    # e.g if the filter dimenson = 28x5, and we have 5 filters we end up with 5x140, one vector per filter.
    reshaped_filters = filters_array.transpose().reshape((nb_filters, -1))
    #e.g if the filter dimension = 28x5 the flattened filter will be 140
    flattened_filter_size = reshaped_filters.shape[1]
    
    # we fill in the conv_filter matrix by stacking reshaped_filters in the right place:
    #the nb_columns_after_conv will be the number of times we perform the convolutional operation
    #in this case the filter has as many rows as the input so it makes it easier as we only have
    #to make it fit "nb_columns_after_conv" with a stride of 1.
    #e.g if the image is 28x19 and the filter is 28x5 the nb_columns_after_conv will be 15, as we 
    #will perform the operation 15 times.
    # we will fill the matrix by number of filters in this case 5 by 5 so we will perform the operation
    #15 times = which equals 75 rows.
    
    #we will stack results in convultion_matrix_from_filters with 
    #dimension 75 x 28*19 = 75 x (vectorized input = 532 or =>
    #((nb_input_columns - nb_filters)x nb_input_rows) + (filter_rows x filter_columsd) = (19-5)*28 + 140= 532)
    #as we use a stride of 1 every 5(filters) rows we will move 140 positions as this is the filter dimension
    convultion_matrix_from_filters = np.zeros((nb_columns_after_conv * nb_filters, nb_columns_inputs * nb_rows))
    
    for convolution_nb in range(nb_columns_after_conv):
        convultion_matrix_from_filters[
            # stacks 'convolution_nb' conv for each filter
            convolution_nb * nb_filters: (convolution_nb + 1) * nb_filters,
            convolution_nb * nb_rows: convolution_nb * nb_rows + flattened_filter_size
        ] = reshaped_filters
    return convultion_matrix_from_filters

def build_convolution_matrix_from_inputs(vectorized_input, d, k, nf):
    #we need to stack column by column. vectorized_input.shape = (d*k , 1)
#     vectorized_input = vectorized_input.T.reshape((-1,1))
    flattened_filter_size = d * k
    # n_len, vectorized_input.shape = (n_columns * d, 1)
    nb_columns = int(vectorized_input.shape[0] / d)  
    convolutions_number = nb_columns - k + 1
    #e.g 5 filters, 15 convolutions, filter column_n = k = 5, filter row_n = 28 = d
    #matrix_MX.shape = (15*5, 5*28*5) = (75, 140*5)
    #15 convolutions per filter(5), by vectorized filter(5*28) by filter (5)
    matrix_MX = np.zeros((nf * convolutions_number, k * nf * d))
    
    #this gives the different slices of vectorized input by convolution (15). 
    #each element should be then 5x28 (filter column_n x filter row_n)
    #e.g: [0:140], [28:168], [56:196], [84:224], [112:256], ...., [392:532]
    inputs_slices_for_convolution = [
        vectorized_input[d * conv_nb: d * conv_nb + flattened_filter_size]
        for conv_nb in range(convolutions_number)
    ]
    
    for full_matrix_row in range(matrix_MX.shape[0]): #matrix_MX.shape[0] = n_filters * n_convolutions
        # number in matrix_MX indices we are looking at.
        # filter_nb is in [0, 1, 2, ..., nf - 1], it translates the filter
        filter_nb = full_matrix_row % nf  
        
        # this will give us the vectorized slice vec(X:i,i+k) to put in the matrix, as
        # matrix_MX.shape[0] = convolutions_number * nf and there are convolutions_number
        # vectorized slices.
        #e.g full_matrix_row (0,...,4 // nf )= 0, (5,...,9 // nf )=1, ... , (70,...,74 // nf )=15
        vectorized_slice_to_put = inputs_slices_for_convolution[full_matrix_row // nf]
        
        matrix_MX[full_matrix_row, filter_nb * flattened_filter_size: (filter_nb + 1) * flattened_filter_size] = (
            vectorized_slice_to_put.reshape(-1)
        )
    return matrix_MX

def ComputeLoss(Ys_batch, P_batch):
    n = Ys_batch.shape[1]
    prediction_labels = np.sum(np.multiply(Ys_batch, P_batch), axis=0)
    return -np.sum(np.log(prediction_labels)) / n

def ForwardPass(network, X_batch):
    """
    X_batch has already been flattened column wise (first column, then second and so on...)
    """
    vectorized_batch_inputs = dict()
    filter_convolution_matrices = dict()
    nb_layers = len(network.F)
    
    # first conv layer
    n_len = X_batch.shape[0] // network.F[0].shape[0]
    filter_convolution_matrices[0] = build_matrix_from_filters(network.F[0], n_len)
    vectorized_first_convolution_array = np.maximum(0., np.matmul(filter_convolution_matrices[0], X_batch))
    vectorized_batch_inputs[0] = vectorized_first_convolution_array
    
    # second conv layer
    n_len_1 = vectorized_first_convolution_array.shape[0] // network.F[0].shape[2]
    filter_convolution_matrices[1] = build_matrix_from_filters(network.F[1], n_len_1)
    vectorized_second_convolution_array = np.maximum(
        0., np.matmul(filter_convolution_matrices[1], vectorized_first_convolution_array)
    )
    vectorized_batch_inputs[1] = vectorized_second_convolution_array
    
    # last NN layer
    s_batch = np.matmul(network.W, vectorized_second_convolution_array)
    p_batch = softmax(s_batch, axis=0)
    p_argmax = np.argmax(p_batch, axis=0)
    
    return vectorized_batch_inputs, p_batch, p_argmax, filter_convolution_matrices

def BackwardPass(network, X_batch, Ys_batch):
    analytical_gradients = {}
    analytical_gradients['W'] = np.zeros_like(network.W)
    analytical_gradients['F'] = {i : np.zeros_like(network.F[i]).reshape(-1) for i in range(len(network.F))}

    n_layers = len(network.F)
    X, P_batch, P_argmax, MF_matrix = ForwardPass(network, X_batch)
    
    loss = ComputeLoss(Ys_batch, P_batch)
    G_batch = -(Ys_batch - P_batch)

    n = X_batch.shape[1]
    analytical_gradients['W'] = np.dot(G_batch, X[n_layers - 1].T) / n
    
    # propagate the gradients through the fully connected layer and second RELU function
    G_batch = np.dot(network.W.T, G_batch)
    ind_2 = (X[n_layers - 1] > 0).astype(np.float64)
    # element by element multiplication
    G_batch = np.multiply(G_batch, ind_2)
    
    # compute the gradient w.r.t. the second layer convolutional filters for j = 1,...,n
    d, k, nf = network.F[n_layers-1].shape
    for j in range(n):
        g_j = G_batch[:, j]
        x_j = X[0][:, j]
        MX = build_convolution_matrix_from_inputs(x_j, d, k, nf)
        v = np.dot(g_j.T, MX)
        analytical_gradients['F'][n_layers - 1] += (1 / n) * v
    analytical_gradients['F'][n_layers - 1] = analytical_gradients['F'][n_layers - 1].reshape((nf, k, d)).T
    
    # propagate the gradient to the previous layer through the second convolutional 
    # layer and first RELU operation
    G_batch = np.dot(MF_matrix[n_layers-1].T, G_batch)
    ind_1 = (X[0] > 0).astype(np.float64)
    # element by element multiplication
    G_batch = np.multiply(G_batch, ind_1)
    
    # compute the gradient w.r.t. the first layer convolutional filters for j = 1,...,n
    d, k, nf = network.F[0].shape
    for j in range(n):
        g_j = G_batch[:, j]
        x_j = X_batch[:, j]
        MX = build_convolution_matrix_from_inputs(x_j, d, k, nf)
        v = np.dot(g_j.T, MX)
        analytical_gradients['F'][0] += (1 / n) * v
    analytical_gradients['F'][0] = analytical_gradients['F'][0].reshape((nf, k, d)).T

    return loss, P_argmax, analytical_gradients

def numericalGradients(network, X_batch, Ys_batch, h=1e-4):
    h = np.float64(h)
    numerical_gradients = {}
    numerical_gradients['W'] = np.zeros_like(network.W)
    numerical_gradients['F'] = {i : np.zeros_like(network.F[i]) for i in range(len(network.F))}

    #numerical gradients W
    for i in range(network.W.shape[0]):
        for j in range(network.W.shape[1]):
            initial_weight_value = copy.copy(network.W[i][j])
            network.W[i][j] = initial_weight_value + h
            _, P_batch, _, _ = ForwardPass(network, X_batch)
            l2 = ComputeLoss(Ys_batch, P_batch)

            # as I have previously modified the weights(+h) I need to substract twice - h
            network.W[i][j] = initial_weight_value - h
            _, P_batch, _, _ = ForwardPass(network, X_batch)
            l1 = ComputeLoss(Ys_batch, P_batch)

            #initial weights
            network.W[i][j] = initial_weight_value
            numerical_gradients['W'][i][j] = (l2-l1) / (2*h)

    #numerical gradients F
    for key, value in numerical_gradients['F'].items():
        for i in range(numerical_gradients['F'][key].shape[0]):
            for j in range(numerical_gradients['F'][key].shape[1]):
                for k in range(numerical_gradients['F'][key].shape[2]):
                # + h, I could have deep copied
                    initial_weight_value = copy.copy(network.F[key][i][j][k])
                    network.F[key][i][j][k] = initial_weight_value + h
                    _, P_batch, _, _ = ForwardPass(network, X_batch)
                    l2 = ComputeLoss(Ys_batch, P_batch)

                    # as I have previously modified the weights(+h) I need to substract twice - h
                    network.F[key][i][j][k] = initial_weight_value - h
                    _, P_batch, _, _ = ForwardPass(network, X_batch)
                    l1 = ComputeLoss(Ys_batch, P_batch)

                    #initial weights
                    network.F[key][i][j][k] = initial_weight_value
                    numerical_gradients['F'][key][i][j][k] = (l2-l1) / (2*h)
    
    return numerical_gradients

def VerifyGradients(network, analitical_gradients, numerical_gradients):    
    eps = 1e-30
    max_relative_error = {}
    absolute_v = np.maximum(eps, np.absolute(analitical_gradients['W']) + np.absolute(numerical_gradients['W']))
    max_relative_error['W']= np.amax(np.absolute(analitical_gradients['W'] - numerical_gradients['W'])/absolute_v)

    max_relative_error['F'] = {i : None for i in range(len(analitical_gradients['F']))}
    for i in range(len(analitical_gradients['F'])):
        absolute_v = np.maximum(
            eps, np.absolute(analitical_gradients['F'][i]) + np.absolute(numerical_gradients['F'][i])
        )
        max_relative_error['F'][i]= np.amax(
            np.absolute(analitical_gradients['F'][i]-numerical_gradients['F'][i]) / absolute_v)
    print(max_relative_error)

def createBatches(network, X, Y, shuffle_data=False):
    n_batch = network.model_param['batch_size']
    Npts = X.shape[1]
    Xbatch = np.zeros((X.shape[0], n_batch, Npts // n_batch))
    Ybatch = np.zeros((Y.shape[0], n_batch, Npts // n_batch))
    
    if shuffle_data==True:
        sel = np.random.choice(X.shape[1], size=X.shape[1], replace=False)
        X = X[:,sel]
        Y = Y[:,sel]
    #e.g if the batch size is 100, and the total number of examples is 19798
    # we will end up with 197 batches each of 100 observations leaving 98 observations out.
    for j in range(Npts // n_batch):
        j_start = j * n_batch
        j_end = (j + 1) * n_batch
        Xbatch[:,:,j] = X[:, j_start:j_end]
        Ybatch[:,:,j] = Y[:, j_start:j_end]
    return Xbatch, Ybatch

def plot_confusion_matrix(cm_training, cm_validation,
                          y_label_t, y_predict_t, 
                          y_label_v, y_predict_v, save_plot='none'):
    
    
    accuracy_t = np.trace(cm_training) / float(np.sum(cm_training))
    accuracy_v = np.trace(cm_validation) / float(np.sum(cm_validation))
    misclass_t = 1 - accuracy_t
    misclass_v = 1 - accuracy_v
    title = 'Confusion Matrix'
    cmap = plt.get_cmap('Blues')
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6)) 
    cm = ax1.imshow(cm_training, interpolation='nearest', cmap=cmap)
    ax1.figure.colorbar(cm, ax=ax1)
    ax1.set_title('Training ' + title)
    tick_marks_t = list(set(y_label_t))
    ax1.set_xticks(np.arange(len(tick_marks_t)))
    ax1.set_xticklabels(tick_marks_t)
    ax1.set_yticks(np.arange(len(tick_marks_t)))
    ax1.set_yticklabels(tick_marks_t)
    ax1.xaxis.set_tick_params(rotation=45)
    thresht = cm_training.max() / 2
    for i, j in itertools.product(range(cm_training.shape[0]), range(cm_training.shape[1])):
        ax1.text(j, i, "{:,}".format(cm_training[i, j]),
                 horizontalalignment="center",
                 color="white" if cm_training[i, j] > thresht else "black")
    ax1.set_ylabel('True label')
    ax1.set_xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}\nNpts={:0}'.format(accuracy_t, 
                                                                                           misclass_t, 
                                                                                           len(y_label_t)))

    ax2.set_title('Validation ' + title)
    cm2= ax2.imshow(cm_validation, interpolation='nearest', cmap=cmap)
    ax2.figure.colorbar(cm2, ax=ax2)
    tick_marks_v = list(set(y_label_v))
    ax2.set_xticks(tick_marks_v)
    ax2.set_xticklabels(tick_marks_v)
    ax2.set_yticks(tick_marks_v)
    ax2.xaxis.set_tick_params(rotation=45)
    threshv = cm_validation.max() / 2
    for i, j in itertools.product(range(cm_validation.shape[0]), range(cm_validation.shape[1])):
        ax2.text(j, i, "{:,}".format(cm_validation[i, j]),
                 horizontalalignment="center",
                 color="white" if cm_validation[i, j] > threshv else "black")
    plt.tight_layout()   
    ax2.set_ylabel('True label')
    ax2.set_xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}\nNpts={:0}'.format(accuracy_v,
                                                                                           misclass_v, 
                                                                                           len(y_label_v)))
    
    if save_plot != 'none':
        os.makedirs(save_plot, exist_ok=True)
        plt.savefig(save_plot)
        plt.show()
    plt.show()

def plot_results(results_dict, save_file='none'):
    num_iteration= list(results_dict['loss_validation'].keys())
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(num_iteration, list(results_dict['loss_training'].values()),
             label = 'training loss', color='tab:blue')
    ax1.plot(num_iteration, list(results_dict['loss_validation'].values()),
             label = 'validation loss', color='tab:green')
    ax1.legend(loc='best')
    ax1.set_title('Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('update step')
    ax2.plot(num_iteration, list(results_dict['accuracy_training'].values()), 
            label='training accuracy', color='tab:blue')
    ax2.plot(num_iteration, list(results_dict['accuracy_validation'].values()),
             label='validation accuracy', color='tab:green')
    ax2.set_title('Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('update step')
    ax2.legend(loc='best')
    if save_file == 'none':
        plt.show()
    else:
        plt.savefig(save_file)
        plt.show()

def train_model(network, shuffle_data=False, balance_data=False,
                weights_update='SGD', print_conf_matrix=False, 
                save_cm= 'none', save_results_file = 'none', 
                update_epoch = 100, update_cm = 500):
    
    #get training data
    X_training = network.data['training']['X']
    Y_training = network.data['training']['Y']
    y_training = network.data['training']['y']
    #get validation data
    X_validation = network.data['validation']['X']
    Y_validation = network.data['validation']['Y']
    y_validation = network.data['validation']['y']
    
    Npts_val = X_validation.shape[1]
    d, Npts = X_training.shape
    #get model parameters
    n_batch = network.model_param['batch_size']
    rho = network.model_param['rho']
    eta = network.model_param['eta']
    n_update = network.model_param['n_updates']
    epochs = network.model_param['epochs']
    
    n_batches_per_epoch = Npts//n_batch
    total_num_iterations = n_batches_per_epoch * epochs
    
    Results, best_model = {}, {}
    loss_training, loss_validation, accuracy_training, accuracy_validation = {}, {}, {}, {}
    cm_training_, cm_validation_ = {}, {}
    
    best_model['W'] = np.zeros_like(network.W)
    best_model['F'] = {i : np.zeros_like(network.F[i]) for i in range(len(network.F))}
    
    #we balance the data undersampling. We will take the same number of observations,
    #per class, the number will be determined by the class with fewer observations.
    if balance_data:
        # everything good here
        labels_count = np.asarray([[x, list(y_training).count(x)] for x in set(y_training)])
        min_class_instance = labels_count[:,0][np.argmin(labels_count[:,1])]
        min_instances_frequency = min(labels_count[:,1])
        K = len(labels_count)
        
        X_training_imb = X_training[:]
        Y_training_imb = Y_training[:]
        y_training_imb = y_training[:]
        
        network.model_param['batch_size'] = min_instances_frequency
        n_batch = network.model_param['batch_size']
        balanced_Npts = min_instances_frequency * K
        n_batches_per_epoch = (balanced_Npts)//n_batch
        epochs = int(epochs * (Npts / balanced_Npts))
        network.model_param['epochs'] = epochs
        total_num_iterations = n_batches_per_epoch * epochs
        n_update = n_batches_per_epoch
        
    if weights_update == 'momentum':
        momentum = {}
        momentum['W'] = np.zeros_like(network.W)
        momentum['F'] = {i : np.zeros_like(network.F[i]) for i in range(len(network.F))}
        
    #initialize iteration counter
    iteration = 0 
    best_model_val_acc = 0.5
    tic = time.time()
    print(f'epochs = {epochs}')
    for epoch in range (epochs):
        #=============================GENERATE=DATA=BATCHES============================================# 
        if balance_data:
            shuffled_indices = np.random.choice(Npts, size=Npts, replace=False)
            balanced_data_indexs = list(itertools.chain.from_iterable(
        [list(np.where(y_training_imb[shuffled_indices]==i)[0][:min_instances_frequency]) for i in range(K)]  
            ))
            X_training = X_training_imb[:, shuffled_indices][:, balanced_data_indexs]
            Y_training = Y_training_imb[:, shuffled_indices][:, balanced_data_indexs]
            y_training = y_training_imb[shuffled_indices][balanced_data_indexs]
        if shuffle_data:
            Xbatch, Ybatch = createBatches(network, X_training, Y_training, shuffle_data=False)
        else:
            Xbatch, Ybatch = createBatches(network, X_training, Y_training)
        #====================================START=TRAINING===========================================#
        for batch_n in range(n_batches_per_epoch):
            x_miniBatch = Xbatch[:, :, batch_n]
            y_miniBatch = Ybatch[:, :, batch_n]
            #===================LOSS=&=TRAINING=DATA=ACCURACY=========================================#
            miniBatch_loss, y_pred, grads = BackwardPass(network, x_miniBatch, y_miniBatch)
            miniBatch_accuracy = np.count_nonzero(np.argmax(y_miniBatch, axis=0) == y_pred) / n_batch 
            #===================UPDATE=WEIGHTS=======================================================#
            #Vanilla gradient descent update
            if weights_update == 'SGD':
                network.W +=  - eta  * grads['W']
                for layer in range(len(network.F)):
                    network.F[layer] += - eta * grads['F'][layer]
            #momentum
            if weights_update == 'momentum':
                momentum['W'] = rho * momentum['W'] - eta  * grads['W']
                network.W += momentum['W']  
                for layer in range(len(network.F)):
                    momentum['F'][layer] = rho * momentum['F'][layer] - eta  * grads['F'][layer]
                    network.F[layer] += momentum['F'][layer]
            #===================VALIDATION=DATA=ACCURACY=============================================#
            if iteration  % (n_update) == 0:
                # train computations
                _, P_train, y_pred_t, _ = ForwardPass(network, X_training)
                train_loss = ComputeLoss(Y_training, P_train)
                train_acc = np.count_nonzero(y_training == y_pred_t) / X_training.shape[1]
                loss_training[iteration] = train_loss
                accuracy_training[iteration] = train_acc
                
                # val computations
                _, P_val, y_pred_v, _ = ForwardPass(network, X_validation)
                val_loss = ComputeLoss(Y_validation, P_val)
                val_acc = np.count_nonzero(y_validation == y_pred_v) / Npts_val
                loss_validation[iteration] = val_loss
                accuracy_validation[iteration] = val_acc
                #==========================SELECT=BEST=MODEL==============================================#
                if val_acc > best_model_val_acc:
                    best_model_val_acc = val_acc
                    best_model['training_loss'] = train_loss
                    best_model['val_loss'] = val_loss
                    best_model['cm_training'] = confusion_matrix(y_training, y_pred_t)
                    best_model['cm_validation'] = confusion_matrix(y_validation, y_pred_v)
                    best_model['training_acc'] = train_acc
                    best_model['val_acc'] = val_acc
                    best_model['grads'] = grads
                    best_model['W'] = network.W 
                    for layer in range(len(network.F)):
                        best_model['F'][layer] = network.F[layer]
                    best_model['iteration'] = iteration
                    
                if epoch % update_epoch == 0:
                    print(
                        f'epoch = {epoch}/{epochs}, iteration: {iteration}/{total_num_iterations},'
                        f' Loss: Training: {train_loss}, Validation: {val_loss}'
                        f'\nAccuracy: Training: {round(train_acc*100, 2)}%,'
                        f' Validation: {round(val_acc*100, 2)}%. '
                        )  
                    
                #print confusion matrix.
                if print_conf_matrix:
                    cm_training = confusion_matrix(y_training, y_pred_t)
                    cm_validation = confusion_matrix(y_validation, y_pred_v)
                    if (epoch % update_cm == 0 or iteration == (total_num_iterations-n_update)):
                        print(f'confusion matrix epoch: {epoch}')
                        if save_cm == 'none':
                            plot_confusion_matrix(cm_training, cm_validation,
                                                  y_training, y_pred_t,
                                                  y_validation, y_pred_v)
                        plot_confusion_matrix(cm_training, cm_validation, 
                                              y_training, y_pred_t, y_validation, 
                                              y_pred_v, save_plot= 'cm_Results/'+ str(iteration)+save_cm)
                    cm_training_[iteration] = cm_training
                    cm_validation_[iteration] = cm_validation
            iteration += 1        
    tac = time.time() 
    time_minutes = (tac - tic)/60
    print(f'training took {time_minutes} minutes')
    
    Results['loss_training'] = loss_training
    Results['loss_validation'] = loss_validation
    Results['cm_training'] = cm_training_
    Results['cm_validation'] = cm_validation_
    Results['accuracy_training'] = accuracy_training
    Results['accuracy_validation'] = accuracy_validation
    Results['time'] = time_minutes
    
    if save_results_file == 'none':
        return Results, best_model
    
    np.savez( 'models_Results/'+save_results_file, Results = Results, best_model = best_model)
    
    return best_model, Results
