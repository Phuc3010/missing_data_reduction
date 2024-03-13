import numpy as np
import torch
import re

def generate_nan(X, non_missing_cols = None, missing_rate = 0.2, mechanism='mcar'):
    input_shape = X.shape
    data = X.copy().reshape(input_shape[0], -1)
    rows, cols = data.shape
    c = np.zeros(cols, dtype=bool)
    c[non_missing_cols:] = True
    if mechanism == 'mcar':
        v = np.random.uniform(size=(rows, cols))
        mask = (v<=missing_rate)*c
    elif mechanism=='mnar':
        sample_cols = np.random.choice(np.arange(start=non_missing_cols, stop=cols), 2)
        m1, m2 = np.median(data[:, sample_cols], axis=0)
        v = np.random.uniform(size=(rows, cols))
        m1 = data[:, sample_cols[0]] <= m1
        m2 = data[:, sample_cols[1]] >= m2
        m = (m1*m2)[:, np.newaxis]
        mask = m*(v<=missing_rate)*c
    data[mask] = np.nan
    data = data.reshape(input_shape)
    return data

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def normalization (data, parameters=None):
    _, dim = data.shape
    norm_data = data.copy()
    if parameters is None:
  
    # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']
    # For each dimension
    for i in range(dim):
        min_val[i] = np.nanmin(norm_data[:,i])
        norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
        max_val[i] = np.nanmax(norm_data[:,i])
        norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
      
    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val, 'max_val': max_val}
   
    
    # For each dimension
    for i in range(dim):
        norm_data[:,i] = norm_data[:,i] - min_val[i]
        norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
    norm_parameters = parameters    
      
    return norm_data, norm_parameters

def renormalization (norm_data, norm_parameters):
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()
    
    for i in range(dim):
        renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
        renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    return renorm_data

def rounding (imputed_data, data_x):
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()
  
    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
    # Only for the categorical variable
    if len(np.unique(temp)) < 20:
        rounded_data[:, i] = np.round(rounded_data[:, i])  
    return rounded_data

def rmse_loss (ori_data, imputed_data, data_m):
    ori_data, norm_parameters = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)
  # Only for missing values
    nominator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)
    denominator = np.sum(1-data_m)
  
    rmse = np.sqrt(nominator/float(denominator))
    return rmse

def xavier_init(size):
    w = torch.nn.init.xavier_normal_(torch.empty(size=size))
    return w
      
def binary_sampler(p, rows, cols):
    unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
    binary_random_matrix = 1*(unif_random_matrix < p)
    return binary_random_matrix

def uniform_sampler(low, high, rows, cols):
    return np.random.uniform(low, high, size = [rows, cols])       

def sample_batch_index(total, batch_size):
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def calculate_report(coef_f, coef_impute):
    return np.linalg.norm(coef_f-coef_impute,2)/np.linalg.norm(coef_f, 2)

def get_missing_data(X, non_missing_cols, missing_rate=0.2, missing_mechanism='mcar'):
    X_missing = generate_nan(X, non_missing_cols=non_missing_cols, missing_rate=missing_rate, mechanism=missing_mechanism)
    return X_missing

def rmse(actual, predicted):
    squared_error = np.square(actual - predicted)
    mean_squared_error = np.mean(squared_error)
    root_mean_squared_error = np.sqrt(mean_squared_error)
    return root_mean_squared_error