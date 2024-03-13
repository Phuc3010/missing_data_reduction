from sklearn.impute import IterativeImputer, KNNImputer
from autoprognosis.plugins.imputers import Imputers
from fancyimpute import SoftImpute
from math import sqrt
import numpy as np
import time
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

class Imputer(object):
    def __init__(self):
        pass
    
    def get(self, name):
        assert name in ['softimpute', 'mice', 'gain', 'knn']
        if name == 'softimpute':
            return SoftImpute(verbose=False)
        elif name == 'mice':
            return IterativeImputer()
        elif name == 'gain':
            return Imputers().get(name)
        elif name == 'knn':
            return KNNImputer()
    
    def fit_transform(self, X, X_missing, non_misisng_cols):
        
        X_missing = self.scaler.fit_transform(X_missing).to_numpy()
        X_scaled = self.scaler.transform(X).to_numpy()
        start = time.time()
        if self.reduction is not None:
            X_proj = self.reduction.fit_transform(X_missing[:, :non_misisng_cols])
        else:
            X_proj = X_scaled[:, :non_misisng_cols]
        X_reduced = self.imputer.fit_transform(np.hstack((X_proj, X_scaled[:, non_misisng_cols:])))
        time_take = time.time() - start
        n_take = self.reduction.n_take
        error = sqrt(mean_squared_error(X_scaled[:, non_misisng_cols:], X_reduced[:, n_take:]))
        return X_reduced , np.array([error, time_take])        