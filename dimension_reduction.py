import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_val_score

class PCA():
    def __init__(self, explain_ratio=0.95, pca_type="pca1") -> None:
        assert pca_type in ['pca1', 'pca2']
        self.explain_per = explain_ratio
        self.scaler = StandardScaler()
        self.pca_type = pca_type
        self.proj_matrix = None
    
    def __pca1(self, data):
        eig_vals, eig_vectors = np.linalg.eig(np.cov(data,rowvar = False, ddof = 0))
        eig_vals = eig_vals.real
        eig_vectors = eig_vectors.real
        e_indices = np.argsort(eig_vals)[::-1]
        eigenvectors_sorted = eig_vectors[:,e_indices]
        variance_explained = np.array([i/sum(eig_vals) for i in eig_vals])#percentage of variance_explained
        self.n_take = np.where(np.cumsum(variance_explained)>self.explain_per)[0][0] + 1 # chose st total percentage of variance_explained > 99%
        print('n components chosen:', self.n_take)
        self.proj_matrix = eigenvectors_sorted[:, :self.n_take]
    
    def __pca2(self, data):
        U, S, VT = np.linalg.svd(data, full_matrices=False)
        n_take = np.where(np.cumsum(S)/sum(S)>self.explain_per/100) 
        n_take = np.min(n_take)+1
        print('n components chosen:', n_take)
        self.n_take = n_take
        V = VT.T
        self.proj_matrix = V[:, :n_take]
    
    def fit(self, data):
        if self.pca_type == "pca1":
            data = self.scaler.fit_transform(data)
            self.__pca1(data)
        else:
            data = self.scaler.fit_transform(data)
            self.__pca2(data)
    
    def fit_transform(self, data):
        if self.pca_type == "pca1":
            self.__pca1(data)
            reduced_matrix = data @ self.proj_matrix
        else:
            self.__pca2(data)
            reduced_matrix = data.dot(self.proj_matrix)

        return reduced_matrix
    
    def transform(self, data):
        assert self.proj_matrix is not None
        not_reduced_part = data
        if self.pca_type == "pca1":
            data = self.scaler.transform(data)
            reduced_matrix = data @ self.proj_matrix
        else:
            data = self.scaler.transform(data)
            reduced_matrix = data.dot(self.proj_matrix)     
        return reduced_matrix
    
    def inverse_transform(self, data):
        if self.pca_type == 'pca1':
            return data @ self.proj_matrix.T
        else:
            return data.dot(self.proj_matrix.T)
    
class SVD(object):
    def __init__(self, explain_ratio=0.95):
        self.explain_ratio = explain_ratio
        self.model = TruncatedSVD()
        self.n_take = None
    
    def _get_n_take(self, data):
        self.model = TruncatedSVD(data.shape[1])
        self.model.fit(data)
        self.n_take = np.where(np.cumsum(self.model.explained_variance_ratio_)>self.explain_ratio)[0][0] + 1
        
    def fit(self, data):
        self._get_n_take(data)
        print('n components chosen:', self.n_take)
        self.model = TruncatedSVD(self.n_take)
        self.model.fit(data)
    
    def transform(self, data):
        reduced = self.model.transform(data)
        return reduced
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data):
        return self.model.inverse_transform(data)
    
class kernelPCA(object):
    def __init__(self, explain_ratio=0.95):
        self.model = KernelPCA(kernel='rbf', fit_inverse_transform=True, n_jobs=4)
        self.n_take = None
        self.scaler = StandardScaler()
        self.explain_ratio = 0.95
    
    def _get_n_take(self, data):
        kpca_transform  = self.model.fit_transform(data)
        explained_variance = np.var(kpca_transform, axis=0)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        self.n_take = np.where(np.cumsum(explained_variance_ratio)>self.explain_ratio)[0][0] + 1# chose st total percentage of variance_explained > 99%
        
    def fit(self, data):
        self._get_n_take(data)
        data = self.scaler.fit_transform(data)
        self.model = KernelPCA(kernel='rbf', n_components=self.n_take, fit_inverse_transform=True, n_jobs=4)
        self.model.fit(data)
    
    def transform(self, data):
        data = self.scaler.transform(data)
        reduced = self.model.transform(data)
        return reduced
    
    def inverse_transform(self, data):
        return self.scaler.scale_*self.model.inverse_transform(data) + self.scaler.mean_
    
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
class Reducer(object):
    def __init__(self):
        pass
    
    def get(self, name):
        assert name in ['pca', 'svd', 'kernelpca']
        if name == 'pca':
            return PCA()
        elif name == 'svd':
            return SVD()
        elif name == 'kernelpca':
            return kernelPCA()
        else:
            return NotImplemented