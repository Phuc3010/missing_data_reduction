import numpy as np

class Trainer(object):
    def __init__(self, imputer, reducer, classifier):
        self.imputer = imputer
        self.reducer = reducer
        self.classifier = classifier
    
    def fit(self, x_miss, y, non_missing, x_orig=None):
        input_shape = x_miss.shape
        x_miss = x_miss.reshape(y.shape[0], -1)
        x_non_missing, x_missing = x_miss[:, :non_missing], x_miss[:, non_missing:]
        x_projected = self.reducer.fit_transform(x_non_missing)
        n_take = self.reducer.n_take
        x_imputed = self.imputer.fit_transform(np.hstack((x_projected, x_missing)))
        x_imputed = x_imputed.to_numpy() if type(x_imputed) != np.ndarray else x_imputed
        x_imputed = x_imputed[:, n_take:]
        x_recovered = self.reducer.inverse_transform(x_projected)
        x_imputed = np.hstack((x_recovered, x_imputed))
        x_imputed = x_imputed.reshape(*input_shape)
        if x_orig is not None:
            rmse_score = np.sqrt(np.mean((x_imputed-x_orig)**2))
        x_imputed = x_imputed.astype(np.float32)
        self.classifier.fit(x_imputed, y)
        self.non_missing = non_missing
        if x_orig is not None:
            return rmse_score
        return self
    
    def score(self, x_test, y_test, model):
        if np.sum(np.isnan(x_test)) == 0:
            accuracy = model.score(x_test, y_test)
        else:
            x_miss = x_test
            non_missing = self.non_missing
            x_non_missing, x_missing = x_miss[:, :non_missing], x_miss[:, non_missing:]
            x_projected = self.reducer.transform(x_non_missing)
            x_imputed = self.imputer.transform(np.hstack((x_projected, x_missing)))
            x_imputed = x_imputed[:, non_missing:]
            x_recovered = self.reducer.inverse_transform(x_projected)
            x_imputed = np.hstack((x_recovered, x_imputed))
            accuracy = self.score(x_imputed, y_test)
        return accuracy