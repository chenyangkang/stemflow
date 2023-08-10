import numpy as np
import warnings
from .dummy_model import dummy_model1
from sklearn.base import BaseEstimator


class Hurdle(BaseEstimator):
    def __init__(self, classifier, regressor):
        '''
        The input classifier should have function:
        1. predict
        
        and the regressor should have
        1. predict
        
        '''
        self.classifier = classifier
        self.regressor = regressor
        
    
    def fit(self, X_train, y_train, sample_weight=None):
        '''
        y_train should be a continued feature
        '''
        binary_ =np.unique(np.where(y_train>0, 1, 0))
        if len(binary_)==1:
            warnings.warn('Warning: only one class presented. Replace with dummy classifier & regressor.')
            self.classifier = dummy_model1(binary_[0])
            self.regressor = dummy_model1(binary_[0])
            return
        
        new_dat = np.concatenate([np.array(X_train), np.array(y_train).reshape(-1,1)], axis=1)
        if not type(sample_weight)==type(None):
            self.classifier.fit(new_dat[:,:-1], np.where(new_dat[:,-1]>0, 1, 0), sample_weight=sample_weight)
        else:
            self.classifier.fit(new_dat[:,:-1], np.where(new_dat[:,-1]>0, 1, 0))
        self.regressor.fit(new_dat[new_dat[:,-1]>0,:][:,:-1], new_dat[new_dat[:,-1]>0,:][:,-1])
        
    def predict(self, X_test):
        cls_res = self.classifier.predict(X_test)
        reg_res = self.regressor.predict(X_test)
        # reg_res = np.where(reg_res>=0, reg_res, 0) ### we constrain the reg value to be positive
        res = np.where(cls_res>0, reg_res, 0)
        return res.reshape(-1,1)
    
    def predict_proba(self, X_test):
        '''
        This method output a numpy array with shape (n_sample, 2)
        However, user should notice that this is only for structuring the sklearn predict_proba-like method
        Only the res[:,1] is meaningful, aka the last dimension in the two dimensions. The first dimension is always zero.
        '''
        a = np.zeros(len(X_test)).reshape(-1,1)
        b = self.predict(X_test).reshape(-1,1)
        res = np.concatenate([a, b], axis=1)
        return res
    
    # def predict_proba(self, X_test):
    #     cls_res = self.classifier.predict_proba(X_test)[:,1]
    #     reg_res = self.regressor.predict(X_test)
    #     res = cls_res * reg_res
    #     return res
        
