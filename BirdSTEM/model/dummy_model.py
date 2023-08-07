import numpy as np

class dummy_model1():
    def __init__(self, the_value):
        self.the_value = float(the_value)
        pass
    def fit(self, X_train, y_train):
        pass
    def predict(self,X_test):
        return np.array([self.the_value] * len(X_test))
    def predict_proba(self,X_test):
        if self.the_value==0:
            return np.array([[1,0]] * len(X_test))
        elif self.the_value==1:
            return np.array([[0,1]] * len(X_test))
