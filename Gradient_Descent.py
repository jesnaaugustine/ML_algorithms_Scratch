import numpy as np
import pandas as pd
class GradientDescent:

    def __init__(self,eta=0.0001,n_iter=50,random_state=1):
        self.eta =eta
        self.n_iter =n_iter
        self.random_state = random_state
        
    
    def fit(self,X,y):
        rge = np.random.RandomState(self.random_state)
        self.w_ = rge.normal(loc=0.0, scale=0.01,size=X.shape[1])
        self.b_= np.float64(0.)
        self.losses =[]

        for n in range(self.n_iter):
            y_predict = np.dot(X,self.w_)+self.b_
            error = y-y_predict
            mse = np.mean(error ** 2)
            db = np.sum(error)*2/len(error)
            dw = np.dot(error,X)*2/len(error)
            self.w_+=dw*self.eta
            self.b_+=db*self.eta
            self.losses.append(mse)
