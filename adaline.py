import torch
import numpy as np
device_mps = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class ADALINE:
    def __init__(self,lerrate =0.001):
        self.lerrate =lerrate
        
    def fit(self,X,y,epcho =100):
        self.n_features = X.shape[1]

        self.w = torch.zeros((self.n_features,1),dtype=torch.float32,device=device_mps)
        self.b =torch.zeros(1,dtype=torch.float32,device=device_mps)
        X_n =X.float().to(device_mps)
        y_n =y.float().reshape(X.shape[0],-1).to(device_mps)
        
        for i in range(epcho):
            output =torch.mm(X_n,self.w)

            error = output - y_n
            dw = torch.mm(error.T,X_n)
            self.w -=self.lerrate*dw.reshape(-1,1)
            db = output - y_n
            self.b -=self.lerrate*torch.sum(db)
        return self
    
    def predict(self,X):
        X_n = X.float().to(device_mps)
        out = torch.matmul(X_n,self.w+self.b)
        prediction = torch.where(out>0,1,-1)
        return prediction

