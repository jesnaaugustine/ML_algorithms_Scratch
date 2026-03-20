import numpy as np
import torch
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") -- this is to check whether NVIDIA GPU avalibl;e or not
## 
'''
for mackbook(apple GPU integred in Apple chip). it can check by MPS (Metal Performance Shaders)

🔹 What is MPS?

Apple's GPU acceleration backend

Works on:

M1, M2, M3 Macs

Much faster than CPU (not as fast as high-end CUDA GPUs, but still good)
'''
device_mps = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Perceptron:
    def __init__(self,learn_rate =0.001):
        self.learn_rate =learn_rate
    
    def fit(self,X,y,epchos =100):
        self.no_featres =X.shape[1]
        self.weight =np.zeros((self.no_featres,),dtype=np.float64)
        self.bias =np.zeros(1,dtype=np.float64)
        self.errors=[]
        for i in range(epchos):
            for j in range(X.shape[0]):
                output = np.dot(X[j,:],self.weight)+self.bias
                prediction = np.sign(output)
                error =max(0,-1*np.dot(output,y[j]))
                self.errors.append(error)
                if y[j]!=prediction:
                    self.weight+=self.learn_rate*y[j]*X[j,:]
                    self.bias+=self.learn_rate*y[j]


    def predict(self,X):
        out= np.dot(X,self.weight)+self.bias
        return np.sign(out)

class Perceptron_Pytorch:
    def __init__(self,learn_rate =0.001):
        self.learn_rate =learn_rate
    def fit(self,X,y,epchos =100):
        X_n=X.float().to(device_mps)
        y_n=y.float().to(device_mps)
        self.no_featres =X.shape[1]
        self.weight=torch.zeros((self.no_featres, 1), dtype=torch.float32, device=device_mps)
        self.bias =torch.zeros(1,dtype=torch.float32, device=device_mps)
        print(X_n.device)
        print(self.weight.device)
        self.errors=[]
        for i in range(epchos):
            for j in range(X_n.shape[0]):
                output = torch.matmul(X_n[j].reshape(1,self.no_featres),self.weight)+self.bias
                prediction = torch.sign(output)
                error =max(0,-1*output,y_n[j])
                self.errors.append(error)
                if y_n[j]!=prediction:
                    self.weight+=(self.learn_rate*y_n[j]*X_n[j,:]).reshape((self.no_featres, 1))
                    self.bias+=self.learn_rate*y_n[j]


    def predict(self,X):
        X_n = X.float().to(device_mps)
        out= torch.mm(X_n,self.weight)+self.bias
        return torch.sign(out)