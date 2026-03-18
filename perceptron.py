import numpy as np


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


