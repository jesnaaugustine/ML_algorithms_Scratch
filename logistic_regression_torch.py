import torch
import torch.nn.functional as F
import numpy as np

class LogisticRegression2(torch.nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.linear =torch.nn.Linear(num_features,1)
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()

    def forward(self,x):
        logits = self.linear(x)
        probs = torch.sigmoid(logits)
        return probs
    

        
    