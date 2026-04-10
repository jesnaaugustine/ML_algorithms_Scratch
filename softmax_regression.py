import torch
import torch.nn.functional as F
import numpy as np

class LogisticRegression2(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()
        self.linear =torch.nn.Linear(num_features,num_classes)
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()

    def forward(self,x):
        logits = self.linear(x)
        probs = torch.softmax(logits,dim =1)
        return probs,logits