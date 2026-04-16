# helper functions and can be reused another project

import os
import torch
import numpy as np
import random
import time

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
class MyDataset(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.y =y
    
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]
    def __len__(self):
        return self.X.shape[0]
    

def my_dataloader(train_set,test_set,train_target,test_target,batch_size,valid_fraction=0.1):
    valid_fraction = int(train_set.shape[0]*valid_fraction)
    train_dataset=Dataset(train_set[:valid_fraction],train_target[:valid_fraction])
    valid_dataset=Dataset(train_set[valid_fraction:],train_target[:valid_fraction:])
    test_dataset=Dataset(test_set,test_target)

    train_loader =DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    valid_loader =DataLoader(valid_dataset,batch_size=batch_size,shuffle=True)
    test_loader =DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    return train_loader,valid_loader,test_loader



def train_model(model,num_epoch,train_loader,valid_loader,test_loader,optimizer,device):
    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
    for epoch in range(num_epoch):
        model.train()



