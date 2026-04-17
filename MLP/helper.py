# helper functions and can be reused another project

import os
import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt

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
    print(valid_fraction)
    train_dataset=MyDataset(train_set[:valid_fraction],train_target[:valid_fraction])
    valid_dataset=MyDataset(train_set[valid_fraction:],train_target[valid_fraction:])
    test_dataset=MyDataset(test_set,test_target)

    train_loader =DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    valid_loader =DataLoader(valid_dataset,batch_size=batch_size,shuffle=True)
    test_loader =DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

    return train_loader,valid_loader,test_loader

def comp_accuracy(model,dataloader,device):
    with torch.no_grad():
        correct_pred =0
        num_examples =0
        for i, (features,target) in enumerate(dataloader):
            features =features.to(device)
            target = target.float().to(device)

            logit = model.forward(features)
            predicted =torch.argmax(logit,dim=1)
            num_examples+=target.size(0)
            correct_pred+=(predicted==target).sum()
        return (correct_pred.float()/num_examples)*100

def train_model(model,num_epoch,train_loader,valid_loader,test_loader,optimizer,device):
    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
    for epoch in range(num_epoch):
        model.train()
        for batch_idx,(features,target) in enumerate(train_loader):
            features = features.to(device)
            target = target.to(device)

            # ## FORWARD AND BACK PROP
            logits = model.forward(features)
            loss = torch.nn.functional.cross_entropy(logits, target)
            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()

            # ## LOGGING
            minibatch_loss_list.append(loss.item())
            if not batch_idx % 50:
                print(f'Epoch: {epoch+1:03d}/{num_epoch:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')
            
        model.eval()
        
        with torch.no_grad():  # save memory during inference
            train_acc = comp_accuracy(model, train_loader, device=device)
            valid_acc = comp_accuracy(model, valid_loader, device=device)
            print(f'Epoch: {epoch+1:03d}/{num_epoch:03d} '
                f'| Train: {train_acc :.2f}% '
                f'| Validation: {valid_acc :.2f}%')
            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())

        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')
    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    test_acc = comp_accuracy(model, test_loader, device=device)
    print(f'Test accuracy {test_acc :.2f}%')

    return minibatch_loss_list, train_acc_list, valid_acc_list

def plot_accuracy(train_acc_list, valid_acc_list, results_dir):

    num_epochs = len(train_acc_list)

    plt.plot(np.arange(1, num_epochs+1),
             train_acc_list, label='Training')
    plt.plot(np.arange(1, num_epochs+1),
             valid_acc_list, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    image_path = os.path.join(results_dir, 'plot_acc_training_validation.pdf')
    plt.savefig(image_path)
    plt.clf()

def plot_training_loss(minibatch_loss_list, num_epochs, iter_per_epoch,
                       results_dir, averaging_iterations=100):

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_loss_list)),
             (minibatch_loss_list), label='Minibatch Loss')

    if len(minibatch_loss_list) > 1000:
        ax1.set_ylim([
            0, np.max(minibatch_loss_list[1000:])*1.5
            ])
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    ax1.plot(np.convolve(minibatch_loss_list,
                         np.ones(averaging_iterations,)/averaging_iterations,
                         mode='valid'),
             label='Running Average')
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))

    newpos = [e*iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()
    image_path = os.path.join(results_dir, 'plot_training_loss.pdf')
    plt.savefig(image_path)
    plt.clf()


