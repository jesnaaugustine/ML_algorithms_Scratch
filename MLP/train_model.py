import argparse
import torch
import logging
import os
import time
import yaml

from helper import set_all_seed,my_dataloader,train_model,plot_accuracy,plot_training_loss,comp_accuracy
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--setting_path',type =str,required=True)
parser.add_argument('--output_path',type =str,required=True)
args = parser.parse_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

with open(args.setting_path) as file:
    SETTINGS= yaml.load(file,Loader=yaml.FullLoader)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
logpath = os.path.join(args.output_path, 'training.log')
logger.addHandler(logging.FileHandler(logpath, 'a'))
print = logger.info

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cpu')
print(f'Using {device}')

set_all_seed(SETTINGS['random seed'])

data = pd.read_csv(SETTINGS['data path'])
X =data[SETTINGS['features']]
y =data[SETTINGS['target']]

y=y.map(SETTINGS['map'])

X = torch.tensor(X.values,dtype =torch.float)
y = torch.tensor(y.values,dtype=torch.float)
shuffle_idx = torch.randperm(X.size(0),dtype =torch.int)
X,y = X[shuffle_idx],y[shuffle_idx]

perc80 = int(shuffle_idx.size(0)*0.8)
X_train,X_test = X[:perc80],X[perc80:]
y_train,y_test = y[:perc80],y[perc80:]

mu,sigma = X_train.mean(dim =0),X_train.std(dim=0)

X_train =(X_train-mu)/sigma
X_test =(X_test-mu)/sigma

train_loader,valid_loader,test_loader = my_dataloader(X_train,X_test,y_train,y_test,batch_size=SETTINGS['batch size'])
for x_batch,y_batch in train_loader:
    print(f'X tain shape: {x_batch.shape}')
    print(f'Y tain shape: {y_batch.shape}')
    break
for x_batch,y_batch in valid_loader:
    print(f'X tain shape: {x_batch.shape}')
    print(f'Y tain shape: {y_batch.shape}')
    break

###Model
class MLP(torch.nn.Module):
    def __init__(self,num_features,num_classes,num_hidden):
        super().__init__()
        self.num_classes=num_classes
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_features,num_hidden,bias=False),#not taking bias unit to avaoid redendance since we are using batchnorm
            torch.nn.BatchNorm1d(num_hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(SETTINGS['drop_prob']),
            torch.nn.Linear(num_hidden,num_classes)

        )
    
    def forward(self,X):
        return self.model(X)
    
model= MLP(num_features=X_train.shape[1],num_hidden=SETTINGS['hidden layer size'],num_classes=SETTINGS['num class labels'])
model=model.to(device)
optimizer=torch.optim.SGD(model.parameters(),lr =SETTINGS['learning rate'])

minibatch_loss_list, train_acc_list, valid_acc_list = train_model(
    model=model,
    num_epoch=SETTINGS['num epochs'],
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    device=device)

test_acc = comp_accuracy(model, test_loader, device=device)
print(f'Test accuracy {test_acc :.2f}%')

plot_training_loss(minibatch_loss_list=minibatch_loss_list,
                   num_epochs=SETTINGS['num epochs'],
                   iter_per_epoch=len(train_loader),
                   results_dir=args.output_path)
plot_accuracy(train_acc_list=train_acc_list,
              valid_acc_list=valid_acc_list,
              results_dir=args.output_path)

results_dict = {'train accuracies': train_acc_list,
                'validation accuracies': valid_acc_list,
                'test accuracy': test_acc.item(),
                'settings': SETTINGS}

output_path = os.path.join(args.output_path, 'results_dict.yaml')
with open(output_path, 'w') as file:
    yaml.dump(results_dict, file)





