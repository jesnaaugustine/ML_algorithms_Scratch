import argparse
import torch
import logging
import os
import time
import yaml

from helper import set_all_seed,my_dataloader
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--setting_path',type =str,required=True)
parser.add_argument('--output_path',type =str,required=True)
args = parser.parse_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

with open(args.setting_path) as file:
    SETTINGS= yaml.load(file)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
logpath = os.path.join(args.results_path, 'training.log')
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

y.map(SETTINGS['map'])

X = torch.tensor(X.values,dtype =torch.float)
y = torch.tensor(y.values,dtype=torch.float)
shuffle_idx = torch.randperm(X.size(0),dtype =torch.int)
X,y = X[shuffle_idx],y[shuffle_idx]

perc80 = int(shuffle_idx.size(0)*0.8)
X_train,X_test = X[shuffle_idx[:perc80]],X[shuffle_idx[perc80:]]
y_train,y_test = y[shuffle_idx[:perc80]],y[shuffle_idx[perc80:]]

mu,sigma = X_train.mean(dim =0),X_train.std(dim=0)

X_train =(X_train-mu)/sigma
X_test =(X_test-mu)/sigma

train_loader,valid_loader,test_loader = my_dataloader(X_train,X_test,y_train,y_test,batch_size=SETTINGS['batch size'])
for x_batch,y_batch in train_loader:
    print(f'X tain shape: {x_batch.shape}')
    print(f'Y tain shape: {y_batch.shape}')
    break

###Model


