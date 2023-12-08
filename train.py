import argparse
import numpy as np
import os
import pandas as pd
import smtplib
import torch
import torch.nn as nn
import models

from email.message import EmailMessage
from matplotlib.pylab import plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


parser = argparse.ArgumentParser(description='Model trainer.')

parser.add_argument('dataset', type=str, help='Dataset to use', choices=['I-80', 'US-101'])
parser.add_argument('t_o', type=int, help='Observation horizon', choices=[3000, 4000, 5000])
parser.add_argument('t_p', type=int, help='Prediction horizon', choices=[1500, 2000, 2500, 3000, 3500])
parser.add_argument('method', type=str, help='Method/model to use', choices=['LSTM', 'HMM', 'LSTMHMM'])

args = parser.parse_args()

dataset = args.dataset
t_o = args.t_o
t_p = args.t_p
method = args.method

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

base_dir = '/home/mma6789/Stuff/Studies/sem3/ms_project' #@param {type: 'string'}

dataset = args.dataset
t_o = args.t_o
t_p = args.t_p
method = args.method

timesteps = t_o // 100
variables = 5


print('Loading class weights...')
class_weights = np.load(f'{base_dir}/data/processed/{dataset}/{t_o}_{t_p}_class_weights.npy')
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)


print('Loading data splits...')
train = np.load(f'{base_dir}/data/processed/{dataset}/{t_o}_{t_p}_train.npy')
valid = np.load(f'{base_dir}/data/processed/{dataset}/{t_o}_{t_p}_valid.npy')
test = np.load(f'{base_dir}/data/processed/{dataset}/{t_o}_{t_p}_test.npy')

train_X, train_y = np.split(train, [-3], axis=1)
valid_X, valid_y = np.split(valid, [-3], axis=1)
test_X, test_y = np.split(test, [-3], axis=1)

temp = np.empty((len(train_X), timesteps, variables))
for i in range(len(train_X)):
    temp[i] = np.array(np.split(train_X[i], timesteps))
train_X = temp

temp = np.empty((len(valid_X), timesteps, variables))
for i in range(len(valid_X)):
    temp[i] = np.array(np.split(valid_X[i], timesteps))
valid_X = temp

temp = np.empty((len(test_X), timesteps, variables))
for i in range(len(test_X)):
    temp[i] = np.array(np.split(test_X[i], timesteps))
test_X = temp


print('Creating tensors...')
train_X = torch.tensor(train_X).float()
train_y = torch.tensor(train_y).float()
valid_X = torch.tensor(valid_X).float()
valid_y = torch.tensor(valid_y).float()
test_X = torch.tensor(test_X).float()
test_y = torch.tensor(test_y).float()


print('Creating datasets...')
class LSTMDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
train_dataset = LSTMDataset(train_X, train_y)
valid_dataset = LSTMDataset(valid_X, valid_y)
test_dataset = LSTMDataset(test_X, test_y)


print('Creating loaders...')
batch_size = 192

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def train_epoch():
    model.train(True)
    running_loss = 0.0
    y_true = []
    y_pred = []
    
    for batch_index, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1} train')):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        
        y_true += torch.argmax(y_batch, dim=1).flatten().tolist()
        y_pred += torch.argmax(output, dim=1).cpu().detach().numpy().tolist()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_batch_loss = running_loss / len(train_loader)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    matrix = confusion_matrix(y_true, y_pred)
    class_acc = matrix.diagonal() / matrix.sum(axis=1)
    
    train_loss.append(avg_batch_loss)
    train_acc.append(acc)
    train_f1.append(f1)
    train_class_acc.append(class_acc)
    
    print('Train epoch results:')
    print(f'Loss: {avg_batch_loss}')
    print(f'Acc: {acc}')
    print(f'F1: {f1}')
    print(f'Class Acc: {class_acc}')


def validate_epoch():
    model.train(False)
    running_loss = 0.0
    y_true = []
    y_pred = []
    
    for batch_index, batch in enumerate(tqdm(valid_loader, desc=f'Epoch {epoch + 1} valid')):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
            
        y_true += torch.argmax(y_batch, dim=1).flatten().tolist()
        y_pred += torch.argmax(output, dim=1).cpu().detach().numpy().tolist()
            
    avg_batch_loss = running_loss / len(valid_loader)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    matrix = confusion_matrix(y_true, y_pred)
    class_acc = matrix.diagonal() / matrix.sum(axis=1)
    
    valid_loss.append(avg_batch_loss)
    valid_acc.append(acc)
    valid_f1.append(f1)
    valid_class_acc.append(class_acc)
    
    print('Valid epoch results:')
    print(f'Loss: {avg_batch_loss}')
    print(f'Acc: {acc}')
    print(f'F1: {f1}')
    print(f'Class Acc: {class_acc}')
    print('***************************************************')


print('Initializing model...')
if method == 'LSTM':
    input_size = variables
    hidden_size = 150
    num_layers = 1
    output_size = 3

    model = models.Three_LSTM(input_size, hidden_size, num_layers, output_size)
    model.to(device)
elif method == 'HMM':
    input_size = variables
    hidden_states = 30
    output_size = 3
    
    model = models.Three_HMM(input_size, hidden_states, output_size)
    model.to(device)
elif method == 'LSTMHMM':
    input_size = variables
    hidden_size = 150
    hidden_states = 30
    output_size = 3
    
    model = models.Three_LSTMHMM(input_size, hidden_size, hidden_states, output_size)
    model.to(device)


print('Training model...')
learning_rate = 0.0002
num_epochs = 140

loss_function = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_loss = []
train_acc = []
train_f1 = []
train_class_acc = []
valid_loss = []
valid_acc = []
valid_f1 = []
valid_class_acc = []

for epoch in tqdm(range(num_epochs), desc='Training progress'):
    train_epoch()
    validate_epoch()


print('Saving model...')
torch.save(model, f'{base_dir}/models/{method}/{dataset}/model_{t_o}_{t_p}_{learning_rate}.pt')


print('Saving metrics...')
metrics = pd.DataFrame(data={
    'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1, 'train_class_acc': train_class_acc,
    'valid_loss': valid_loss, 'valid_acc': valid_acc, 'valid_f1': valid_f1, 'valid_class_acc': valid_class_acc,
})

metrics.to_json(f'{base_dir}/metrics/{method}/{dataset}/training_{t_o}_{t_p}_{learning_rate}.json')


print('Emailing results...')
msg = EmailMessage()

msg.set_content(metrics.to_string())
msg['Subject'] = f'Testing metrics of {method} on {dataset} with {learning_rate}'

msg['From'] = 'mma6789@psu.edu'
msg['To'] = 'mahmad97taha@gmail.com'

s = smtplib.SMTP('localhost')
s.send_message(msg)
s.quit()
