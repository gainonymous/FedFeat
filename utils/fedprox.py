import copy
import numpy as np
import matplotlib.pyplot as plt
import IPython.display
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.validate import validate

from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Optional

# device = self._device
criterion = nn.CrossEntropyLoss()


def train_client_fedprox(id, client_loader, global_model, num_local_epochs, lr, device, mu = 0.1):
    # Clone the global model for local update
    local_model = copy.deepcopy(global_model)
    local_model = local_model.to(device)
    local_model.train()

    global_model = copy.deepcopy(global_model).to(device)  # Needed for the proximal term
    for param in global_model.parameters():
        param.requires_grad = False  # Don't update global weights

    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    for epoch in range(num_local_epochs):
        for (x, y) in client_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = local_model(x)
            loss = criterion(out, y)

            # 🔧 FedProx proximal term: ||w - w_global||^2
            prox_reg = 0.0
            for param_local, param_global in zip(local_model.parameters(), global_model.parameters()):
                prox_reg += ((param_local - param_global) ** 2).sum()

            loss += (mu / 2) * prox_reg
            loss.backward()
            optimizer.step()

    return local_model


# Client Trining method for FeadFeat
def train_client_fedfeat_fedprox(id, client_loader, global_model, num_local_epochs, lr, device, mu = 0.1):
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()

    global_model = copy.deepcopy(global_model).to(device)  # Needed for the proximal term
    for param in global_model.parameters():
        param.requires_grad = False  # Don't update global weights

    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    features_list = []
    labels_list = []
    torch.cuda.empty_cache()
    # for (i, (x,y)) in enumerate(client_loader):
    #         x = x.to(device)
    #         # x = x.view(x.size(0), -1)
    #         features = local_model.feature_extractor(x)
    #         features_list.extend(features)
    #         labels_list.extend(y.tolist())
            
    # 🔧 FedProx training loop
    for epoch in range(num_local_epochs):
        for x, y in client_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = local_model(x)
            loss = criterion(out, y)

            # Add FedProx proximal term
            prox_reg = 0.0
            for param_local, param_global in zip(local_model.parameters(), global_model.parameters()):
                prox_reg += ((param_local - param_global) ** 2).sum()

            loss += (mu / 2) * prox_reg
            loss.backward()
            optimizer.step()

    # 🔍 After training: collect updated features
    with torch.no_grad():
        for x, y in client_loader:
            x = x.to(device)
            features = local_model.feature_extractor(x)
            features_list.extend(features.cpu().detach())
            labels_list.extend(y.tolist())

    return local_model, features_list, labels_list

def running_model_avg(current, next, scale):
    if current == None:
        current = next
        for key in current:
            current[key] = current[key] * scale
    else:
        for key in current:
            current[key] = current[key] + (next[key] * scale)
    return current

# For managing the Features and Corresponding label for Server Retraining
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = {
            'feature': torch.tensor(self.features[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.int64)
        }
        return sample

def fedprox_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, filename, test_loader, device):
    round_accuracy = []
    for t in range(max_rounds):
        print("starting round {}".format(t))

        # choose clients
        clients = np.random.choice(np.arange(100), num_clients_per_round, replace = False)
        print("clients: ", clients)

        global_model.eval()
        global_model = global_model.to(device)
        running_avg = None

        for i,c in enumerate(clients):
            # train local client
            print("round {}, starting client {}/{}, id: {}".format(t, i+1,num_clients_per_round, c))
            local_model = train_client_fedprox(c, client_train_loader[c], global_model, num_local_epochs, lr, device)

            # add local model parameters to running average
            running_avg = running_model_avg(running_avg, local_model.state_dict(), 1/num_clients_per_round)
        
        # set global model parameters for the next step
        global_model.load_state_dict(running_avg)

        # validate
        val_acc = validate(global_model, test_loader, device)
        print("round {}, validation acc: {}".format(t, val_acc))
        round_accuracy.append(val_acc)

        # if (t % 10 == 0):
        #   np.save(filename+'_{}'.format(t)+'.npy', np.array(round_accuracy))

    return np.array(round_accuracy)

# FedAvg for FedFeat method
def fedprox_fedfeat_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, filename, test_loader, device, num_global_epochs, global_lr):
    round_accuracy = []
    for t in range(max_rounds):
        print("starting round {}".format(t))

        # choose clients
        clients = np.random.choice(np.arange(100), num_clients_per_round, replace = False)
        print("clients: ", clients)
        global_model.to(device)
        global_model.train()
        running_avg = None
        optimizer_global = torch.optim.SGD(global_model.parameters(), lr = global_lr)
        global_feature_list = []
        global_label_list = []

        for i,c in enumerate(clients):
            # train local client
            print("round {}, starting client {}/{}, id: {}".format(t, i+1,num_clients_per_round, c))
            local_model, features_list, labels_list = train_client_fedfeat_fedprox(c, client_train_loader[c], global_model, num_local_epochs, lr, device)
            global_feature_list.extend(features_list)
            global_label_list.extend(labels_list)
            # add local model parameters to running average
            running_avg = running_model_avg(running_avg, local_model.state_dict(), 1/num_clients_per_round)
        
        # set global model parameters for the next step    
        global_model.load_state_dict(running_avg)

        # Create a CustomDataset instance
        dataset = CustomDataset(global_feature_list, global_label_list)
        # print(dataset)
        # Define batch size
        batch_size = 128
        # Create a DataLoader
        server_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Iterate through the DataLoader in your training loop
        num_global_epochs = num_global_epochs
        # Note: Make sure to replace 'features' and 'labels' with your actual data.
        for epoch in range(num_global_epochs):
            for batch in server_loader:
                batch_features = batch['feature']
                batch_labels = batch['label']
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                optimizer_global.zero_grad()
                out = global_model.classifier(batch_features)
                loss = criterion(out, batch_labels)
                loss.backward()
                optimizer_global.step()
        # validate
        global_model.eval()        
        val_acc = validate(global_model, test_loader, device)
        print("round {}, validation acc: {}".format(t, val_acc))
        round_accuracy.append(val_acc)

        # if (t % 10 == 0):
        #   np.save(filename+'_{}'.format(t)+'.npy', np.array(round_accuracy))

    return np.array(round_accuracy)
