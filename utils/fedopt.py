import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.validate import validate

from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Optional

# device = self._device
criterion = nn.CrossEntropyLoss()



def train_client_fedopt(id, client_loader, global_model, num_local_epochs, lr, device):
    local_model = copy.deepcopy(global_model)
    local_model = local_model.to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    for epoch in range(num_local_epochs):
        for (i, (x,y)) in enumerate(client_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = local_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    return local_model


# Client Trining method for FeadFeat
def train_client_fedopt_fedfeat(id, client_loader, global_model, num_local_epochs, lr, device):
    local_model = copy.deepcopy(global_model)
    local_model.to(device)
    local_model.train()
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
            
    for epoch in range(num_local_epochs):
        for (i, (x,y)) in enumerate(client_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = local_model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

    for (i, (x,y)) in enumerate(client_loader):
            x = x.to(device)
            # x = x.view(x.size(0), -1)
            features = local_model.feature_extractor(x)
            features_list.extend(features.cpu().detach())
            labels_list.extend(y.tolist())        

    return local_model, features_list, labels_list

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

def fedopt_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, filename, test_loader, device):
    round_accuracy = []

    # Initialize server optimizer (e.g., Adam) on global model parameters
    server_optimizer = torch.optim.Adam(global_model.parameters(), lr=0.001)

    for t in range(max_rounds):
        print(f"starting round {t}")

        # Choose clients randomly
        clients = np.random.choice(np.arange(100), num_clients_per_round, replace=False)
        print("clients: ", clients)

        global_model.to(device)
        global_model.train()

        # Store deltas from clients
        aggregated_deltas = None

        # Get current global weights as baseline
        global_weights = copy.deepcopy(global_model.state_dict())

        for i, c in enumerate(clients):
            print(f"round {t}, starting client {i+1}/{num_clients_per_round}, id: {c}")

            # Train local client model starting from global model
            local_model = train_client_fedopt(c, client_train_loader[c], global_model, num_local_epochs, lr, device)

            # Compute delta = local weights - global weights
            local_weights = local_model.state_dict()
            delta = {}
            for key in global_weights:
                delta[key] = local_weights[key] - global_weights[key]

            # Aggregate deltas weighted equally
            scale = 1.0 / num_clients_per_round
            if aggregated_deltas is None:
                aggregated_deltas = {key: delta[key] * scale for key in delta}
            else:
                for key in delta:
                    aggregated_deltas[key] += delta[key] * scale

        # Apply server optimizer step using aggregated delta as "gradient"
        server_optimizer.zero_grad()
        # Copy aggregated delta into global model parameters' grad fields
        for name, param in global_model.named_parameters():
            if name in aggregated_deltas:
                # Note: aggregated_deltas are tensors on CPU or device - ensure same device
                param.grad = -aggregated_deltas[name].to(device)  # negative because we do gradient descent
            else:
                param.grad = None

        server_optimizer.step()

        # Validate global model
        global_model.eval()
        val_acc = validate(global_model, test_loader, device)
        print(f"round {t}, validation acc: {val_acc}")
        round_accuracy.append(val_acc)

        # Optionally save intermediate results
        # if (t % 10 == 0):
        #     np.save(f"{filename}_{t}.npy", np.array(round_accuracy))

    return np.array(round_accuracy)

# FedOpt for FedFeat method
def fedopt_fedfeat_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, filename, test_loader, device, num_global_epochs, global_lr):
    round_accuracy = []
    
    # server_optimizer = torch.optim.Adam(global_model.parameters(), lr=0.001)
    server_optimizer = torch.optim.Adam(global_model.parameters(), lr=0.001, weight_decay=1e-5)

    for t in range(max_rounds):
        print(f"starting round {t}")
        
        # select clients
        clients = np.random.choice(np.arange(100), num_clients_per_round, replace=False)
        print("clients:", clients)
        
        global_model.to(device)
        global_model.train()
        
        aggregated_deltas = None
        global_weights = copy.deepcopy(global_model.state_dict())
        optimizer_global = torch.optim.SGD(global_model.parameters(), lr = global_lr)
        global_feature_list = []
        global_label_list = []
        
        for i, c in enumerate(clients):
            print(f"round {t}, starting client {i+1}/{num_clients_per_round}, id: {c}")
            
            local_model, features_list, labels_list = train_client_fedopt_fedfeat(
                c, client_train_loader[c], global_model, num_local_epochs, lr, device)
            
            global_feature_list.extend(features_list)
            global_label_list.extend(labels_list)
            
            # Compute delta = local_model weights - global_model weights
            local_weights = local_model.state_dict()
            delta = {}
            for key in global_weights:
                delta[key] = local_weights[key] - global_weights[key]
            
            scale = 1.0 / num_clients_per_round
            if aggregated_deltas is None:
                aggregated_deltas = {key: delta[key] * scale for key in delta}
            else:
                for key in delta:
                    aggregated_deltas[key] += delta[key] * scale
        
        # Apply server optimizer update with aggregated delta as gradient
        server_optimizer.zero_grad()
        for name, param in global_model.named_parameters():
            if name in aggregated_deltas:
                param.grad = -aggregated_deltas[name].to(device)  # negative for gradient descent
            else:
                param.grad = None
        server_optimizer.step()
        
        # Now train classifier on aggregated features (same as your FedAvg code)
        dataset = CustomDataset(global_feature_list, global_label_list)
        batch_size = 128
        server_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # num_global_epochs = 20
        for epoch in range(num_global_epochs):
            for batch in server_loader:
                batch_features = batch['feature'].to(device)
                batch_labels = batch['label'].to(device)
                optimizer_global.zero_grad()
                out = global_model.classifier(batch_features)
                loss = criterion(out, batch_labels)
                loss.backward()
                optimizer_global.step()
        
        # validate
        global_model.eval()
        val_acc = validate(global_model, test_loader, device)
        print(f"round {t}, validation acc: {val_acc}")
        round_accuracy.append(val_acc)
        
        # Optional saving
        # if (t % 10 == 0):
        #     np.save(f"{filename}_{t}.npy", np.array(round_accuracy))
    
    return np.array(round_accuracy)

