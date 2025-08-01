import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.validate import validate

from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Optional
import gc
# device = self._device
criterion = nn.CrossEntropyLoss()



def train_client_feddyn(id, client_loader, global_model, lambda_i, w_global_prev, mu, num_local_epochs, lr, device):
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    lambda_i = [lam.to(device).float() for lam in lambda_i]
    w_global_prev = w_global_prev.to(device).float()

    for epoch in range(num_local_epochs):
        for (x, y) in client_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = local_model(x)
            loss = criterion(out, y)
            loss.backward()

            with torch.no_grad():
                for param, param_global, lam in zip(local_model.parameters(), w_global_prev.parameters(), lambda_i):
                    if param.grad is not None:
                        param.grad += (-lam + mu * (param - param_global))

            # Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=5)
            optimizer.step()
            # Cleanup batch tensors
            del x, y, out, loss
            torch.cuda.empty_cache()

    new_lambda_i = []
    with torch.no_grad():
        for param_local, param_global, lam in zip(local_model.parameters(), w_global_prev.parameters(), lambda_i):
            new_lambda_i.append((lam - mu * (param_global - param_local)).detach())
    # ✅ Move model to CPU before returning to free GPU
    local_model_cpu = copy.deepcopy(local_model).to("cpu")

    # ✅ Cleanup everything from GPU
    del local_model, optimizer, lambda_i, w_global_prev
    torch.cuda.empty_cache()
    gc.collect()

    return local_model_cpu, new_lambda_i




# Client Trining method for FeadFeat
def train_client_feddyn_fedfeat(id, client_loader, global_model, lambda_i, w_global_prev, mu, num_local_epochs, lr, device):
    local_model = copy.deepcopy(global_model)
    local_model = local_model.to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    features_list = []
    labels_list = []
    torch.cuda.empty_cache()

    # Collect initial features before training (optional)
    # for (x, y) in client_loader:
    #     x = x.to(device)
    #     features = local_model.feature_extractor(x)
    #     features_list.extend(features)
    #     labels_list.extend(y.tolist())

    for epoch in range(num_local_epochs):
        for (x, y) in client_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = local_model(x)
            loss = criterion(out, y)
            loss.backward()

            # Add FedDyn correction to gradients
            with torch.no_grad():
                for param, param_global, lam in zip(local_model.parameters(), w_global_prev.parameters(), lambda_i):
                    lam = lam.to(param.device)
                    param_global = param_global.to(param.device)
                    if param.grad is not None:
                        param.grad += -lam + mu * (param - param_global)
            
            optimizer.step()

    # Collect features after training
    for (x, y) in client_loader:
        x = x.to(device)
        features = local_model.feature_extractor(x)
        features_list.extend(features.cpu().detach())
        labels_list.extend(y.tolist())

    # Update lambda_i (correction vector) after local training
    new_lambda_i = []
    with torch.no_grad():
        for param_local, param_global, lam in zip(local_model.parameters(), w_global_prev.parameters(), lambda_i):
            lam = lam.to(param.device)
            param_global = param_global.to(param.device)
            new_lambda_i.append(lam - mu * (param_global - param_local))

    return local_model, features_list, labels_list, new_lambda_i


def running_model_avg(current, next, scale):
    if current == None:
        current = {k: v.clone() * scale for k, v in next.items()}
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

def feddyn_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, filename, test_loader, device, mu = 0.5):
    round_accuracy = []
    num_clients = len(client_train_loader)  # total clients, assuming 100 here or adjust

    # Initialize lambda_i for each client as list of tensors (same shapes as model params)
    lambda_dict = {}
    for c in range(num_clients):
        lambda_dict[c] = [torch.zeros_like(param) for param in global_model.parameters()]
    
    w_global_prev = copy.deepcopy(global_model)

    for t in range(max_rounds):
        print(f"starting round {t}")

        # choose clients
        clients = np.random.choice(np.arange(num_clients), num_clients_per_round, replace=False)
        print("clients: ", clients)

        global_model.eval()
        # global_model = global_model.to(device)
        
        # Store updated local models and lambdas
        local_models = []
        updated_lambdas = {}

        for i, c in enumerate(clients):
            print(f"round {t}, starting client {i+1}/{num_clients_per_round}, id: {c}")

            local_model, new_lambda_i = train_client_feddyn(
                c,
                client_train_loader[c],
                global_model,
                lambda_dict[c],
                w_global_prev,
                mu,
                num_local_epochs,
                lr,
                device
            )

            local_models.append(local_model.state_dict())
            updated_lambdas[c] = new_lambda_i
            ## Cleanup
            del local_model, new_lambda_i
            torch.cuda.empty_cache()
            gc.collect()
        # Aggregate local models by averaging parameters
        running_avg = None
        for local_state in local_models:
            running_avg = running_model_avg(running_avg, local_state, 1/num_clients_per_round)
        
        global_model.load_state_dict(running_avg)
        w_global_prev = copy.deepcopy(global_model)

        # Update lambdas for selected clients
        for c in updated_lambdas:
            lambda_dict[c] = updated_lambdas[c]

        # Validate
        global_model.eval()
        val_acc = validate(global_model, test_loader, device)
        print(f"round {t}, validation acc: {val_acc}")
        round_accuracy.append(val_acc)

        # Optional: save intermediate results
        # if (t % 10 == 0):
        #     np.save(filename + f'_{t}.npy', np.array(round_accuracy))

    return np.array(round_accuracy)


# FedDyn for FedFeat method
def feddyn_fedfeat_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, filename, test_loader, device, num_global_epochs, global_lr, mu=0.1):
    round_accuracy = []
    num_clients = len(client_train_loader) 
    # Initialize lambda_i for each client as list of tensors (same shapes as model params)
    lambda_dict = {}
    for c in range(num_clients):
        lambda_dict[c] = [torch.zeros_like(param) for param in global_model.parameters()]
    
    w_global_prev = copy.deepcopy(global_model).to(device)

    for t in range(max_rounds):
        print(f"starting round {t}")

        # Sample clients
        clients = np.random.choice(np.arange(100), num_clients_per_round, replace=False)
        print("clients:", clients)

        global_model.to(device)
        global_model.train()

        running_avg = None
        optimizer_global = torch.optim.SGD(global_model.parameters(), lr = global_lr)

        # Store updated local models and lambdas
        local_models = []
        updated_lambdas = {}

        global_feature_list = []
        global_label_list = []

        for i, c in enumerate(clients):
            print(f"round {t}, starting client {i+1}/{num_clients_per_round}, id: {c}")

            local_model, features_list, labels_list, new_lambda_i = train_client_feddyn_fedfeat(
                c,
                client_train_loader[c],
                global_model,
                lambda_dict[c],
                w_global_prev,
                mu,
                num_local_epochs,
                lr,
                device
            )
            global_feature_list.extend(features_list)
            global_label_list.extend(labels_list)
            local_models.append(local_model.state_dict())
            updated_lambdas[c] = new_lambda_i

            del local_model, new_lambda_i
            torch.cuda.empty_cache()
        # Aggregate local models by averaging parameters
        running_avg = None
        for local_state in local_models:
            running_avg = running_model_avg(running_avg, local_state, 1/num_clients_per_round)
        
        global_model.load_state_dict(running_avg)
        w_global_prev = copy.deepcopy(global_model).to(device)

        # Update lambdas for selected clients
        for c in updated_lambdas:
            lambda_dict[c] = updated_lambdas[c]

        # ----- Server-side training on features -----
        dataset = CustomDataset(global_feature_list, global_label_list)
        server_loader = DataLoader(dataset, batch_size=128, shuffle=True)

        for epoch in range(num_global_epochs):  # Server-side epochs
            for batch in server_loader:
                batch_features = batch['feature'].to(device)
                batch_labels = batch['label'].to(device)
                optimizer_global.zero_grad()
                out = global_model.classifier(batch_features)
                loss = criterion(out, batch_labels)
                loss.backward()
                optimizer_global.step()

        # ----- Validation -----
        global_model.eval()
        val_acc = validate(global_model, test_loader, device)
        print(f"round {t}, validation acc: {val_acc}")
        round_accuracy.append(val_acc)

    return np.array(round_accuracy)


