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


def train_client_fednova(
    client_id,
    client_loader,
    global_model,
    num_local_epochs,
    lr,
    device,
    momentum=0.9
):
    # Clone global model to client
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    # Initialize optimizer (SGD with momentum)
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum)

    # Save initial (global) model weights
    with torch.no_grad():
        initial_state = {k: v.clone().detach() for k, v in global_model.state_dict().items()}

    local_steps = 0

    # Local training loop
    for epoch in range(num_local_epochs):
        for x, y in client_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = local_model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            local_steps += 1

    # Compute weight delta: Δ_i = w_i - w_global
    with torch.no_grad():
        local_state = {k: v.clone().detach() for k, v in local_model.state_dict().items()}
        delta = {k: (local_state[k] - initial_state[k]) for k in initial_state}

    # Compute normalization factor (per FedNova)
    if momentum == 0:
        normalizer = lr * local_steps
    else:
        normalizer = lr * (1 - (1 - momentum) ** local_steps) / momentum

    # Normalize the delta
    normalized_delta = {k: (v / normalizer).cpu() for k, v in delta.items()}

    return normalized_delta, normalizer, local_steps  


def train_client_fednova_fedfeat(
    client_id,
    client_loader,
    global_model,
    num_local_epochs,
    lr,
    device,
    momentum=0.9
):
    # Clone global model to client
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    # Initialize optimizer (SGD with momentum)
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum)

    # Save initial (global) model weights
    with torch.no_grad():
        initial_state = {k: v.clone().detach() for k, v in global_model.state_dict().items()}

    local_steps = 0

    # Local training loop
    for epoch in range(num_local_epochs):
        for x, y in client_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = local_model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            local_steps += 1
    
    # Extract features after training
    features_list = []
    labels_list = []
    with torch.no_grad():
        for x, y in client_loader:
            x = x.to(device)
            features = local_model.feature_extractor(x)
            features_list.extend(features.detach().cpu())
            labels_list.extend(y.tolist())

    # Compute weight delta: Δ_i = w_i - w_global
    with torch.no_grad():
        local_state = {k: v.clone().detach() for k, v in local_model.state_dict().items()}
        delta = {k: (local_state[k] - initial_state[k]) for k in initial_state}

    # Compute normalization factor (per FedNova)
    if momentum == 0:
        normalizer = lr * local_steps
    else:
        normalizer = lr * (1 - (1 - momentum) ** local_steps) / momentum

    # Normalize the delta
    normalized_delta = {k: (v / normalizer).cpu() for k, v in delta.items()}

    return normalized_delta, normalizer, local_steps, features_list, labels_list  




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

def fednova_experiment(
    global_model,
    num_clients_per_round,
    num_local_epochs,
    lr,
    client_train_loader,
    max_rounds,
    filename,
    test_loader,
    device,
    momentum=0.9
):
    round_accuracy = []
    num_clients = len(client_train_loader)

    for t in range(max_rounds):
        print(f"\n--- Starting Round {t} ---")

        # Sample clients randomly
        clients = np.random.choice(np.arange(num_clients), num_clients_per_round, replace=False)
        print("Selected clients:", clients)

        global_model.to(device)
        global_model.eval()
        global_state = global_model.state_dict()

        # Initialize aggregate update
        agg_update = {k: torch.zeros_like(v).float().to(device) for k, v in global_state.items()}

        # Total data for weighted averaging
        total_samples = sum(len(client_train_loader[c].dataset) for c in clients)

        for i, c in enumerate(clients):
            print(f" -> Client {i+1}/{num_clients_per_round}, ID: {c}")

            # Get normalized update and normalizer
            normalized_delta, normalizer, local_steps = train_client_fednova(
                c,
                client_train_loader[c],
                global_model,
                num_local_epochs,
                lr,
                device,
                momentum
            )

            # Sample-based weighting
            client_weight = len(client_train_loader[c].dataset) / total_samples

            # Reconstruct full delta (FedNova: normalized * normalizer)
            for k in agg_update:
                delta = normalized_delta[k].to(device) * normalizer
                agg_update[k] += client_weight * delta

        # Update global model
        for k in global_state:
            global_state[k] = global_state[k].float().to(device) + agg_update[k]

        global_model.load_state_dict(global_state)

        # Evaluate
        global_model.eval()
        val_acc = validate(global_model, test_loader, device)
        print(f"Round {t}, Validation Accuracy: {val_acc:.4f}")
        round_accuracy.append(val_acc)

        # Optional: Save accuracy
        # if t % 10 == 0:
        #     np.save(f"{filename}_round{t}.npy", np.array(round_accuracy))

    return np.array(round_accuracy)



# FedNova for FedFeat method
def fednova_fedfeat_experiment(
    global_model,
    num_clients_per_round,
    num_local_epochs,
    lr,
    client_train_loader,
    max_rounds,
    filename,
    test_loader,
    device,
    num_global_epochs,
    global_lr,
    momentum=0.9
):
    round_accuracy = []
    optimizer_global = torch.optim.SGD(global_model.parameters(), lr=global_lr)

    num_clients = len(client_train_loader)

    for t in range(max_rounds):
        print(f"\n--- Starting Round {t} ---")

        # Sample clients randomly
        clients = np.random.choice(np.arange(num_clients), num_clients_per_round, replace=False)
        print("Selected clients:", clients)

        global_model.to(device)
        global_model.eval()
        global_state = global_model.state_dict()

        # Initialize aggregate update
        agg_update = {k: torch.zeros_like(v).float().to(device) for k, v in global_state.items()}

        # Total data for weighted averaging
        total_samples = sum(len(client_train_loader[c].dataset) for c in clients)
        global_feature_list = []
        global_label_list = []

        for i, c in enumerate(clients):
            print(f"Round {t}, client {i+1}/{num_clients_per_round}, ID: {c}")

            normalized_delta, normalizer, local_steps, features_list, labels_list = train_client_fednova_fedfeat(
                c,
                client_train_loader[c],
                global_model,
                num_local_epochs,
                lr,
                device,
                momentum
            )

            # Sample-based weighting
            client_weight = len(client_train_loader[c].dataset) / total_samples

            # Reconstruct full delta (FedNova: normalized * normalizer)
            for k in agg_update:
                delta = normalized_delta[k].to(device) * normalizer
                agg_update[k] += client_weight * delta

            global_feature_list.extend(features_list)
            global_label_list.extend(labels_list)

        # Update global model
        for k in global_state:
            global_state[k] = global_state[k].float().to(device) + agg_update[k]

        global_model.load_state_dict(global_state)

        # Server-side FedFeat training on aggregated features
        dataset = CustomDataset(global_feature_list, global_label_list)
        server_loader = DataLoader(dataset, batch_size=128, shuffle=True)

        global_model.train()
        for epoch in range(num_global_epochs):
            for batch in server_loader:
                batch_features = batch['feature'].to(device)
                batch_labels = batch['label'].to(device)

                optimizer_global.zero_grad()
                out = global_model.classifier(batch_features)  # assuming a classifier attribute
                loss = criterion(out, batch_labels)
                loss.backward()
                optimizer_global.step()

        # Validation
        global_model.eval()
        val_acc = validate(global_model, test_loader, device)
        print(f"Round {t}, validation accuracy: {val_acc:.4f}")
        round_accuracy.append(val_acc)

        # Optional saving
        # if t % 10 == 0:
        #     np.save(f"{filename}_{t}.npy", np.array(round_accuracy))

    return np.array(round_accuracy)


