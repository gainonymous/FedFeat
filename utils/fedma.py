import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.validate import validate

from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Optional
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# device = self._device
criterion = nn.CrossEntropyLoss()



def train_client_fedma(id, client_loader, global_model, num_local_epochs, lr, device):
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
def train_client_fedma_fedfeat(id, client_loader, global_model, num_local_epochs, lr, device):
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


def match_and_average_layers(current_layer: np.ndarray, next_layer: np.ndarray, scale: float) -> np.ndarray:
    """
    FedMA-style neuron matching and averaging.
    - Uses Hungarian algorithm for weights (shape ≥ 2D).
    - Directly averages biases (1D).
    - Skips scalar (0D) parameters.
    """
    shape = current_layer.shape

    # Case 1: scalar (e.g., batch norm scalar gamma/beta)
    if len(shape) == 0:
        return current_layer  # Or handle differently if needed

    # Case 2: 1D bias vector – directly average
    if len(shape) == 1:
        return current_layer + scale * next_layer

    # Case 3: 2D+ (fully connected or conv filters) – do matching
    N = shape[0]
    cur_flat = current_layer.reshape(N, -1)
    nxt_flat = next_layer.reshape(N, -1)

    # Normalize rows for cosine similarity
    def _normalize(mat):
        mat = mat.detach().cpu().numpy()
        norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        return mat / norm

    cur_norm = _normalize(cur_flat)
    nxt_norm = _normalize(nxt_flat)

    # Cost matrix (1 - cosine similarity)
    cost = 1 - (cur_norm @ nxt_norm.T)

    # Hungarian matching
    row_idx, col_idx = linear_sum_assignment(cost)

    # Align next layer neurons based on matching
    aligned = next_layer[col_idx]

    # Weighted average
    return current_layer + scale * aligned


def running_fedma_avg(client_models, scale_factors):
    """
    Perform FedMA aggregation over multiple client models.
    `client_models`: List of dicts, each containing NumPy arrays of model parameters.
    `scale_factors`: List of floats, one per client (usually 1 / num_clients).
    """
    aggregated = {}

    # Initialize with the first client
    base_model = client_models[0]
    for key in base_model:
        aggregated[key] = base_model[key] * scale_factors[0]

    # Match and merge remaining client models
    for client_idx in range(1, len(client_models)):
        current_model = client_models[client_idx]
        scale = scale_factors[client_idx]

        for key in base_model:
            cur = aggregated[key]
            nxt = current_model[key]

            # Use FedMA layer-wise matching and averaging
            merged = match_and_average_layers(cur, nxt, scale)
            aggregated[key] = merged

    # Convert final aggregated model back to PyTorch tensors safely
    return {
        k: torch.from_numpy(v).float() if isinstance(v, np.ndarray)
        else torch.tensor(v, dtype=torch.float32)
        for k, v in aggregated.items()
    }

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

def fedma_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, filename, test_loader, device):
    
    round_accuracy = []

    for t in range(max_rounds):
        print(f"Starting round {t}")
        clients = np.random.choice(np.arange(100), num_clients_per_round, replace=False)
        print("Selected clients:", clients)

        global_model.eval()
        global_model = global_model.to(device)

        client_models = []

        for i, c in enumerate(clients):
            print(f"Round {t}, client {i+1}/{num_clients_per_round}, ID: {c}")
            local_model = train_client_fedma(c, client_train_loader[c], global_model, num_local_epochs, lr, device)
            client_models.append(copy.deepcopy(local_model.state_dict()))

        # Perform FedMA aggregation: match + average
        scale = 1 / num_clients_per_round
        aggregated_model = running_fedma_avg(client_models, scale_factors=[scale]*num_clients_per_round)

        # Update global model with matched-averaged parameters
        global_model.load_state_dict(aggregated_model)

        # Validate on global model
        val_acc = validate(global_model, test_loader, device)
        print(f"Round {t}, validation acc: {val_acc}")
        round_accuracy.append(val_acc)

    return np.array(round_accuracy)


# FedMa for FedFeat method
def fedma_fedfeat_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, filename, test_loader, device, num_global_epochs, global_lr):
    
    round_accuracy = []
    
    for t in range(max_rounds):
        print(f"Starting round {t}")
        clients = np.random.choice(np.arange(100), num_clients_per_round, replace=False)
        print("Selected clients:", clients)

        global_model.to(device)
        global_model.train()
        optimizer_global = torch.optim.SGD(global_model.parameters(), lr = global_lr)

        global_feature_list = []
        global_label_list = []
        client_models = []

        for i, c in enumerate(clients):
            print(f"Round {t}, client {i+1}/{num_clients_per_round}, ID: {c}")

            # Train client and extract features
            local_model, features_list, labels_list = train_client_fedma_fedfeat(
                c, client_train_loader[c], global_model, num_local_epochs, lr, device
            )

            # Collect for server-side classifier update
            global_feature_list.extend(features_list)
            global_label_list.extend(labels_list)

            # Store local model for FedMA aggregation
            client_models.append(copy.deepcopy(local_model.state_dict()))

        # === FedMA Aggregation ===
        scale_factors = [1 / num_clients_per_round] * num_clients_per_round
        fedma_state_dict = running_fedma_avg(client_models, scale_factors)
        global_model.load_state_dict(fedma_state_dict)

        # === FedFeat Classifier Training ===
        dataset = CustomDataset(global_feature_list, global_label_list)
        server_loader = DataLoader(dataset, batch_size=128, shuffle=True)
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

        # === Validation ===
        global_model.eval()
        val_acc = validate(global_model, test_loader, device)
        print(f"Round {t}, validation acc: {val_acc}")
        round_accuracy.append(val_acc)

    return np.array(round_accuracy)

