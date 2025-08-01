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
cos_sim = torch.nn.CosineSimilarity(dim=-1)
criterion = torch.nn.CrossEntropyLoss()

def get_representation(model, x):
    """
    Extract intermediate representation from model x.
    Tries to call `feature_extractor` if available,
    else falls back to hooking or flattening before classifier.
    """
    if hasattr(model, "feature_extractor"):
        model.eval()
        with torch.no_grad():
            rep = model.feature_extractor(x)
            # If representation has spatial dims, flatten them to vector
            if rep.ndim > 2:
                rep = torch.flatten(rep, start_dim=1)
        return rep

    else:
        # Fallback for full models without explicit extractor
        # Option 1: Use forward hook (complex, one-time setup)
        # Option 2: Flatten input and pass through model excluding final layer (if accessible)

        # Example fallback: flatten input and pass through all layers except last
        # This depends heavily on model architecture

        # As a last resort, just flatten input
        return x.flatten(1)


def cos_sim(a, b):
    # Compute cosine similarity between two tensors [B, D]
    # Returns: [B]
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return (a_norm * b_norm).sum(dim=1)

def contrastive_loss(rep_local, rep_global, rep_negatives, temperature=0.5):
    """
    Compute contrastive loss between local and global representations.

    Args:
        rep_local: Tensor of shape [batch_size, feature_dim]
        rep_global: Tensor of shape [batch_size, feature_dim]
        rep_negatives: list of Tensors, each [batch_size, feature_dim]
        temperature: scaling factor

    Returns:
        loss: scalar tensor
    """
    # Positive similarities (local vs global)
    pos_sim = cos_sim(rep_local, rep_global) / temperature  # [B]

    # Negative similarities (local vs negatives)
    neg_sims = []
    for neg_rep in rep_negatives:
        neg_sims.append(cos_sim(rep_local, neg_rep) / temperature)  # [B]

    if len(neg_sims) > 0:
        neg_sims = torch.stack(neg_sims, dim=1)  # [B, N_neg]
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sims], dim=1)  # [B, 1+N_neg]
    else:
        # No negatives: logits = pos_sim only
        logits = pos_sim.unsqueeze(1)  # [B, 1]

    # Labels: positives are index 0
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=rep_local.device)

    # CrossEntropyLoss expects [B, C], labels of shape [B]
    loss = F.cross_entropy(logits, labels)

    return loss

def train_client_moon(client_id, client_loader, global_model, prev_global_models, num_local_epochs, lr, device, mu=1.0, temperature=0.5):
    # Clone the global model
    local_model = copy.deepcopy(global_model).to(device)
    global_model = copy.deepcopy(global_model).to(device).eval()
    prev_global_models = [copy.deepcopy(m).to(device).eval() for m in prev_global_models]

    for param in global_model.parameters():
        param.requires_grad = False
    for m in prev_global_models:
        for param in m.parameters():
            param.requires_grad = False

    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    for epoch in range(num_local_epochs):
        for (x, y) in client_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = local_model(x)
            loss_cls = criterion(logits, y)

            # Get representations for contrastive loss
            rep_local = get_representation(local_model, x)
            rep_global = get_representation(global_model, x)
            rep_negatives = [get_representation(m, x) for m in prev_global_models]

            loss_contrast = contrastive_loss(rep_local, rep_global, rep_negatives, temperature)

            total_loss = loss_cls + mu * loss_contrast
            total_loss.backward()
            optimizer.step()

    return local_model


# Client Trining method for FeadFeat
def train_client_moon_fedfeat(
    client_id,
    client_loader,
    global_model,
    prev_global_models,
    num_local_epochs,
    lr,
    device,
    mu=1.0,
    temperature=0.5
):
    # Clone the models
    local_model = copy.deepcopy(global_model).to(device)
    global_model = copy.deepcopy(global_model).to(device).eval()
    prev_global_models = [copy.deepcopy(m).to(device).eval() for m in prev_global_models]

    # Freeze global and previous models
    for param in global_model.parameters():
        param.requires_grad = False
    for m in prev_global_models:
        for param in m.parameters():
            param.requires_grad = False

    # Training mode
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    for epoch in range(num_local_epochs):
        for x, y in client_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # Classification loss
            logits = local_model(x)
            loss_cls = criterion(logits, y)

            # Feature extraction
            rep_local = get_representation(local_model, x)
            rep_global = get_representation(global_model, x)
            rep_negatives = [get_representation(m, x) for m in prev_global_models]

            # Contrastive loss
            loss_contrast = contrastive_loss(rep_local, rep_global, rep_negatives, temperature)

            # Total loss
            total_loss = loss_cls + mu * loss_contrast
            total_loss.backward()
            optimizer.step()

    # Evaluation mode for feature collection
    local_model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for x, y in client_loader:
            x = x.to(device)
            features = local_model.feature_extractor(x)
            features_list.extend(features.detach().cpu())
            labels_list.extend(y.cpu().tolist())

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

def moon_experiment(
    global_model,
    num_clients_per_round,
    num_local_epochs,
    lr,
    client_train_loader,
    max_rounds,
    filename,
    test_loader,
    device,
    prev_models_to_keep=3  # number of previous global models to store for contrastive loss
):
    round_accuracy = []

    # Keep a list of previous global models (initially empty)
    prev_global_models = []

    for t in range(max_rounds):
        print(f"starting round {t}")

        # Sample clients randomly
        clients = np.random.choice(np.arange(len(client_train_loader)), num_clients_per_round, replace=False)
        print("clients: ", clients)

        global_model.eval()
        global_model = global_model.to(device)
        running_avg = None

        for i, c in enumerate(clients):
            print(f"round {t}, starting client {i+1}/{num_clients_per_round}, id: {c}")

            # Pass prev_global_models to the client trainer
            local_model = train_client_moon(
                c,
                client_train_loader[c],
                global_model,
                prev_global_models,
                num_local_epochs,
                lr,
                device
            )

            # Running average update
            running_avg = running_model_avg(running_avg, local_model.state_dict(), 1 / num_clients_per_round)

        # Update global model
        global_model.load_state_dict(running_avg)

        # Update the list of previous global models
        prev_global_models.append(copy.deepcopy(global_model).to(device).eval())
        if len(prev_global_models) > prev_models_to_keep:
            prev_global_models.pop(0)  # keep only last few

        # Validate
        val_acc = validate(global_model, test_loader, device)
        print(f"round {t}, validation acc: {val_acc}")
        round_accuracy.append(val_acc)

        # Optional save
        # if t % 10 == 0:
        #   np.save(filename + f'_{t}.npy', np.array(round_accuracy))

    return np.array(round_accuracy)


# FedAvg for FedFeat method
def moon_fedfeat_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, filename, test_loader, device, num_global_epochs, global_lr):
    round_accuracy = []
    optimizer_global = torch.optim.SGD(global_model.parameters(), lr=global_lr)
    prev_global_models = []
    max_prev_models = 3

    for t in range(max_rounds):
        print("starting round {}".format(t))

        clients = np.random.choice(np.arange(len(client_train_loader)), num_clients_per_round, replace=False)
        print("clients: ", clients)
        global_model.to(device)
        global_model.train()
        running_avg = None

        global_feature_list = []
        global_label_list = []

        for i, c in enumerate(clients):
            print("round {}, starting client {}/{}, id: {}".format(t, i + 1, num_clients_per_round, c))

            local_model, features_list, labels_list = train_client_moon_fedfeat(
                c,
                client_train_loader[c],
                global_model,
                prev_global_models,
                num_local_epochs,
                lr,
                device,
            )

            global_feature_list.extend(features_list)
            global_label_list.extend(labels_list)

            running_avg = running_model_avg(running_avg, local_model.state_dict(), 1 / num_clients_per_round)

        global_model.load_state_dict(running_avg)

        # Update prev_global_models with latest global model (copy to CPU)
        prev_global_models.append(copy.deepcopy(global_model).cpu())
        if len(prev_global_models) > max_prev_models:
            prev_global_models.pop(0)

        # Server-side FedFeat training
        dataset = CustomDataset(global_feature_list, global_label_list)
        server_loader = DataLoader(dataset, batch_size=128, shuffle=True)

        global_model.train()
        for epoch in range(num_global_epochs):
            for batch in server_loader:
                batch_features = batch['feature'].to(device)
                batch_labels = batch['label'].to(device)
                optimizer_global.zero_grad()
                out = global_model.classifier(batch_features)
                loss = criterion(out, batch_labels)
                loss.backward()
                optimizer_global.step()

        # Validation
        global_model.eval()
        val_acc = validate(global_model, test_loader, device)
        print("round {}, validation acc: {}".format(t, val_acc))
        round_accuracy.append(val_acc)

    return np.array(round_accuracy)

