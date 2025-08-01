import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.validate import validate
import torch.optim as optim
from typing import List

from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Optional

# device = self._device
criterion = nn.CrossEntropyLoss()


def train_client_scaffold(client_id, client_loader, global_model, c_global, c_local, num_local_epochs, lr, device):
    """
    Trains a client using SCAFFOLD algorithm.
    Compatible with MLP, CNN, ResNet, ViT.

    Args:
        client_id (int): Identifier for the current client.
        client_loader (DataLoader): DataLoader for the client's local data.
        global_model (nn.Module): The global model received from the server.
        c_global (dict): Dictionary of control variates from the global model.
        c_local (dict): Dictionary of control variates for this client.
        num_local_epochs (int): Number of local epochs to train.
        lr (float): Learning rate for local training.
        device (torch.device): Device to train on (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: A tuple containing:
            - model_delta (dict): Dictionary of parameter differences (local_model - initial_weights).
            - updated_c_local (dict): Updated local control variates for this client.
            - delta_c (dict): Difference in local control variates (updated_c_local - initial c_local).
    """

    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Store initial parameters to compute model delta
    initial_weights = {
        name: param.detach().clone()
        for name, param in local_model.named_parameters()
        if param.requires_grad
    }

    # Compute gradient correction (control variate difference)
    c_diff = {}
    for name in initial_weights:
        # Ensure c_global and c_local have the same keys and are on the correct device
        if name not in c_global or name not in c_local:
            print(f"Warning: Key '{name}' not found in c_global or c_local. This might indicate an issue with control variate initialization.")
            # Handle missing keys, e.g., by initializing with zeros
            c_global[name] = torch.zeros_like(initial_weights[name]).to(device)
            c_local[name] = torch.zeros_like(initial_weights[name]).to(device)

        c_diff[name] = c_global[name].to(device) - c_local[name].to(device)

    # print(f"\n--- Client {client_id} Training Start ---")
    # print(f"Initial learning rate: {lr}")
    # print(f"Number of local epochs: {num_local_epochs}")
    # print(f"Device: {device}")

    # Training loop
    for epoch in range(num_local_epochs):
        total_loss = 0.0
        num_batches = 0
        for batch_idx, (x, y) in enumerate(client_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = local_model(x)
            loss = criterion(output, y)
            loss.backward()

            # Apply control variate correction
            with torch.no_grad():
                for name, param in local_model.named_parameters():
                    if param.grad is not None and name in c_diff:
                        correction = c_diff[name]
                        # Ensure correction is on the same device as param.grad
                        if correction.device != param.grad.device:
                            correction = correction.to(param.grad.device)
                        param.grad += correction

                        # Debugging: Print gradient norms
                        # if batch_idx % 10 == 0: # Print every 10 batches for brevity
                        #     print(f"  Batch {batch_idx}, Param: {name}, Grad Norm (before correction): {param.grad.norm().item():.4f}")
                        #     print(f"  Batch {batch_idx}, Param: {name}, Correction Norm: {correction.norm().item():.4f}")
                        #     print(f"  Batch {batch_idx}, Param: {name}, Grad Norm (after correction): {(param.grad + correction).norm().item():.4f}")


            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        # print(f"Client {client_id}, Epoch {epoch+1}/{num_local_epochs}, Avg Loss: {avg_loss:.4f}")

    # Compute model delta
    model_delta = {}
    for name, param in local_model.named_parameters():
        if name in initial_weights:
            model_delta[name] = (param.detach() - initial_weights[name])
            # Debugging: Print model_delta norm
            # print(f"  Model Delta Norm for {name}: {model_delta[name].norm().item():.4f}")

    # Update c_local and compute delta_c
    updated_c_local = {}
    delta_c = {}
    for name in model_delta:
        if name in c_local:
            coef = 1.0 / (lr * num_local_epochs)
            # Ensure all tensors are on the same device before operations
            c_local_val = c_local[name].to(device)
            c_global_val = c_global[name].to(device)
            model_delta_val = model_delta[name].to(device)

            updated = c_local_val - c_global_val + coef * model_delta_val
            updated_c_local[name] = updated.detach()
            delta_c[name] = (updated - c_local_val).detach() # This is (updated_c_local - initial c_local)

            # Debugging: Print delta_c norm
            # print(f"  Delta C Norm for {name}: {delta_c[name].norm().item():.4f}")

    # print(f"--- Client {client_id} Training End ---\n")

    return model_delta, updated_c_local, delta_c


# Client Trining method for FeadFeat
def train_client_scaffold_fedfeat(client_id, client_loader, global_model, c_global, c_local, num_local_epochs, lr, device):
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)

    # Store initial weights to compute delta_w later
    initial_weights = {name: param.clone().detach() for name, param in local_model.named_parameters()}

    # For feature collection (FedFeat)
    features_list = []
    labels_list = []

    # Before training: collect features
    torch.cuda.empty_cache()
    # with torch.no_grad():
    #     for x, y in client_loader:
    #         x = x.to(device)
    #         features = local_model.feature_extractor(x)
    #         features_list.extend(features.cpu().detach())
    #         labels_list.extend(y.tolist())
    local_model.train()

    # Training loop with SCAFFOLD correction
    for epoch in range(num_local_epochs):
        for x, y in client_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = local_model(x)
            loss = criterion(out, y)
            loss.backward()

            # SCAFFOLD gradient correction
            with torch.no_grad():
                for p, cg, cl in zip(local_model.parameters(), c_global, c_local):
                    p.grad += 0.01 * (cl - cg)
                    # print(p.grad)
            optimizer.step()
    # Compute model delta
    model_delta = [param.data.clone() - initial_weights[name].clone() for name, param in local_model.named_parameters()]
    # print(c_local)
    # Update local control variate
    updated_c_local = []
    delta_c = []
    for delta_w, cg, cl in zip(model_delta, c_global, c_local):
        new_cl = (cl - cg + (delta_w / (lr * num_local_epochs))).detach()
        updated_c_local.append(new_cl)
        delta_c.append((new_cl - cl).detach())

    # After training: collect features again
    with torch.no_grad():
        for x, y in client_loader:
            x = x.to(device)
            features = local_model.feature_extractor(x)
            features_list.extend(features.cpu().detach())
            labels_list.extend(y.tolist())

    return model_delta, updated_c_local, delta_c, features_list, labels_list


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


def scaffold_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, filename, test_loader, device):
    """
    Orchestrates the federated learning process using the SCAFFOLD algorithm.

    Args:
        global_model (nn.Module): The initial global model.
        num_clients_per_round (int): Number of clients to select in each round.
        num_local_epochs (int): Number of local epochs for each client.
        lr (float): Learning rate for client training.
        client_train_loader (list): List of DataLoaders, one for each client's local data.
        max_rounds (int): Maximum number of communication rounds.
        filename (str): Filename for saving results (not directly used in this snippet, but kept for context).
        test_loader (DataLoader): DataLoader for the global test set.
        device (torch.device): Device to run the experiment on.

    Returns:
        list: A list of validation accuracies after each communication round.
    """
    round_accuracy = []
    global_model.to(device)
    global_model.train()

    # Initialize global and local control variates.
    # IMPORTANT: Ensure control variates only exist for parameters that require gradients,
    # consistent with how 'initial_weights' is built in train_client_scaffold.
    global_c = {
        name: torch.zeros_like(param).to(device)
        for name, param in global_model.named_parameters()
        if param.requires_grad # Only include trainable parameters
    }
    local_cs = {
        cid: {
            name: torch.zeros_like(param).to(device)
            for name, param in global_model.named_parameters()
            if param.requires_grad # Only include trainable parameters
        }
        for cid in range(len(client_train_loader))
    }

    print("SCAFFOLD Experiment Started.")

    for round_num in range(max_rounds):
        print(f"\n--- Round {round_num + 1}/{max_rounds} ---")
        # Randomly select clients for the current round
        clients = np.random.choice(range(len(client_train_loader)), num_clients_per_round, replace=False)
        print("Selected clients:", clients)

        model_deltas = []
        control_deltas = [] # Stores (updated_c_local - initial c_local) for selected clients

        for cid in clients:
            # Call the client training function from the Canvas
            model_delta, updated_c_local, delta_c = train_client_scaffold(
                cid, client_train_loader[cid], global_model, global_c, local_cs[cid],
                num_local_epochs, lr, device
            )
            # Update the client's local control variates for the next round
            local_cs[cid] = updated_c_local
            model_deltas.append(model_delta)
            control_deltas.append(delta_c)

        # Server-side aggregation
        with torch.no_grad():
            # Aggregate model deltas (FedAvg style)
            for name, param in global_model.named_parameters():
                if param.requires_grad: # Only update trainable parameters
                    # Stack deltas from participating clients for this parameter
                    stacked_model_deltas = torch.stack([delta[name] for delta in model_deltas if name in delta])
                    # Update global model parameters by adding the average of client model deltas
                    param.data += stacked_model_deltas.mean(dim=0)

            # Aggregate control variate deltas (SCAFFOLD specific)
            # The global control variate is updated by adding the average of the
            # 'delta_c' (c_new - c_old) from the participating clients.
            for name in global_c:
                # Stack delta_c from participating clients for this control variate
                stacked_control_deltas = torch.stack([delta[name] for delta in control_deltas if name in delta])
                # Update global_c by adding the average of these delta_c values
                # The previous scaling factor (num_clients_per_round / len(client_train_loader)) was incorrect
                global_c[name] += stacked_control_deltas.mean(dim=0)

        # Evaluate global model after aggregation
        global_model.eval()
        acc = validate(global_model, test_loader, device)
        print(f"Validation accuracy after round {round_num + 1}: {acc:.4f}")
        round_accuracy.append(acc)
        global_model.train() # Set back to train mode for next round

    print("\nSCAFFOLD Experiment Finished.")
    return round_accuracy



# FScaffold for FedFeat method
def scaffold_fedfeat_experiment(global_model, num_clients_per_round, num_local_epochs, lr, client_train_loader, max_rounds, filename, test_loader, device, num_global_epochs, global_lr):
    round_accuracy = []

    global_model.to(device)
    global_model.train()

    # Initialize global control variate
    global_c = [torch.zeros_like(p).to(device) for p in global_model.parameters()]
    num_total_clients = len(client_train_loader)

    # Initialize local control variates for each client
    local_cs = {
        cid: [torch.zeros_like(p).to(device) for p in global_model.parameters()]
        for cid in range(num_total_clients)
    }
    optimizer_global = torch.optim.SGD(global_model.parameters(), lr = global_lr)
    for t in range(max_rounds):
        print(f"--- Round {t} ---")

        # Choose clients
        clients = np.random.choice(np.arange(100), num_clients_per_round, replace=False)
        print("Selected clients:", clients)

        global_feature_list = []
        global_label_list = []
        y_deltas = []
        c_deltas = []

        for cid in clients:
            local_c = local_cs[cid]
            # Train client with scaffold + collect features
            model_delta, updated_c_local, delta_c, features_list, labels_list = train_client_scaffold_fedfeat(
                client_id=cid,
                client_loader=client_train_loader[cid],
                global_model=global_model,
                c_global=global_c,
                c_local=local_c,
                num_local_epochs=num_local_epochs,
                lr=lr,
                device=device
            )

            y_deltas.append(model_delta)
            c_deltas.append(delta_c)
            local_cs[cid] = updated_c_local

            # Collect features and labels
            global_feature_list.extend(features_list)
            global_label_list.extend(labels_list)

        # Aggregate model deltas
        with torch.no_grad():
            for idx, param in enumerate(global_model.parameters()):
                if param.requires_grad:
                    delta_stack = torch.stack([delta[idx] for delta in y_deltas])
                    avg_delta = delta_stack.mean(dim=0)
                    param.data += avg_delta  # DO NOT replace weights — apply delta

            # Update global control variate
            for idx in range(len(global_c)):
                delta_stack = torch.stack([delta[idx] for delta in c_deltas])
                global_c[idx] += (num_clients_per_round / num_total_clients) * delta_stack.mean(dim=0)

        # ---------- Server-side fine-tuning using FedFeat ----------
        # Create a dataset and DataLoader from collected features
        dataset = CustomDataset(global_feature_list, global_label_list)
        server_loader = DataLoader(dataset, batch_size=128, shuffle=True)

        # num_global_epochs = 20
        for epoch in range(num_global_epochs):
            for batch in server_loader:
                features = batch['feature'].to(device)
                labels = batch['label'].to(device)
                optimizer_global.zero_grad()
                out = global_model.classifier(features)
                loss = criterion(out, labels)
                loss.backward()
                optimizer_global.step()

        # ---------- Evaluation ----------
        global_model.eval()
        val_acc = validate(global_model, test_loader, device)
        print(f"Round {t}, Validation Accuracy: {val_acc:.4f}")
        round_accuracy.append(val_acc)

        # Optionally save every 10 rounds
        # if t % 10 == 0:
        #     np.save(f"{filename}_{t}.npy", np.array(round_accuracy))

    return np.array(round_accuracy)

