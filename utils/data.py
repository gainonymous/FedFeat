import random

import torch
import torchvision


def fetch_dataset_mnist():
    """ Collect MNIST """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )

    test_data = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )

    return train_data, test_data

def fetch_dataset_cifar10():
    """ Collect CIFAR10 """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_data = torchvision.datasets.CIFAR10(
        './data', train=True, download=True, transform=transform
    )

    test_data = torchvision.datasets.CIFAR10(
        './data', train=False, download=True, transform=transform
    )

    return train_data, test_data

def fetch_dataset_cifar100():
    """ Collect CIFAR100 """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])

    train_data = torchvision.datasets.CIFAR100(
        './data', train=True, download=True, transform=transform
    )

    test_data = torchvision.datasets.CIFAR100(
        './data', train=False, download=True, transform=transform
    )

    return train_data, test_data

def fetch_dataset_fashionmnist():
    """ Collect FashionMNIST """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_data = torchvision.datasets.FashionMNIST(
        './data', train=True, download=True, transform=transform
    )

    test_data = torchvision.datasets.FashionMNIST(
        './data', train=False, download=True, transform=transform
    )

    return train_data, test_data


def data_to_tensor(data):
    """ Loads dataset to memory, applies transform"""
    loader = torch.utils.data.DataLoader(data, batch_size=len(data))
    img, label = next(iter(loader))
    return img, label



def iid_partition_loader(data, bsz=10, n_clients=100):
    """ partition the dataset into a dataloader for each client, iid style
    """
    m = len(data)
    assert m % n_clients == 0
    m_per_client = m // n_clients
    assert m_per_client % bsz == 0

    client_data = torch.utils.data.random_split(
        data,
        [m_per_client for x in range(n_clients)]
    )
    client_loader = [
        torch.utils.data.DataLoader(x, batch_size=bsz, shuffle=True)
        for x in client_data
    ]
    return client_loader


def noniid_partition_loader(
    data, bsz=10, m_per_shard=300, n_shards_per_client=2
):
    """ semi-pathological client sample partition
    1. sort examples by label, form shards of size 300 by grouping points
       successively
    2. each client is 2 random shards
    most clients will have 2 digits, at most 4
    """

    # load data into memory
    img, label = data_to_tensor(data)
    print(img.shape)
    print(label.shape)
    # sort
    idx = torch.argsort(label)
    img = img[idx]
    label = label[idx]

    # split into n_shards of size m_per_shard
    m = len(data)
    print(m)
    assert m % m_per_shard == 0
    n_shards = m // m_per_shard
    shards_idx = [
        torch.arange(m_per_shard*i, m_per_shard*(i+1))
        for i in range(n_shards)
    ]
    random.shuffle(shards_idx)  # shuffle shards

    # pick shards to create a dataset for each client
    assert n_shards % n_shards_per_client == 0
    n_clients = n_shards // n_shards_per_client
    client_data = [
        torch.utils.data.TensorDataset(
            torch.cat([img[shards_idx[j]] for j in range(
                i*n_shards_per_client, (i+1)*n_shards_per_client)]),
            torch.cat([label[shards_idx[j]] for j in range(
                i*n_shards_per_client, (i+1)*n_shards_per_client)])
        )
        for i in range(n_clients)
    ]

    # make dataloaders
    client_loader = [
        torch.utils.data.DataLoader(x, batch_size=bsz, shuffle=True)
        for x in client_data
    ]
    return client_loader

def noniid_partition_loader_cifar(
    data, bsz=10, m_per_shard=250, n_shards_per_client=2
):
    """ semi-pathological client sample partition
    1. sort examples by label, form shards of size 300 by grouping points
       successively
    2. each client is 2 random shards
    most clients will have 2 digits, at most 4
    """

    # load data into memory
    img, label = data_to_tensor(data)
    print(img.shape)
    print(label.shape)
    # sort
    idx = torch.argsort(label)
    img = img[idx]
    label = label[idx]

    # split into n_shards of size m_per_shard
    m = len(data)
    print(m)
    assert m % m_per_shard == 0
    n_shards = m // m_per_shard
    shards_idx = [
        torch.arange(m_per_shard*i, m_per_shard*(i+1))
        for i in range(n_shards)
    ]
    random.shuffle(shards_idx)  # shuffle shards

    # pick shards to create a dataset for each client
    assert n_shards % n_shards_per_client == 0
    n_clients = n_shards // n_shards_per_client
    client_data = [
        torch.utils.data.TensorDataset(
            torch.cat([img[shards_idx[j]] for j in range(
                i*n_shards_per_client, (i+1)*n_shards_per_client)]),
            torch.cat([label[shards_idx[j]] for j in range(
                i*n_shards_per_client, (i+1)*n_shards_per_client)])
        )
        for i in range(n_clients)
    ]

    # make dataloaders
    client_loader = [
        torch.utils.data.DataLoader(x, batch_size=bsz, shuffle=True)
        for x in client_data
    ]
    return client_loader


def label_partition_loader_cifar100(dataset, bsz=32, classes_per_client=5, num_clients=100):
    """
    Partition CIFAR-100 into 100 clients, each with 10 classes (with class overlap).
    
    Args:
        dataset: CIFAR-100 torchvision dataset (or similar)
        bsz: batch size for each client
        classes_per_client: how many unique classes per client
        num_clients: number of clients (default = 100)
        
    Returns:
        List of DataLoader objects (one per client)
    """
    from collections import defaultdict
    import random

    # Map from each class label to its sample indices
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)

    all_classes = list(class_to_indices.keys())  # [0, 1, ..., 99]

    client_loaders = []

    for _ in range(num_clients):
        # Randomly select `classes_per_client` classes for this client
        chosen_classes = random.sample(all_classes, classes_per_client)

        # Collect all sample indices for these classes
        indices = []
        for cls in chosen_classes:
            indices.extend(class_to_indices[cls])

        # Create subset and dataloader
        subset = torch.utils.data.Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=bsz, shuffle=True)
        client_loaders.append(loader)

    return client_loaders



