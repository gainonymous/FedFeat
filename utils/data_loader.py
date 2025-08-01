import random
import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import view_10_color, num_params
from utils.data import fetch_dataset_cifar100, fetch_dataset_cifar10, fetch_dataset_mnist, fetch_dataset_fashionmnist, label_partition_loader_cifar100, iid_partition_loader, noniid_partition_loader, noniid_partition_loader_cifar

bsz = 10
def get_dataset_by_name(name, root='./data'):
    """
    Unified dataset fetcher by name.
    """
    name = name.lower()
    if name == 'mnist':
        train_data, test_data = fetch_dataset_mnist()
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1000, shuffle=False) # inference bsz=1000
        debug_loader = torch.utils.data.DataLoader(train_data, bsz)
        img, label = next(iter(debug_loader))
        view_10(img, label)

        iid_client_train_loader = iid_partition_loader(train_data, bsz = bsz)
        noniid_client_train_loader = noniid_partition_loader(train_data, bsz = bsz)


    elif name == 'fashionmnist':
        train_data, test_data = fetch_dataset_fashionmnist()
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1000, shuffle=False) # inference bsz=1000
        debug_loader = torch.utils.data.DataLoader(train_data, bsz)
        img, label = next(iter(debug_loader))
        view_10(img, label)

        iid_client_train_loader = iid_partition_loader(train_data, bsz = bsz)
        noniid_client_train_loader = noniid_partition_loader(train_data, bsz = bsz)


    elif name == 'cifar10':
        train_data, test_data = fetch_dataset_cifar10()
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1000, shuffle=False) # inference bsz=1000
        debug_loader = torch.utils.data.DataLoader(train_data, bsz)
        img, label = next(iter(debug_loader))
        view_10_color(img, label)

        iid_client_train_loader = iid_partition_loader(train_data, bsz = bsz)
        noniid_client_train_loader = noniid_partition_loader_cifar(train_data, bsz = bsz)


    elif name == 'cifar100':
        train_data, test_data = fetch_dataset_cifar100()
        test_loader = torch.utils.data.DataLoader(test_data, batch_size = 1000, shuffle=False) # inference bsz=1000
        debug_loader = torch.utils.data.DataLoader(train_data, bsz)
        img, label = next(iter(debug_loader))
        view_10_color(img, label)

        iid_client_train_loader = iid_partition_loader(train_data, bsz = bsz)
        noniid_client_train_loader = label_partition_loader_cifar100(train_data, bsz = bsz)


    else:
        raise ValueError(f"Unknown dataset name: {name}")
    
    return iid_client_train_loader, noniid_client_train_loader, test_loader

def inspect_client_distribution(name, iid_client_train_loader, noniid_client_train_loader):
    """
    Unified dataset fetcher by name.
    """
    name = name.lower()
    # iid
    label_dist = torch.zeros(100)
    for (x,y) in iid_client_train_loader[1]:
        label_dist+= torch.sum(F.one_hot(y, num_classes=100), dim=0)
    print("iid: ", label_dist)
    if name == 'mnist' or name == 'fashionmnist':
        view_10(x,y)
    elif name == 'cifar10' or name == 'cifar100':   
        view_10_color(x,y) 
    else:
        raise ValueError(f"Unknown dataset name: {name}")    

    # non-iid
    label_dist = torch.zeros(100)
    for (x,y) in noniid_client_train_loader[1]:
        label_dist+= torch.sum(F.one_hot(y,num_classes=100), dim=0)
    print("non-iid: ", label_dist)
    if name == 'mnist' or name == 'fashionmnist':
        view_10(x,y)
    elif name == 'cifar10' or name == 'cifar100':     
        view_10_color(x,y) 
    else:
        raise ValueError(f"Unknown dataset name: {name}")
