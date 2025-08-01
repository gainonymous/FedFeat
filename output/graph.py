import numpy as np
import matplotlib.pyplot as plt

method = "CNN"
num_clients = 10

cnn = np.load(f'cifar100_acc_cnn_FedAvg_FedFeat_noniid_10.npy')
resnet = np.load(f'cifar100_acc_resnet_FedAvg_FedFeat_noniid_10.npy')

print(f"CNN {max(cnn)}")
print(f"ResNet {max(resnet)}")